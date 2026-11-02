# runner.py
import os
import sys
import json
import re
import math
import argparse
from typing import Dict, Any, List, Tuple

# Add project root to path for direct execution
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import gymnasium as gym
import numpy as np
import torch
import random
from gymnasium.wrappers import RecordVideo, TimeLimit
import highway_env  # noqa: F401  # Register environment

from src.utils.sampler import Sampler
from src.rule_system.scorer import RuleScorer
from src.planning.planner import CausalPlanner
from src.utils.common import load_yaml
from src.rule_system.storage import Neo4jWriter
from src.world_model.structure_learner import HybridStructureFunctionLearner
from src.utils.visualization import visualize_inference_results
from src.planning.retriever import LLMRuleRetriever
from config.config import CONFIG
from datetime import datetime


# =========================
# Utility Functions
# =========================
def get_ego_xy(ego: Dict[str, Any]) -> Tuple[float, float, float, float]:
    x = ego.get("x", ego.get("ego_x", 0.0))
    y = ego.get("y", ego.get("ego_y", 0.0))
    vx = ego.get("vx", ego.get("ego_vx", 0.0))
    vy = ego.get("vy", ego.get("ego_vy", 0.0))
    return float(x), float(y), float(vx), float(vy)

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_tokens(text: str) -> List[str]:
    text = re.sub(r"[^a-zA-Z0-9_]+", " ", str(text))
    return text.lower().split()


def flatten_state_from_sample(sample: Dict[str, Any]) -> List[float]:
    flat = []
    for macro_key in ["Environment", "Traffic", "Ego", "Action", "Outcome"]:
        macro = sample.get(macro_key, {})
        if isinstance(macro, dict):
            for v in macro.values():
                flat.append(float(v) if isinstance(v, (int, float)) else 0.0)
    return flat


def update_vocab(rule_path: str, variable_path: str, vocab_path: str):
    """Sync/expand vocab.json based on existing rules & variable names."""
    if os.path.exists(vocab_path):
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
    else:
        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}

    next_idx = max(vocab.values()) + 1 if vocab else 0
    tokens = set()

    if os.path.exists(rule_path):
        with open(rule_path, "r") as f:
            rules = json.load(f)
        for r in rules:
            for field in ["rule", "raw_text", "description", "trigger", "effect", "cause"]:
                val = r.get(field)
                if isinstance(val, str):
                    tokens.update(extract_tokens(val))
                elif isinstance(val, list):
                    for v in val:
                        tokens.update(extract_tokens(v))

    if os.path.exists(variable_path):
        import yaml
        with open(variable_path, "r") as f:
            var_dict = yaml.safe_load(f)
        for _, micros in var_dict.items():
            for v in micros:
                tokens.add(v.lower())

    for tok in sorted(tokens):
        if tok not in vocab:
            vocab[tok] = next_idx
            next_idx += 1

    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    print(f"[✓] Updated vocab.json with {len(tokens)} tokens (now {len(vocab)} entries).")


def _fmt_num(v, nd=2) -> str:
    """Safely format numbers: None or NaN -> 'nan'"""
    if v is None:
        return "nan"
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return f"{float(v):.{nd}f}"


# =========================
# Inference Runner
# =========================
class InferenceRunner:
    # highway-env MetaAction: [LANE_LEFT, LANE_RIGHT, IDLE, FASTER, SLOWER]
    ACTION_MAP_TO_ID = {
        "LANE_LEFT": 0, "lane_left": 0, "turn_left": 0, "go_left": 0,
        "LANE_RIGHT": 1, "lane_right": 1, "turn_right": 1, "go_right": 1,
        "IDLE": 2, "keep_lane": 2, "maintain": 2, "stay": 2, "idle": 2,
        "FASTER": 3, "accelerate": 3, "increase_speed": 3, "speed_up": 3, "faster": 3,
        "SLOWER": 4, "brake": 4, "decelerate": 4, "reduce_speed": 4, "slow_down": 4, "slower": 4,
    }

    def __init__(self, env, config_path: str):
        self.env = env
        self.config = load_yaml(config_path)

        # Configuration items
        self.top_k = int(self.config["planner"].get("top_k", 5))
        self.rule_path = self.config.get("rule_path", "data/extracted_rules.json")
        self.variable_cfg_path = "config/variable_config.yaml"
        self.vocab_path = self.config["rule_scorer"]["vocab_path"]
        self.model_ckpt = self.config["rule_scorer"]["model_path"]

        # 1) Sync vocab (offline rule library + variable names)
        print("[INFO] Syncing vocab.json with extracted rules...")
        update_vocab(self.rule_path, self.variable_cfg_path, self.vocab_path)

        # 2) Sample once to determine state_dim (consistent with training)
        obs0, _ = self.env.unwrapped.reset(seed=9999)
        self.sampler = Sampler(env=self.env)
        warm_sample = self.sampler.sample_episode(episode_id=-1, obs=obs0)
        state_dim = len(flatten_state_from_sample(warm_sample))
        print(f"[INFO] Detected state_dim={state_dim}")

        # 3) Module initialization
        self.structure_function = HybridStructureFunctionLearner()  # Optional for leaf heuristics
        self.rule_scorer = RuleScorer(state_dim=state_dim, vocab_path=self.vocab_path)
        self.rule_scorer.load_partial_state_dict(self.model_ckpt)
        self.rule_scorer.eval()

        # Use LLM + Neo4j rule retriever, replacing local rule_engine.match_rules
        self.rule_retriever = LLMRuleRetriever(
            uri=self.config["neo4j"]["uri"],
            user=self.config["neo4j"]["user"],
            password=self.config["neo4j"]["password"],
            llm_model=self.config.get("llm", {}).get("model", "gpt-4.1-nano"),
            max_rules=self.config.get("llm", {}).get("max_rules", 30),
            debug=True,
        )

        # Your MCTS/planner (internally uses PUCT to fuse rule priors, or has fallback strategy)
        self.planner = CausalPlanner(
            structure_function=self.structure_function,
            rule_engine=None,                 # Already replaced by retriever
            rule_scorer=self.rule_scorer,
            action_space=["LANE_LEFT", "LANE_RIGHT", "IDLE", "FASTER", "SLOWER"],
        )

        # Optional: Neo4j write-back for episode/step results
        if self.config.get("use_neo4j", False):
            self.neo4j_writer = Neo4jWriter(
                uri=self.config["neo4j"]["uri"],
                user=self.config["neo4j"]["user"],
                password=self.config["neo4j"]["password"]
            )
        else:
            self.neo4j_writer = None

    # -----------------------------
    # Rule Scoring -> TopK -> Candidate Actions
    # -----------------------------
    def _score_and_select_rules(self, sample: Dict[str, Any]) -> Tuple[List[Dict], List[str]]:
        # 1) Dynamic retrieval from Neo4j (LLM generates read-only Cypher)
        retrieved = self.rule_retriever.retrieve(
            sample, k=self.top_k * 3, target_groups=["Outcome", "Ego", "Traffic"]

        )

        # ==== Debug output: view LLM's Cypher prompt / query / params / matched rules preview ====
        try:
            dbg = self.rule_retriever.get_last_debug()
            print("\n[DEBUG] --- Cypher PROMPT ---\n", dbg.get("prompt", ""))
            print("\n[DEBUG] --- Cypher QUERY ---\n", dbg.get("cypher", ""))
            print("[DEBUG] Params:", json.dumps(dbg.get("params", {}), ensure_ascii=False, indent=2))
            print("[DEBUG] Fallback used:", dbg.get("fallback_used"))
            print("[DEBUG] Rows preview:", json.dumps(dbg.get("rows_preview", []), ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"[DEBUG] print retriever debug failed: {e}")
    # ===============================================================
        if not retrieved:
            return [], ["IDLE", "FASTER"]

        # 2) RuleScorer scores on "current state" (unified field name rule_text)
        for r in retrieved:
            r["rule_text"] = r.get("rule_text") or r.get("raw_text") or r.get("rule") or r.get("text") or ""

        scored_raw = self.rule_scorer.score(retrieved, [sample])

        # —— Uniformly enrich to [{"rule_text","score"}] —— #
        enriched = []
        if isinstance(scored_raw, (list, tuple)) and scored_raw and isinstance(scored_raw[0], (float, int)):
            for i, r in enumerate(retrieved):
                enriched.append({"rule_text": r["rule_text"], "score": float(scored_raw[i])})
        elif isinstance(scored_raw, (list, tuple)) and scored_raw and isinstance(scored_raw[0], dict):
            for i, it in enumerate(scored_raw):
                rt = it.get("rule_text") or retrieved[i]["rule_text"]
                sc = float(it.get("score", 0.0))
                enriched.append({"rule_text": rt, "score": sc})
        else:
            arr = np.array(scored_raw).reshape(-1)
            for i, r in enumerate(retrieved):
                enriched.append({"rule_text": r["rule_text"], "score": float(arr[i])})

        top_rules = sorted(enriched, key=lambda x: x["score"], reverse=True)[: self.top_k]

        # 3) Rules -> Candidate actions (deduplicated)
        actions = []
        for r in top_rules:
            a = self._parse_rule_to_action_name(r.get("rule_text", ""))
            if a and a not in actions:
                actions.append(a)
        if not actions:
            actions = ["IDLE", "FASTER"]
        # Optional: print candidate actions and top rules for this step
        print("[DEBUG] candidate_actions:", actions)
        print("[DEBUG] top_rules(head):", json.dumps(top_rules[:5], ensure_ascii=False, indent=2))
        return top_rules, actions

    @staticmethod
    def _normalize_action_text(text: str) -> str:
        t = (text or "").lower()
        if any(k in t for k in ["brake", "decelerate", "reduce speed", "slow down", "slowing down", "slower"]):
            return "SLOWER"
        if any(k in t for k in ["accelerate", "increase speed", "increases", "speed up", "faster"]):
            return "FASTER"
        if any(k in t for k in ["lane change left", "turn left", "go left", "lane_left"]):
            return "LANE_LEFT"
        if any(k in t for k in ["lane change right", "turn right", "go right", "lane_right"]):
            return "LANE_RIGHT"
        if any(k in t for k in ["keep lane", "maintain", "maintaining","stay", "idle"]):
            return "IDLE"
        return "IDLE"

    def _parse_rule_to_action_name(self, rule_text: str) -> str:
        return self._normalize_action_text(rule_text)

    def _action_name_to_id(self, name: str) -> int:
        return self.ACTION_MAP_TO_ID.get(name, 3)  # Default FASTER to prevent stalling

    # -----------------------------
    # Single Episode Inference (max 20 steps, ends earlier on collision; logs once per step)
    # -----------------------------
    def run_episode(self, episode_id: int, max_safe_actions: int = 70) -> Dict[str, Any]:
        obs, _ = self.env.reset(seed=1000 + episode_id)
        done = False
        trajectory = []
        history_frames: List[Dict[str, Any]] = []
        step_count = 0
        total_reward = 0.0

        # —— Maintain "predicted next step vs actual H steps" full sequence —— #
        H = 10                       # Evaluation lookahead steps (adjustable)
        pred_queue = []              # Save predictions "for H steps ahead"
        true_hist: List[np.ndarray] = []   # Actual position sequence (only real)
        matured_errs: List[float] = []     # Matured (due) errors for ADE/FDE


        # Inference timestep dt
        cfg = getattr(self.env, "config", None) or getattr(self.env.unwrapped, "config", {})
        pol_f = float(cfg.get("policy_frequency", 15) or 15)
        dt = 1.0 / pol_f if pol_f > 0 else 1.0 / 15.0

        while (not done) and (step_count < max_safe_actions):
            # 1) Sample current structured state
            sample = self.sampler.sample_episode(episode_id=episode_id, obs=obs, last_action_id=None)

            # Maintain history
            history_frames.append(sample)
            if hasattr(self.structure_function, "max_lag"):
                history_frames = history_frames[-self.structure_function.max_lag:]
            else:
                history_frames = history_frames[-3:]

            # 2) Dynamic rule retrieval -> scoring -> candidate action set
            top_rules, candidate_action_names = self._score_and_select_rules(sample)

            # 3) Planning (MCTS/PUCT), fallback to greedy rule action on failure
            try:
                plan_out = self.planner.plan(
                    sample=sample,
                    candidate_actions=candidate_action_names,
                    top_rules=top_rules,
                    top_k=self.top_k,
                    history_frames=history_frames
                )

                action_name = plan_out[0] if isinstance(plan_out, list) else plan_out
                if not isinstance(action_name, str):
                    inv = {v: k for k, v in self.ACTION_MAP_TO_ID.items()}
                    action_name = inv.get(int(action_name), "IDLE")
            except Exception as e:
                print(f"[WARN] Planner failed, fallback to greedy: {e}")
                action_name = candidate_action_names[0]

            print("[DBG] candidates:", candidate_action_names[:5])
            print("[DBG] chosen by greedy:", action_name)

            action_id = self._action_name_to_id(action_name)

            # —— Use current frame to "extrapolate H steps ahead" position as prediction —— #
            ego_now = sample.get("Ego", {})
            x0, y0, vx0, vy0 = get_ego_xy(ego_now)
            pred_xy_H = np.array([x0 + vx0 * dt * H, y0 + vy0 * dt * H], dtype=float)
            pred_queue.append(pred_xy_H)

            # 4) Execute action (Gymnasium five-tuple unpacking)
            out = self.env.step(action_id)
            if len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = terminated or truncated
            else:
                obs, reward, done, info = out

            total_reward += float(reward)

            # Sample "next frame" structured state, record actual position
            next_sample = self.sampler.sample_episode(episode_id=episode_id, obs=obs, last_action_id=action_id)
            ego_next = next_sample.get("Ego", {})
            x1, y1, _, _ = get_ego_xy(ego_next)
            true_next_xy = np.array([x1, y1], dtype=float)
            true_hist.append(true_next_xy)

            # When actual trajectory reaches "predicted moment", settle error (prediction from t-H compared to current actual)
            if len(true_hist) >= H:
                pred_due = pred_queue[-H]        # Prediction from H steps ago (aligned to current moment)
                true_xy_H = true_hist[-1]        # Current actual position
                matured_errs.append(np.linalg.norm(pred_due - true_xy_H))

            # Online calculation of current ADE/FDE (based on matured errors)
            ade_running = float(np.mean(matured_errs)) if matured_errs else float("nan")
            fde_running = float(matured_errs[-1]) if matured_errs else float("nan")


            # —— Log: once per step —— #
            print(f"[DEBUG] ep={episode_id:02d} step={step_count:03d} "
                  f"act={action_name}({action_id}) reward={float(reward):.3f} "
                  f"Rsum={total_reward:.3f} ADE={_fmt_num(ade_running, 3)} FDE={_fmt_num(fde_running, 3)}")

            trajectory.append({
                "step": step_count,
                "sample": sample,
                "top_rules": top_rules,
                "candidate_actions": candidate_action_names,
                "chosen_action_name": action_name,
                "chosen_action_id": action_id,
                "reward": float(reward),
                "running_reward": float(total_reward),
                "ADE_step": ade_running,
                "FDE_step": fde_running,
                "done": bool(done)
            })

            step_count += 1

            # Early termination condition: reached 20 steps (max_safe_actions) or environment reports done (collision/termination)
            if step_count >= max_safe_actions:
                # No collision, end naturally at 20 steps
                break

        # —— Episode end: calculate "proper" ADE / FDE using full sequence —— #
        # —— Episode end: ADE / FDE based on matured errors (multi-step lookahead) —— #
        if matured_errs:
            ade_final = float(np.mean(matured_errs))
            fde_final = float(matured_errs[-1])
        else:
            ade_final = float("nan")
            fde_final = float("nan")


        result = {
            "episode_id": episode_id,
            "trajectory": trajectory,
            "total_reward": float(total_reward),
            "steps": step_count,
            "ADE": ade_final,
            "FDE": fde_final
        }

        if self.neo4j_writer:
            try:
                self.neo4j_writer.write_episode(result)
            except Exception as e:
                print(f"[WARN] Neo4j write failed: {e}")

        return result

    # -----------------------------
    # Multiple Episodes
    # -----------------------------
    def run_multiple(self, n_episodes: int, max_safe_actions: int = 20) -> List[Dict[str, Any]]:
        results = []
        for ep in range(n_episodes):
            print(f"\n[Runner] Running Episode {ep}")
            res = self.run_episode(ep, max_safe_actions=max_safe_actions)
            results.append(res)
            print(f"[✓] Episode {ep} completed: "
                  f"reward={_fmt_num(res.get('total_reward'), 2)}, "
                  f"steps={res.get('steps')}, "
                  f"ADE={_fmt_num(res.get('ADE'), 3)}, "
                  f"FDE={_fmt_num(res.get('FDE'), 3)}")
        return results


# =========================
# main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/inference_runner.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_safe_actions", type=int, default=65, help="Maximum actions (step limit) without collision")
    args = parser.parse_args()

    # ==== Video uses "action count" as limit (not seconds) ====
    MAX_SAFE_ACTIONS = int(args.max_safe_actions)

    # Rendering/control frequency (for rendering smoothness, length controlled by step count)
    POLICY_HZ = 15
    SIM_HZ = 15

    # Environment parameters: duration set to 20 (consistent with MAX_SAFE_ACTIONS), slightly reduced resolution + offscreen rendering to avoid lag
    env_cfg = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["x", "y", "vx", "vy"],
        },
        "duration": MAX_SAFE_ACTIONS,       # With TimeLimit, constrained to "20 steps"
        "policy_frequency": POLICY_HZ,
        "simulation_frequency": SIM_HZ,
        "screen_width": 800,
        "screen_height": 270,
        "offscreen_rendering": True,
        "initial_lane_id": "random",
        "initial_spacing": "random",
        "ego_spacing": "random",
    }


    raw_env = gym.make(CONFIG["env_name"], render_mode="rgb_array", config=env_cfg)
    set_global_seed(20241025)
    # Let recorder render at 15 FPS to avoid lag
    if not hasattr(raw_env, "metadata") or raw_env.metadata is None:
        raw_env.metadata = {}
    raw_env.metadata["render_fps"] = POLICY_HZ

    # TimeLimit: double insurance, max 20 steps
    raw_env = TimeLimit(raw_env, max_episode_steps=MAX_SAFE_ACTIONS)

    # Video recording
    video_root = f"result/videos/{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(video_root, exist_ok=True)
    env = RecordVideo(
        raw_env,
        video_folder=video_root,
        name_prefix="episode",
        episode_trigger=lambda ep: ep < args.episodes,
    )

    try:
        runner = InferenceRunner(env, args.config)
        results = runner.run_multiple(args.episodes, max_safe_actions=MAX_SAFE_ACTIONS)

        # Write results
        result_dir = "result/inference"
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, "inference_results.json")

        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[✓] Inference results saved to {result_path}")

        # Visualization
        try:
            visualize_inference_results(result_path, output_dir=result_dir)
            print("[✓] Visualization images generated")
        except Exception as e:
            print(f"[WARN] Visualization failed: {e}")

    finally:
        # Important: close env to trigger RecordVideo proper disk write
        try:
            env.close()
        except Exception:
            pass
