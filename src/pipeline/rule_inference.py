# runner.py
import os
import sys
import json
import re
import math
import copy
import time
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
from src.rule_system.acquirer import OnlineRuleAcquirer
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


# Replacement for flatten_state_from_sample
def flatten_state_from_sample(sample: Dict[str, Any]) -> List[float]:
    """
    Flatten state in a deterministic order to ensure alignment with training state_dim.
    Prioritize the order from variable_config.yaml; fallback to built-in order if missing.
    """
    from src.utils.common import load_yaml
    try:
        var_cfg = load_yaml("config/variable_config.yaml") or {}
    except Exception:
        var_cfg = {}

    FEATURE_ORDER = {
        "Environment": var_cfg.get(
            "Environment",
            ["weather_type", "visibility", "road_friction", "road_type", "time_of_day", "speed_limit"],
        ),
        "Traffic": var_cfg.get(
            "Traffic",
            [
                "headway_distance",
                "lead_speed",
                "lead_rel_speed",
                "lead_ttc",
                "lead_thw",
                "rear_distance",
                "rear_speed",
                "rear_rel_speed",
                "rear_ttc",
                "left_front_distance",
                "left_front_speed",
                "left_rear_distance",
                "left_rear_speed",
                "right_front_distance",
                "right_front_speed",
                "right_rear_distance",
                "right_rear_speed",
                "lane_density_left",
                "lane_density_right",
                "available_gap_left",
                "available_gap_right",
                "cutin_risk",
            ],
        ),
        "Ego": var_cfg.get(
            "Ego",
            [
                "ego_x",
                "ego_y",
                "ego_vx",
                "ego_vy",
                "ego_speed",
                "ego_acceleration",
                "yaw_rate",
                "jerk",
                "lane_id",
                "lateral_offset",
            ],
        ),
        "Action": var_cfg.get("Action", ["action_type"]),
        "Outcome": var_cfg.get(
            "Outcome",
            ["risk_score", "comfort_score", "energy_consumption", "success_flag", "violation_flag"],
        ),
    }

    flat: List[float] = []
    for group in ["Environment", "Traffic", "Ego", "Action", "Outcome"]:
        g = sample.get(group, {}) or {}
        for key in FEATURE_ORDER[group]:
            v = g.get(key, 0.0)
            try:
                flat.append(float(v))
            except Exception:
                flat.append(0.0)
    return flat


def update_vocab(rule_path: str, variable_path: str, vocab_path: str):
    """Sync/extend vocab.json based on existing rules & variable names."""
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
            # Include action as well
            for field in ["rule", "raw_text", "description", "trigger", "effect", "cause", "action"]:
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

    print(f"[Done] Updated vocab.json with {len(tokens)} tokens (now {len(vocab)} entries).")


def _fmt_num(v, nd=2) -> str:
    """Safely format numeric value: None or NaN -> 'nan'"""
    if v is None:
        return "nan"
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return f"{float(v):.{nd}f}"


def estimate_ttc_from_kinematics_obs(obs: np.ndarray, lane_width: float = 4.0) -> float:
    """
    Estimate TTC (Time-To-Collision with lead vehicle in same lane) from highway-env Kinematics observation.
    Convention: obs[0] is ego; features=["x","y","vx","vy"] with absolute=True.
    TTC = dx / (v_ego - v_lead) when dx>0 and v_ego>v_lead, else inf.
    """
    try:
        arr = np.asarray(obs, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 3:
            return float("inf")

        ex, ey, evx = arr[0, 0], arr[0, 1], arr[0, 2]

        best_dx = None
        best_lead_vx = None
        for i in range(1, arr.shape[0]):
            x, y, vx = arr[i, 0], arr[i, 1], arr[i, 2]
            dx = x - ex
            dy = y - ey
            if dx <= 0:
                continue
            if abs(dy) > (lane_width * 0.5):
                continue
            if (best_dx is None) or (dx < best_dx):
                best_dx = dx
                best_lead_vx = vx

        if best_dx is None or best_lead_vx is None:
            return float("inf")

        rel = evx - best_lead_vx
        if rel <= 1e-6:
            return float("inf")
        return float(best_dx / rel)

    except Exception:
        return float("inf")


def get_crash_flag(info: dict, env) -> int:
    """Check if episode crashed: prefer info['crashed'], otherwise try env.unwrapped.vehicle.crashed"""
    try:
        if isinstance(info, dict) and ("crashed" in info):
            return int(bool(info.get("crashed")))
    except Exception:
        pass
    try:
        v = getattr(getattr(env, "unwrapped", env), "vehicle", None)
        if v is not None and hasattr(v, "crashed"):
            return int(bool(v.crashed))
    except Exception:
        pass
    return 0


# =========================
# Inference Runner
# =========================
class InferenceRunner:
    # highway-env MetaAction: [LANE_LEFT, LANE_RIGHT, IDLE, FASTER, SLOWER]
    ACTION_MAP_TO_ID = {
        "LANE_LEFT": 0,
        "lane_left": 0,
        "turn_left": 0,
        "go_left": 0,
        "LANE_RIGHT": 1,
        "lane_right": 1,
        "turn_right": 1,
        "go_right": 1,
        "IDLE": 2,
        "keep_lane": 2,
        "maintain": 2,
        "stay": 2,
        "idle": 2,
        "FASTER": 3,
        "accelerate": 3,
        "increase_speed": 3,
        "speed_up": 3,
        "faster": 3,
        "SLOWER": 4,
        "brake": 4,
        "decelerate": 4,
        "reduce_speed": 4,
        "slow_down": 4,
        "slower": 4,
    }

    def __init__(self, env, config_path: str):
        self.env = env
        self.config = load_yaml(config_path)

        # === NEW: Planning/reward related default hyperparameters (can be overridden in config) ===
        self.horizon = int(self.config.get("planner", {}).get("horizon", 8))
        self.gamma = float(self.config.get("planner", {}).get("gamma", 0.98))
        rwd_cfg = self.config.get("reward", {})
        self.w_target = float(rwd_cfg.get("w_target_speed", 0.10))
        self.w_risk = float(rwd_cfg.get("w_risk", 1.00))
        self.w_energy = float(rwd_cfg.get("w_energy", 0.20))
        self.w_comfort = float(rwd_cfg.get("w_comfort", 0.10))
        self.speed_target = float(rwd_cfg.get("speed_target", 31.0))
        self.hard_risk_th = float(rwd_cfg.get("hard_risk_th", 0.95))
        self.hard_penalty = float(rwd_cfg.get("hard_penalty", 5.0))
        self.lane_step = float(rwd_cfg.get("lane_step", 4.0))  # Approximate y spacing per lane

        # === NEW: Hyperparameters for RE / PE (can be overridden in config) ===
        mcfg = self.config.get("metrics", {})
        self.delta_ttc = float(mcfg.get("delta_ttc", 3.0))  # delta
        self.lambda_risk = float(mcfg.get("lambda_risk", 1.0))  # lambda
        self.lane_width = float(mcfg.get("lane_width", self.lane_step))

        # Configuration items
        self.top_k = int(self.config["planner"].get("top_k", 5))
        self.rule_path = self.config.get("rule_path", "data/extracted_rules.json")
        self.variable_cfg_path = "config/variable_config.yaml"
        self.vocab_path = self.config["rule_scorer"]["vocab_path"]
        self.model_ckpt = self.config["rule_scorer"]["model_path"]

        prior_cfg = self.config.get("prior", {})
        self.use_action_head_prior = bool(prior_cfg.get("use_action_head_prior", False))
        self.prior_alpha = float(prior_cfg.get("alpha", 1.0))  # Prior strength
        self.prior_top_rules = int(prior_cfg.get("top_rules", self.top_k))
        self.prior_top_actions = int(prior_cfg.get("top_actions", 5))  # Number of actions entering MPC

        self.action_bias = {
            "IDLE": -3.0,  # Explicitly penalize prolonged inaction
            "FASTER": +1.3,  # Moderately encourage acceleration (bounded by risk constraints)
            "SLOWER": -1.1,  # Slight penalty (avoid constant braking)
            "LANE_LEFT": +2.6,  # Encourage meaningful lane changes
            "LANE_RIGHT": +2.6,
        }

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
        self.structure_function = HybridStructureFunctionLearner()  # F_i: World model / metric model
        self.rule_scorer = RuleScorer(state_dim=state_dim, vocab_path=self.vocab_path)
        self.rule_scorer.load_partial_state_dict(self.model_ckpt)
        self.rule_scorer.eval()

        # Rule retriever using LLM + Neo4j
        self.rule_retriever = LLMRuleRetriever(
            uri=self.config["neo4j"]["uri"],
            user=self.config["neo4j"]["user"],
            password=self.config["neo4j"]["password"],
            llm_model=self.config.get("llm", {}).get("model", "gpt-4.1-nano"),
            max_rules=self.config.get("llm", {}).get("max_rules", 1000),
            debug=True,
        )

        # Your MCTS/planner
        self.planner = CausalPlanner(
            structure_function=self.structure_function,
            rule_engine=None,
            rule_scorer=self.rule_scorer,
            action_space=["LANE_LEFT", "LANE_RIGHT", "IDLE", "FASTER", "SLOWER"],
            hard_risk_th=self.hard_risk_th,
            hard_penalty=self.hard_penalty,
            speed_target=self.speed_target,
            w_target=self.w_target,
            w_risk=self.w_risk,
            w_energy=self.w_energy,
            w_comfort=self.w_comfort,
            lane_step=getattr(self, "lane_step", 4.0),
            action_bias=self.action_bias,
        )

        # Optional: Neo4j write back episode/step results
        if self.config.get("use_neo4j", False):
            self.neo4j_writer = Neo4jWriter(
                uri=self.config["neo4j"]["uri"],
                user=self.config["neo4j"]["user"],
                password=self.config["neo4j"]["password"],
            )
        else:
            self.neo4j_writer = None

        # Online Rule Acquisition (self-evolution loop, Section 3.2.2)
        if self.neo4j_writer:
            acq_cfg = self.config.get("acquirer", {})
            self.acquirer = OnlineRuleAcquirer(
                neo4j_writer=self.neo4j_writer,
                llm_model=self.config.get("llm", {}).get("model", "gpt-4.1-nano"),
                miss_eta=float(acq_cfg.get("miss_eta", 0.0)),
                verify_horizon=int(acq_cfg.get("verify_horizon", 5)),
                window_size=int(acq_cfg.get("window_size", 50)),
                novelty_threshold=float(acq_cfg.get("novelty_threshold", 0.85)),
            )
        else:
            self.acquirer = None

    # -----------------------------
    # Rule scoring -> TopK -> Candidate actions
    # -----------------------------
    def _score_and_select_rules(self, sample: Dict[str, Any]) -> Tuple[List[Dict], List[str]]:
        retrieved = self.rule_retriever.retrieve(
            sample, k=self.top_k * 3, target_groups=["Outcome", "Ego", "Traffic"]
        )

        if not retrieved:
            return [], ["IDLE", "FASTER"]

        for r in retrieved:
            r["rule_text"] = r.get("rule_text") or r.get("raw_text") or r.get("rule") or r.get("text") or ""
            if "action" not in r:
                r["action"] = ""

        scored_raw = self.rule_scorer.score(retrieved, [sample])

        enriched: List[Dict[str, Any]] = []
        if isinstance(scored_raw, (list, tuple)) and scored_raw and isinstance(scored_raw[0], (float, int)):
            for i, r in enumerate(retrieved):
                enriched.append(
                    {
                        "rule_text": r["rule_text"],
                        "raw_text": r.get("raw_text", r["rule_text"]),
                        "score": float(scored_raw[i]),
                        "action": r.get("action", ""),
                        "rule_id": r.get("rule_id", None),
                        "rule_type": r.get("rule_type", None),
                    }
                )
        elif isinstance(scored_raw, (list, tuple)) and scored_raw and isinstance(scored_raw[0], dict):
            for i, it in enumerate(scored_raw):
                enriched.append(
                    {
                        "rule_text": it.get("rule_text") or retrieved[i]["rule_text"],
                        "score": float(it.get("score", 0.0)),
                        "action": retrieved[i].get("action", ""),
                        "rule_id": retrieved[i].get("rule_id", None),
                        "rule_type": retrieved[i].get("rule_type", None),
                    }
                )
        else:
            arr = np.array(scored_raw).reshape(-1)
            for i, r in enumerate(retrieved):
                enriched.append(
                    {
                        "rule_text": r["rule_text"],
                        "score": float(arr[i]),
                        "action": r.get("action", ""),
                        "rule_id": r.get("rule_id", None),
                        "rule_type": r.get("rule_type", None),
                    }
                )

        top_rules = sorted(enriched, key=lambda x: x["score"], reverse=True)[: self.top_k]

        actions: List[str] = []
        for r in top_rules:
            a = (self.extract_action_from_rule(r) or "").upper()
            if a in self.ACTION_MAP_TO_ID and a not in actions:
                actions.append(a)

        filtered_actions: List[str] = []
        for a in actions:
            if a == "SLOWER" and (not self.safe_to_slow(sample)):
                continue
            filtered_actions.append(a)
        actions = filtered_actions

        if not actions:
            actions = ["IDLE", "FASTER"]

        if ("LANE_LEFT" not in actions) and ("LANE_RIGHT" not in actions):
            actions += ["LANE_LEFT", "LANE_RIGHT"]

        actions = list(dict.fromkeys(actions))

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
        if any(k in t for k in ["keep lane", "maintain", "maintaining", "stay", "idle"]):
            return "IDLE"
        return "IDLE"

    def try_parse_rule_json(self, text: str) -> dict:
        try:
            s = (text or "").strip()
            if s.startswith("{") and s.endswith("}"):
                return json.loads(s)
        except Exception:
            pass
        return {}

    def extract_action_from_rule(self, r: dict) -> str:
        a = (r.get("action") or "").strip()
        if a:
            return a
        data = self.try_parse_rule_json(r.get("rule_text", r.get("raw_text", "")) or "")
        a = (data.get("action") or "").strip() if isinstance(data, dict) else ""
        if a:
            return a
        return self._parse_rule_to_action_name(r.get("rule_text", ""))

    def safe_to_slow(self, sample: Dict[str, Any]) -> bool:
        T = sample.get("Traffic", {})
        rear_d = float(T.get("rear_distance", 1e9))
        rear_ttc = float(T.get("rear_ttc", 1e9))
        rear_rel = float(T.get("rear_rel_speed", 0.0))  # >0 means rear vehicle is faster
        return (rear_d > 25.0) and (rear_ttc > 4.0) and (rear_rel <= 0.0)

    def _parse_rule_to_action_name(self, rule_text: str) -> str:
        return self._normalize_action_text(rule_text)

    def _action_name_to_id(self, name: str) -> int:
        return self.ACTION_MAP_TO_ID.get(name, 3)  # Default FASTER to prevent stalling

    # === NEW: Compact state & prediction/reward utilities ===
    def _compact_state(self, sample: Dict[str, Any]) -> Dict[str, float]:
        T = sample.get("Traffic", {})
        E = sample.get("Ego", {})
        O = sample.get("Outcome", {})
        return {
            "ego_x": float(E.get("ego_x", 0.0)),
            "ego_y": float(E.get("ego_y", 0.0)),
            "ego_vx": float(E.get("ego_vx", E.get("ego_speed", 0.0))),
            "ego_vy": float(E.get("ego_vy", 0.0)),
            "headway_distance": float(T.get("headway_distance", 1e6)),
            "lead_ttc": float(T.get("lead_ttc", 1e9)),
            "lead_rel_speed": float(T.get("lead_rel_speed", 0.0)),
            "rear_distance": float(T.get("rear_distance", 1e6)),
            "risk_score": float(O.get("risk_score", 0.5)),
            "energy_consumption": float(O.get("energy_consumption", 0.1)),
            "comfort_score": float(O.get("comfort_score", 0.8)),
        }

    def _apply_action_dynamics_fallback(self, s: Dict[str, float], action: str, dt: float):
        ax = 0.0
        if action == "FASTER":
            ax = +1.5
        elif action == "SLOWER":
            ax = -1.5

        vx = max(0.0, s["ego_vx"] + ax * dt)
        x = s["ego_x"] + vx * dt

        vy = s["ego_vy"]
        y = s["ego_y"]
        if action == "LANE_LEFT":
            vy = 1.0
            y = y + self.lane_step * 0.25
        elif action == "LANE_RIGHT":
            vy = -1.0
            y = y - self.lane_step * 0.25

        s["ego_x"], s["ego_y"], s["ego_vx"], s["ego_vy"] = x, y, vx, vy
        s["lead_ttc"] = max(1.0, s["lead_ttc"] + (1.0 if ax <= 0 else -1.0) * 50.0)
        s["headway_distance"] = max(1.0, s["headway_distance"] + (1.0 if ax <= 0 else -1.0) * 2.0)

        risk = 1.0 / (1.0 + s["lead_ttc"] / 1000.0)
        energy = max(0.0, 0.05 + (s["ego_vx"] ** 2) * 0.0005 + (abs(ax) * 0.02))
        comfort = max(0.0, min(1.0, 1.0 - abs(ax) * 0.05 - abs(vy) * 0.02))

        s["risk_score"] = float(risk)
        s["energy_consumption"] = float(energy)
        s["comfort_score"] = float(comfort)

    def _predict_one_step_with_Fi(
        self, sample: Dict[str, Any], action: str, dt: float, history_frames: List[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        s = self._compact_state(sample)
        try:
            if hasattr(self.structure_function, "predict_next"):
                next_struct = self.structure_function.predict_next(
                    sample=sample, action=action, dt=dt, history_frames=history_frames
                )

                def _get2(d, g, k, default):
                    try:
                        v = d.get(g, {}).get(k, default)
                        return float(v) if isinstance(v, (int, float)) else default
                    except Exception:
                        return default

                s["ego_x"] = _get2(next_struct, "Ego", "ego_x", s["ego_x"])
                s["ego_y"] = _get2(next_struct, "Ego", "ego_y", s["ego_y"])
                s["ego_vx"] = _get2(next_struct, "Ego", "ego_vx", s["ego_vx"])
                s["ego_vy"] = _get2(next_struct, "Ego", "ego_vy", s["ego_vy"])

                s["headway_distance"] = _get2(next_struct, "Traffic", "headway_distance", s["headway_distance"])
                s["lead_ttc"] = _get2(next_struct, "Traffic", "lead_ttc", s["lead_ttc"])
                s["lead_rel_speed"] = _get2(next_struct, "Traffic", "lead_rel_speed", s["lead_rel_speed"])
                s["rear_distance"] = _get2(next_struct, "Traffic", "rear_distance", s["rear_distance"])

                s["risk_score"] = _get2(next_struct, "Outcome", "risk_score", s["risk_score"])
                s["energy_consumption"] = _get2(next_struct, "Outcome", "energy_consumption", s["energy_consumption"])
                s["comfort_score"] = _get2(next_struct, "Outcome", "comfort_score", s["comfort_score"])

                for k in list(s.keys()):
                    if not np.isfinite(s[k]):
                        s[k] = self._compact_state(sample).get(k, s[k])
            else:
                self._apply_action_dynamics_fallback(s, action, dt)
        except Exception as e:
            print(f"[WARN] F_i prediction failed, fallback used: {e}")
            self._apply_action_dynamics_fallback(s, action, dt)
        return s

    def _step_reward(self, s: Dict[str, float]) -> float:
        speed = float(s["ego_vx"])
        risk = float(s["risk_score"])
        energy = float(s["energy_consumption"])
        comfort = float(s["comfort_score"])

        r_target = -abs(speed - self.speed_target) * self.w_target
        r_risk = -risk * self.w_risk
        r_energy = -energy * self.w_energy
        r_comfort = +comfort * self.w_comfort

        r = r_target + r_risk + r_energy + r_comfort
        if risk >= self.hard_risk_th:
            r -= self.hard_penalty
        return float(r)

    def _greedy_plan_sequence(self, sample, candidate_actions, history_frames=None, H=None):
        H = int(H or self.horizon)
        dt = 1.0 / float(
            (getattr(self.env, "config", {}) or getattr(self.env.unwrapped, "config", {})).get("policy_frequency", 15)
            or 15
        )
        s_sim = copy.deepcopy(sample)
        hf = list(history_frames or [])
        seq = []
        for _ in range(H):
            best_a, best_v = None, -1e9
            for a in (candidate_actions or ["IDLE", "FASTER", "SLOWER", "LANE_LEFT", "LANE_RIGHT"]):
                s_next = self._predict_one_step_with_Fi(s_sim, a, dt, history_frames=hf)
                v = self._step_reward(s_next) + self.action_bias.get(a, 0.0)
                if v > best_v:
                    best_v, best_a = v, a
            seq.append(best_a or "IDLE")
            s_next = self._predict_one_step_with_Fi(s_sim, seq[-1], dt, history_frames=hf)
            s_sim = {
                "Ego": {
                    "ego_x": s_next["ego_x"],
                    "ego_y": s_next["ego_y"],
                    "ego_vx": s_next["ego_vx"],
                    "ego_vy": s_next["ego_vy"],
                    "ego_speed": s_next["ego_vx"],
                },
                "Traffic": {
                    "headway_distance": s_next["headway_distance"],
                    "lead_ttc": s_next["lead_ttc"],
                    "lead_rel_speed": s_next["lead_rel_speed"],
                    "rear_distance": s_next["rear_distance"],
                },
                "Outcome": {
                    "risk_score": s_next["risk_score"],
                    "energy_consumption": s_next["energy_consumption"],
                    "comfort_score": s_next["comfort_score"],
                },
            }
            hf = (hf + [s_sim])[-getattr(self.structure_function, "max_lag", 3) :]
        return seq

    # -----------------------------
    # Single episode inference (up to max_safe_actions steps)
    # -----------------------------
    def run_episode(self, episode_id: int, max_safe_actions: int = 100) -> Dict[str, Any]:
        obs, _ = self.env.reset(seed=1000 + episode_id + getattr(self, "seed_offset", 0) * 10000)
        # Self-evolution: at episode boundary, migrate current ep buffer to prior-window
        if self.acquirer is not None:
            self.acquirer.reset_episode()
        done = False
        trajectory = []
        history_frames: List[Dict[str, Any]] = []
        current_plan: List[str] = []
        plan_ptr: int = 0
        step_count = 0
        total_reward = 0.0
        REPLAN_EVERY = 3

        # ===== Metrics accumulators =====
        plan_time_total = 0.0  # T_plan
        risk_accum = 0.0  # R_risk
        compl_viol_sum = 0.0  # Sum of v_{k,t}
        compl_denom_sum = 0.0  # Sum of K_t (total number of evaluable rules)
        crashed_ep = 0  # CR episode flag

        def need_replan(sample: Dict[str, Any]) -> bool:
            if plan_ptr >= len(current_plan):
                return True
            O = sample.get("Outcome", {})
            risk_now = float(O.get("risk_score", 0.0))
            if risk_now >= self.hard_risk_th:
                return True
            return (step_count % REPLAN_EVERY) == 0

        # -- ADE / FDE evaluation (your version) -- #
        H_eval = max(4, self.horizon)
        stride = max(1, H_eval // 1)
        pending_preds = []  # Each item: {"t_gt": step_idx, "pred_seq": np.ndarray[[H,2]]}
        per_t_ade: List[float] = []
        per_t_fde: List[float] = []
        true_hist: List[np.ndarray] = []

        cfg = getattr(self.env, "config", None) or getattr(self.env.unwrapped, "config", {})
        pol_f = float(cfg.get("policy_frequency", 15) or 15)
        dt = 1.0 / pol_f if pol_f > 0 else 1.0 / 15.0

        def rollout_positions_from(sample0, planned_actions, H, dt_local, history_frames_local):
            seq_pred = []
            s = copy.deepcopy(sample0)
            hf = list(history_frames_local or [])
            acts = list(planned_actions or [])
            if len(acts) < H:
                acts = acts + ["IDLE"] * (H - len(acts))
            for _ in range(H):
                a = acts[_]
                s_next = self._predict_one_step_with_Fi(s, a, dt_local, history_frames=hf)
                seq_pred.append([s_next["ego_x"], s_next["ego_y"]])
                s = {
                    "Ego": {
                        "ego_x": s_next["ego_x"],
                        "ego_y": s_next["ego_y"],
                        "ego_vx": s_next["ego_vx"],
                        "ego_vy": s_next["ego_vy"],
                        "ego_speed": s_next["ego_vx"],
                    },
                    "Traffic": {
                        "headway_distance": s_next["headway_distance"],
                        "lead_ttc": s_next["lead_ttc"],
                        "lead_rel_speed": s_next["lead_rel_speed"],
                        "rear_distance": s_next["rear_distance"],
                    },
                    "Outcome": {
                        "risk_score": s_next["risk_score"],
                        "energy_consumption": s_next["energy_consumption"],
                        "comfort_score": s_next["comfort_score"],
                    },
                }
                hf = (hf + [s])[-getattr(self.structure_function, "max_lag", 3) :]
            return np.asarray(seq_pred, dtype=float)

        while (not done) and (step_count < max_safe_actions):
            # ===== T_plan timing start (excluding env.step) =====
            t_plan0 = time.perf_counter()

            sample = self.sampler.sample_episode(episode_id=episode_id, obs=obs, last_action_id=None)

            history_frames.append(sample)
            if hasattr(self.structure_function, "max_lag"):
                history_frames = history_frames[-self.structure_function.max_lag :]
            else:
                history_frames = history_frames[-3:]

            top_rules, candidate_action_names = self._score_and_select_rules(sample)

            # === Self-evolution: acquire new rule on miss (Section 3.2.2) ===
            llm_invoked_this_step = False
            if self.acquirer:
                self.acquirer.total_steps += 1
                self.acquirer.update_window(sample)
                if self.acquirer.is_miss(top_rules):
                    llm_invoked_this_step = True
                    _predict = lambda s, a: self._predict_one_step_with_Fi(
                        s, a, dt, history_frames=history_frames
                    )
                    new_rule = self.acquirer.try_acquire(sample=sample, predict_fn=_predict)
                    if new_rule:
                        top_rules = [new_rule] + top_rules
                        a_new = new_rule.get("action", "").upper()
                        if a_new in self.ACTION_MAP_TO_ID and a_new not in candidate_action_names:
                            candidate_action_names = [a_new] + candidate_action_names

            # === action prior (preserve original logic) ===
            action_prior = {a: 0.0 for a in candidate_action_names}
            if self.use_action_head_prior and top_rules:
                rule_texts = [r.get("raw_text", r.get("rule_text", "")) for r in top_rules[: self.prior_top_rules]]
                try:
                    device = next(self.rule_scorer.parameters()).device
                    state_vec = torch.tensor([flatten_state_from_sample(sample)], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        for a in candidate_action_names:
                            a_idx = self.rule_scorer.get_action_index(a)
                            chunk = 256
                            vals = []
                            for i in range(0, len(rule_texts), chunk):
                                rlogit, alogit = self.rule_scorer(state_vec, rule_texts[i : i + chunk])
                                s_rule = torch.sigmoid(rlogit).flatten()
                                s_act = torch.softmax(alogit, dim=-1)[:, a_idx]
                                vals.append((s_rule * s_act).mean().item())
                            action_prior[a] = float(np.mean(vals)) if vals else 0.0
                except Exception as e:
                    print(f"[WARN] action-head prior failed: {e}")

            if self.use_action_head_prior and len(candidate_action_names) > self.prior_top_actions:
                candidate_action_names = sorted(
                    candidate_action_names, key=lambda a: action_prior.get(a, 0.0), reverse=True
                )[: self.prior_top_actions]

            # === Generate/update plan ===
            if need_replan(sample):
                try:
                    if hasattr(self.planner, "plan_sequence_beam"):
                        current_plan = self.planner.plan_sequence_beam(
                            sample=sample,
                            candidate_actions=candidate_action_names,
                            top_rules=top_rules,
                            history_frames=history_frames,
                            H=self.horizon,
                            beam_width=3,
                        )
                    else:
                        raise AttributeError("planner has no plan_sequence_beam")
                except Exception as e:
                    print(f"[WARN] plan_sequence failed, use greedy fallback: {e}")
                    current_plan = self._greedy_plan_sequence(
                        sample=sample,
                        candidate_actions=candidate_action_names,
                        history_frames=history_frames,
                        H=self.horizon,
                    )
                plan_ptr = 0

            action_name = (current_plan[plan_ptr] if (plan_ptr < len(current_plan)) else "IDLE")
            plan_ptr += 1
            action_id = self._action_name_to_id(action_name)

            print("[DBG] candidates:", candidate_action_names[:5])
            print("[DBG] chosen action:", action_name)

            # === F_i one-step probe + safety override (Section 3.2.1 verification step) ===
            try:
                s_probe = self._predict_one_step_with_Fi(sample, action_name, dt, history_frames=history_frames)
                print(
                    "[DBG] Fi next vx, risk, energy, comfort:",
                    round(float(s_probe.get("ego_vx", float("nan"))), 3),
                    round(float(s_probe.get("risk_score", float("nan"))), 3),
                    round(float(s_probe.get("energy_consumption", float("nan"))), 3),
                    round(float(s_probe.get("comfort_score", float("nan"))), 3),
                )
                step_rwd = self._step_reward(s_probe)
                print("[DBG] step_reward (ours):", round(step_rwd, 3))

                if float(s_probe.get("risk_score", 0.0)) >= self.hard_risk_th:
                    safer = None
                    trial = []
                    if "SLOWER" in candidate_action_names and self.safe_to_slow(sample):
                        trial.append("SLOWER")
                    for z in ["LANE_LEFT", "LANE_RIGHT", "IDLE", "FASTER"]:
                        if z in candidate_action_names and z not in trial:
                            trial.append(z)
                    for z in trial:
                        s_tmp = self._predict_one_step_with_Fi(sample, z, dt, history_frames=history_frames)
                        if s_tmp["risk_score"] < s_probe["risk_score"]:
                            safer = z
                            break
                    if safer is not None:
                        print(f"[SAFETY] Override risky action {action_name} -> {safer}")
                        action_name = safer
                        action_id = self._action_name_to_id(action_name)

            except Exception as e:
                print("[DBG] Fi probe failed:", e)

            # ===== Compl: Calculate v_{k,t} based on rule->action consistency =====
            v_step = 0.0  # Number of rules not adopted this step
            k_step = 0.0  # Number of rules evaluated this step
            for r in (top_rules or []):
                a_rec = (self.extract_action_from_rule(r) or "").upper()
                if a_rec in self.ACTION_MAP_TO_ID:
                    k_step += 1.0
                    if a_rec != action_name:
                        v_step += 1.0
            compl_viol_sum += v_step
            compl_denom_sum += k_step

            # -- Roll out H_eval step prediction sequence from current frame -- #
            t_gt = len(true_hist)
            if (step_count % stride) == 0:
                plan_suffix = current_plan[plan_ptr - 1 : plan_ptr - 1 + H_eval]
                pred_seq = rollout_positions_from(
                    sample0=sample,
                    planned_actions=plan_suffix,
                    H=H_eval,
                    dt_local=dt,
                    history_frames_local=history_frames,
                )
                pending_preds.append({"t_gt": t_gt, "pred_seq": pred_seq})

            # ===== T_plan: Cumulative decision time (excluding env.step) =====
            plan_time_total += (time.perf_counter() - t_plan0)

            # 4) Execute action
            out = self.env.step(action_id)
            if len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = terminated or truncated
            else:
                obs, reward, done, info = out

            # ===== CR: Episode-level crash flag =====
            crashed_ep = max(crashed_ep, get_crash_flag(info, self.env))

            # ===== RE: Risk accumulation (estimate TTC from current obs) =====
            ttc_t = estimate_ttc_from_kinematics_obs(obs, lane_width=self.lane_width)
            risk_term = max(0.0, self.delta_ttc - ttc_t) if np.isfinite(ttc_t) else 0.0
            risk_accum += float(risk_term)

            try:
                print("[DBG] env_reward:", round(float(reward), 3))
            except Exception:
                print("[DBG] env_reward: <nan>")

            total_reward += float(reward)

            # Record actual position
            next_sample = self.sampler.sample_episode(episode_id=episode_id, obs=obs, last_action_id=action_id)
            ego_next = next_sample.get("Ego", {})
            x1, y1, _, _ = get_ego_xy(ego_next)
            true_next_xy = np.array([x1, y1], dtype=float)
            true_hist.append(true_next_xy)

            # Settle due predictions
            while pending_preds and (len(true_hist) >= (pending_preds[0]["t_gt"] + H_eval)):
                item = pending_preds.pop(0)
                t0 = item["t_gt"]
                gt_seq = np.asarray(true_hist[t0 : t0 + H_eval], dtype=float)
                pr_seq = np.asarray(item["pred_seq"], dtype=float)

                gt_rel = gt_seq - gt_seq[0:1, :]
                pr_rel = pr_seq - pr_seq[0:1, :]

                errs = np.linalg.norm(pr_rel - gt_rel, axis=1)
                per_t_ade.append(float(errs.mean()))
                per_t_fde.append(float(errs[-1]))

            ade_running = float(np.mean(per_t_ade)) if per_t_ade else float("nan")
            fde_running = float(np.mean(per_t_fde)) if per_t_fde else float("nan")

            compl_running = (1.0 - (compl_viol_sum / compl_denom_sum)) if (compl_denom_sum > 0) else float("nan")
            compl_step = (1.0 - (v_step / k_step)) if (k_step > 0) else float("nan")

            print(
                f"[DEBUG] ep={episode_id:02d} step={step_count:03d} "
                f"act={action_name}({action_id}) reward={float(reward):.3f} "
                f"Rsum={total_reward:.3f} ADE={_fmt_num(ade_running, 3)} FDE={_fmt_num(fde_running, 3)} "
                f"TTC={_fmt_num(ttc_t if np.isfinite(ttc_t) else 1e9, 3)} risk_term={_fmt_num(risk_term, 3)} "
                f"Compl_step={_fmt_num(compl_step, 3)} Compl_run={_fmt_num(compl_running, 3)}"
            )
            # print(
            #     f"[DEBUG] ep={episode_id:02d} step={step_count:03d} "
            #     f"act={action_name}({action_id}) reward={float(reward):.3f} "
            #     f"Rsum={total_reward:.3f} ADE={_fmt_num(ade_running, 3)} FDE={_fmt_num(fde_running, 3)} "
            #     f"TTC={_fmt_num(ttc_t if np.isfinite(ttc_t) else 1e9, 3)} risk_term={_fmt_num(risk_term, 3)}"
            # )

            trajectory.append(
                {
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
                    "done": bool(done),
                    # ===== step-level metrics (for CSV) =====
                    "T_plan_cum": float(plan_time_total),
                    "ttc": float(ttc_t) if np.isfinite(ttc_t) else float("inf"),
                    "risk_term": float(risk_term),
                    "risk_accum": float(risk_accum),
                    "compl_v_step": float(v_step),
                    "compl_k_step": float(k_step),
                    "crashed_flag": int(crashed_ep),
                    "compl_step": float(1.0 - v_step / k_step) if k_step > 0 else float("nan"),
                    "compl_running": float(1.0 - compl_viol_sum / compl_denom_sum)
                          if compl_denom_sum > 0 else float("nan"),
                    "llm_invoked": bool(llm_invoked_this_step),
                }
            )

            step_count += 1
            if step_count >= max_safe_actions:
                break

        # -- Episode end: ADE / FDE -- #
        ade_final = float(np.mean(per_t_ade)) if per_t_ade else float("nan")
        fde_final = float(np.mean(per_t_fde)) if per_t_fde else float("nan")

        # ===== Episode Metrics =====
        compl_final = (1.0 - (compl_viol_sum / compl_denom_sum)) if (compl_denom_sum > 0) else float("nan")
        cr_ep = int(crashed_ep)

        R_task = float(total_reward)
        R_risk = float(risk_accum)
        RE = float(R_task - self.lambda_risk * R_risk)

        PE = float(RE / plan_time_total) if plan_time_total > 1e-9 else float("nan")

        result = {
            "episode_id": episode_id,
            "trajectory": trajectory,
            "total_reward": float(total_reward),
            "steps": step_count,
            "ADE": ade_final,
            "FDE": fde_final,
            # ===== new metrics =====
            "Compl": float(compl_final),
            "CR_ep": int(cr_ep),
            "R_task": float(R_task),
            "R_risk": float(R_risk),
            "RE": float(RE),
            "T_plan": float(plan_time_total),
            "PE": float(PE),
            "delta_ttc": float(self.delta_ttc),
            "lambda_risk": float(self.lambda_risk),
            "llm_invocation_rate": self.acquirer.invocation_rate if self.acquirer else 0.0,
            "llm_invoked_count": self.acquirer.llm_invoked_count if self.acquirer else 0,
            "acquired_rules_count": self.acquirer.accepted_count if self.acquirer else 0,
        }

        if self.neo4j_writer:
            try:
                self.neo4j_writer.write_episode(result)
            except Exception as e:
                print(f"[WARN] Neo4j write failed: {e}")

        return result

    # -----------------------------
    # Multiple episodes
    # -----------------------------
    def run_multiple(self, n_episodes: int, max_safe_actions: int = 20) -> List[Dict[str, Any]]:
        results = []
        for ep in range(n_episodes):
            print(f"\n[Runner] Running Episode {ep}")
            res = self.run_episode(ep, max_safe_actions=max_safe_actions)
            results.append(res)
            print(
                f"[Done] Episode {ep} completed: "
                f"reward={_fmt_num(res.get('total_reward'), 2)}, "
                f"steps={res.get('steps')}, "
                f"ADE={_fmt_num(res.get('ADE'), 3)}, "
                f"FDE={_fmt_num(res.get('FDE'), 3)}, "
                f"Compl={_fmt_num(res.get('Compl'), 3)}, "
                f"CR_ep={res.get('CR_ep')}, "
                f"RE={_fmt_num(res.get('RE'), 3)}, "
                f"PE={_fmt_num(res.get('PE'), 6)}"
            )
        return results


# =========================
# main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/inference_runner.yaml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_safe_actions", type=int, default=150, help="Maximum number of actions to execute without collision (step limit)")
    parser.add_argument("--seed_offset", type=int, default=0, help="Offset for env-reset seeds to vary trajectories across multi-seed runs")
    parser.add_argument("--out_tag", type=str, default="", help="Suffix for inference_results filename (e.g. seed0)")
    args = parser.parse_args()

    MAX_SAFE_ACTIONS = int(args.max_safe_actions)
    POLICY_HZ = 15
    SIM_HZ = 15

    env_cfg = {
        "observation": {
            "type": "Kinematics",
            "features": ["x", "y", "vx", "vy"],
            "absolute": True,
        },
        "action": {"type": "DiscreteMetaAction"},
        "duration": MAX_SAFE_ACTIONS,
        "policy_frequency": POLICY_HZ,
        "simulation_frequency": SIM_HZ,
        "vehicles_count": 24,
        "initial_spacing": 1.2,
        "collision_terminal": True,
        "offroad_terminal": False,
        "normalize_reward": False,
        "collision_reward": -5.0,
        "high_speed_reward": 0.15,
        "lane_change_reward": 2,
        "right_lane_reward": 3,
        "merge_reward": 4.0,
        "reward_speed_range": [20, 30],
        "screen_width": 800,
        "screen_height": 270,
        "offscreen_rendering": True,
    }

    raw_env = gym.make(CONFIG["env_name"], render_mode="rgb_array", config=env_cfg)
    set_global_seed(20241025 + args.seed_offset * 7919)

    if not hasattr(raw_env, "metadata") or raw_env.metadata is None:
        raw_env.metadata = {}
    raw_env.metadata["render_fps"] = POLICY_HZ

    raw_env = TimeLimit(raw_env, max_episode_steps=MAX_SAFE_ACTIONS)

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
        runner.seed_offset = args.seed_offset
        results = runner.run_multiple(args.episodes, max_safe_actions=MAX_SAFE_ACTIONS)

        result_dir = "result/inference/Roundabout"
        os.makedirs(result_dir, exist_ok=True)
        tag = ("_" + args.out_tag) if args.out_tag else ""
        result_path = os.path.join(result_dir, f"inference_results{tag}.json")

        # ===== 1. Save complete JSON =====
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Done] Inference results saved to {result_path}")

        # ===== 2. Episode summary metrics (including Compl/CR/RE/PE) =====
        import csv

        episode_metrics_path = os.path.join(result_dir, "episode_metrics.csv")
        with open(episode_metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode_id",
                    "steps",
                    "total_reward",
                    "ADE",
                    "FDE",
                    "Compl",
                    "CR_ep",
                    "R_task",
                    "R_risk",
                    "RE",
                    "T_plan",
                    "PE",
                    "delta_ttc",
                    "lambda_risk",
                ],
            )
            writer.writeheader()
            for res in results:
                writer.writerow(
                    {
                        "episode_id": res.get("episode_id"),
                        "steps": res.get("steps"),
                        "total_reward": res.get("total_reward"),
                        "ADE": res.get("ADE"),
                        "FDE": res.get("FDE"),
                        "Compl": res.get("Compl"),
                        "CR_ep": res.get("CR_ep"),
                        "R_task": res.get("R_task"),
                        "R_risk": res.get("R_risk"),
                        "RE": res.get("RE"),
                        "T_plan": res.get("T_plan"),
                        "PE": res.get("PE"),
                        "delta_ttc": res.get("delta_ttc"),
                        "lambda_risk": res.get("lambda_risk"),
                    }
                )
        print(f"[Done] Episode-level metrics saved to {episode_metrics_path}")

        # ===== 3. Step-level metrics (including Compl/CR/RE components, T_plan) =====
        step_metrics_path = os.path.join(result_dir, "step_metrics.csv")
        with open(step_metrics_path, "w", newline="") as f:
            fieldnames = [
                "episode_id",
                "step",
                "ADE_step",
                "FDE_step",
                "reward",
                "running_reward",
                "chosen_action_name",
                "chosen_action_id",
                "done",
                "ttc",
                "risk_term",
                "risk_accum",
                "compl_v_step",
                "compl_k_step",
                "crashed_flag",
                "T_plan_cum",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for res in results:
                ep_id = res.get("episode_id")
                traj = res.get("trajectory", [])
                for item in traj:
                    writer.writerow(
                        {
                            "episode_id": ep_id,
                            "step": item.get("step"),
                            "ADE_step": item.get("ADE_step"),
                            "FDE_step": item.get("FDE_step"),
                            "reward": item.get("reward"),
                            "running_reward": item.get("running_reward"),
                            "chosen_action_name": item.get("chosen_action_name"),
                            "chosen_action_id": item.get("chosen_action_id"),
                            "done": item.get("done"),
                            "ttc": item.get("ttc"),
                            "risk_term": item.get("risk_term"),
                            "risk_accum": item.get("risk_accum"),
                            "compl_v_step": item.get("compl_v_step"),
                            "compl_k_step": item.get("compl_k_step"),
                            "crashed_flag": item.get("crashed_flag"),
                            "T_plan_cum": item.get("T_plan_cum"),
                        }
                    )
        print(f"[Done] Step-level metrics saved to {step_metrics_path}")

        # ===== 4. Visualization (if available) =====
        try:
            visualize_inference_results(result_path, output_dir=result_dir)
            print("[Done] Visualization images generated")
        except Exception as e:
            print(f"[WARN] Visualization failed: {e}")

    finally:
        try:
            env.close()
        except Exception:
            pass
