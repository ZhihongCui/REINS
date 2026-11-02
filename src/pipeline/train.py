# train_pipeline.py
import os
import sys
import re
import json
import torch
import torch.nn as nn

# Add project root to path for direct execution
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.ndimage import gaussian_filter1d
from hashlib import md5


# Module imports
from src.utils.common import load_yaml, build_variable_stats
from src.utils.sampler import Sampler
from src.world_model.structure_learner import HybridStructureFunctionLearner
from src.rule_system.extractor import RuleExtractor
from src.rule_system.scorer import build_rule_scorer, RuleScoringDataset
from src.utils.vocab_builder import build_vocab_from_yaml_rules, build_vocab_from_raw_text, save_vocab
from config.config import CONFIG
from src.rule_system.storage import Neo4jWriter, Rule
from src.rule_system.calibrator import ThresholdCalibrator

# ============ Generate Stable ruleID ============
def _stable_rule_id(r: dict) -> str:
    key = "|".join([
        str(r.get("rule_text_clean") or r.get("rule_text") or r.get("raw_text") or "").strip(),
        str(r.get("cause")),
        str(r.get("effect")),
        str(r.get("action", "")),
        str(r.get("rule_type_hint", "Unknown"))
    ])
    return "R_" + md5(key.encode("utf-8")).hexdigest()[:12]

# ============ Collate function ============
def collate_fn(batch):
    state_vecs, rule_texts, labels = zip(*batch)
    state = torch.stack(state_vecs)  # [B, D]
    state = torch.nan_to_num(state, nan=0.0, posinf=1e3, neginf=-1e3)
    state = torch.clamp(state, -1e3, 1e3)

    labels = torch.stack(labels).float()   # [B] or [B,1]
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)       # -> [B,1]

    return state, list(rule_texts), labels


# ============ Clean LLM-generated rule text (for RuleScorer only) ============
def clean_rule_text(rule_text: str) -> str:
    rule_text = (rule_text or "").lower()
    rule_text = re.sub(r"^[0-9]+\.\s*", "", rule_text)
    rule_text = rule_text.split("#")[0]
    rule_text = rule_text.strip()
    rule_text = re.sub(r"\bif\b", "[if]", rule_text)
    rule_text = re.sub(r"\bthen\b", "[then]", rule_text)
    return rule_text.strip()



def _flatten_state(sample: Dict[str, Any]) -> List[float]:
    try:
        var_cfg = load_yaml("config/variable_config.yaml") or {}
    except Exception:
        var_cfg = {}
    FEATURE_ORDER = {
        "Environment": var_cfg.get("Environment", ["weather_type","visibility","road_friction","road_type","time_of_day","speed_limit"]),
        "Traffic":     var_cfg.get("Traffic", ["headway_distance","lead_speed","lead_rel_speed","lead_ttc","lead_thw",
                                               "rear_distance","rear_speed","rear_rel_speed","rear_ttc",
                                               "left_front_distance","left_front_speed","left_rear_distance","left_rear_speed",
                                               "right_front_distance","right_front_speed","right_rear_distance","right_rear_speed",
                                               "lane_density_left","lane_density_right","available_gap_left","available_gap_right","cutin_risk"]),
        "Ego":         var_cfg.get("Ego", ["ego_x","ego_y","ego_vx","ego_vy","ego_speed","ego_acceleration","yaw_rate","jerk","lane_id","lateral_offset"]),
        "Action":      var_cfg.get("Action", ["action_type"]),
        "Outcome":     var_cfg.get("Outcome", ["risk_score","comfort_score","energy_consumption","success_flag","violation_flag"]),
    }
    flat = []
    for g in ["Environment","Traffic","Ego","Action","Outcome"]:
        d = sample.get(g, {}) or {}
        for k in FEATURE_ORDER[g]:
            v = d.get(k, 0.0)
            try: flat.append(float(v))
            except: flat.append(0.0)
    return flat


def _norm_cause(cause_field: Any) -> str:
    """Normalize cause to a string usable in rule_id"""
    if isinstance(cause_field, list):
        return "_".join([str(x) for x in cause_field])
    return str(cause_field)

@torch.no_grad()
def evaluate_rule_scorer(rule_scorer, dataset, device, tag="val", out_dir="result/figures/Roundabout"):
    """
    Evaluation metrics:
      - BCE baseline (theoretical lower bound for constant prediction) vs actual val_loss
      - ROC-AUC / PR-AUC (average precision)
      - Brier score (calibration)
      - Logit distribution (mean/variance/extrema)
      - OOV rate (UNK ratio)
    Also performs 3 corruption controls: shuffle label / shuffle text / shuffle state
    """
    os.makedirs(out_dir, exist_ok=True)

    # === 1) Collect all predictions ===
    y_true, y_prob, logits = [], [], []
    unk_tokens, total_tokens = 0, 0

    # Iterate directly over dataset without DataLoader
    for i in range(len(dataset)):
        state_vec, rule_text, label_t = dataset[i]       # Single sample before collate_fn
        # Record OOV (UNK) ratio
        # Assuming scorer has tokenizer.encode(text) or vocab mapping
        try:
            ids = rule_scorer.tokenizer.encode(rule_text)  # Modify if your implementation differs
        except Exception:
            # Fallback: use empty implementation
            ids = []
        total_tokens += len(ids)
        unk_id = rule_scorer.vocab.get("[UNK]", 1) if hasattr(rule_scorer, "vocab") else None
        if unk_id is not None:
            unk_tokens += sum(int(t == unk_id) for t in ids)

        # Forward pass
        sv = torch.nan_to_num(state_vec, nan=0.0, posinf=1e3, neginf=-1e3).unsqueeze(0).to(device)
        rl = [rule_text]
        logit, _ = rule_scorer(sv, rl)   # [1,1]
        p = torch.sigmoid(logit).item()
        logits.append(float(logit.item()))
        y_prob.append(float(p))
        y_true.append(float(label_t.item()))

    y_true = np.array(y_true, dtype=float)
    y_prob = np.array(y_prob, dtype=float)
    logits = np.array(logits, dtype=float)
    pos_ratio = float(np.mean(y_true)) if len(y_true) else 0.0

    # === 2) Calculate metrics ===
    # Constant baseline cross-entropy (for comparing whether your train/val loss is below this baseline)
    eps = 1e-12
    ce_baseline = -(pos_ratio*np.log(pos_ratio+eps) + (1-pos_ratio)*np.log(1-pos_ratio+eps))
    # AUC / PR-AUC / Brier
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    try:
        ap = average_precision_score(y_true, y_prob)
    except Exception:
        ap = float("nan")
    try:
        brier = brier_score_loss(y_true, y_prob)
    except Exception:
        brier = float("nan")

    # Logit statistics & OOV
    logit_stats = dict(mean=float(np.mean(logits)),
                       std=float(np.std(logits)),
                       min=float(np.min(logits)),
                       max=float(np.max(logits)))
    oov_rate = (unk_tokens / max(1, total_tokens)) if total_tokens > 0 else float("nan")

    # === 3) Corruption control experiments ===
    def _metric_under_corruption(corrupt: str) -> Dict[str, float]:
        y_prob_c = []
        for i in range(len(dataset)):
            state_vec, rule_text, label_t = dataset[i]
            if corrupt == "shuffle_label":
                # Keep input unchanged, only randomize labels -> AUC/PR not meaningful, observe CE and Brier
                pass
            elif corrupt == "shuffle_text":
                # Replace with text from a random sample
                j = np.random.randint(0, len(dataset))
                _, rule_text, _ = dataset[j]
            elif corrupt == "shuffle_state":
                j = np.random.randint(0, len(dataset))
                state_vec, _, _ = dataset[j]

            sv = torch.nan_to_num(state_vec, nan=0.0, posinf=1e3, neginf=-1e3).unsqueeze(0).to(device)
            logit, _ = rule_scorer(sv, [rule_text])
            y_prob_c.append(float(torch.sigmoid(logit).item()))
        y_prob_c = np.array(y_prob_c, dtype=float)
        out = {}
        try: out["AUC"] = roc_auc_score(y_true, y_prob_c)
        except: out["AUC"] = float("nan")
        try: out["AP"] = average_precision_score(y_true, y_prob_c)
        except: out["AP"] = float("nan")
        try: out["Brier"] = brier_score_loss(y_true, y_prob_c)
        except: out["Brier"] = float("nan")
        # Constant baseline is compared in main results
        return out

    corrupt_results = {
        "shuffle_text": _metric_under_corruption("shuffle_text"),
        "shuffle_state": _metric_under_corruption("shuffle_state"),
        # Label shuffling is better for recalculating loss; providing AUC/AP/Brier as reference
        "shuffle_label": _metric_under_corruption("shuffle_label"),
    }

    # === 4) Output and save ===
    summary = {
        "pos_ratio": pos_ratio,
        "ce_baseline": ce_baseline,
        "AUC": float(auc),
        "AP": float(ap),
        "Brier": float(brier),
        "logit_stats": logit_stats,
        "oov_rate": float(oov_rate),
        "corrupt": corrupt_results
    }
    with open(os.path.join(out_dir, f"rule_scorer_eval_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[Eval]", json.dumps(summary, indent=2))
    return summary


# ============ Main pipeline ============
def train_pipeline(num_train_episodes=30, num_epochs=5):
    # ==================================================
    # Step 1: Sampling data from Env.
    # ==================================================
    print("=" * 60)
    print("[Step 1] Sampling data from env-name")

    # config = {
    #     "observation": {
    #         "type": "Kinematics",
    #         "vehicles_count": 8,
    #         "features": ["x", "y", "vx", "vy"],
    #     },
    #     "duration": 150
    # }
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 12,
            "features": ["x", "y", "vx", "vy"],
            "absolute": True,
        },
        "simulation_frequency": 15,
        "policy_frequency": 15,
        "duration": 200,
        "action": {"type": "DiscreteMetaAction"},
        # "controlled_vehicles": 1,
        # "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "initial_spacing": "uniform",
        "collision_reward": -1,
        "high_speed_reward": 0.4,
        "right_lane_reward": 0.1,
        "lane_change_reward": -0.05,
        "reward_speed_range": [20, 30],
        "normalize_reward": True,
        "offroad_terminal": True,
        "initial_lane_id": "random",
        # "initial_spacing": "random",
        "ego_spacing": "random",
        "spawn_probability": 0.5,
    }

    env = gym.make(CONFIG["env_name"], render_mode=None, config=config)
    env = env.unwrapped
    obs, _ = env.reset()

    sampler = Sampler(env=env)
    samples = sampler.sample_multiple_episodes(n=num_train_episodes)
    print(f"[Done] Collected {len(samples)} episodes.")

    # One sub-directory per scenario (data/samples/Highway, Merge, Roundabout, Intersection).
    _scenario_dirs = {"highway": "Highway", "merge": "Merge", "roundabout": "Roundabout", "intersection": "Intersection"}
    scenario_dir = _scenario_dirs.get(CONFIG["env_name"].split("-")[0], CONFIG["env_name"])
    save_dir = os.path.join("data", "samples", scenario_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"samples_{CONFIG['env_name']}_ep{num_train_episodes}.json")
    sampler.save_samples(samples, save_path)
    print(f"[Done] Saved raw sampled episodes -> {save_path}")

    # ==================================================
    # Step 2: Learn structure (Pa(X)) + function (f_i)
    # ==================================================
    print("=" * 60)
    print("[Step 2] Learning structure & functions (Hybrid SCM)")

    structure_learner = HybridStructureFunctionLearner(
        max_lag=3,
        fi_model_type="transformer",
        fi_epochs=300,
        use_llm_prior=True
    )
    structure_learner.train(samples)
    causal_edges = structure_learner.infer_edges()
    print(f"[Done] Inferred {len(causal_edges)} causal edges.")

    # ==================================================
    # Step 3: Extract numeric-parameterized rules via LLM (+ stats)
    # ==================================================
    print("=" * 60)
    print("[Step 3] Extracting rules from causal graph")

    # Calculate mu/std/quantiles for each variable, used to prompt LLM to produce parameterizable templates & later calibration
    var_stats = build_variable_stats(samples)

    # Let RuleExtractor produce structured rules with params.thresholds / variables / action
    rule_extractor = RuleExtractor()
    # RuleExtractor accepts variable_stats as an optional argument; falls back to per-variable stats from disk when omitted.
    rules = rule_extractor.extract_rules(causal_edges, variable_stats=var_stats)
    print(f"[Done] Extracted {len(rules)} rules from LLM (structured).")

    # ==================================================
    # Step 3.5: Calibrate thresholds to numeric values
    # ==================================================
    calibrator = ThresholdCalibrator(samples)
    calibrated_rules = []
    for r in rules:
        effect_name = r.get("effect", "risk_score")
        calibrated = calibrator.calibrate_rule(r, effect_name)
        calibrated_rules.append(calibrated)

    # ==================================================
    # Step 3.6: Add a clean text field for RuleScorer only; keep structure intact
    #           Compatible with existing tokenizer: write 'rule' as clean text
    # ==================================================
    enriched_rules = []
    for r in calibrated_rules:
        rt = r.get("rule_text") or r.get("raw_text") or r.get("rule", "")
        rule_text_clean = clean_rule_text(rt) if isinstance(rt, str) else ""
        r["rule_text_clean"] = rule_text_clean
        r["rule"] = rule_text_clean  # Compatible with build_vocab_from_raw_text & RuleScoringDataset reading
        enriched_rules.append(r)

    print(f"[Done] Structured rules kept & clean text added: {len(enriched_rules)}")

    os.makedirs("data", exist_ok=True)
    with open("data/extracted_rules.json", "w") as f:
        json.dump(enriched_rules, f, indent=2, ensure_ascii=False)
    print("[Done] Saved structured rules -> data/extracted_rules.json")

    # ==================================================
    # Step 4: Build vocab (from variables + clean rule text)
    # ==================================================
    print("=" * 60)
    print("[Step 4] Building vocabulary")

    vocab_yaml = build_vocab_from_yaml_rules(
        "config/variable_config.yaml", "config/rule_index_definitions.yaml"
    )
    vocab_llm = build_vocab_from_raw_text("data/extracted_rules.json")  # Reads 'rule' field

    # Merge
    for token in vocab_llm:
        if token not in vocab_yaml:
            vocab_yaml[token] = len(vocab_yaml)
    save_vocab(vocab_yaml, "config/vocab.json")
    print(f"[Done] Vocabulary built with {len(vocab_yaml)} tokens.")

    # ==================================================
    # Step 5: Prepare RuleScoringDataset (train with clean text only)
    # ==================================================
    print("=" * 60)
    print("[Step 5] Preparing rule scoring dataset")

    sample_timestep = samples[0]["trajectory"][0]
    state_dim = len(_flatten_state(sample_timestep))
    print(f"[INFO] State vector dimension = {state_dim}")

    rule_scorer = build_rule_scorer(
        method="transformer",
        state_dim=state_dim,
        vocab_path="config/vocab.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rule_scorer = rule_scorer.to(device)

    # Point raw_text to clean text only (other structure fields unchanged, not written to disk)
    train_ready_rules = []
    for r in enriched_rules:
        rr = dict(r)
        rr["raw_text"] = r.get("rule_text_clean", r.get("rule_text", r.get("raw_text", "")))
        train_ready_rules.append(rr)

    full_dataset = RuleScoringDataset(samples, train_ready_rules, rule_scorer.vocab)
    val_size = int(0.2 * len(full_dataset)) if len(full_dataset) > 0 else 0
    train_size = max(0, len(full_dataset) - val_size)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size]) if val_size > 0 else (full_dataset, [])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn) if val_size > 0 else None

    labels = [label.item() for _, _, label in full_dataset]
    pos_ratio = np.mean(labels) if labels else 0.0
    print(f"[INFO] Total training samples: {len(full_dataset)}")
    print(f"[INFO] Positive ratio: {pos_ratio:.4f}")

    # ==================================================
    # Step 6: Train RuleScorer
    # ==================================================
    print("=" * 60)
    print("[Step 6] Training RuleScorer model")

    optimizer = torch.optim.Adam(rule_scorer.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val = float("inf")
    patience, bad_epochs = 5, 0
    epoch_losses = []
    epoch_step_losses = {}

    os.makedirs("result/checkpoints/Roundabout", exist_ok=True)
    os.makedirs("result/figures/Roundabout", exist_ok=True)
    os.makedirs("result/logs/Roundabout", exist_ok=True)

    for epoch in range(num_epochs):
        rule_scorer.train()
        total_loss = 0.0
        step_losses = []

        for step, (state_vec, rule_texts, labels_t) in enumerate(train_loader, 1):
            labels_smooth = 0.05 + 0.9 * labels_t  # label smoothing
            # scores = rule_scorer(state_vec, rule_texts)
            # loss = criterion(scores, labels_smooth)
            state_vec    = state_vec.to(device)
            labels_t     = labels_t.to(device)
            rule_logit, _ = rule_scorer(state_vec, rule_texts)  # logits
            loss = criterion(rule_logit, labels_smooth)         # BCEWithLogitsLoss expects logits


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rule_scorer.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            step_losses.append(loss.item())

            if step % 20 == 0 or step == len(train_loader):
                print(f"[Epoch {epoch+1}/{num_epochs} | Step {step}/{len(train_loader)}] loss={loss.item():.4f}")

        avg_train_loss = total_loss / max(1, len(train_loader))
        epoch_step_losses[epoch + 1] = step_losses

        # Validation
        val_loss, auc = 0.0, 0.0
        if val_loader and len(val_loader) > 0:
            rule_scorer.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for state_vec, rule_texts, labels_v in val_loader:
                    state_vec = state_vec.to(device)
                    labels_v = labels_v.to(device)

                    rule_logit, _ = rule_scorer(state_vec, rule_texts)
                    val_loss += criterion(rule_logit, labels_v).item()
                    y_true.extend(labels_v.cpu().numpy())
                    y_pred.extend(torch.sigmoid(rule_logit).cpu().numpy())

                    # scores = rule_scorer(state_vec, rule_texts)
                    # val_loss += criterion(scores, labels_v).item()
                    # y_true.extend(labels_v.cpu().numpy())
                    # y_pred.extend(torch.sigmoid(scores).cpu().numpy())
            val_loss /= max(1, len(val_loader))
            try:
                auc = roc_auc_score(y_true, y_pred)
            except Exception:
                auc = float("nan")

        epoch_losses.append(val_loss)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train={avg_train_loss:.4f}  Val={val_loss:.4f}  AUC={auc:.4f}")

        if val_loader and val_loss < best_val - 1e-4:
            best_val = val_loss
            bad_epochs = 0
            torch.save(rule_scorer.state_dict(), "result/checkpoints/Roundabout/rule_scorer.pt")
        elif val_loader:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[Early Stop] Best Val Loss = {best_val:.4f}")
                break

        scheduler.step()

    # ==================================================
    # Step 6.5: Evaluate RuleScorer performance
    # ==================================================
    print("=" * 60)
    print("[Step 6.5] Evaluating RuleScorer performance")

    rule_scorer.eval()
    eval_ds = val_ds if (val_ds and len(val_ds) > 0) else full_dataset
    evaluate_rule_scorer(rule_scorer, eval_ds, device, tag="val", out_dir="result/figures/Roundabout")


    # ==================================================
    # Step 7: Save results (epoch + step level)
    # ==================================================
    np.savetxt("result/logs/Roundabout/loss_history_epoch.txt", np.array(epoch_losses), fmt="%.6f")

    first_n = 3
    steps_per_epoch = len(train_loader)
    epoch_step_losses_arr = [
        step_losses[i * steps_per_epoch : (i + 1) * steps_per_epoch]
        for i in range(len(epoch_losses))
    ]
    selected_epochs = list(range(min(first_n, len(epoch_losses))))

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_epochs)))
    for idx, ep in enumerate(selected_epochs):
        if ep < len(epoch_step_losses_arr):
            raw_losses = epoch_step_losses_arr[ep]
            if len(raw_losses) > 0:
                smooth_losses = gaussian_filter1d(raw_losses, sigma=8)
                plt.plot(range(1, len(smooth_losses) + 1), smooth_losses,
                         color=colors[idx], linewidth=2, label=f"Epoch {ep+1}")
    plt.xlabel("Step"); plt.ylabel("Smoothed Training Loss")
    plt.title(f"Step-wise Smoothed Training Loss (First {first_n} Epochs)")
    plt.legend(title="Epoch", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("result/figures/Roundabout/step_loss_smoothed_firstN.png", dpi=300)
    plt.close()
    print(f"[Done] Saved smoothed step-wise loss curves -> result/figures/Roundabout/step_loss_smoothed_firstN.png")

    # ==================================================
    # Step 7.5: Use selector (guards + AND + S_rule/S_act) to choose best rule
    # ==================================================
    print("=" * 60)
    print("[Step 7.5] Selecting best rules with learned scorer")

    # Wrap two callbacks: s_rule_fn / s_act_fn
    def s_rule_fn(rule, state_dict) -> float:
        txt = rule.get("raw_text", rule.get("rule", ""))
        st = torch.tensor([_flatten_state(state_dict)], dtype=torch.float32, device=device)
        with torch.no_grad():
            sr, _ = rule_scorer(st, [txt])
        return float(sr.squeeze(0).squeeze(-1).item())

    def s_act_fn(rule, state_dict, action_idx: int) -> float:
        txt = rule.get("raw_text", rule.get("rule", ""))
        st = torch.tensor([_flatten_state(state_dict)], dtype=torch.float32, device=device)
        with torch.no_grad():
            _, s_act = rule_scorer.score_with_action(st, [txt], action_idx)  # [B,1]
        return float(s_act.squeeze(0).squeeze(-1).item())

    # Take a current state for demonstration
    demo_state = samples[0]["trajectory"][0]  # Can also use current timestep from online environment
    action_list = ["LANE_LEFT","LANE_RIGHT","IDLE","FASTER","SLOWER"]

    selector_out = rule_extractor.select_best_rules(
        rules=enriched_rules,             # Use threshold-calibrated and cleaned rules
        state=demo_state,                 # Current state (dict structure)
        candidate_actions=action_list,
        s_rule_fn=s_rule_fn,
        s_act_fn=s_act_fn,
        alpha=0.6, beta=0.3, gamma=0.1,   # Adjustable weights
        action_order=action_list,         # Synchronized with above
        action_priority=["SLOWER","LANE_LEFT","LANE_RIGHT","IDLE","FASTER"]  # Safety first
    )

    print("[Selector] Global best:", selector_out["global_best"])


    # ==================================================
    # Step 8: Write results to Neo4j (text uses clean version for readability; structure still in JSON)
    # ==================================================
    print("=" * 60)
    print("[Step 8] Writing causal graph and rules to Neo4j")

    neo4j_writer = Neo4jWriter(uri="bolt://localhost:7687", user="neo4j")  # Password from NEO4J_PASSWORD env var
    neo4j_writer.initialize(clear=False)

    # Write five categories of variables (automatically includes group & level)
    neo4j_writer.write_variables(structure_learner.variables)

    # Write rules (rule_id normalized; original text uses rule_text_clean)
    rule_objs = []
    for r in enriched_rules:
        rid = _stable_rule_id(r)  # Replace original rid generation
        text_for_db = r.get("rule_text_clean") or r.get("rule_text") or r.get("raw_text", "")
        rule_objs.append(
            Rule(
                rule_id=rid,
                cause=r.get("cause", ""),
                effect=r.get("effect", ""),
                rule_type=r.get("rule_type_hint", "Unknown"),
                confidence=float(r.get("weight", 0.5) or 0.5),
                raw_text=text_for_db,
                cause_group=r.get("cause_group", "Unknown"),
                effect_group=r.get("effect_group", "Unknown"),
                action=r.get("action", "")  # Newly added
            )
        )
    neo4j_writer.write_rules(rule_objs)
    neo4j_writer.close()

    print("[Done] Pipeline completed.")

if __name__ == "__main__":
    train_pipeline(num_train_episodes=30, num_epochs=5)
