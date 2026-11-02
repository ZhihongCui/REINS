# modules/structure_function_learner_hybrid.py

import os
import json  # [FIX] ensure JSON is imported for LLM prior parsing
import yaml
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.llm.caller import call_gpt
from src.world_model.dynamics.mlp import FunctionMLP
from src.world_model.dynamics.transformer import FunctionTransformer


# ------------------ Utilities ------------------

def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x)

def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu, std = x.mean(), x.std()
    if std < eps:
        return x * 0.0
    return (x - mu) / (std + eps)

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------ f_i Training Dataset ------------------

class FiDataset(Dataset):
    """
    Function learning dataset for a single target variable X_i:
    Input: time-lagged values of parent variable set Pa(X_i) (can be flattened or kept as sequence)
    Output: X_i(t)
    """
    def __init__(self,
                 episode_series: List[Dict[str, List[float]]],
                 target: str,
                 parents: List[str],
                 max_lag: int,
                 zscore: bool = True,
                 as_sequence: bool = False):
        self.X = []
        self.Y = []
        self.as_sequence = as_sequence
        self.parents = list(parents)  # Fix parent variable order
        self.max_lag = max_lag

        # [FIX] To ensure dimension stability, we no longer shrink to parents_in_ep,
        # but pad missing parent variables with zero vectors (after standardization: zero mean, zero variance -> direct 0).
        for series in episode_series:
            if target not in series:
                continue
            T = len(series[target])
            if T <= max_lag:
                continue

            # Build z_map: use zero sequence when parent variable is missing
            z_map: Dict[str, np.ndarray] = {}

            # Target variable
            tgt_arr = np.asarray(series[target], dtype=float)
            tgt_arr = _zscore(tgt_arr) if zscore else tgt_arr
            z_map[target] = tgt_arr

            # Each parent variable
            for p in self.parents:
                if p in series:
                    arr = np.asarray(series[p], dtype=float)
                    arr = _zscore(arr) if zscore else arr
                    # Align length
                    if len(arr) < T:
                        arr = np.concatenate([arr, np.repeat(arr[-1], T - len(arr))])
                    z_map[p] = arr
                else:
                    # Missing -> all zeros sequence
                    z_map[p] = np.zeros((T,), dtype=float)

            # Construct samples per timestep (t uses current frame as y, previous max_lag frames as x)
            for t in range(max_lag, T):
                if self.as_sequence:
                    # Shape [L, P], arranged by increasing lag (1..L)
                    seq = []
                    for lag in range(1, max_lag + 1):
                        row = [float(z_map[p][t - lag]) for p in self.parents]
                        seq.append(row)
                    self.X.append(np.asarray(seq, dtype=float))    # [L, P]
                else:
                    # Flatten [P*L], parent variable dimension is outer, lag is inner
                    row = []
                    for p in self.parents:
                        for lag in range(1, max_lag + 1):
                            row.append(float(z_map[p][t - lag]))
                    self.X.append(np.asarray(row, dtype=float))     # [P*L]

                self.Y.append(float(z_map[target][t]))

        self.X = np.asarray(self.X, dtype=float)
        self.Y = np.asarray(self.Y, dtype=float)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        # Keep interface consistent
        return x, y


# ------------------ Main Learner ------------------

@dataclass
class FiSpec:
    parents: List[str]
    model_type: str                   # "mlp" or "transformer"
    input_dim: int                    # MLP input dimension (P*L)
    num_parents: int                  # Used by Transformer
    max_lag: int
    path: Optional[str] = None        # Save path


class HybridStructureFunctionLearner:
    """
    Hybrid structure + function learning (complete version):
    1) LLM prior: provide candidate parent set and confidence for each target Y
    2) Group ridge regression (parent-specific penalty): select Pa(Y) and coefficients from data (linear f-hat)
    3) Train nonlinear f_Y on selected Pa(Y) (choose between MLP/Transformer)
    4) Export causal_graph.yaml + save f_Y models
    """

    def __init__(
        self,
        max_lag: int = 3,
        lambda_base: float = 1.0,   # Group ridge base penalty
        penalty_power: float = 2.0, # Penalty exponent p: lambda_j = lambda_base * (1 - conf_j)^p
        top_k: int = 8,
        min_importance: float = 1e-3,
        target_groups: Optional[List[str]] = None,  # Default: only learn Outcome
        verbose: bool = True,
        use_llm_prior: bool = True,
        llm_model: str = "gpt-4.1-nano",
        fi_model_type: str = "transformer",  # "mlp" | "transformer"
        fi_hidden: int = 64,
        fi_dropout: float = 0.0,
        fi_epochs: int = 200,
        fi_batch_size: int = 256,
        fi_lr: float = 1e-3,
        save_dir: str = "result/scm/Roundabout"
    ):
        self.max_lag = max_lag
        self.lambda_base = lambda_base
        self.penalty_power = penalty_power
        self.top_k = top_k
        self.min_importance = min_importance
        self.target_groups = target_groups or ["Outcome", "Ego", "Traffic"]
        self.verbose = verbose
        self.use_llm_prior = use_llm_prior
        self.llm_model = llm_model

        self.fi_model_type = fi_model_type
        self.fi_hidden = fi_hidden
        self.fi_dropout = fi_dropout
        self.fi_epochs = fi_epochs
        self.fi_batch_size = fi_batch_size
        self.fi_lr = fi_lr
        self.save_dir = save_dir

        self.variables: Dict[str, List[str]] = {}
        self.numeric_vars: List[str] = []
        self.parents: Dict[str, List[Tuple[str, float]]] = {}   # target -> [(parent, importance)]
        self.coef_: Dict[str, Dict[str, np.ndarray]] = {}       # target -> parent -> coef_lags
        self.bias_: Dict[str, float] = {}
        self.llm_prior_: Dict[str, Dict[str, float]] = {}       # target -> {parent: confidence}
        self.fi_specs: Dict[str, FiSpec] = {}
        self.fi_models: Dict[str, nn.Module] = {}

        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------- public API -------------------

    def train(self, dataset: List[Dict[str, Any]],
              variable_path: str = "config/variable_config.yaml",
              causal_yaml_path: str = "config/causal_graph.yaml") -> None:
        """
        End-to-end: learn structure + learn functions
        """
        # 0) Load variable space
        self._load_variables(variable_path)

        # 1) episode -> time series array for each variable (per-episode standardization done downstream)
        episode_series = self._episodes_to_series(dataset)

        # 2) Target variables (default: Outcome)
        targets = []
        for g in self.target_groups:
            targets.extend(self.variables.get(g, []))
        # Keep only numeric types (string variables like weather_type will be filtered out)
        targets = [t for t in targets if t in self.numeric_vars]

        # 3) LLM prior (optional, controlled by use_llm_prior)
        if self.use_llm_prior:
            self.llm_prior_ = self._build_llm_prior(targets, variable_path)

        if self.verbose:
            print(f"[Hybrid] targets={targets}, max_lag={self.max_lag}, lambda_base={self.lambda_base}, fi={self.fi_model_type}")

        # 4) Structure learning: use group ridge regression to select parents and record linear coefficients (interpretable)
        for tgt in targets:
            parents, coef_map, bias = self._learn_target_parents(tgt, episode_series)
            self.parents[tgt] = parents
            self.coef_[tgt] = coef_map
            self.bias_[tgt] = bias
            if self.verbose:
                desc = ", ".join([f"{p}({imp:.3f})" for p, imp in parents])
                print(f"[Hybrid][Structure] {tgt} <- {desc}")

        # 5) Function learning: train nonlinear f_Y on selected Pa(Y)
        for tgt in targets:
            pa = [p for p, _ in self.parents.get(tgt, [])]
            if len(pa) == 0:
                if self.verbose:
                    print(f"[Hybrid][Function] skip {tgt} (no parents selected)")
                continue
            self._train_fi_for_target(tgt, pa, episode_series)

        # 6) Save causal graph configuration
        self._save_causal_yaml(causal_yaml_path)

        # 7) Save all f_i models
        self._save_all_fi_models()

    def infer_edges(self) -> List[Dict[str, Any]]:
        edges = []
        group_of = {v: g for g, vs in self.variables.items() for v in vs}
        for tgt, plist in self.parents.items():
            for p, imp in plist:
                sign = np.sign(self.coef_[tgt][p].sum())  # Simple direction estimate: sign of sum of lagged coefficients
                edges.append({
                    "cause": p,
                    "effect": tgt,
                    "weight": float(imp),
                    "direction": "positive" if sign >= 0 else "negative",
                    "cause_group": group_of.get(p, "Unknown"),
                    "effect_group": group_of.get(tgt, "Unknown"),
                    "rule_type_hint": self._map_rule_type(group_of.get(p), group_of.get(tgt))
                })
        return edges

    # ------------------- core steps -------------------

    def _load_variables(self, variable_path: str) -> None:
        with open(variable_path, "r") as f:
            self.variables = yaml.safe_load(f)
        numeric = []
        for group, vs in self.variables.items():
            for v in vs:
                numeric.append(v)
        # Don't filter types here, numeric filtering done in episodes_to_series
        self.numeric_vars = list(dict.fromkeys(numeric))

    def _episodes_to_series(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, List[float]]]:
        episodes_series = []
        for ep in dataset:
            series = defaultdict(list)
            for timestep in ep.get("trajectory", []):
                for group, micro in timestep.items():
                    if not isinstance(micro, dict):
                        continue
                    for var, val in micro.items():
                        if _is_number(val):
                            series[var].append(float(val))
            # Align lengths
            series = {k: v for k, v in series.items() if len(v) > 0}
            if series:
                max_len = max(len(v) for v in series.values())
                for k, v in series.items():
                    if len(v) < max_len:
                        v.extend([v[-1]] * (max_len - len(v)))
                episodes_series.append(series)
        if self.verbose:
            print(f"[Hybrid] built time-series for {len(episodes_series)} episodes")
        return episodes_series

    # ---------- LLM prior ----------
    def _build_llm_prior(self, targets: List[str], variable_path: str) -> Dict[str, Dict[str, float]]:
        with open(variable_path, "r") as f:
            var_cfg = yaml.safe_load(f)

        all_vars = []
        for group, vs in var_cfg.items():
            all_vars.extend(vs)
        all_vars = list(dict.fromkeys(all_vars))

        prior: Dict[str, Dict[str, float]] = {}
        for tgt in targets:
            prompt = f"""
You are assisting causal discovery in highway driving.
Variables: {', '.join(all_vars)}.
Target variable: {tgt}.
List 3-10 most likely direct causes (parents) of {tgt} from the variable list above.
Return strict JSON: {{"parents":[{{"var":"<name>","confidence":<0..1>}}]}}
"""
            try:
                ans = call_gpt(prompt, model=self.llm_model, temperature=0.2, max_tokens=300)
                js = json.loads(ans)
                pairs = js.get("parents", [])
                prior[tgt] = {p["var"]: float(p.get("confidence", 0.5)) for p in pairs if "var" in p}
            except Exception as e:
                if self.verbose:
                    print(f"[Hybrid][LLM] prior failed for {tgt}: {e}")
                prior[tgt] = {}
        # Add fallback section here
        # Fallback: if some targets have no LLM response, use default confidence 0.5
        for tgt in targets:
            if tgt not in prior or len(prior[tgt]) == 0:
                prior[tgt] = {v: 0.5 for g, vs in self.variables.items() for v in vs if v != tgt}

        if self.verbose:
            print(f"[Hybrid][LLM] prior merged for {len(targets)} targets.")
        return prior

    # ---------- Supervised matrix: lag expansion ----------
    # ---------- Supervised: lag expansion (with strong constraint) ----------
    def _build_supervised(self, episode_series, target) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X_list, y_list = [], []

        # 1) Identify the group that target variable belongs to
        target_group = None
        for g, vs in self.variables.items():
            if target in vs:
                target_group = g
                break

        # 2) Hierarchy mapping table (strong constraint: only allow causation between these groups)
        # Whitelist of groups for parent selection (target_group -> allowed parent groups)
        valid_hierarchy = {
            "Environment": [],                                  # Exogenous: no parents
            "Action":      [],                                  # Exogenous: no parents (decouple policy learning from causal learning)

            "Traffic":     ["Environment", "Ego"],              # Weather/road conditions affect traffic; Ego behavior changes headway/density
            "Ego":         ["Environment", "Traffic", "Action"],# Perceived traffic + external conditions + action jointly determine Ego
            "Outcome":     ["Ego", "Traffic", "Action", "Environment"]  # Outcome affected by state, action, and external factors
        }
        allowed_groups = valid_hierarchy.get(target_group, [])

        # 3) Mixed candidate parent set: manual hierarchy UNION LLM prior
        parent_names = []
        for g in allowed_groups:
            parent_names.extend(self.variables.get(g, []))

        # Semantic prior from LLM (if available)
        llm_candidates = []
        if self.use_llm_prior and target in self.llm_prior_:
            llm_candidates = list(self.llm_prior_[target].keys())

        # Merge and deduplicate
        parent_names = list(dict.fromkeys(parent_names + llm_candidates))
        parent_names = [p for p in parent_names if p != target]

        # 4) Filter parent variables that appear in at least one episode
        exist_set = set()
        for series in episode_series:
            for p in parent_names:
                if p in series:
                    exist_set.add(p)
        used_parents = [p for p in parent_names if p in exist_set]

        # 5) Build time series matrix (pad missing with zeros, expand per timestep)
        for series in episode_series:
            if target not in series:
                continue
            T = len(series[target])
            if T <= self.max_lag:
                continue

            z_series = {}
            tgt_arr = np.asarray(series[target], dtype=float)
            tgt_arr = _zscore(tgt_arr)
            z_series[target] = tgt_arr

            for p in used_parents:
                if p in series:
                    arr = np.asarray(series[p], dtype=float)
                    arr = _zscore(arr)
                    if len(arr) < T:
                        arr = np.concatenate([arr, np.repeat(arr[-1], T - len(arr))])
                    z_series[p] = arr
                else:
                    z_series[p] = np.zeros((T,), dtype=float)

            for t in range(self.max_lag, T):
                row = []
                for p in used_parents:
                    for lag in range(1, self.max_lag + 1):
                        row.append(z_series[p][t - lag])
                X_list.append(row)
                y_list.append(z_series[target][t])

        if len(X_list) == 0:
            return np.zeros((0, 0)), np.zeros((0,)), []

        X = np.asarray(X_list, dtype=float)
        y = np.asarray(y_list, dtype=float)
        return X, y, used_parents

    # ---------- Group ridge regression (parent-specific penalty + soft constraint) ----------
    def _solve_group_weighted_ridge(self, X: np.ndarray, y: np.ndarray,
                                    parents: List[str], llm_conf: Dict[str, float],
                                    target: Optional[str] = None) -> np.ndarray:
        if X.size == 0:
            return np.zeros((0,))
        d = X.shape[1]
        L = self.max_lag
        assert d == len(parents) * L

        D = np.zeros((d,), dtype=float)

        # Define valid hierarchy edges (for soft constraint)
        valid_edges = {
            # Environment is exogenous, influences downstream:
            ("Environment", "Traffic"),
            ("Environment", "Ego"),
            ("Environment", "Outcome"),

            # Traffic influences Ego and Outcome:
            ("Traffic", "Ego"),
            ("Traffic", "Outcome"),

            # Ego in turn influences Traffic (lane change/speed affects local traffic), also influences Outcome:
            ("Ego", "Traffic"),
            ("Ego", "Outcome"),

            # Action is exogenous intervention, only points to controlled state and outcome:
            ("Action", "Ego"),
            ("Action", "Outcome"),

            # (Optional) If you want to allow action to directly change traffic, not just indirectly through Ego:
            # ("Action", "Traffic"),
        }


        for i, p in enumerate(parents):
            parent_group = self._group_of(p)
            target_group = self._group_of(target) if target else None

            # Get confidence from LLM prior (default 0.5)
            conf_llm = float(llm_conf.get(p, 0.5))

            # -------------------------------
            # Mixed mode: LLM + manual constraint -> unified confidence
            # -------------------------------
            if (parent_group, target_group) in valid_edges:
                # Valid semantic path -> weighted fusion (prior 70%, hierarchy 30%)
                conf_final = 0.7 * conf_llm + 0.3
            else:
                # Invalid hierarchy -> reduce confidence (prior 30%, hierarchy 0.0)
                conf_final = 0.3 * conf_llm

            # Build lambda_j: penalty inversely proportional to confidence
            lam_j = self.lambda_base * (max(0.0, 1.0 - conf_final) ** self.penalty_power)

            sl = slice(i * L, (i + 1) * L)
            D[sl] = lam_j

            if self.verbose and self.fi_epochs == 5:  # Optional debug output
                print(f"    [Ridge] {p}->{target}: conf_llm={conf_llm:.2f}, conf_final={conf_final:.2f}, lambda={lam_j:.3f}")

        XtX = X.T @ X
        XtX += np.diag(D)
        w = np.linalg.solve(XtX, X.T @ y)
        return w

    # Helper function
    def _group_of(self, var: str) -> str:
        for g, vs in self.variables.items():
            if var in vs:
                return g
        return "Unknown"


    def _learn_target_parents(self, target: str, episode_series) -> Tuple[List[Tuple[str, float]], Dict[str, np.ndarray], float]:
        X, y, parent_names = self._build_supervised(episode_series, target)
        if X.shape[0] == 0 or X.shape[1] == 0:
            return [], {}, 0.0

        # Get LLM confidence mapping (empty dict if not available)
        conf_map = self.llm_prior_.get(target, {}) if self.use_llm_prior else {}
        w = self._solve_group_weighted_ridge(X, y, parent_names, conf_map, target=target)

        if self.verbose:
            top_conf = {p: conf_map.get(p, 0.5) for p in parent_names}
            print(f"[Hybrid][LLM+Data] {target} merged confidence: {json.dumps(top_conf, indent=2)}")


        L = self.max_lag
        coef_map, importances = {}, {}
        for i, p in enumerate(parent_names):
            sl = slice(i * L, (i + 1) * L)
            wl = w[sl]
            coef_map[p] = wl.copy()
            importances[p] = float(np.sum(np.abs(wl)))

        sorted_parents = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        selected = [(p, imp) for p, imp in sorted_parents if imp >= self.min_importance]
        if len(selected) > self.top_k:
            selected = selected[: self.top_k]
        return selected, coef_map, 0.0

    # ---------- Train f_i ----------
    def _train_fi_for_target(self, target: str, parents: List[str], episode_series) -> None:
        as_sequence = (self.fi_model_type == "transformer")
        ds = FiDataset(
            episode_series=episode_series,
            target=target,
            parents=parents,
            max_lag=self.max_lag,
            zscore=True,
            as_sequence=as_sequence
        )
        if len(ds) < 50:
            if self.verbose:
                print(f"[Hybrid][Function] {target}: not enough samples ({len(ds)}), skip.")
            return

        dl = DataLoader(ds, batch_size=self.fi_batch_size, shuffle=True, drop_last=False)
        device = _device()

        if self.fi_model_type == "mlp":
            input_dim = len(parents) * self.max_lag
            model = FunctionMLP(input_dim=input_dim, hidden_dim=self.fi_hidden, dropout=self.fi_dropout).to(device)
        elif self.fi_model_type == "transformer":
            model = FunctionTransformer(num_vars=len(parents), model_dim=self.fi_hidden, dropout=0.2).to(device)
            # [FIX] Align with the signature we implemented in function_models:
            # FunctionTransformer(num_parents: int, seq_len: int, d_model: int)
            # model = FunctionTransformer(num_parents=len(parents), seq_len=self.max_lag, d_model=self.fi_hidden).to(device)
        else:
            raise ValueError(f"Unknown fi_model_type: {self.fi_model_type}")

        opt = optim.Adam(model.parameters(), lr=self.fi_lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        model.train()
        epoch_losses = []
        for epoch in range(self.fi_epochs):
            losses = []
            for xb, yb in dl:
                xb = xb.to(device)   # [B, P*L] or [B, L, P]
                yb = yb.to(device)

                if self.fi_model_type == "mlp":
                    # xb: [B, P*L]
                    # pred = model(xb).squeeze(1)
                    pred = model(xb)
                else:
                    # transformer: xb: [B, L, P]
                    # function_models.FunctionTransformer expects [B, L, P]
                    # pred = model(xb).squeeze(1)
                    pred = model(xb)

                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())

            avg_loss = float(np.mean(losses))
            epoch_losses.append(avg_loss)
            if self.verbose and (epoch + 1) % max(1, self.fi_epochs // 2) == 0:
                print(f"[Hybrid][Function] {target} :: epoch {epoch+1}/{self.fi_epochs}  loss={np.mean(losses):.4f}")

        # Save loss curve for each target variable
        loss_path = os.path.join(self.save_dir, f"loss_fi_{target}.txt")
        np.savetxt(loss_path, np.array(epoch_losses), fmt="%.6f")
        if self.verbose:
            print(f"[Hybrid][Function] saved loss curve -> {loss_path}")

        # Save spec & model
        spec = FiSpec(
            parents=parents,
            model_type=self.fi_model_type,
            input_dim=len(parents) * self.max_lag,
            num_parents=len(parents),
            max_lag=self.max_lag,
            path=os.path.join(self.save_dir, f"fi_{target}.pt")
        )
        self.fi_specs[target] = spec
        self.fi_models[target] = model

        # Save single model immediately
        torch.save({
            "model_state_dict": model.state_dict(),
            "spec": spec.__dict__
        }, spec.path)
        if self.verbose:
            print(f"[Hybrid][Function] saved f_{target} -> {spec.path}")

    # ------------------- save / utils -------------------

    def _map_rule_type(self, cg: str, eg: str) -> str:
        cg = (cg or "").strip()
        eg = (eg or "").strip()

        # Behavioral: action-driven influence (note: does not include Action->Traffic)
        if cg == "Action" and eg in ("Ego", "Outcome"):
            return "Behavioral"

        # Scene: scene/traffic influence on vehicle or outcome, and Environment->Traffic
        if cg == "Environment" and eg in ("Traffic", "Ego", "Outcome"):
            return "Scene"
        if cg == "Traffic" and eg in ("Ego", "Outcome"):
            return "Scene"

        # Physical: changes caused by vehicle's own state (including feedback to traffic)
        if cg == "Ego" and eg in ("Traffic", "Ego", "Outcome"):
            return "Physical"

        return "Unknown"


    def _save_causal_yaml(self, path: str) -> None:
        edges = self.infer_edges()
        content = {
            "Variables": self.variables,
            "NumericVariables": self.numeric_vars,
            "Edges": edges,
            "MaxLag": self.max_lag,
            "LambdaBase": self.lambda_base,
            "PenaltyPower": self.penalty_power,
            "TopK": self.top_k,
            "MinImportance": self.min_importance,
            "UseLLMPrior": self.use_llm_prior,
            "FiModelType": self.fi_model_type
        }
        with open(path, "w") as f:
            yaml.dump(content, f, allow_unicode=True)
        if self.verbose:
            print(f"[Hybrid] Saved causal graph with {len(edges)} edges -> {path}")

    def _save_all_fi_models(self):
        # Aggregate specs into an index file
        idx = {tgt: spec.__dict__ for tgt, spec in self.fi_specs.items()}
        with open(os.path.join(self.save_dir, "fi_index.json"), "w") as f:
            json.dump(idx, f, indent=2, ensure_ascii=False)
        if self.verbose:
            print(f"[Hybrid] Saved fi_index.json with {len(idx)} models")

    # Runtime prediction for a target (using trained f_i)
    def predict_target(self, target: str, history_frames: List[Dict[str, Any]]) -> Optional[float]:
        if target not in self.fi_models or target not in self.fi_specs:
            return None
        spec = self.fi_specs[target]
        model = self.fi_models[target].to(_device())
        model.eval()

        # Build input from the last max_lag frames
        frames = history_frames[-spec.max_lag:]
        if len(frames) < spec.max_lag:
            return None

        # Flatten dictionary
        def flat(ts):
            flat = {}
            for g, m in ts.items():
                if isinstance(m, dict):
                    for k, v in m.items():
                        flat[k] = v
            return flat

        # [FIX] Build input with the same parent variable order as during training; pad missing variables with 0
        seq = []
        for lag in range(1, spec.max_lag + 1):
            f = flat(frames[-lag])
            row = [float(f.get(p, 0.0) or 0.0) for p in spec.parents]
            seq.append(row)  # [P] for this lag
        seq = np.asarray(seq, dtype=float)  # [L, P]

        with torch.no_grad():
            if spec.model_type == "mlp":
                # [1, P*L]
                x = torch.tensor(seq.T.reshape(1, -1), dtype=torch.float32, device=_device())
                y = model(x).item()
            else:
                # [1, L, P]
                x = torch.tensor(seq.reshape(1, spec.max_lag, len(spec.parents)),
                                 dtype=torch.float32, device=_device())
                y = model(x).item()
        return y

        # ======= NEW: Forward simulation for inference (called by MPC/PUCT) =======
    def predict_next(
        self,
        sample: Dict[str, Any],
        action: str,
        dt: float,
        history_frames: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Predict next step grouped state:
        - Prioritize using trained f_i to override (risk/energy/comfort/ego_x/y/vx/vy etc.)
        - Use conservative kinematics and heuristics when no model available
        - Environment is exogenous: pass through
        - Action: provide both string and index/one-hot for downstream convenience
        Return structure consistent with sampler/runner: {Environment:{}, Traffic:{}, Ego:{}, Action:{}, Outcome:{}}
        """
        def _getg(d, g, k, default=0.0):
            try:
                v = d.get(g, {}).get(k, default)
                return float(v) if isinstance(v, (int, float)) else v
            except Exception:
                return default

        # --- 0) Extract necessary current values ---
        ego_x  = _getg(sample, "Ego", "ego_x", 0.0)
        ego_y  = _getg(sample, "Ego", "ego_y", 0.0)
        ego_vx = _getg(sample, "Ego", "ego_vx", _getg(sample, "Ego", "ego_speed", 0.0))
        ego_vy = _getg(sample, "Ego", "ego_vy", 0.0)

        headway = _getg(sample, "Traffic", "headway_distance", 1e6)
        lead_ttc = _getg(sample, "Traffic", "lead_ttc", 1e9)
        lead_rel_speed = _getg(sample, "Traffic", "lead_rel_speed", 0.0)
        rear_distance  = _getg(sample, "Traffic", "rear_distance", 1e6)
        lane_id = _getg(sample, "Ego", "lane_id", 0.0)
        lat_off = _getg(sample, "Ego", "lateral_offset", 0.0)
        gap_L = _getg(sample, "Traffic", "available_gap_left", 1e6)
        gap_R = _getg(sample, "Traffic", "available_gap_right", 1e6)
        cutin_risk = _getg(sample, "Traffic", "cutin_risk", 0.0)

        risk = _getg(sample, "Outcome", "risk_score", 0.5)
        energy = _getg(sample, "Outcome", "energy_consumption", 0.1)
        comfort = _getg(sample, "Outcome", "comfort_score", 0.8)

        # --- 1) Process action ---
        a_str = (action if isinstance(action, str) else str(action)).upper()
        action_map = {"LANE_LEFT":0, "LANE_RIGHT":1, "IDLE":2, "FASTER":3, "SLOWER":4}
        a_idx = action_map.get(a_str, 2)
        a_onehot = [1.0 if i==a_idx else 0.0 for i in range(5)]

        # --- 2) Heuristic kinematics (fallback; f_i will override if available) ---
        ax = 0.0
        if a_str == "FASTER": ax = +1.5
        elif a_str == "SLOWER": ax = -1.5

        vx_next = max(0.0, ego_vx + ax * dt)
        x_next  = ego_x + vx_next * dt

        vy_next = ego_vy
        y_next  = ego_y
        lat_next = lat_off
        lane_next = lane_id

        # Lightweight lateral displacement (don't jump lanes directly, leave room for f_i)
        lane_width = 4.0
        lateral_step = 0.25 * lane_width
        if a_str == "LANE_LEFT":
            vy_next = +1.0
            lat_next = lat_off + lateral_step
        elif a_str == "LANE_RIGHT":
            vy_next = -1.0
            lat_next = lat_off - lateral_step

        # --- 3) Rough update of traffic variable trends (fallback; f_i will override if available) ---
        # Acceleration usually decreases headway/decreases TTC, deceleration is opposite
        lead_ttc_next = max(1.0, lead_ttc + (50.0 if ax <= 0 else -50.0))
        headway_next  = max(1.0, headway + (2.0 if ax <= 0 else -2.0))

        # Cut-in risk: higher risk with smaller gap; acceleration slightly increases, deceleration slightly decreases (conservative)
        def _norm_gap(g): return max(0.0, min(1.0, (20.0 - g) / 20.0))
        cutin_blend = max(_norm_gap(gap_L), _norm_gap(gap_R))
        cutin_next = 1.0 - np.exp(-(0.6*cutin_blend + 0.2*(ax>0) + 0.0*(ax<0)))

        # --- 4) Assemble pseudo "next frame" to feed f_i for override ---
        if history_frames is None or len(history_frames) == 0:
            history_frames = [sample] * max(1, self.max_lag)
        else:
            history_frames = list(history_frames)

        pseudo_next = {
            "Environment": sample.get("Environment", {}).copy(),  # Exogenous: pass through
            "Action": {"action_type": a_str, "action_index": a_idx, "action_onehot": a_onehot},
            "Ego": {
                "ego_x": x_next, "ego_y": y_next,
                "ego_vx": vx_next, "ego_vy": vy_next,
                "ego_speed": vx_next,     # Align with vx
                "lane_id": lane_next, "lateral_offset": lat_next
            },
            "Traffic": {
                "headway_distance": headway_next,
                "lead_ttc": lead_ttc_next,
                "lead_rel_speed": lead_rel_speed,
                "rear_distance": rear_distance,
                "available_gap_left": gap_L,
                "available_gap_right": gap_R,
                "cutin_risk": cutin_next
            },
            "Outcome": {
                "risk_score": risk,
                "energy_consumption": energy,
                "comfort_score": comfort
            }
        }

        hist_for_fi = (history_frames + [pseudo_next])[-self.max_lag:]

        def _try_fi(name: str, default_val: float) -> float:
            y = self.predict_target(name, hist_for_fi)
            if y is None or not np.isfinite(y):  # No model or unstable -> use fallback
                return default_val
            return float(y)

        # --- 5) Use f_i to override key targets (override if available, otherwise keep heuristics) ---
        # Outcome variables
        pseudo_next["Outcome"]["risk_score"]          = _try_fi("risk_score",          pseudo_next["Outcome"]["risk_score"])
        pseudo_next["Outcome"]["energy_consumption"]  = _try_fi("energy_consumption",  pseudo_next["Outcome"]["energy_consumption"])
        pseudo_next["Outcome"]["comfort_score"]       = _try_fi("comfort_score",       pseudo_next["Outcome"]["comfort_score"])
        # Ego kinematics (if you also trained these targets)
        pseudo_next["Ego"]["ego_vx"] = _try_fi("ego_vx", pseudo_next["Ego"]["ego_vx"])
        pseudo_next["Ego"]["ego_vy"] = _try_fi("ego_vy", pseudo_next["Ego"]["ego_vy"])
        pseudo_next["Ego"]["ego_x"]  = _try_fi("ego_x",  pseudo_next["Ego"]["ego_x"])
        pseudo_next["Ego"]["ego_y"]  = _try_fi("ego_y",  pseudo_next["Ego"]["ego_y"])
        pseudo_next["Ego"]["ego_speed"] = pseudo_next["Ego"]["ego_vx"]

        # Optional: if you trained predictable Traffic variables, also try to override
        pseudo_next["Traffic"]["headway_distance"] = _try_fi("headway_distance", pseudo_next["Traffic"]["headway_distance"])
        pseudo_next["Traffic"]["lead_ttc"]         = _try_fi("lead_ttc",         pseudo_next["Traffic"]["lead_ttc"])

        return pseudo_next

