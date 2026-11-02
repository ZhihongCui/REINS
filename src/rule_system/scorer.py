import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Union, Tuple, Optional
from torch.utils.data import Dataset
from src.rule_system.engine import parse_rule_to_tokens


# ======================================================
# Rule Scorer Model (Dual-head: S_rule + S_act)
#   - Head 1 head_rule: Rule-state matching score S_rule in (0,1)
#   - Head 2 head_act: 5 action logits (LANE_LEFT, LANE_RIGHT, IDLE, FASTER, SLOWER)
#   - Compatible with old interface score(): only returns S_rule, minimizing changes to existing calls
#   - New interfaces score_with_action() / score_both()
# ======================================================
class RuleScorer(nn.Module):
    """
    Transformer-based rule scoring model.
    S_rule: Evaluates the matching degree between rules and current driving state (0~1)
    S_act: Evaluates the consistency between rule-implied action and current candidate action (softmax probability)
    """
    def __init__(self, state_dim: int,
                 vocab_path: str = "config/vocab.json",
                 embed_dim: int = 128, num_layers: int = 2, num_heads: int = 4,
                 max_rule_len: int = 32, dropout: float = 0.2, noise_std: float = 0.01):
        super().__init__()

        # Load vocabulary
        assert os.path.exists(vocab_path), f"Vocab file not found: {vocab_path}"
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        self.vocab_size = len(self.vocab)
        self.embed_dim = embed_dim
        self.max_rule_len = max_rule_len
        self.noise_std = noise_std

        # ===== Action enumeration (fixed mapping, used uniformly externally) =====
        self.action_to_idx = {"LANE_LEFT": 0, "LANE_RIGHT": 1, "IDLE": 2, "FASTER": 3, "SLOWER": 4}
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}

        # ===== Embedding layer =====
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embed_dim,
            padding_idx=self.vocab.get("[PAD]", 0)
        )

        # ===== State encoder =====
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # ===== Transformer Encoder =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ===== Dual scoring heads (replacing original scoring_head) =====
        self.head_rule = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)   # -> sigmoid => S_rule
        )
        self.head_act = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 5)   # logits for 5 actions
        )

    # ---------------- Core forward ----------------
    def forward(self, state_vec: torch.Tensor, rule_strs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        state_vec: [B_s, D]
        rule_strs: list[str] length = B_r
        Returns:
          s_rule:    [B, 1]    (after sigmoid)
          act_logits:[B, 5]    (before softmax)
        Compatible batch alignment rules:
        - If B_s == 1 and B_r > 1: duplicate state to B_r
        - If B_r == 1 and B_s > 1: duplicate rule to B_s
        - Otherwise requires B_s == B_r
        """
        device = state_vec.device
        B_s = state_vec.size(0)
        B_r = len(rule_strs)

        # --- Align batch sizes ---
        if B_s != B_r:
            if B_s == 1 and B_r > 1:
                state_vec = state_vec.expand(B_r, -1)      # duplicate state
            elif B_r == 1 and B_s > 1:
                rule_strs = rule_strs * B_s               # duplicate rule
            else:
                raise ValueError(f"Batch mismatch: states={B_s}, rules={B_r}. "
                                 f"Either pass 1 state with many rules, 1 rule with many states, "
                                 f"or make them equal.")
        B = max(B_s, B_r)

        # 1) Tokenize
        token_lists = [parse_rule_to_tokens(rule, self.vocab, self.max_rule_len) for rule in rule_strs]
        for i, t in enumerate(token_lists):
            if not t:
                token_lists[i] = [self.vocab.get("[UNK]", 1)]

        max_len = max(len(t) for t in token_lists)
        pad_id = self.vocab.get("[PAD]", 0)
        padded_tokens = [t + [pad_id] * (max_len - len(t)) for t in token_lists]
        token_tensor = torch.tensor(padded_tokens, dtype=torch.long, device=device)  # [B, T]

        # 2) Embed tokens + encode state
        rule_embed = self.token_embedding(token_tensor)  # [B, T, D]
        state_embed = self.state_encoder(state_vec).unsqueeze(1)  # [B, 1, D]

        # Add noise during training
        if self.training and self.noise_std > 0:
            rule_embed = rule_embed + torch.randn_like(rule_embed) * self.noise_std

        # 3) concat & encode
        transformer_input = torch.cat([state_embed, rule_embed], dim=1)  # [B, 1+T, D]
        transformer_output = self.transformer_encoder(transformer_input)  # [B, 1+T, D]
        pooled = transformer_output[:, 0, :]  # [B, D] take CLS position

        # 4) Dual head output
        rule_logit = self.head_rule(pooled)             # [B, 1] raw logits (no sigmoid)
        act_logits = self.head_act(pooled)              # [B, 5]
        return rule_logit, act_logits


    # ---------------- Compatible with old interface: only returns S_rule ----------------
    def score(self, rules: List[Dict], samples: List[Dict]) -> List[Dict]:
        """
        Compatible with old interface: only returns rule-state matching S_rule.
        Returns: [{rule_text, cause, effect, score}, ...]  (expanded for each sample)
        """
        device = next(self.parameters()).device
        results = []
        for sample in samples:
            state_vec = self._flatten_state(sample)
            state_tensor = torch.tensor([state_vec], dtype=torch.float32, device=device)

            rule_texts = [r.get("raw_text", r.get("rule", "")) for r in rules]
            chunk = 256
            all_scores = []
            with torch.no_grad():
                for i in range(0, len(rule_texts), chunk):
                    rule_logit, _ = self(state_tensor, rule_texts[i:i+chunk])
                    scores = torch.sigmoid(rule_logit).cpu().numpy().flatten().tolist()
                    all_scores.extend(scores)

            for rule, sc in zip(rules, all_scores):
                results.append({
                    "rule_text": rule.get("raw_text", rule.get("rule", "")),
                    "cause": rule.get("cause", ""),
                    "effect": rule.get("effect", ""),
                    "score": float(sc)  # For compatibility, field name remains score (i.e., S_rule)
                })
        return results

    # ---------------- New interface: given candidate action, returns (S_rule, S_act) ----------------
    @torch.no_grad()
    def score_with_action(self, state_vec: torch.Tensor, rule_strs: List[str], action_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        state_vec: [B_s, D] or [1, D]
        rule_strs: list[str] length B_r
        action_idx: 0..4, corresponding to self.action_to_idx
        Returns:
            s_rule: [B,1]  rule-state matching
            s_act : [B,1]  rule-action consistency (softmax probability for given action_idx)
        """
        rule_logit, act_logits = self(state_vec, rule_strs)
        s_rule = torch.sigmoid(rule_logit)              # [B,1]
        s_act_full = torch.softmax(act_logits, dim=-1)  # [B,5]
        s_act = s_act_full[:, action_idx:action_idx+1]
        return s_rule, s_act


    # ---------------- New interface: batch output by action name ----------------
    @torch.no_grad()
    def score_both(self, rules: List[Dict], samples: List[Dict], action_str: str) -> List[Dict]:
        """
        Returns [{rule_text, cause, effect, s_rule, s_act}], for combined scoring in MCTS/decision making.
        action_str: "LANE_LEFT"/"LANE_RIGHT"/"IDLE"/"FASTER"/"SLOWER"
        """
        device = next(self.parameters()).device
        action_idx = self.get_action_index(action_str)
        out = []
        for sample in samples:
            state_vec = self._flatten_state(sample)
            state_tensor = torch.tensor([state_vec], dtype=torch.float32, device=device)

            rule_texts = [r.get("raw_text", r.get("rule", "")) for r in rules]
            chunk = 256
            all_s_rule, all_s_act = [], []
            for i in range(0, len(rule_texts), chunk):
                rule_logit, s_act = self.score_with_action(state_tensor, rule_texts[i:i+chunk], action_idx)
                all_s_rule.extend(s_rule.detach().cpu().numpy().flatten().tolist())
                all_s_act.extend(s_act.detach().cpu().numpy().flatten().tolist())

            for rule, sr, sa in zip(rules, all_s_rule, all_s_act):
                out.append({
                    "rule_text": rule.get("raw_text", rule.get("rule", "")),
                    "cause": rule.get("cause", ""),
                    "effect": rule.get("effect", ""),
                    "s_rule": float(sr),
                    "s_act":  float(sa)
                })
        return out

    # ---------------- Utility: action name <-> index ----------------
    def get_action_index(self, action: str) -> int:
        a = (action or "").upper()
        if a not in self.action_to_idx:
            raise ValueError(f"Unknown action: {action}. Valid: {list(self.action_to_idx.keys())}")
        return self.action_to_idx[a]

    # ==================================================
    # Load checkpoint compatible with vocab size changes
    # ==================================================
    def load_partial_state_dict(self, ckpt_path: str):
        print(f"[INFO] Loading RuleScorer weights from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_state = self.state_dict()

        for name, param in ckpt.items():
            if name in model_state and model_state[name].shape != param.shape:
                print(f"[WARN] Shape mismatch for {name}: checkpoint {tuple(param.shape)} vs {tuple(model_state[name].shape)}")
                if "token_embedding.weight" in name:
                    old_num, dim = param.shape
                    new_num, _ = model_state[name].shape
                    model_state[name][:old_num, :] = param
                    print(f"[OK] Partially loaded {old_num}/{new_num} embeddings.")
                else:
                    continue
            else:
                model_state[name] = param

        self.load_state_dict(model_state, strict=False)
        print("[OK] RuleScorer weights loaded successfully (shape-safe mode).")

    # ==================================================
    # Utility functions
    # ==================================================
    def _flatten_state(self, timestep: Dict) -> List[float]:
        flat = []
        for macro_key in ["Environment", "Traffic", "Ego", "Action", "Outcome"]:
            macro = timestep.get(macro_key, {})
            if isinstance(macro, dict):
                for v in macro.values():
                    flat.append(float(v) if isinstance(v, (int, float)) else 0.0)
        return flat


# ======================================================
# Rule Scoring Dataset
#   - Compatible with old triplet (state_vec, rule_text, label)
#   - Optionally when supervise_action=True, additionally returns action_idx (for second head supervision)
#     Requires each timestep to provide chosen_action_name (string)
# ======================================================
class RuleScoringDataset(Dataset):
    """
    Dataset for training RuleScorer.
    Each sample contains:
      - Basic: (state_vec, rule_text, y_rule)
      - Optional: (state_vec, rule_text, y_rule, y_act_idx) when supervise_action=True
    """
    def __init__(self, samples: List[Dict], rules: List[Dict], vocab: Dict,
                 max_rule_len: int = 32, supervise_action: bool = False):
        self.samples = []
        self.vocab = vocab
        self.max_rule_len = max_rule_len
        self.supervise_action = supervise_action

        # Action mapping consistent with the model (in case of external use)
        self.action_to_idx = {"LANE_LEFT": 0, "LANE_RIGHT": 1, "IDLE": 2, "FASTER": 3, "SLOWER": 4}

        for episode in samples:
            for timestep in episode.get("trajectory", []):
                state_vec = self.flatten_state(timestep)
                # Rule relevance label: (cause present) AND (effect non-zero) -> 1.0
                for rule in rules:
                    cause = rule.get("cause")
                    effect = rule.get("effect")
                    cause_vars = cause if isinstance(cause, list) else [cause]
                    cause_val = any(self._get_val(timestep, c) != 0 for c in cause_vars)
                    effect_val = self._get_val(timestep, effect)
                    y_rule = 1.0 if cause_val and effect_val != 0 else 0.0

                    if self.supervise_action:
                        act_name = (timestep.get("chosen_action_name") or timestep.get("action") or "").upper()
                        y_act = self.action_to_idx.get(act_name, 2)  # Default IDLE=2, to avoid errors when missing
                        self.samples.append((state_vec, rule.get("raw_text", ""), y_rule, y_act))
                    else:
                        self.samples.append((state_vec, rule.get("raw_text", ""), y_rule))

    def _get_val(self, timestep, var):
        for macro in ["Environment", "Traffic", "Ego", "Action", "Outcome"]:
            macro_dict = timestep.get(macro, {})
            if isinstance(macro_dict, dict) and var in macro_dict:
                return macro_dict[var]
        return 0.0

    def flatten_state(self, timestep):
        flat = []
        for macro_key in ["Environment", "Traffic", "Ego", "Action", "Outcome"]:
            macro = timestep.get(macro_key, {})
            if isinstance(macro, dict):
                for v in macro.values():
                    flat.append(float(v) if isinstance(v, (int, float)) else 0.0)
        return flat

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.supervise_action:
            state_vec, rule_text, y_rule, y_act = self.samples[idx]
            state_vec = torch.tensor(state_vec, dtype=torch.float32)
            y_rule = torch.tensor(y_rule, dtype=torch.float32)  # [1] for BCELoss
            y_act = torch.tensor(y_act, dtype=torch.long)                     # [] for CE
            return state_vec, rule_text, y_rule, y_act
        else:
            state_vec, rule_text, y_rule = self.samples[idx]
            state_vec = torch.tensor(state_vec, dtype=torch.float32)
            y_rule = torch.tensor(y_rule, dtype=torch.float32) # [1]
            return state_vec, rule_text, y_rule


# ======================================================
# Builder
# ======================================================
def build_rule_scorer(method: str = "transformer", state_dim: int = 30, **kwargs):
    if method == "transformer":
        return RuleScorer(state_dim=state_dim, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported rule scorer method: {method}")
