# modules/rule_extractor.py
# -*- coding: utf-8 -*-
import json
import re
from typing import List, Dict, Union, Optional, Any

from src.llm.caller import call_gpt
from src.utils.common import load_yaml


CANON_ACTIONS = {"LANE_LEFT", "LANE_RIGHT", "IDLE", "FASTER", "SLOWER"}

ACTION_MAP = {
    "brake": "SLOWER", "decelerate": "SLOWER", "reduce speed": "SLOWER", "slow down": "SLOWER",
    "accelerate": "FASTER", "increase speed": "FASTER", "speed up": "FASTER",
    "keep lane": "IDLE", "maintain": "IDLE", "stay": "IDLE", "idle": "IDLE",
    "lane change left": "LANE_LEFT", "turn left": "LANE_LEFT", "go left": "LANE_LEFT", "lane_left":"LANE_LEFT",
    "lane change right": "LANE_RIGHT", "turn right": "LANE_RIGHT", "go right": "LANE_RIGHT", "lane_right":"LANE_RIGHT",
}

# —— Added after CANON_ACTIONS / ACTION_MAP —— #

# Variables that "imply" lateral lane change, safety gap, etc.
LANE_LEFT_KEYS  = {"left_front", "left_rear", "available_gap_left", "lane_density_left", "left", "lane_left"}
LANE_RIGHT_KEYS = {"right_front", "right_rear", "available_gap_right", "lane_density_right", "right", "lane_right"}
LONGIT_FRONT_KEYS = {"lead_ttc", "lead_thw", "headway_distance", "lead_speed", "lead_rel_speed",
                     "headway", "front_distance", "front_speed"}
LONGIT_REAR_KEYS  = {"rear_ttc", "rear_distance", "rear_speed", "rear_rel_speed"}

def _has_any(d: dict, keys: set) -> bool:
    return any(k in d for k in keys)

def _get_op_val(ths: dict, key: str):
    """Extract (op, value) from thresholds[key], return (None, None) if not found"""
    x = ths.get(key, {})
    if not isinstance(x, dict):
        return None, None
    return x.get("op"), x.get("value")


def _canon_action(text: str) -> str:
    """map free-form action text to highway 5-class actions"""
    t = (text or "").lower()
    for k, v in ACTION_MAP.items():
        if k in t:
            return v
    return ""


def _dedup_keep_order(items: List[Any]) -> List[Any]:
    seen, out = set(), []
    for x in items:
        key = json.dumps(x, sort_keys=True, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
        if key not in seen:
            seen.add(key); out.append(x)
    return out


class RuleExtractor:
    """
    Training phase: Expand [structured executable rules] based on causal edges (cause→effect).
    - Force JSONL output (one object per line)
    - Support 7 types: IF_THEN, AND_THEN, THRESHOLD, PRIORITY, MULTI_ACTION, WEIGHTED_ACTION, TEMPORAL_CHAIN
    - Parameterized thresholds (abs / z / q) + local calibration to concrete values
    - Unified actions: LANE_LEFT / LANE_RIGHT / IDLE / FASTER / SLOWER
    """

    def __init__(self,
                 template_path="src/llm/prompts/causal_to_rule.txt",
                 rule_index_config="config/rule_index_definitions.yaml",
                 model_name="gpt-4.1-nano",
                 temperature=0.5):
        self.model_name = model_name
        self.temperature = temperature
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = f.read()
        self.rule_index_dict = load_yaml(rule_index_config)

    # ----------------------- public API -----------------------

    def extract_rules(self,
                      causal_edges: List[Dict[str, Union[str, List[str]]]],
                      variable_stats: Optional[Dict[str, Dict[str, float]]] = None,
                      diversity_retries: int = 2) -> List[Dict]:
        """
        Expand rules from structural edges and perform local calibration on threshold parameters.
        :param causal_edges: [{"cause":..., "effect":..., "weight":..., "direction":..., "rule_type_hint":...}, ...]
        :param variable_stats: {var: {"mean","std","p20","p50","p80","min","max"}}
        :param diversity_retries: If rule types are monotonous (all IF_THEN), auto-retry with higher temperature
        """
        all_rules: List[Dict] = []

        for edge in causal_edges:
            causes = edge.get("cause") or edge.get("causes")
            effect = edge.get("effect")
            weight = edge.get("weight", None)
            direction = edge.get("direction", "positive")
            rule_type_hint = edge.get("rule_type_hint", "Unknown")
            if not causes or not effect:
                print(f"[!] Skipped invalid edge: {edge}")
                continue

            cause_list = causes if isinstance(causes, list) else [str(causes)]
            cause_text = " AND ".join(map(str, cause_list))

            # 1) Compose prompt (only replace {cause}/{effect}), and attach related variable statistics
            base_prompt = self._fill_template(cause_text, str(effect))
            prompt = self._attach_stats_to_prompt(base_prompt, cause_list, str(effect), variable_stats)

            # 2) Call LLM, JSON-only, diversity retry
            temp = self.temperature
            collected: List[Dict] = []
            for attempt in range(1 + diversity_retries):
                lines = self._call_llm_lines(prompt, temperature=temp)
                parsed = self._parse_lines_json_first(lines)
                if parsed:
                    collected.extend(parsed)
                # Diversity check: whether there are non-IF_THEN types
                types = {str(p.get("rule_index", "IF_THEN")).upper() for p in parsed}
                if any(t in {"THRESHOLD", "PRIORITY", "MULTI_ACTION", "WEIGHTED_ACTION", "TEMPORAL_CHAIN"} for t in types):
                    break
                temp = min(1.0, temp + 0.2)

            # 3) Normalize + inject meta information
            fixed = []
            for r in collected:
                rr = self._normalize_rule_dict(r)
                rr.update({
                    "raw_text": rr.get("rule_text", ""),
                    "cause": causes,
                    "effect": effect,
                    "weight": float(weight) if weight is not None else None,
                    "direction": direction,
                    "type": rr.get("rule_index", "UNKNOWN"),
                    "rule_type_hint": rule_type_hint,
                    "source": "LLM+Data",
                    "rule": rr.get("rule_text", "").replace("\n", " ").strip(),
                })
                fixed.append(rr)

            # 4) Local calibration (convert {form: z/q} to value/op/unit), use our own inference to forcibly determine action, and write confidence
            calibrated = [self._calibrate_rule(r, variable_stats) for r in fixed]

            calibrated = [self._finalize_action_and_conf(r, variable_stats) for r in calibrated]

            # 5) Deduplicate + final collection
            calibrated = _dedup_keep_order(calibrated)
            for r in calibrated:
                all_rules.append(r)
                print(f"[✔] Rule parsed: {r.get('rule_text','').strip()}")

        print(f"[✓] Extracted {len(all_rules)} total rules.")
        return all_rules

    # ===== New: Action inference based on threshold/variable semantics/causal edges =====
    def _infer_action_from_rule(self, rule: Dict, variable_stats: Optional[Dict[str, Dict[str, float]]] = None) -> str:
        """
        Infer one of 5 action classes based on params.thresholds + cause/effect semantics.
        - Longitudinal: headway_distance small / lead_ttc small / lead_rel_speed > 0 => SLOWER
                      : headway_distance large / lead_ttc large / lead_rel_speed < 0 => FASTER
        - Lateral (left): available_gap_left large, left_front/rear_distance sufficient, and left side sparser => LANE_LEFT
        - Lateral (right): similarly => LANE_RIGHT
        - Other / insufficient info => IDLE
        """
        ths = ((rule.get("params") or {}).get("thresholds") or {})
        causes = rule.get("cause") or rule.get("causes") or []
        effects = rule.get("effect") or rule.get("effects") or []
        cause_set = set(causes if isinstance(causes, list) else [causes])
        effect_set = set(effects  if isinstance(effects,  list) else [effects])

        # 1) First check if clearly doing "lane change decision"
        if _has_any(ths, LANE_LEFT_KEYS) and not _has_any(ths, LANE_RIGHT_KEYS):
            # Left gap sufficient + front/rear distance sufficient + left sparser
            ok_gap_op, ok_gap_val = _get_op_val(ths, "available_gap_left")
            lf_op, lf_val = _get_op_val(ths, "left_front_distance")
            lr_op, lr_val = _get_op_val(ths, "left_rear_distance")

            left_sparse = False
            ld_l_op, ld_l_val = _get_op_val(ths, "lane_density_left")
            ld_r_op, ld_r_val = _get_op_val(ths, "lane_density_right")
            if (ld_l_op in {"<", "<="}) and (ld_r_op in {">", ">="}):
                left_sparse = True

            if (ok_gap_op in {">", ">="}) and (lf_op in {">", ">="}) and (lr_op in {">", ">="}) and left_sparse:
                return "LANE_LEFT"

        if _has_any(ths, LANE_RIGHT_KEYS) and not _has_any(ths, LANE_LEFT_KEYS):
            ok_gap_op, ok_gap_val = _get_op_val(ths, "available_gap_right")
            rf_op, rf_val = _get_op_val(ths, "right_front_distance")
            rr_op, rr_val = _get_op_val(ths, "right_rear_distance")

            right_sparse = False
            ld_r_op, ld_r_val = _get_op_val(ths, "lane_density_right")
            ld_l_op, ld_l_val = _get_op_val(ths, "lane_density_left")
            if (ld_r_op in {"<", "<="}) and (ld_l_op in {">", ">="}):
                right_sparse = True

            if (ok_gap_op in {">", ">="}) and (rf_op in {">", ">="}) and (rr_op in {">", ">="}) and right_sparse:
                return "LANE_RIGHT"

        # 2) Then check longitudinal control (SLOWER / FASTER)
        # SLOWER: headway distance small or TTC small, and lead_rel_speed > 0 (I'm faster)
        hd_op, hd_val = _get_op_val(ths, "headway_distance")
        ttc_op, ttc_val = _get_op_val(ths, "lead_ttc")
        rel_op, rel_val = _get_op_val(ths, "lead_rel_speed")

        slow_sign = False
        if (hd_op in {"<", "<="}) or (ttc_op in {"<", "<="}):
            slow_sign = True
        if slow_sign and (rel_op in {">", ">="}):
            return "SLOWER"

        # FASTER: headway large / TTC large, and lead_rel_speed < 0 (I'm slower)
        fast_sign = False
        if (hd_op in {">", ">="}) or (ttc_op in {">", ">="}):
            fast_sign = True
        if fast_sign and (rel_op in {"<", "<="}):
            return "FASTER"

        # 3) If effect is ego_lane, and neither side satisfies, keep IDLE; if effect is ego_speed but signal insufficient, also IDLE
        return "IDLE"

    # ----------------------- prompt helpers -----------------------

    def _fill_template(self, cause_text: str, effect_text: str) -> str:
        # Only replace {cause}/{effect}, avoid conflicts between braces and JSON
        t = self.template
        t = re.sub(r"\{cause\}", str(cause_text), t)
        t = re.sub(r"\{effect\}", str(effect_text), t)
        return t

    def _attach_stats_to_prompt(self, prompt: str, cause_vars: List[str], effect_var: str,
                                variable_stats: Optional[Dict[str, Dict[str, float]]]) -> str:
        if not variable_stats:
            return prompt
        keys = [v for v in cause_vars + [effect_var] if v in variable_stats]
        if not keys:
            return prompt
        stats_pack = {k: variable_stats[k] for k in keys}
        hint = "\n\nDATA STATS (related variables):\n" + json.dumps(stats_pack, ensure_ascii=False, indent=2)
        hint += "\nReturn ONLY JSON objects, one per line."
        hint += "\nValid rule_index: [\"IF_THEN\",\"AND_THEN\",\"THRESHOLD\",\"PRIORITY\",\"MULTI_ACTION\",\"WEIGHTED_ACTION\",\"TEMPORAL_CHAIN\"]"
        return prompt + hint

    def _call_llm_lines(self, prompt: str, temperature: float) -> List[str]:
        raw = ""
        try:
            raw = call_gpt(prompt=prompt, model=self.model_name, temperature=temperature)
        except Exception as e:
            print(f"[✘] LLM generation failed: {e}")
        lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
        if not lines:
            # Minimal fallback: guarantee at least one IF_THEN text
            lines = [json.dumps({
                "rule_text": "IF {cause} THEN {effect}",
                "rule_index": "IF_THEN",
                "variables": [],
                "params": {},
                "action": "IDLE"
            }, ensure_ascii=False)]
        return lines

    # ----------------------- parsing -----------------------

    def _parse_lines_json_first(self, lines: List[str]) -> List[Dict]:
        out = []
        for ln in lines:
            obj = self._try_parse_json(ln)
            if obj is None:
                obj = self._parse_text_line(ln)  # Text fallback
            if obj:
                out.append(obj)
        return out

    def _try_parse_json(self, s: str) -> Optional[Dict]:
        # Light fix: trailing comma, single/double quotes
        t = s.strip()
        t = re.sub(r",\s*}", "}", t)
        t = re.sub(r",\s*]", "]", t)
        try:
            obj = json.loads(t)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        obj.setdefault("rule_text", "")
        obj.setdefault("rule_index", "IF_THEN")
        obj.setdefault("variables", [])
        obj.setdefault("params", {})
        obj.setdefault("action", "IDLE")
        obj.setdefault("guards", {})
        return obj

    def _parse_text_line(self, text: str) -> Optional[Dict]:
        text = (text or "").strip()
        if not text:
            return None
        parsed = self._parse_rule_text(text)  # Reuse your original text parser
        return {
            "rule_text": parsed.get("raw_text", text),
            "rule_index": parsed.get("rule_index", "IF_THEN"),
            "variables": self._extract_variables_from_conditions(parsed.get("condition", [])),
            "params": {},
            "action": parsed.get("action", "IDLE"),
        }

    def _extract_variables_from_conditions(self, cond_list) -> List[str]:
        vars_found = []
        for c in cond_list or []:
            tokens = re.split(r"[^\w]+", str(c))
            for tok in tokens:
                tok = tok.strip()
                if tok and tok.lower() not in {"if","then","and","or","within","seconds"}:
                    vars_found.append(tok)
        return _dedup_keep_order(vars_found)

    # ----------------------- text rule parser (your base) -----------------------

    def _parse_rule_text(self, text: str) -> Dict:
        text = text.strip()
        raw_text = text
        rule_index = "UNKNOWN"
        condition, action = [], ""

        if "PRIORITIZE" in text and "OVER" in text:
            rule_index = "PRIORITY"
            try:
                prefix, decision = text.split("PRIORITIZE")
                a1, a2 = decision.strip().split("OVER")
                condition = [prefix.replace("IF", "").strip()]
                action = {"priority": [a1.strip(), a2.strip()]}
            except Exception:
                pass
        elif "{" in text and "}" in text:
            rule_index = "WEIGHTED_ACTION"
            try:
                trigger, action_block = text.split("THEN")
                trigger = trigger.replace("IF", "").strip()
                condition = [c.strip() for c in trigger.split("AND")]
                action_str = action_block.strip().strip("{}")
                action = {k.strip(): float(v.strip()) for k, v in [pair.split(":") for pair in action_str.split(",")]}
            except Exception:
                pass
        elif "THEN" in text and "AND" in text and "{" not in text:
            try:
                trigger, action_part = text.split("THEN")
                trigger = trigger.replace("IF", "").strip()
                conditions = [c.strip() for c in trigger.split("AND")]
                actions = [a.strip() for a in action_part.strip().split("AND")]
                if len(actions) >= 2:
                    rule_index = "MULTI_ACTION"
                    condition = conditions
                    action = actions
                else:
                    rule_index = "AND_THEN"
                    condition = conditions
                    action = action_part.strip()
            except Exception:
                pass
        elif "WITHIN" in text and "THEN" in text:
            rule_index = "TEMPORAL_CHAIN"
            try:
                left, time_clause = text.split("WITHIN")
                trigger, response = left.split("THEN")
                condition = [trigger.replace("IF", "").strip()]
                action = {"response": response.strip(), "time_window": time_clause.strip()}
            except Exception:
                pass
        elif any(op in text for op in [">=", "<=", ">", "<"]) and "THEN" in text:
            rule_index = "THRESHOLD"
            try:
                trigger, action_part = text.split("THEN")
                trigger = trigger.replace("IF", "").strip()
                condition = [c.strip() for c in trigger.split("AND")]
                action = action_part.strip()
            except Exception:
                pass
        elif "IF" in text and "THEN" in text:
            try:
                trigger, action_part = text.split("THEN")
                if "AND" in trigger:
                    rule_index = "AND_THEN"
                    condition = [c.strip() for c in trigger.replace("IF", "").split("AND")]
                else:
                    rule_index = "IF_THEN"
                    condition = [trigger.replace("IF", "").strip()]
                action = action_part.strip()
            except Exception:
                pass
        else:
            condition = [text]
            action = "UNKNOWN"

        # Action normalization
        if isinstance(action, dict) and "priority" in action:
            # Leave for subsequent normalization
            pass
        elif isinstance(action, list):
            # MULTI_ACTION handled by subsequent normalization
            pass
        elif isinstance(action, str):
            ca = _canon_action(action)
            action = ca or "IDLE"

        return {"raw_text": raw_text, "rule_index": rule_index, "condition": condition, "action": action}

    # ----------------------- normalization & calibration -----------------------

    def _normalize_rule_dict(self, r: Dict) -> Dict:
        r = dict(r)
        r["rule_text"] = str(r.get("rule_text", "")).strip()
        r["rule_index"] = str(r.get("rule_index", "UNKNOWN")).strip().upper()
        # normalize action into canonical 5 classes when possible
        act = r.get("action", "IDLE")
        if isinstance(act, dict) and "priority" in act:
            seq = []
            for a in act["priority"]:
                a2 = a if a in CANON_ACTIONS else _canon_action(str(a))
                if a2 in CANON_ACTIONS and a2 not in seq:
                    seq.append(a2)
            r["action"] = {"priority": seq} if len(seq) >= 2 else (seq[0] if seq else "IDLE")
        elif isinstance(act, list):
            seq = []
            for a in act:
                a2 = a if a in CANON_ACTIONS else _canon_action(str(a))
                if a2 in CANON_ACTIONS and a2 not in seq:
                    seq.append(a2)
            r["action"] = seq if seq else "IDLE"
        elif isinstance(act, str):
            a2 = act if act in CANON_ACTIONS else _canon_action(act)
            r["action"] = a2 or "IDLE"
        else:
            r["action"] = "IDLE"

        # params/variables
        params = r.get("params") or {}
        r["params"] = params if isinstance(params, dict) else {}
        variables = r.get("variables") or []
        r["variables"] = variables if isinstance(variables, list) else []
        r["guards"] = r.get("guards", {}) if isinstance(r.get("guards", {}), dict) else {}  # <<< ADD

        if r["rule_index"] in {"TEMPORAL_CHAIN", "UNKNOWN"}:
            r.setdefault("meta", {})["only_prior"] = True
        return r

    def _finalize_action_and_conf(self, r: Dict, variable_stats: Optional[Dict[str, Dict[str, float]]]) -> Dict:
        """
        Normalize + infer + confidence:
        1) First normalize LLM's action using _canon_action.
        2) Force inference based on left/right lateral & longitudinal cues in rule_text / variables (override if determinable).
        3) Add slight confidence bonus based on "richness" of thresholds.
        """
        r = dict(r)

        # ---------- 0) Preprocessing ----------
        text = str(r.get("rule_text") or r.get("raw_text") or "").lower()
        var_list = [str(v).lower() for v in (r.get("variables") or [])]

        def has_kw(s: str) -> bool:
            return (s in text) or any(s in v for v in var_list)

        # ---------- 1) First normalize LLM's action ----------
        act0 = r.get("action", "IDLE")
        canon_act = "IDLE"
        if isinstance(act0, str):
            canon_act = act0 if act0 in CANON_ACTIONS else (_canon_action(act0) or "IDLE")
        elif isinstance(act0, list) and act0:
            # Multi-action: take the first recognizable one
            for x in act0:
                cand = x if x in CANON_ACTIONS else _canon_action(str(x))
                if cand in CANON_ACTIONS:
                    canon_act = cand
                    break
        elif isinstance(act0, dict) and "priority" in act0 and act0["priority"]:
            for x in act0["priority"]:
                cand = x if x in CANON_ACTIONS else _canon_action(str(x))
                if cand in CANON_ACTIONS:
                    canon_act = cand
                    break

        # ---------- 2) Forced inference based on cues ----------
        # Lateral cues
        left_signals  = any(has_kw(k) for k in LANE_LEFT_KEYS)
        right_signals = any(has_kw(k) for k in LANE_RIGHT_KEYS)

        # Longitudinal cues
        front_risk = any(has_kw(k) for k in LONGIT_FRONT_KEYS)
        rear_risk  = any(has_kw(k) for k in LONGIT_REAR_KEYS)

        # Text hint keywords
        says_slow = any(k in text for k in ["slow", "decelerate", "brake", "reduce speed", "decrease speed"])
        says_fast = any(k in text for k in ["accelerate", "speed up", "increase speed", "faster"])
        says_keep = any(k in text for k in ["keep lane", "maintain", "stay", "idle"])

        chosen, conf = None, 0.0

        # 2.1 Lateral priority (prioritize L/R when clear lateral signals present)
        if left_signals and not right_signals:
            chosen, conf = "LANE_LEFT", 0.85
        elif right_signals and not left_signals:
            chosen, conf = "LANE_RIGHT", 0.85
        elif left_signals and right_signals:
            if has_kw("available_gap_left") and not has_kw("available_gap_right"):
                chosen, conf = "LANE_LEFT", 0.80
            elif has_kw("available_gap_right") and not has_kw("available_gap_left"):
                chosen, conf = "LANE_RIGHT", 0.80
            # If both sides present without significant difference, defer to longitudinal logic

        # 2.2 Longitudinal (front risk→SLOWER; rear risk→FASTER; both→default SLOWER)
        if chosen is None:
            if front_risk and not rear_risk:
                chosen, conf = "SLOWER", 0.75
            elif rear_risk and not front_risk:
                chosen, conf = "FASTER", 0.70
            elif front_risk and rear_risk:
                chosen, conf = "SLOWER", 0.65

        # 2.3 Text hint fallback
        if chosen is None and says_slow:
            chosen, conf = "SLOWER", 0.65
        if chosen is None and says_fast:
            chosen, conf = "FASTER", 0.60
        if chosen is None and says_keep:
            chosen, conf = "IDLE", 0.55

        # 2.4 If still undetermined, and LLM already has canonical action, use it
        if chosen is None and canon_act in CANON_ACTIONS:
            chosen, conf = canon_act, max(conf, 0.55 if canon_act != "IDLE" else 0.50)

        # 2.5 Final fallback
        if chosen is None:
            chosen, conf = "IDLE", 0.50

        # Slightly reduce confidence when lateral but text emphasizes keep/slow/fast
        if chosen in {"LANE_LEFT", "LANE_RIGHT"} and (says_keep or says_slow or says_fast):
            conf = max(0.55, conf - 0.10)

        # ---------- 3) Threshold info "richness" bonus ----------
        ths = ((r.get("params") or {}).get("thresholds") or {})
        # Count involved key variables
        richness = 0
        for k in ths.keys():
            kk = str(k).lower()
            if (kk in LANE_LEFT_KEYS) or (kk in LANE_RIGHT_KEYS) or (kk in LONGIT_FRONT_KEYS) or (kk in LONGIT_REAR_KEYS):
                richness += 1
        bonus = min(0.15, 0.03 * max(0, richness - 2))  # Beyond 2 key variables, +0.03 each, max +0.15

        r["action"] = chosen
        r["confidence"] = round(min(0.95, conf + bonus), 3)

        # Clean text (optional)
        rt = (r.get("rule_text") or r.get("raw_text") or "").strip()
        r["rule_text_clean"] = re.sub(r"\s+", " ", rt)

        return r

    def _calibrate_rule(self, r: Dict, stats: Optional[Dict[str, Dict[str, float]]]) -> Dict:
        """Convert {form: abs/z/q} thresholds to value/op/unit. Skip conversion if stats is empty, but preserve structure."""
        r = dict(r)
        if not stats:
            return r
        params = r.get("params") or {}
        th = params.get("thresholds") or {}
        if not isinstance(th, dict):
            return r

        new_th = {}
        for var, cfg in th.items():
            if not isinstance(cfg, dict):
                continue
            form = str(cfg.get("form", "")).lower()
            op = cfg.get("op", None)
            unit = cfg.get("unit", "")
            value = cfg.get("value", None)

            # Allow fallback: if value is directly given, keep it
            if form == "abs" and (value is not None) and (op in (">","<",">=","<=")):
                new_th[var] = {"form":"abs","value": float(value), "op": op, "unit": unit}
                continue

            s = stats.get(var)
            if not s:
                # No statistics info, keep as-is and mark as prior only
                r.setdefault("meta", {})["only_prior"] = True
                new_th[var] = cfg
                continue

            if form == "z":
                # x > mu + k*sigma
                k = float(cfg.get("k", 1.0))
                mu = float(s.get("mean", 0.0)); sig = float(s.get("std", 1e-8))
                if op == ">" or op == ">=":
                    val = mu + k * sig
                elif op == "<" or op == "<=":
                    val = mu - k * sig
                else:
                    val = mu + k * sig
                    op = ">="
                new_th[var] = {"form":"abs","value": float(val), "op": op, "unit": unit}
            elif form == "q":
                # Quantile: q in (0,1)
                q = float(cfg.get("q", 0.8))
                # Use nearest p20/p50/p80
                if q <= 0.3:
                    val = float(s.get("p20", s.get("min", 0.0)))
                elif q >= 0.7:
                    val = float(s.get("p80", s.get("max", 0.0)))
                else:
                    val = float(s.get("p50", s.get("mean", 0.0)))
                if op not in (">","<",">=","<="):
                    op = ">=" if q >= 0.5 else "<="
                new_th[var] = {"form":"abs","value": float(val), "op": op, "unit": unit}
            else:
                # Unknown form: mark as prior, keep as-is
                r.setdefault("meta", {})["only_prior"] = True
                new_th[var] = cfg

        # Write back
        r.setdefault("params", {})["thresholds"] = new_th
        return r

    # -----------------------------------------------------
    # guards: unified check (range/comparison/boolean)
    # -----------------------------------------------------
    @staticmethod
    def check_guards(guards: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """
        Support three types:
        1) Pure boolean: {"lane_left_available": true}
        2) Comparison: {"speed_limit": {"op":"<=","value":110,"unit":"km/h"}}
        3) Compound key path: use a.b to access state["a"]["b"]
        """
        if not guards:
            return True

        def get_val_from_state(path: str):
            cur = state
            for key in path.split("."):
                if not isinstance(cur, dict) or key not in cur:
                    return None
                cur = cur[key]
            return cur

        def cmp(v, op, tgt):
            try:
                v = float(v); tgt = float(tgt)
            except Exception:
                return False
            if op == ">":  return v >  tgt
            if op == ">=": return v >= tgt
            if op == "<":  return v <  tgt
            if op == "<=": return v <= tgt
            if op == "==": return v == tgt
            if op == "!=": return v != tgt
            return False

        for k, spec in guards.items():
            # 1) bool
            if isinstance(spec, bool):
                v = get_val_from_state(k)
                if isinstance(v, bool):
                    if v is not spec:
                        return False
                else:
                    # If environment doesn't have this boolean, consider as not satisfied
                    return False
            # 2) Comparison object
            elif isinstance(spec, dict):
                op = spec.get("op"); tgt = spec.get("value")
                v = get_val_from_state(k)
                if v is None or op is None or tgt is None or not cmp(v, op, tgt):
                    return False
            # 3) Other types not supported yet, default to not pass
            else:
                return False
        return True


    # -----------------------------------------------------
    # AND merger: merge same-source rules by thresholds into AND_THEN
    # -----------------------------------------------------
    def merge_and_rules(self, rules: List[Dict]) -> List[Dict]:
        if not rules:
            return rules

        def key_of(r):
            # Merge by same cause and action (avoid merging different actions into one)
            return (json.dumps(r.get("cause"), sort_keys=True, ensure_ascii=False),
                    json.dumps(r.get("effect"), sort_keys=True, ensure_ascii=False),
                    r.get("action","IDLE"))

        from collections import defaultdict
        buckets = defaultdict(list)
        for r in rules:
            buckets[key_of(r)].append(r)

        merged = []
        for k, group in buckets.items():
            if len(group) == 1:
                merged.append(group[0]); continue

            # Collect mergeable thresholds
            base = dict(group[0])  # clone
            base_params = base.setdefault("params", {})
            base_th = base_params.setdefault("thresholds", {})
            ok_merge = True

            for r in group[1:]:
                th = ((r.get("params") or {}).get("thresholds") or {})
                # Merge by variable: if same variable conflicts, conservatively take stricter threshold (< takes min, > takes max)
                for var, cond in th.items():
                    if not isinstance(cond, dict) or "op" not in cond or "value" not in cond:
                        ok_merge = False; break
                    if var not in base_th:
                        base_th[var] = cond; continue
                    # Conflict handling
                    a = base_th[var]; b = cond
                    if a.get("op") in {">",">="} and b.get("op") in {">",">="}:
                        # Both ">=" direction, take larger value (stricter)
                        try:
                            base_th[var]["value"] = max(float(a["value"]), float(b["value"]))
                            base_th[var]["op"]    = ">="
                        except Exception:
                            ok_merge = False; break
                    elif a.get("op") in {"<","<="} and b.get("op") in {"<","<="}:
                        # Both "<=" direction, take smaller value (stricter)
                        try:
                            base_th[var]["value"] = min(float(a["value"]), float(b["value"]))
                            base_th[var]["op"]    = "<="
                        except Exception:
                            ok_merge = False; break
                    else:
                        # Opposite directions: cannot safely merge
                        ok_merge = False; break
                if not ok_merge:
                    break

            if ok_merge and len(base_th) >= 2:
                base["rule_index"] = "AND_THEN"
                base["rule_text"]  = self._synthesize_rule_text_from_thresholds(base)  # Readability
                merged.append(base)
            else:
                merged.extend(group)  # Abandon merge, keep original

        return _dedup_keep_order(merged)

    def _synthesize_rule_text_from_thresholds(self, r: Dict) -> str:
        th = ((r.get("params") or {}).get("thresholds") or {})
        conds = []
        for var, c in th.items():
            op = c.get("op","?"); val = c.get("value","?")
            unit = c.get("unit","")
            unit_str = f" {unit}" if unit else ""
            conds.append(f"{var} {op} {val}{unit_str}")
        act = r.get("action","IDLE")
        if conds:
            return f"IF " + " AND ".join(conds) + f" THEN {act}"
        return r.get("rule_text") or ""


    # -----------------------------------------------------
    # Selector: combine S_rule / S_act / confidence + guards
    # -----------------------------------------------------
    def select_best_rules(
        self,
        rules: List[Dict],
        state: Dict[str, Any],
        candidate_actions: List[str],
        s_rule_fn,      # callable(rule, state) -> float in [0,1]
        s_act_fn,       # callable(rule, state, action_idx) -> float in [0,1]
        alpha=0.6, beta=0.3, gamma=0.1,
        action_order: Optional[List[str]] = None,
        action_priority: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Return:
        {
        "per_action_best": { "LANE_LEFT": {rule...,"score":...}, ... },
        "global_best": {rule...,"score":..., "action": "..."}
        }
        """
        action_order   = action_order   or ["LANE_LEFT","LANE_RIGHT","SLOWER","FASTER","IDLE"]
        action_priority= action_priority or ["SLOWER","LANE_LEFT","LANE_RIGHT","IDLE","FASTER"]  # Safety > lane change > efficiency

        # Filter: first merge AND
        rules = self.merge_and_rules(rules)

        # Select best per action
        per_action_best = {}
        for act in action_order:
            best = None
            for r in rules:
                if r.get("action") != act:
                    continue
                # guards check
                if not self.check_guards(r.get("guards", {}), state):
                    continue
                # Score
                try:
                    # action_idx: index of action in action_order
                    aidx = action_order.index(act)
                except Exception:
                    aidx = 0
                sr = float(s_rule_fn(r, state))  # [0,1]
                sa = float(s_act_fn(r, state, aidx))  # [0,1]
                prior = float(r.get("confidence", 0.5))
                score = alpha*sr + beta*sa + gamma*prior
                item = dict(r); item["score"] = score
                if (best is None) or (score > best["score"]):
                    best = item
            if best:
                per_action_best[act] = best

        # Global best: sort by score, tie-break by priority when scores are close
        global_best = None
        if per_action_best:
            # First find max
            global_best = max(per_action_best.values(), key=lambda x: x["score"])
            # Tie-break
            # Find candidates with difference < 1e-6, pick by action_priority order
            eps = 1e-6
            ties = [v for v in per_action_best.values() if abs(v["score"] - global_best["score"]) < eps]
            if len(ties) > 1:
                def pri(a):
                    try: return action_priority.index(a.get("action","IDLE"))
                    except: return len(action_priority)
                ties.sort(key=pri)
                global_best = ties[0]

        return {
            "per_action_best": per_action_best,
            "global_best": global_best
        }


