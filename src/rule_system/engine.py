# modules/rule_engine.py

import operator
from typing import List, Dict, Any
import yaml
import torch
import json
import re
from src.utils.vocab_builder import clean_rule_text, extract_tokens_from_expr


class RuleEngine:
    def __init__(self, rule_config_path: str):
        with open(rule_config_path, 'r') as f:
            self.rules = yaml.safe_load(f)

    def match_rules(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Match current observation to rules and return triggered rules.
        """
        matched_rules = []

        for rule_category, rule_list in self.rules.items():
            for rule in rule_list:
                try:
                    triggered = self._is_triggered(rule["trigger"], observation)
                    if triggered:
                        matched_rules.append({
                            "category": rule_category,
                            "index": rule["index"],
                            "trigger": rule["trigger"],
                            "effect": rule["effect"],
                            "description": rule.get("description", "")
                        })
                except Exception as e:
                    print(f"[RuleEngine] Error matching rule: {rule.get('description', '')} → {e}")
        return matched_rules

    def _is_triggered(self, trigger_expr, obs: Dict[str, Any]) -> bool:
        """
        Determine whether the trigger condition is satisfied.
        Support both single expression and list of expressions (AND logic).
        """
        if isinstance(trigger_expr, str):
            return self._eval_expr(trigger_expr, obs)
        elif isinstance(trigger_expr, list):
            return all(self._eval_expr(expr, obs) for expr in trigger_expr)
        else:
            raise ValueError(f"Unsupported trigger type: {type(trigger_expr)}")

    def _eval_expr(self, expr: str, obs: Dict[str, Any]) -> bool:
        """
        Evaluate a single trigger expression, like 'lead_speed > 120'
        """
        try:
            # Support basic syntax like: var > value
            tokens = expr.replace("↑", ">").replace("↓", "<").split()

            if len(tokens) != 3:
                raise ValueError(f"Invalid expression: {expr}")

            var_name, op_str, value = tokens
            value = float(value)

            val = obs.get(var_name, None)
            if val is None:
                return False

            ops = {
                ">": operator.gt,
                "<": operator.lt,
                ">=": operator.ge,
                "<=": operator.le,
                "==": operator.eq,
                "!=": operator.ne
            }

            op_func = ops.get(op_str)
            if not op_func:
                raise ValueError(f"Unsupported operator: {op_str}")

            return op_func(float(val), value)

        except Exception as e:
            # print(f"[RuleEngine] Failed to evaluate expression '{expr}': {e}")
            return False


# Load vocabulary (load only once)
# modules/rule_engine.py

with open("config/vocab.json", "r") as f:
    VOCAB: Dict[str, int] = json.load(f)

def parse_rule_to_tokens(rule_text: str, vocab: Dict[str, int], max_len: int = 32) -> List[int]:
    """
    Clean and tokenize rule string, return list of token IDs.
    Never returns an empty list (if parsing fails, returns [UNK]).
    """
    rule_text = rule_text.lower().strip()
    rule_text = rule_text.replace("↑", ">").replace("↓", "<")
    rule_text = rule_text.replace(" and ", " AND ").replace(" or ", " OR ")

    # Use a relaxed regex to extract tokens
    tokens = re.findall(r"[a-z0-9_><=!]+", rule_text)

    # Filter empty tokens
    tokens = [t for t in tokens if t]

    # Fallback to [UNK] if empty
    if not tokens:
        tokens = ["[UNK]"]

    # Convert to vocab indices
    token_ids = [vocab.get(t, vocab.get("[UNK]", 1)) for t in tokens[:max_len]]

    # Additional fallback
    if len(token_ids) == 0:
        token_ids = [vocab.get("[UNK]", 1)]

    # (Optional) Debug output
    # print(f"[DEBUG] Parsed rule: '{rule_text}' → {tokens} → {token_ids}")

    return token_ids




