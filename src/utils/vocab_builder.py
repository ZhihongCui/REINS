import yaml
import json
import re
from typing import Dict, List
from collections import Counter

# --------------------------
# Clean rule text
# --------------------------
def clean_rule_text(rule_text: str) -> str:
    rule_text = rule_text.lower()
    rule_text = re.sub(r"^[0-9]+\.\s*", "", rule_text)  # Remove leading numbering
    rule_text = rule_text.split("#")[0]  # Remove comments
    rule_text = rule_text.strip()

    # Standardize logical keywords
    rule_text = re.sub(r"\bif\b", "[if]", rule_text)
    rule_text = re.sub(r"\bthen\b", "[then]", rule_text)
    rule_text = rule_text.replace(" and ", " AND ")
    rule_text = rule_text.replace(" or ", " OR ")

    # Standardize symbols: up/down arrows => > <
    rule_text = rule_text.replace("↑", ">")
    rule_text = rule_text.replace("↓", "<")

    # Remove duplicate spaces
    rule_text = re.sub(r"\s+", " ", rule_text)
    return rule_text.strip()


# --------------------------
# Token extraction utility
# --------------------------

def extract_tokens_from_expr(expr: str) -> list:
    expr = expr.lower()
    expr = expr.replace("↑", ">").replace("↓", "<")
    expr = expr.replace("and", "AND").replace("or", "OR")

    # Support up/down arrows, underscore, AND, OR and other logical symbols, preserve comparison operators
    tokens = re.findall(r"[a-z0-9_↑↓<>!=]+", expr)

    cleaned_tokens = []
    for t in tokens:
        t = t.strip(".,;:%()[]/")  # Remove extra symbols
        if not t:
            continue
        # Skip pure numbers or decimals
        if t.replace('.', '', 1).isdigit():
            continue
        cleaned_tokens.append(t)

    # If empty, return a fallback token to prevent max() error later
    return cleaned_tokens if cleaned_tokens else ["[UNK]"]








# --------------------------
# 1. Build vocabulary from YAML rules
# --------------------------
def build_vocab_from_yaml_rules(var_path: str, rule_path: str) -> Dict[str, int]:
    with open(var_path, 'r') as f:
        var_data = yaml.safe_load(f)

    with open(rule_path, 'r') as f:
        rule_data = yaml.safe_load(f)

    vocab = {}
    idx = 1

    for tok in ["[if]", "[then]"]:
        vocab[tok] = idx
        idx += 1

    for op in [">", "<", ">=", "<=", "==", "!=", "AND"]:
        vocab[op] = idx
        idx += 1

    for group in ["Environment", "Traffic", "Ego", "Action", "Outcome"]:
        for var in var_data.get(group, []):
            if var not in vocab:
                vocab[var] = idx
                idx += 1

    for group in ["Physical", "Behavioral", "Scene"]:
        for rule in rule_data.get(group, []):
            triggers = rule.get("trigger", [])
            if isinstance(triggers, str):
                triggers = [triggers]
            for expr in triggers:
                for tok in extract_tokens_from_expr(expr):
                    if tok not in vocab:
                        vocab[tok] = idx
                        idx += 1

            effect = rule.get("effect", [])
            if isinstance(effect, str):
                effect = [effect]
            elif isinstance(effect, dict):
                effect = list(effect.keys())
            for e in effect:
                for tok in extract_tokens_from_expr(e):
                    if tok not in vocab:
                        vocab[tok] = idx
                        idx += 1

    return vocab

# --------------------------
# 2. Build vocabulary from LLM JSON rules
# --------------------------
def build_vocab_from_raw_text(rule_json_path: str, min_freq: int = 1) -> Dict[str, int]:
    with open(rule_json_path, "r") as f:
        rules = json.load(f)

    counter = Counter()
    for rule in rules:
        # Prefer using cleaned rule field, otherwise use raw_text
        text = rule.get("rule", rule.get("raw_text", ""))
        text = clean_rule_text(text)  # Auto-clean here
        tokens = extract_tokens_from_expr(text)
        counter.update(tokens)

    vocab = {"[PAD]": 0, "[UNK]": 1}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)

    print(f"[INFO] Build vocab from {rule_json_path} with {len(vocab)} tokens")
    return vocab

# --------------------------
# Save vocabulary
# --------------------------
def save_vocab(vocab: Dict[str, int], output_path: str = "config/vocab.json"):
    with open(output_path, "w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
