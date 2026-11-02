# config.py
import yaml

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = {
    "env_name": "roundabout-v0", # "merge-v0"
    "num_train_episodes": 100,
    "structure_model": "transformer",  # or "gnn" mlp
    "rule_scorer": "transformer",
    "output_dir": "result/",
    "seed": 42,
    "openai_engine": "gpt-4.1-nano",
    "max_pairs": 1000,
    "use_cache": True
}

