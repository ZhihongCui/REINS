import json
import time
from typing import List, Tuple, Dict
from pathlib import Path

from src.llm.caller import call_gpt

PROMPT_TEMPLATE_PATH = Path("src/llm/prompts/variable_to_causal.txt")


class CausalGraphBuilder:
    def __init__(self, sample_path: str, llm_engine="gpt-4.1-nano", max_pairs=100):
        """
        Module for building causal graphs.
        :param sample_path: Path containing variable snapshots saved by the sampler
        :param llm_engine: Name of the LLM to call
        :param max_pairs: Limit on the number of variable pair combinations to avoid exponential growth
        """
        self.sample_path = sample_path
        self.llm_engine = llm_engine
        self.max_pairs = max_pairs
        self.nodes = []
        self.edges = []

        with open(PROMPT_TEMPLATE_PATH, "r") as f:
            self.prompt_template = f.read()

    def load_samples(self) -> List[Dict[str, Dict[str, float]]]:
        with open(self.sample_path, "r") as f:
            return json.load(f)

    def extract_variables(self, samples: List[Dict]) -> List[str]:
        """
        Extract all variable names from samples (flattened)
        """
        variables = set()
        for sample in samples:
            for macro, micro_dict in sample.items():
                if macro in ["episode_id"]:
                    continue
                for micro in micro_dict:
                    variables.add(micro)
        return sorted(list(variables))

    def generate_variable_pairs(self, variables: List[str]) -> List[Tuple[str, str]]:
        """
        Generate all variable pairs (directed)
        """
        pairs = []
        for var_a in variables:
            for var_b in variables:
                if var_a != var_b:
                    pairs.append((var_a, var_b))
        return pairs[:self.max_pairs]

    def ask_llm_causality(self, var_a: str, var_b: str) -> bool:
        """
        Use GPT to determine whether a causal relationship exists from var_a to var_b
        """
        prompt = self.prompt_template.replace("{A}", var_a).replace("{B}", var_b)
        try:
            answer = call_gpt(prompt, engine=self.llm_engine)
            return answer.strip().lower().startswith("yes")
        except Exception as e:
            print(f"[LLM] Error for pair ({var_a}, {var_b}): {e}")
            return False

    def build_graph(self):
        samples = self.load_samples()
        self.nodes = self.extract_variables(samples)
        var_pairs = self.generate_variable_pairs(self.nodes)

        print(f"[GraphBuilder] Total variables: {len(self.nodes)}")
        print(f"[GraphBuilder] Total variable pairs to check: {len(var_pairs)}")

        self.edges = []
        for a, b in var_pairs:
            time.sleep(1)  # optional: avoid rate limiting
            if self.ask_llm_causality(a, b):
                self.edges.append((a, b, "CAUSES"))
                print(f"[OK] {a} -> {b}")
            else:
                print(f"[ ] {a} !-> {b}")

    def get_graph(self) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        return self.nodes, self.edges

    def export_graph_json(self, path: str):
        """
        Optional: Save the graph in JSON format
        """
        graph_data = {
            "nodes": self.nodes,
            "edges": [{"from": src, "to": dst, "type": rel} for (src, dst, rel) in self.edges]
        }
        with open(path, "w") as f:
            json.dump(graph_data, f, indent=2)
        print(f"[GraphBuilder] Graph saved to {path}")
