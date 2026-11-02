# evaluate.py

import numpy as np
import json
from typing import List, Dict


class Evaluator:
    def __init__(self):
        self.metrics = {
            "ADE": [],
            "FDE": [],
            "success_rate": [],
            "violation_rate": [],
            "energy_consumption": [],
            "comfort_score": [],
        }

    def _compute_ade_fde(self, trajectory: List[Dict]) -> (float, float):
        positions = [step["sample"]["Ego"]["lateral_offset"] for step in trajectory]
        ground_truth = [0.0] * len(positions)  # Assume target lane center is 0.0

        errors = np.abs(np.array(positions) - np.array(ground_truth))
        ade = np.mean(errors)
        fde = errors[-1]
        return ade, fde

    def _compute_success(self, trajectory: List[Dict]) -> bool:
        return all(step["sample"]["Outcome"]["success_flag"] for step in trajectory)

    def _compute_violation(self, trajectory: List[Dict]) -> bool:
        return any(step["sample"]["Outcome"]["violation_flag"] for step in trajectory)

    def _aggregate_outcome(self, trajectory: List[Dict], key: str) -> float:
        return np.mean([step["sample"]["Outcome"][key] for step in trajectory])

    def evaluate_episode(self, result: Dict):
        traj = result["trajectory"]

        ade, fde = self._compute_ade_fde(traj)
        success = self._compute_success(traj)
        violation = self._compute_violation(traj)
        energy = self._aggregate_outcome(traj, "energy_consumption")
        comfort = self._aggregate_outcome(traj, "comfort_score")

        self.metrics["ADE"].append(ade)
        self.metrics["FDE"].append(fde)
        self.metrics["success_rate"].append(1.0 if success else 0.0)
        self.metrics["violation_rate"].append(1.0 if violation else 0.0)
        self.metrics["energy_consumption"].append(energy)
        self.metrics["comfort_score"].append(comfort)

        print(f"[Episode {result['episode_id']}] ADE={ade:.3f}, FDE={fde:.3f}, "
              f"Success={success}, Violation={violation}, "
              f"Energy={energy:.3f}, Comfort={comfort:.3f}")

    def summarize(self):
        print("\n========== Evaluation Summary ==========")
        for k, v in self.metrics.items():
            avg = np.mean(v)
            std = np.std(v)
            print(f"{k}: {avg:.4f} ± {std:.4f}")
        print("========================================")

    def export(self, save_path: str):
        summary = {k: float(np.mean(v)) for k, v in self.metrics.items()}
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Evaluator] Saved summary to {save_path}")
