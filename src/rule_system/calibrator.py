# modules/threshold_calibrator.py
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class ThresholdCalibrator:
    """
    Converts LLM-generated rules with placeholder thresholds into concrete numerical values.
    Supports three forms: abs (absolute value), z (mu + k*sigma), q (quantile).
    Uses one-dimensional scanning over quantile sets, selecting thresholds that
    maximize the safety-outcome separability (AUC-like surrogate, Eq. 3.4).
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        # Flatten time series to frame-level samples
        self.frames = []
        for ep in samples:
            for t in ep.get("trajectory", []):
                self.frames.append(t)

    def _get(self, frame: Dict[str, Any], var: str) -> Optional[float]:
        for g in ["Environment", "Traffic", "Ego", "Action", "Outcome"]:
            d = frame.get(g, {})
            if isinstance(d, dict) and var in d and isinstance(d[var], (int, float)):
                return float(d[var])
        return None

    def _effect_signal(self, frame: Dict[str, Any], effect: str) -> float:
        # Target signal: numerical value of effect variable or non-zero boolean
        val = self._get(frame, effect)
        if val is None:
            return 0.0
        return float(val)

    def _scan_candidates(self, xs: np.ndarray, form: str) -> List[float]:
        xs = xs[np.isfinite(xs)]
        if len(xs) == 0:
            return []
        if form == "abs":
            # Select several quantile points as candidates
            qs = np.quantile(xs, [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
            return list(np.unique(qs.tolist()))
        elif form == "z":
            mu, sigma = xs.mean(), xs.std() + 1e-8
            ks = [-1.5, -1.0, -0.5, 0.0, +0.5, +1.0, +1.5, +2.0]
            return [mu + k * sigma for k in ks]
        elif form == "q":
            qs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
            return list(np.quantile(xs, qs))
        else:
            return []

    def calibrate_rule(self, rule: Dict[str, Any], effect_name: str) -> Dict[str, Any]:
        """
        Returns the updated rule (with concrete values written back to params.thresholds).
        Scoring function: AUC-like surrogate that splits samples on each threshold candidate
        and selects the one maximizing the safety-outcome mean difference.
        """
        params = rule.get("params", {})
        th = params.get("thresholds", {})
        if not th:
            return rule  # No thresholds to calibrate

        # Collect sample values and effect for each variable
        cache = {}
        for var, cfg in th.items():
            xs, ys = [], []
            for f in self.frames:
                x = self._get(f, var)
                y = self._effect_signal(f, effect_name)
                if x is not None:
                    xs.append(x); ys.append(y)
            cache[var] = (np.asarray(xs, float), np.asarray(ys, float))

        # Scan candidate thresholds independently for each variable, select the one with maximum benefit
        for var, cfg in th.items():
            xs, ys = cache[var]
            if xs.size == 0:
                continue
            form = str(cfg.get("form", "abs"))
            cands = self._scan_candidates(xs, form)
            if not cands:
                continue

            best_v, best_score, best_dir = None, -1e9, None
            for v in cands:
                # Try both directions: x > v or x < v
                mask_gt = xs > v
                mask_lt = xs < v
                # Surrogate score: mean-effect-difference between the two split halves
                s_gt = (ys[mask_gt].mean() - ys[~mask_gt].mean()) if mask_gt.any() and (~mask_gt).any() else -1e9
                s_lt = (ys[mask_lt].mean() - ys[~mask_lt].mean()) if mask_lt.any() and (~mask_lt).any() else -1e9
                if s_gt > best_score:
                    best_score, best_v, best_dir = s_gt, float(v), ">"
                if s_lt > best_score:
                    best_score, best_v, best_dir = s_lt, float(v), "<"

            if best_v is not None:
                cfg["value"] = best_v
                cfg["op"] = best_dir  # Record direction
                # Keep unit as is
                th[var] = cfg

        rule["params"] = params
        return rule
