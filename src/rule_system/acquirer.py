import hashlib
import json
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.llm.caller import call_gpt
from src.rule_system.storage import Neo4jWriter, Rule

_SYSTEM_PROMPT = (
    "You are an expert in autonomous driving safety rules. "
    "Given a novel scene state, propose ONE concise and safe driving rule."
)

_PROPOSAL_TEMPLATE = (
    "The autonomous vehicle encountered a novel scene with no matching rule in memory.\n\n"
    "Current scene state:\n{state_json}\n\n"
    "Propose one driving rule as a JSON object:\n"
    '{{\n'
    '  "trigger": "<variable> <op> <threshold>",\n'
    '  "effect": "<outcome variable>",\n'
    '  "action": "<one of: SLOWER FASTER IDLE LANE_LEFT LANE_RIGHT>",\n'
    '  "confidence": <float 0.0-1.0>,\n'
    '  "description": "<one-line description>"\n'
    '}}\n\n'
    "Rules:\n"
    "- trigger: single condition, e.g. \"headway_distance < 15.0\" or \"lead_ttc < 3.0\"\n"
    "- effect: outcome variable most influenced, e.g. \"risk_score\"\n"
    "- action: the recommended discrete action\n"
    "- confidence: your confidence [0, 1]\n\n"
    "Reply with ONLY the JSON object, no extra text.\n"
)

CANON_ACTIONS = {"SLOWER", "FASTER", "IDLE", "LANE_LEFT", "LANE_RIGHT"}


class OnlineRuleAcquirer:
    """
    Online Rule Acquisition — Section 3.2.2 of the REINS paper.

    Implements the self-evolution loop:
      1. Miss Detection      (Eq. 3.3): is_miss()
      2. LLM Rule Proposal   (Eq. 3.5): _propose_rule()
      3. Online Verification (Eq. 3.6): _verify_rule()
      4. Memory Write-Back   (Eq. 3.7): _write_or_merge()
    """

    def __init__(
        self,
        neo4j_writer: Neo4jWriter,
        llm_model: str = "gpt-4.1-nano",
        miss_eta: float = 0.15,
        verify_horizon: int = 5,
        window_size: int = 50,
        risk_th: float = 0.80,
        ttc_th: float = 2.0,
        novelty_threshold: float = 0.85,
    ):
        self.writer = neo4j_writer
        self.llm_model = llm_model
        self.miss_eta = miss_eta
        self.verify_horizon = verify_horizon
        self.window_size = window_size
        self.risk_th = risk_th
        self.ttc_th = ttc_th
        self.novelty_threshold = novelty_threshold

        self._window: List[Dict[str, Any]] = []
        self._prior_window: List[np.ndarray] = []     # state vectors from PRIOR episodes only
        self._cur_ep_buffer: List[np.ndarray] = []    # state vectors from current episode (transient)
        self._curr_vec: Optional[np.ndarray] = None
        self.llm_invoked_count: int = 0
        self.accepted_count: int = 0
        self.total_steps: int = 0

    # ------------------------------------------------------------------
    # 1. Miss Detection  Eq. 3.3
    # ------------------------------------------------------------------
    @staticmethod
    def _state_vector(sample: Dict[str, Any]) -> np.ndarray:
        """Compact 7-dim feature vector capturing scene dynamics."""
        E = sample.get("Ego", {}) or {}
        T = sample.get("Traffic", {}) or {}
        return np.array([
            float(E.get("ego_vx", 0.0)),
            float(E.get("ego_vy", 0.0)),
            float(E.get("ego_y", 0.0)),
            min(1e3, float(T.get("headway_distance", 1e3))),
            min(1e3, float(T.get("lead_ttc", 1e3))),
            float(T.get("lead_rel_speed", 0.0)),
            min(1e3, float(T.get("rear_distance", 1e3))),
        ], dtype=float)

    def _novelty_score(self) -> float:
        """Maximum cosine similarity between the current scene and the
        prior-episode states accumulated so far (1.0 = closely matches a
        previously seen scene, 0.0 = unlike any prior scene). With no prior
        experience accumulated yet, a scene has no match and is unseen.
        """
        if self._curr_vec is None or len(self._prior_window) < 2:
            return 0.0  # no prior experience -> scene is unseen
        W = np.asarray(self._prior_window)
        mu = W.mean(axis=0)
        sigma = W.std(axis=0) + 1e-6
        Wn = (W - mu) / sigma
        cn = (self._curr_vec - mu) / sigma
        dots = Wn @ cn
        norms = (np.linalg.norm(Wn, axis=1) * np.linalg.norm(cn)) + 1e-9
        return float((dots / norms).max())

    def is_miss(self, top_rules: List[Dict[str, Any]]) -> bool:
        """Miss = empty retrieval OR novel scene (max cos-sim < threshold)."""
        if not top_rules:
            return True
        # Optional score-based gate (Eq. 3.3 in the paper); active when miss_eta > 0
        if top_rules and self.miss_eta > 0.0:
            max_score = max(float(r.get("score", 0.0)) for r in top_rules)
            if max_score < self.miss_eta:
                return True
        # Novelty-based: current scene unlike anything in recent window
        return self._novelty_score() < self.novelty_threshold

    # ------------------------------------------------------------------
    # Sliding window  D_t^W
    # ------------------------------------------------------------------
    def update_window(self, sample: Dict[str, Any]) -> None:
        self._window.append(sample)
        if len(self._window) > self.window_size:
            self._window.pop(0)
        vec = self._state_vector(sample)
        self._cur_ep_buffer.append(vec)
        self._curr_vec = vec

    def reset_episode(self) -> None:
        """Called at the start of each new episode.
        Migrates the just-finished episode's state vectors into the prior-episode
        window (so future novelty checks only compare against PAST experience,
        not the temporally-correlated frames of the current trajectory).
        """
        if self._cur_ep_buffer:
            self._prior_window.extend(self._cur_ep_buffer)
            # Cap memory at 20 x window_size to keep things tractable
            cap = max(self.window_size * 20, 1000)
            if len(self._prior_window) > cap:
                self._prior_window = self._prior_window[-cap:]
            self._cur_ep_buffer = []

    # ------------------------------------------------------------------
    # 2. LLM Rule Proposal  Eq. 3.5
    # ------------------------------------------------------------------
    def _summarize_state(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for group in ["Environment", "Traffic", "Ego", "Outcome"]:
            g = sample.get(group, {}) or {}
            out[group] = {
                k: round(float(v), 3) if isinstance(v, (int, float)) else v
                for k, v in list(g.items())[:6]
            }
        return out

    def _propose_rule(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        state_json = json.dumps(self._summarize_state(sample), indent=2)
        prompt = _PROPOSAL_TEMPLATE.format(state_json=state_json)
        response = call_gpt(
            prompt,
            system_prompt=_SYSTEM_PROMPT,
            model=self.llm_model,
            temperature=0.2,
            max_tokens=256,
        )
        if not response:
            return None
        try:
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if not m:
                return None
            rule = json.loads(m.group())
            action = str(rule.get("action", "")).strip().upper()
            if action not in CANON_ACTIONS:
                return None
            rule["action"] = action
            rule["confidence"] = max(0.0, min(1.0, float(rule.get("confidence", 0.5))))
            return rule
        except Exception as e:
            print(f"[Acquirer] Parse error: {e} | response: {response[:120]}")
            return None

    # ------------------------------------------------------------------
    # 3. Online Verification  Eq. 3.6
    # ------------------------------------------------------------------
    def _window_thresholds(self) -> Dict[str, float]:
        """Calibrate predicate thresholds on D_t^W (Eq. 3.6).
        Risk upper bound = 90th-percentile of risk_score in the window;
        TTC lower bound = 10th-percentile of lead_ttc in the window.
        Falls back to the configured defaults if the window is too small.
        """
        if len(self._window) < 10:
            return {"risk_th": self.risk_th, "ttc_th": self.ttc_th}
        risks = [float(s.get("Outcome", {}).get("risk_score", 0.0))
                 for s in self._window if isinstance(s.get("Outcome"), dict)]
        ttcs = [float(s.get("Traffic", {}).get("lead_ttc", 1e3))
                for s in self._window if isinstance(s.get("Traffic"), dict)]
        risk_th = float(np.quantile(risks, 0.90)) if risks else self.risk_th
        ttc_th = float(np.quantile(ttcs, 0.10)) if ttcs else self.ttc_th
        return {"risk_th": risk_th, "ttc_th": ttc_th}

    def _check_predicates(self, s: Dict[str, float]) -> bool:
        """Safety predicates P_tilde with thresholds calibrated on D_t^W (Eq. 3.6)."""
        th = self._window_thresholds()
        risk = s.get("risk_score")
        ttc = s.get("lead_ttc")
        if risk is not None and float(risk) > th["risk_th"]:
            return False
        if ttc is not None and float(ttc) < th["ttc_th"]:
            return False
        return True

    def _verify_rule(
        self,
        proposed: Dict[str, Any],
        sample: Dict[str, Any],
        predict_fn: Callable[[Dict, str], Dict[str, float]],
    ) -> bool:
        """Accept(r_tilde, s_t) = AND_{p in P_tilde} AND_{h=1}^H p(s_hat_{t+h})"""
        action = proposed["action"]
        s = sample
        for _ in range(self.verify_horizon):
            s_next = predict_fn(s, action)
            if not self._check_predicates(s_next):
                return False
            s = {
                "Ego": {
                    "ego_x": s_next.get("ego_x", 0.0),
                    "ego_y": s_next.get("ego_y", 0.0),
                    "ego_vx": s_next.get("ego_vx", 0.0),
                    "ego_vy": s_next.get("ego_vy", 0.0),
                    "ego_speed": s_next.get("ego_vx", 0.0),
                },
                "Traffic": {
                    "headway_distance": s_next.get("headway_distance", 1e6),
                    "lead_ttc": s_next.get("lead_ttc", 1e9),
                    "lead_rel_speed": s_next.get("lead_rel_speed", 0.0),
                    "rear_distance": s_next.get("rear_distance", 1e6),
                },
                "Outcome": {
                    "risk_score": s_next.get("risk_score", 0.5),
                    "energy_consumption": s_next.get("energy_consumption", 0.1),
                    "comfort_score": s_next.get("comfort_score", 0.8),
                },
            }
        return True

    # ------------------------------------------------------------------
    # 4. Memory Write-Back  Eq. 3.7
    # ------------------------------------------------------------------
    @staticmethod
    def _content_hash(proposed: Dict[str, Any]) -> str:
        """kappa(r_tilde) := kappa(trigger, action, effect) — excludes confidence."""
        key = json.dumps(
            {
                "trigger": proposed.get("trigger", ""),
                "action": proposed.get("action", ""),
                "effect": proposed.get("effect", ""),
            },
            sort_keys=True,
        )
        return "online_" + hashlib.sha256(key.encode()).hexdigest()[:12]

    def _write_or_merge(self, proposed: Dict[str, Any], rule_id: str) -> None:
        """Insert new rule or average confidence on collision (Eq. 3.7)."""
        conf = proposed.get("confidence", 0.5)
        raw_text = proposed.get("description", proposed.get("trigger", ""))
        action = proposed.get("action", "IDLE")
        trigger = proposed.get("trigger", "")
        cause = trigger.split()[0] if trigger else "unknown"
        effect = proposed.get("effect", "risk_score")

        with self.writer.driver.session() as session:
            row = session.run(
                "MATCH (r:Rule {rule_id: $rid}) RETURN r.confidence AS c",
                rid=rule_id,
            ).single()

            if row:
                # Confidence averaging: c' <- (c' + c_tilde) / 2
                old_conf = float(row["c"]) if row["c"] is not None else conf
                new_conf = (old_conf + conf) / 2.0
                session.run(
                    "MATCH (r:Rule {rule_id: $rid}) SET r.confidence = $c",
                    rid=rule_id,
                    c=new_conf,
                )
                print(f"[Acquirer] Merged {rule_id}: conf {old_conf:.3f} -> {new_conf:.3f}")
            else:
                rule = Rule(
                    rule_id=rule_id,
                    cause=cause,
                    effect=effect,
                    rule_type="Online",
                    confidence=conf,
                    raw_text=raw_text,
                    cause_group="Traffic",
                    effect_group="Outcome",
                    action=action,
                    description=proposed.get("description", ""),
                )
                self.writer.write_rule(rule)
                print(f"[Acquirer] Inserted {rule_id}: {raw_text[:80]}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def try_acquire(
        self,
        sample: Dict[str, Any],
        predict_fn: Callable[[Dict, str], Dict[str, float]],
    ) -> Optional[Dict[str, Any]]:
        """
        Full pipeline: propose -> verify -> write-back.
        Returns accepted rule dict (ready to prepend to top_rules) or None.
        """
        self.llm_invoked_count += 1

        proposed = self._propose_rule(sample)
        if not proposed:
            return None

        if not self._verify_rule(proposed, sample, predict_fn):
            print(f"[Acquirer] Rejected: {proposed.get('description', '')}")
            return None

        self.accepted_count += 1
        rule_id = self._content_hash(proposed)
        self._write_or_merge(proposed, rule_id)

        return {
            "rule_id": rule_id,
            "rule_text": proposed.get("description", proposed.get("trigger", "")),
            "raw_text": proposed.get("trigger", ""),
            "action": proposed["action"],
            "score": proposed["confidence"],
            "rule_type": "Online",
            "online_acquired": True,
        }

    @property
    def invocation_rate(self) -> float:
        """Fraction of steps where LLM was invoked (should monotonically decrease)."""
        return self.llm_invoked_count / max(1, self.total_steps)
