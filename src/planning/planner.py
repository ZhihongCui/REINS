# modules/causal_planner.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import copy
import json
import math
import os
import random

import numpy as np
import torch

from src.world_model.structure_learner import HybridStructureFunctionLearner
from src.rule_system.scorer import RuleScorer


# =========================
# Internal Tool: Tree Node
# =========================
@dataclass
class _Node:
    state: Dict[str, Any]                               # Current structured state (format from Sampler.sample_episode)
    history: List[Dict[str, Any]]                       # Recent max_lag frames (including current)
    priors: Dict[str, float]                            # Action prior P(a|s)
    candidate_actions: List[str]                        # Candidate actions
    parent: Optional["_Node"] = None
    incoming_action: Optional[str] = None               # Action from parent node to this node
    depth: int = 0

    # Statistics
    N: int = 0                                          # Visit count
    N_a: Dict[str, int] = field(default_factory=dict)   # Visit count for each action
    W_a: Dict[str, float] = field(default_factory=dict) # Accumulated value
    Q_a: Dict[str, float] = field(default_factory=dict) # Average value

    # Child nodes
    children: Dict[str, "_Node"] = field(default_factory=dict)

    def ensure_action_stats(self):
        for a in self.candidate_actions:
            self.N_a.setdefault(a, 0)
            self.W_a.setdefault(a, 0.0)
            self.Q_a.setdefault(a, 0.0)


# =========================
# Planner: Rule Prior + f_i Leaf Value + Shallow PUCT
# =========================
class CausalPlanner:
    """
    Planner matching the new version of runner.py:
      - plan(sample, candidate_actions, top_rules, top_k, history_frames)
      - Only considers candidate_actions during node expansion (rule-constrained action space)
      - Selection uses PUCT, prior comes from RuleScorer (aggregating rule scores to actions)
      - Leaf value uses f_i (risk/comfort/energy) combination, avoiding long rollouts
      - Lightweight forward simulation: use f_i to predict several key variables one step ahead, forming "simulated state"
    """

    CANONICAL_ACTIONS = ["LANE_LEFT", "LANE_RIGHT", "IDLE", "FASTER", "SLOWER"]

    def __init__(
        self,
        structure_function: Optional[HybridStructureFunctionLearner],
        rule_engine: Optional[Any],
        rule_scorer: RuleScorer,
        action_space: List[str],
        device: str = "cpu",
        # --- MCTS & Heuristic Parameters (kept as original) ---
        n_simulations: int = 256,
        max_depth: int = 4,
        c_puct: float = 1.8,
        lambda_blend: float = 0.7,
        risk_hard_thresh: float = 0.90,
        # --- f_i Model Directory (kept as original) ---
        scm_dir: str = "result/scm/Roundabout",
        # === NEW: MPC/Open-loop Scoring Related (aligned with runner, all with default values, backward compatible) ===
        horizon: int = 16,
        gamma: float = 0.98,
        alpha_prior: float = 4.0,       # score = rollout + alpha * prior[a]
        speed_target: float = 31.0,
        w_target: float = 0.10,
        w_risk: float = 1.00,
        w_energy: float = 0.20,
        w_comfort: float = 0.10,
        dt: float = 1.0 / 15.0,
        use_mpc_first: bool = True,     # First use "open-loop H steps + prior" for direct decision; otherwise use original PUCT
        hard_risk_th: float = 0.95,
        hard_penalty: float = 5.0,
        **kwargs,
    ):
        self.structure_function = structure_function or HybridStructureFunctionLearner()
        self.rule_engine = rule_engine
        self.rule_scorer = rule_scorer
        self.action_space = [self._to_canonical(a) for a in action_space]
        self.device = device

        # --- Original MCTS Parameters ---
        self.n_sim = int(n_simulations)
        self.max_depth = int(max_depth)
        self.c_puct = float(c_puct)
        self.lambda_blend = float(lambda_blend)
        self.risk_hard_thresh = float(risk_hard_thresh)

        # f_i model directory
        self.scm_dir = scm_dir
        self._lazy_load_fi_models()

        # === NEW: MPC/Open-loop Scoring Parameters (aligned with runner) ===
        self.horizon = int(horizon)
        self.gamma = float(gamma)
        self.alpha_prior = float(alpha_prior)
        self.speed_target = float(speed_target)
        self.w_target = float(w_target)
        self.w_risk = float(w_risk)
        self.w_energy = float(w_energy)
        self.w_comfort = float(w_comfort)
        self.dt = float(dt)
        self.use_mpc_first = bool(use_mpc_first)
        self.hard_risk_th = float(hard_risk_th)
        self.hard_penalty = float(hard_penalty)

        self.action_bias = kwargs.get("action_bias", {
            "IDLE": -3.0,
            "FASTER": +1.3,
            "SLOWER": -1.1,
            "LANE_LEFT": +3.6,
            "LANE_RIGHT": +3.6,
        })

    # === New in class: Custom Reward (aligned with runner's _step_reward) ===
    def _step_reward_custom(self, s: Dict[str, float]) -> float:
        # -- Extract key quantities from compact state (with default values + safe float conversion) --
        speed   = float(s.get("ego_vx", s.get("ego_speed", 0.0)) or 0.0)
        risk    = float(s.get("risk_score", 0.5) or 0.5)                 # 0~1, higher is more dangerous
        energy  = float(s.get("energy_consumption", 0.1) or 0.1)         # ~[0,2] approximately
        comfort = float(s.get("comfort_score", 0.8) or 0.8)              # 0~1, higher is more comfortable

        # -- Fallback read weights/targets/hard penalty thresholds (aligned with runner naming, backward compatible) --
        speed_target = float(getattr(self, "speed_target", 31.0))
        w_target     = float(getattr(self, "w_target", 0.10))
        w_risk       = float(getattr(self, "w_risk", 1.00))
        w_energy     = float(getattr(self, "w_energy", 0.20))
        w_comfort    = float(getattr(self, "w_comfort", 0.10))

        # Compatible with risk_hard_thresh naming used in __init__ previously
        hard_th      = float(getattr(self, "hard_risk_th",
                            getattr(self, "risk_hard_thresh", 0.95)))
        hard_penalty = float(getattr(self, "hard_penalty", 5.0))

        # -- One-step reward: target speed deviation + risk + energy consumption + comfort --
        r = (
            -abs(speed - speed_target) * w_target
            - risk * w_risk
            - energy * w_energy
            + comfort * w_comfort
        )

        # -- Hard risk penalty (deduct large score when exceeding threshold) --
        if risk >= hard_th:
            r -= hard_penalty

        return float(r)

    # === New in class: Use F_i (or fallback) for one-step prediction to "compact state" ===
    def _compact_from_state(self, st: Dict[str, Any]) -> Dict[str, float]:
        T, E, O = st.get("Traffic", {}), st.get("Ego", {}), st.get("Outcome", {})
        return {
            "ego_x": float(E.get("ego_x", 0.0)),
            "ego_y": float(E.get("ego_y", 0.0)),
            "ego_vx": float(E.get("ego_vx", E.get("ego_speed", 0.0))),
            "ego_vy": float(E.get("ego_vy", 0.0)),
            "headway_distance": float(T.get("headway_distance", 1e6)),
            "lead_ttc": float(T.get("lead_ttc", 1e9)),
            "lead_rel_speed": float(T.get("lead_rel_speed", 0.0)),
            "rear_distance": float(T.get("rear_distance", 1e6)),
            "risk_score": float(O.get("risk_score", 0.5)),
            "energy_consumption": float(O.get("energy_consumption", 0.1)),
            "comfort_score": float(O.get("comfort_score", 0.8)),
        }

    def _apply_fallback_dynamics(self, s: Dict[str, float], action: str, dt: float):
        ax = 0.0
        if action == "FASTER": ax = +1.5
        elif action == "SLOWER": ax = -1.5
        vx = max(0.0, s["ego_vx"] + ax * dt); x = s["ego_x"] + vx * dt
        vy, y = s["ego_vy"], s["ego_y"]
        if action == "LANE_LEFT":  vy = 1.0;  y += 1.0
        if action == "LANE_RIGHT": vy = -1.0; y -= 1.0
        s["ego_x"], s["ego_y"], s["ego_vx"], s["ego_vy"] = x, y, vx, vy
        s["lead_ttc"] = max(1.0, s["lead_ttc"] + (1.0 if ax <= 0 else -1.0) * 50.0)
        s["headway_distance"] = max(1.0, s["headway_distance"] + (1.0 if ax <= 0 else -1.0) * 2.0)
        risk = 1.0 / (1.0 + s["lead_ttc"] / 1000.0)
        energy = max(0.0, 0.05 + (s["ego_vx"] ** 2) * 0.0005 + (abs(ax) * 0.02))
        comfort = max(0.0, min(1.0, 1.0 - abs(ax) * 0.05 - abs(vy) * 0.02))
        s["risk_score"], s["energy_consumption"], s["comfort_score"] = float(risk), float(energy), float(comfort)

    def _predict_one_step(self, state: Dict[str, Any], action: str, dt: float, history: List[Dict[str, Any]]) -> Dict[str, float]:
        s = self._compact_from_state(state)
        try:
            if hasattr(self.structure_function, "predict_next"):
                next_struct = self.structure_function.predict_next(sample=state, action=action, dt=dt, history_frames=history)
                def _get2(d,g,k,default):
                    try:
                        v = d.get(g, {}).get(k, default); return float(v) if isinstance(v,(int,float)) else default
                    except Exception: return default
                s["ego_x"]  = _get2(next_struct, "Ego", "ego_x",  s["ego_x"])
                s["ego_y"]  = _get2(next_struct, "Ego", "ego_y",  s["ego_y"])
                s["ego_vx"] = _get2(next_struct, "Ego", "ego_vx", s["ego_vx"])
                s["ego_vy"] = _get2(next_struct, "Ego", "ego_vy", s["ego_vy"])
                s["headway_distance"] = _get2(next_struct, "Traffic","headway_distance", s["headway_distance"])
                s["lead_ttc"]         = _get2(next_struct, "Traffic","lead_ttc",         s["lead_ttc"])
                s["lead_rel_speed"]   = _get2(next_struct, "Traffic","lead_rel_speed",   s["lead_rel_speed"])
                s["rear_distance"]    = _get2(next_struct, "Traffic","rear_distance",    s["rear_distance"])
                s["risk_score"]         = _get2(next_struct, "Outcome","risk_score",         s["risk_score"])
                s["energy_consumption"] = _get2(next_struct, "Outcome","energy_consumption", s["energy_consumption"])
                s["comfort_score"]      = _get2(next_struct, "Outcome","comfort_score",      s["comfort_score"])
            else:
                self._apply_fallback_dynamics(s, action, dt)
        except Exception:
            self._apply_fallback_dynamics(s, action, dt)
        return s

    # === New in class: H-step open-loop lookahead scoring for an action (MPC) ===
    def _rollout_score(self, sample: Dict[str, Any], first_action: str, history_frames: List[Dict[str, Any]]) -> float:
        total, a, dt = 0.0, first_action, self.dt
        sim_state = copy.deepcopy(sample)
        hf = list(history_frames or [])
        for h in range(self.horizon):
            s_next = self._predict_one_step(sim_state, a, dt, hf)
            total += (self.gamma ** h) * self._step_reward_custom(s_next)
            # Simple closed-loop: next action uses first-order greedy (only looking at one-step reward)
            best_a, best_v = None, -1e9
            for cand in ["IDLE", "FASTER", "SLOWER", "LANE_LEFT", "LANE_RIGHT"]:
                s_tmp = self._predict_one_step(sim_state, cand, dt, hf)
                v = self._step_reward_custom(s_tmp)
                if v > best_v: best_v, best_a = v, cand
            a = best_a
            # Advance "simulated state" and history (fill compact fields back into hierarchical dict)
            sim_state = {
                "Ego":{"ego_x":s_next["ego_x"],"ego_y":s_next["ego_y"],"ego_vx":s_next["ego_vx"],"ego_vy":s_next["ego_vy"],"ego_speed":s_next["ego_vx"]},
                "Traffic":{"headway_distance":s_next["headway_distance"],"lead_ttc":s_next["lead_ttc"],"lead_rel_speed":s_next["lead_rel_speed"],"rear_distance":s_next["rear_distance"]},
                "Outcome":{"risk_score":s_next["risk_score"],"energy_consumption":s_next["energy_consumption"],"comfort_score":s_next["comfort_score"]},
            }
            hf = (hf + [sim_state])[-getattr(self.structure_function, "max_lag", 3):]
        return float(total)

    # === NEW: Plan H-step action sequence at once (plan consistency), Beam Search version ===
    def plan_sequence_beam(
        self,
        sample: Dict[str, Any],
        candidate_actions: Optional[List[str]],
        top_rules: Optional[List[Dict[str, Any]]],
        history_frames: Optional[List[Dict[str, Any]]] = None,
        H: Optional[int] = None,
        beam_width: int = 3,
    ) -> List[str]:
        """
        Returns an action sequence of length H (not recalculated/replaced per step, strictly open-loop rolling),
        used for plan consistency evaluation and execution.
        Score = cumulative discounted reward + alpha * prior (prior added only at first step to avoid repeated amplification)
        """
        cands = self._normalize_candidate_actions(candidate_actions) or ["IDLE", "FASTER"]
        priors_root = self._priors_from_rules(top_rules, cands)
        for a in cands:
            priors_root.setdefault(a, 1e-6)

        H = int(H or self.horizon)
        dt = float(self.dt)
        gamma = float(self.gamma)
        alpha_prior = float(getattr(self, "alpha_prior", 4.0))

        # beam: list of dict(state, hist, actions, score)
        beams = [{
            "state": copy.deepcopy(sample),
            "hist": list(history_frames or []),
            "seq": [],
            "score": 0.0,
            "used_prior": False,
        }]

        for t in range(H):
            new_beams = []
            for b in beams:
                for a in cands:
                    # One-step forward prediction
                    s_next = self._predict_one_step(b["state"], a, dt, b["hist"])
                    r_step = self._step_reward_custom(s_next)

                    # === Action bias (encourage acceleration or lane change) ===
                    r_step += float(getattr(self, "action_bias", {}).get(a, 0.0))

                    # === New: Repeat action penalty (prevent consecutive IDLE / FASTER) ===
                    repeat_penalty = 0.0
                    if len(b["seq"]) >= 2 and b["seq"][-1] == a and b["seq"][-2] == a:
                        repeat_penalty = -0.15  # Apply penalty when same action repeated >= 3 times

                    # === Total score ===
                    sc = b["score"] + (gamma ** t) * (r_step + repeat_penalty)

                    # Only blend prior at first step
                    if (not b["used_prior"]) and t == 0:
                        sc += alpha_prior * float(priors_root.get(a, 0.0))


                    # Advance "simulated state" and "history"
                    sim_state = {
                        "Ego":{"ego_x":s_next["ego_x"],"ego_y":s_next["ego_y"],
                               "ego_vx":s_next["ego_vx"],"ego_vy":s_next["ego_vy"],
                               "ego_speed":s_next["ego_vx"]},
                        "Traffic":{"headway_distance":s_next["headway_distance"],
                                   "lead_ttc":s_next["lead_ttc"],
                                   "lead_rel_speed":s_next["lead_rel_speed"],
                                   "rear_distance":s_next["rear_distance"]},
                        "Outcome":{"risk_score":s_next["risk_score"],
                                   "energy_consumption":s_next["energy_consumption"],
                                   "comfort_score":s_next["comfort_score"]},
                    }
                    hist_next = (b["hist"] + [sim_state])[-getattr(self.structure_function, "max_lag", 3):]

                    new_beams.append({
                        "state": sim_state,
                        "hist": hist_next,
                        "seq": b["seq"] + [a],
                        "score": float(sc),
                        "used_prior": True or b["used_prior"],
                    })

            # Select top beam_width beams with highest scores
            new_beams.sort(key=lambda x: x["score"], reverse=True)
            beams = new_beams[:beam_width]

        # Return the sequence with highest score
        best = max(beams, key=lambda x: x["score"])
        return best["seq"]


    # =========================
    # Main External Interface
    # =========================
    def plan(
        self,
        sample: Dict[str, Any],
        candidate_actions: Optional[List[str]] = None,
        top_rules: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 1,
        history_frames: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        :param sample:         Current step structured state
        :param candidate_actions: Action set filtered by rules
        :param top_rules:      Rule list (containing rule_text & score)
        :param top_k:          Return top K actions
        :param history_frames: Recent frames (length >= max_lag), used for f_i prediction
        """
        cands = self._normalize_candidate_actions(candidate_actions)
        if not cands:
            cands = ["IDLE", "FASTER"]

        # Root action prior (from top_rules of current state)
        self._ctx_state = sample
        priors_root = self._priors_from_rules(top_rules, cands)
        for a in cands:
            priors_root.setdefault(a, 1e-6)

        if getattr(self, "use_mpc_first", False):
            scores = {}
            hf = list(history_frames or [])
            for a in cands:
                rollout = self._rollout_score(sample, a, hf)
                scores[a] = rollout + float(self.alpha_prior) * float(priors_root.get(a, 0.0))
            best = max(scores.items(), key=lambda kv: kv[1])[0]
            return [best][: max(1, top_k)]

        # Root node
        history = list(history_frames or [sample])  # Make a copy
        root = _Node(
            state=copy.deepcopy(sample),
            history=history[-self.structure_function.max_lag:],
            priors=priors_root,
            candidate_actions=cands,
            parent=None,
            incoming_action=None,
            depth=0,
        )
        root.ensure_action_stats()

        # MCTS simulation
        for _ in range(self.n_sim):
            node = root
            path = [node]

            # ------ Selection ------
            while node.children and node.depth < self.max_depth:
                a = self._select_action_puct(node)
                node = node.children[a]
                path.append(node)

            # ------ Expansion ------
            if node.depth < self.max_depth:
                # Expand all candidate actions for current node (recalculate prior using current node's sample)
                self._ctx_state = node.state
                child_priors = self._priors_from_rules(
                    self._rescore_rules_for_node(top_rules, node.state),
                    node.candidate_actions
                )
                for a in node.candidate_actions:
                    child_priors.setdefault(a, 1e-6)
                self._expand(node, child_priors)

            # ------ Evaluation (Leaf Value) ------
            # Pick one action (if just created children are empty, evaluate at current node directly)
            leaf = node
            if leaf.children:
                # Randomly select one child for evaluation (alternatively, average over all children)
                a_leaf = random.choice(list(leaf.children.keys()))
                leaf = leaf.children[a_leaf]

            V, risk_pred = self._leaf_value(leaf)
            # High risk pruning: if risk is too high, reduce V for this path
            if risk_pred is not None and risk_pred > self.risk_hard_thresh:
                V = min(V, 0.0)

            # ------ Backprop ------
            for n in path:
                n.N += 1
                if n.incoming_action is not None and n.parent is not None:
                    a_in = n.incoming_action
                    n.parent.N_a[a_in] = n.parent.N_a.get(a_in, 0) + 1
                    n.parent.W_a[a_in] = n.parent.W_a.get(a_in, 0.0) + V
                    n.parent.Q_a[a_in] = n.parent.W_a[a_in] / max(1, n.parent.N_a[a_in])

        # Select root's best action (by visit count or Q value)
        best = sorted(root.candidate_actions, key=lambda a: (root.N_a.get(a, 0), root.Q_a.get(a, 0.0)), reverse=True)
        if not best:
            best = ["IDLE"]
        return best[: max(1, top_k)]

    # =========================
    # PUCT Selection / Expansion / Evaluation
    # =========================
    def _select_action_puct(self, node: _Node) -> str:
        node.ensure_action_stats()
        total_n = max(1, node.N)
        scores = {}
        for a in node.candidate_actions:
            q = node.Q_a.get(a, 0.0)
            p = node.priors.get(a, 1e-6)
            n_a = node.N_a.get(a, 0)
            u = self.c_puct * p * math.sqrt(total_n) / (1.0 + n_a)
            scores[a] = q + u
        # Select maximum UCB
        a_sel = max(scores.keys(), key=lambda k: scores[k])
        return a_sel

    def _expand(self, node: _Node, child_priors: Dict[str, float]) -> None:
        # Generate a child node for each action: lightweight "simulate one step" to get new state and history
        for a in node.candidate_actions:
            next_state, next_history = self._simulate_one_step(node.state, node.history, a)
            child = _Node(
                state=next_state,
                history=next_history,
                priors=child_priors,             # See _rescore_rules_for_node for state-conditioned priors
                candidate_actions=node.candidate_actions,  # Inherit candidate actions across depth (state-conditioned variant available)
                parent=node,
                incoming_action=a,
                depth=node.depth + 1,
            )
            child.ensure_action_stats()
            node.children[a] = child

    def _leaf_value(self, node: _Node) -> Tuple[float, Optional[float]]:
        """
        V = lambda * max_a P(a|s)  +  (1-lambda) * V_fi(s)
        Returns (V, risk_pred)
        """
        # Strongest confidence from rule prior
        prior_part = max(node.priors.values()) if node.priors else 0.0
        # f_i leaf value
        fi_part, risk_pred = self._leaf_value_from_scm(node.history)
        if fi_part is None:
            V = prior_part
        else:
            V = self.lambda_blend * prior_part + (1.0 - self.lambda_blend) * fi_part
        # Clamp to around [0,1] (not required)
        V = float(np.clip(V, -0.5, 1.5))
        return V, risk_pred

    def _get_numeric(self, state: Dict[str, Any], var: str) -> Optional[float]:
        """Get numeric variable from hierarchical state dict, return None if not found."""
        try:
            for g in ("Environment", "Traffic", "Ego", "Action", "Outcome"):
                d = state.get(g, {})
                if isinstance(d, dict) and var in d:
                    v = d[var]
                    if isinstance(v, (int, float)):
                        return float(v)
            return None
        except Exception:
            return None

    # =========================
    # Rule Prior: Rule -> Action
    # =========================
    # Placed in modules/causal_planner.py, replacing original function
    def _priors_from_rules(self, rules: Optional[List[Dict[str, Any]]], candidate_actions: List[str]) -> Dict[str, float]:
        """
        Aggregate rule list into action prior P(a|s).
        - Text score comes from RuleScorer's score field (0~1)
        - If params.thresholds exists and current state self._ctx_state is provided, apply light weighting based on threshold satisfaction
        - When multiple rules map to the same action, use 1 - product(1 - score_i) for aggregation, avoiding dominance by single outlier value
        """
        priors: Dict[str, float] = {}
        if not rules:
            return priors

        # For reading current state as needed (fallback to None if not set by upper layer)
        state = getattr(self, "_ctx_state", None)

        # First collect all candidate scores for each action, then aggregate
        bucket: Dict[str, List[float]] = {a: [] for a in candidate_actions}

        for r in rules:
            # Get text & action
            text = r.get("rule_text") or r.get("rule") or r.get("raw_text", "")
            action_hint = r.get("action", None)
            a = self._infer_action_from_rule(text)  # Infer from text
            if isinstance(action_hint, str):
                # If rule has action field, prioritize using that field (normalize again)
                a = self._to_canonical(action_hint) or a

            if a not in candidate_actions:
                continue

            # Base score (RuleScorer output)
            base = float(r.get("score", 0.0))
            base = max(0.0, min(1.0, base))

            # Confidence (from graph building/extraction confidence, optional)
            conf = float(r.get("confidence", 1.0))
            conf = max(0.0, min(1.0, conf))

            # Threshold satisfaction (optional): only enabled when rule has params.thresholds and we can get current state
            sat = 1.0
            params = r.get("params") or {}
            thresholds = params.get("thresholds") if isinstance(params, dict) else None
            if thresholds and isinstance(thresholds, dict) and state is not None:
                sats = []
                for var, cfg in thresholds.items():
                    try:
                        op = str(cfg.get("op", "")).strip()
                        val = cfg.get("value", None)
                        x = self._get_numeric(state, var)
                        if x is None or val is None or op not in (">", "<", ">=", "<="):
                            continue
                        ok = ((op == ">" and x > val) or
                            (op == "<" and x < val) or
                            (op == ">=" and x >= val) or
                            (op == "<=" and x <= val))
                        sats.append(1.0 if ok else 0.5)   # Not satisfied gives 0.5 light downweight; you can also change to 0.0 for hard pruning
                    except Exception:
                        pass
                if sats:
                    sat = float(sum(sats) / len(sats))

            # Light fusion: text prior x confidence x threshold satisfaction
            fused = base * (0.8 * conf + 0.2) * (0.8 * sat + 0.2)
            fused = max(1e-6, min(1.0, fused))

            bucket[a].append(fused)

        # Action-level aggregation: p = 1 - product(1 - s_i)
        for a, arr in bucket.items():
            if not arr:
                continue
            prod = 1.0
            for s in arr:
                prod *= (1.0 - s)
            priors[a] = 1.0 - prod

        # Normalization (optional, but usually helps PUCT stability)
        # Before normalization, slightly reduce IDLE prior to encourage active exploration
        if "IDLE" in priors:
            priors["IDLE"] *= 0.35

        # Normalization
        s = sum(priors.values())
        if s > 0:
            for a in priors:
                priors[a] = float(priors[a] / s)
        else:
            eps = 1.0 / max(1, len(candidate_actions))
            priors = {a: eps for a in candidate_actions}

        return priors



    def _rescore_rules_for_node(self, rules: Optional[List[Dict[str, Any]]], state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Re-score rules on node's local state (acceptable overhead for small depth and few candidates)."""
        if not rules:
            return []
        # Unify to rule_text as field name to avoid KeyError
        inp = []
        for r in rules:
            text = r.get("rule_text") or r.get("rule") or r.get("raw_text", "") or ""
            inp.append({"rule_text": text})

        with torch.no_grad():
            rs = self.rule_scorer.score(inp, [state])

        # -- Enrich score back uniformly -- #
        enriched = []
        if isinstance(rs, (list, tuple)) and rs and isinstance(rs[0], (float, int)):
            for i, item in enumerate(inp):
                enriched.append({"rule_text": item["rule_text"], "score": float(rs[i])})
        elif isinstance(rs, (list, tuple)) and rs and isinstance(rs[0], dict):
            for i, it in enumerate(rs):
                rt = it.get("rule_text") or inp[i]["rule_text"]
                sc = float(it.get("score", 0.0))
                enriched.append({"rule_text": rt, "score": sc})
        else:
            import numpy as np
            arr = np.array(rs).reshape(-1)
            for i, item in enumerate(inp):
                enriched.append({"rule_text": item["rule_text"], "score": float(arr[i])})
        return enriched


    # =========================
    # Lightweight Simulation: One-step Prediction of Key Variables Based on f_i
    # =========================
    def _simulate_one_step(
        self,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
        action_name: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Without calling the real environment, only do "lightweight one step" based on f_i:
          - Set Action.action_type to action name
          - Use f_i to predict several key variables in Ego/Outcome, forming next_state
          - Add new state to history, trim length to max_lag
        """
        # Copy current state
        next_state = copy.deepcopy(state)
        next_state.setdefault("Action", {})
        next_state["Action"]["action_type"] = action_name

        # Predict several key variables (if corresponding model doesn't exist/returns None, keep current value)
        keys_predict_ego = ["ego_speed", "ego_acceleration", "yaw_rate", "jerk", "lane_id"]
        keys_predict_out = ["risk_score", "comfort_score", "energy_consumption", "violation_flag"]

        new_ego = next_state.setdefault("Ego", {})
        new_out = next_state.setdefault("Outcome", {})

        # history + next_state (as "current") will be used by predict_target
        hist = list(history)
        # Put a "with action" current frame at the end of history, keeping structure consistent
        curr_with_action = copy.deepcopy(state)
        curr_with_action.setdefault("Action", {})
        curr_with_action["Action"]["action_type"] = action_name
        hist = (hist + [curr_with_action])[-self.structure_function.max_lag:]

        for k in keys_predict_ego:
            val = self.structure_function.predict_target(k, hist)
            if val is not None:
                new_ego[k] = float(val)
        for k in keys_predict_out:
            val = self.structure_function.predict_target(k, hist)
            if val is not None:
                new_out[k] = float(val)

        # As new history, append next_state
        new_history = (history + [next_state])[-self.structure_function.max_lag:]

        return next_state, new_history

    # =========================
    # Leaf Heuristic: f_i Combined Value
    # =========================
    def _leaf_value_from_scm(self, history_frames: List[Dict[str, Any]], w=(0.6, 0.3, 0.1)) -> Tuple[Optional[float], Optional[float]]:
        """
        Use trained f_i to estimate risk/comfort/energy and synthesize leaf value V_fi(s) in [0,1] range
        Returns (V_fi, risk_pred)
        """
        if not self.structure_function or not history_frames:
            return None, None

        risk = self.structure_function.predict_target("risk_score", history_frames)
        comfort = self.structure_function.predict_target("comfort_score", history_frames)
        energy = self.structure_function.predict_target("energy_consumption", history_frames)

        if risk is None and comfort is None and energy is None:
            return None, None

        r = float(risk) if risk is not None else 0.5  # 0~1, higher is more dangerous
        c = float(comfort) if comfort is not None else 0.5  # 0~1, higher is more comfortable
        e = float(energy) if energy is not None else 0.0    # ~[0,2] approximately, linear scaling
        e = max(0.0, min(1.0, e / 2.0))

        alpha, beta, gamma = w
        v = alpha * (1.0 - r) + beta * c - gamma * e
        return float(v), float(r)

    # =========================
    # SCM Model Loading (Offline)
    # =========================
    def _lazy_load_fi_models(self):
        """
        Read fi_index.json and each fi_*.pt from scm_dir, populate into structure_function.
        Skip if models already exist.
        """
        try:
            if getattr(self.structure_function, "fi_models", None):
                # Looks like already trained/loaded
                return

            idx_path = os.path.join(self.scm_dir, "fi_index.json")
            if not os.path.isfile(idx_path):
                print(f"[CausalPlanner] fi_index.json not found at {idx_path}, skip loading f_i models.")
                return

            with open(idx_path, "r") as f:
                idx = json.load(f)

            self.structure_function.fi_models = {}
            self.structure_function.fi_specs = {}

            for tgt, spec in idx.items():
                ckpt_path = spec.get("path") or os.path.join(self.scm_dir, f"fi_{tgt}.pt")
                if not os.path.isfile(ckpt_path):
                    continue
                ckpt = torch.load(ckpt_path, map_location="cpu")
                model_state = ckpt.get("model_state_dict")
                spec_dict = ckpt.get("spec") or spec

                # Restore spec
                from types import SimpleNamespace
                sn = SimpleNamespace(**spec_dict)
                self.structure_function.fi_specs[tgt] = sn

                # Rebuild model based on spec
                if spec_dict.get("model_type", "transformer") == "mlp":
                    from src.world_model.dynamics.mlp import FunctionMLP
                    model = FunctionMLP(input_dim=spec_dict["input_dim"], hidden_dim=self.structure_function.fi_hidden)
                else:
                    from src.world_model.dynamics.transformer import FunctionTransformer
                    # num_vars matches the training-time signature: one input per discovered parent variable
                    model = FunctionTransformer(num_vars=spec_dict["num_parents"], model_dim=self.structure_function.fi_hidden, dropout=0.2)

                model.load_state_dict(model_state, strict=False)
                model.eval()
                self.structure_function.fi_models[tgt] = model

            print(f"[CausalPlanner] Loaded f_i models from {self.scm_dir}: {len(self.structure_function.fi_models)} targets.")

        except Exception as e:
            print(f"[CausalPlanner] Failed to load f_i models: {e}")

    # =========================
    # Parse/Normalize Actions & Rule Mapping
    # =========================
    def _normalize_candidate_actions(self, actions: Optional[List[str]]) -> List[str]:
        if not actions:
            return []
        out, seen = [], set()
        for a in actions:
            c = self._to_canonical(a)
            if c and (c in self.CANONICAL_ACTIONS) and (c not in seen):
                out.append(c)
                seen.add(c)
        return out

    def _to_canonical(self, a: str) -> Optional[str]:
        if not isinstance(a, str):
            return None
        t = a.strip().upper()
        if t in {"LANE_LEFT", "TURN_LEFT", "GO_LEFT"}:
            return "LANE_LEFT"
        if t in {"LANE_RIGHT", "TURN_RIGHT", "GO_RIGHT"}:
            return "LANE_RIGHT"
        if t in {"IDLE", "KEEP_LANE", "MAINTAIN", "STAY"}:
            return "IDLE"
        if t in {"FASTER", "ACCELERATE", "INCREASE_SPEED", "SPEED_UP"}:
            return "FASTER"
        if t in {"SLOWER", "BRAKE", "DECELERATE", "REDUCE_SPEED", "SLOW_DOWN"}:
            return "SLOWER"
        if t in self.CANONICAL_ACTIONS:
            return t
        return None

    def _infer_action_from_rule(self, text: str) -> Optional[str]:
        t = (text or "").lower()
        if any(k in t for k in ["brake", "decelerate", "reduce speed", "slow down", "slower"]):
            return "SLOWER"
        if any(k in t for k in ["accelerate", "increase speed", "speed up", "faster"]):
            return "FASTER"
        if any(k in t for k in ["lane change left", "turn left", "go left", "lane_left"]):
            return "LANE_LEFT"
        if any(k in t for k in ["lane change right", "turn right", "go right", "lane_right"]):
            return "LANE_RIGHT"
        if any(k in t for k in ["keep lane", "maintain", "stay", "idle"]):
            return "IDLE"
        return "IDLE"
