# src/utils/visualization.py
# -*- coding: utf-8 -*-
"""
Unified visualization module for CausalTrajectoryPlanner.
Combines: inference results, rule graphs, training loss, variable hierarchy.
"""

import json
import math
import os
import re
import random
import textwrap
from collections import Counter
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np


# =============================================================================
# Part 1: Inference Results Visualization (from visualize_inference.py)
# =============================================================================

def _num_ok(x: Any) -> bool:
    if x is None:
        return False
    try:
        xf = float(x)
    except Exception:
        return False
    return not math.isnan(xf) and not math.isinf(xf)


def _rule_text_from_item(r: Dict[str, Any]) -> str:
    """Compatible with multiple field names: rule_text/raw_text/rule/text/description."""
    return (
        r.get("rule_text")
        or r.get("raw_text")
        or r.get("rule")
        or r.get("text")
        or r.get("description")
        or ""
    )


def _action_name_from_step(t: Dict[str, Any]) -> str:
    """Compatible with field names: chosen_action_name / action / action_name."""
    return t.get("chosen_action_name") or t.get("action") or t.get("action_name") or ""


def visualize_inference_results(
    result_path: str = "result/inference/inference_results.json",
    output_dir: str = "result/Roundabout",
) -> None:
    """
    Visualize inference results:
      1) Reward curve for each episode (step-by-step reward)
      2) ADE/FDE curve for each episode (step-by-step)
      3) Rule-to-action causal graph for each episode (based on top_rules pointing to selected action at each step)
    File names include episode_id suffix to avoid overwriting.
    """
    if not os.path.exists(result_path):
        print(f"[WARN] Result file not found: {result_path}")
        return

    with open(result_path, "r") as f:
        results = json.load(f)

    if not isinstance(results, list) or len(results) == 0:
        print("[WARN] No inference results found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for ep in results:
        ep_id = ep.get("episode_id", 0)
        traj: List[Dict[str, Any]] = ep.get("trajectory", [])
        if len(traj) == 0:
            print(f"[WARN] Episode {ep_id} has empty trajectory, skip.")
            continue

        # ---------- 1) Reward curve ----------
        steps = [int(t.get("step", i)) for i, t in enumerate(traj)]
        rewards = [float(t.get("running_reward", 0.0)) for t in traj]

        plt.figure(figsize=(8, 5))
        plt.plot(steps, rewards, marker="o")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        ade_final = ep.get("ADE")
        fde_final = ep.get("FDE")
        ade_txt = f"{float(ade_final):.3f}" if _num_ok(ade_final) else "nan"
        fde_txt = f"{float(fde_final):.3f}" if _num_ok(fde_final) else "nan"
        plt.title(f"Reward Curve (Episode {ep_id})\nADE={ade_txt}, FDE={fde_txt}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out1 = os.path.join(output_dir, f"reward_curve_ep{ep_id}.png")
        plt.savefig(out1, dpi=300)
        plt.close()
        print(f"[OK] Saved {os.path.basename(out1)}")

        # ---------- 2) ADE / FDE curve ----------
        ade_curve: List[Tuple[int, float]] = []
        fde_curve: List[Tuple[int, float]] = []
        for t in traj:
            s = int(t.get("step", 0))
            ade_s = t.get("ADE_step", None)
            fde_s = t.get("FDE_step", None)
            if _num_ok(ade_s):
                ade_curve.append((s, float(ade_s)))
            if _num_ok(fde_s):
                fde_curve.append((s, float(fde_s)))

        plt.figure(figsize=(8, 5))
        if len(ade_curve) > 0:
            xs, ys = zip(*ade_curve)
            plt.plot(xs, ys, marker="o", label="ADE")
        if len(fde_curve) > 0:
            xs, ys = zip(*fde_curve)
            plt.plot(xs, ys, marker="s", label="FDE")
        if len(ade_curve) == 0 and len(fde_curve) == 0:
            plt.text(
                0.5, 0.5, "No ADE/FDE per-step data.", ha="center", va="center", transform=plt.gca().transAxes
            )
        plt.xlabel("Step")
        plt.ylabel("Error (distance)")
        plt.title(f"Per-step ADE/FDE (Episode {ep_id})")
        if len(ade_curve) > 0 or len(fde_curve) > 0:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out2 = os.path.join(output_dir, f"ade_fde_curve_ep{ep_id}.png")
        plt.savefig(out2, dpi=300)
        plt.close()
        print(f"[OK] Saved {os.path.basename(out2)}")

        # ---------- 3) Causal graph: Rule -> Action ----------
        edge_weight: Dict[Tuple[str, str], float] = {}
        for t in traj:
            action = _action_name_from_step(t)
            action = str(action).strip()
            if not action:
                continue

            rules_list = t.get("top_rules", [])
            if not isinstance(rules_list, list):
                continue

            rw = t.get("reward", 0.0)
            try:
                rwf = float(rw)
            except Exception:
                rwf = 0.0

            for r in rules_list:
                rule_str = _rule_text_from_item(r)
                rule_str = str(rule_str).strip()
                if not rule_str:
                    continue
                key = (rule_str, action)
                edge_weight[key] = edge_weight.get(key, 0.0) + max(rwf, 0.0)

        if len(edge_weight) == 0:
            print(f"[WARN] Episode {ep_id}: No rule->action edges, skip causal graph.")
            continue

        G = nx.DiGraph()
        for (rule_str, action), w in edge_weight.items():
            G.add_edge(f"rule::{rule_str}", f"action::{action}", weight=float(w))

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.6, seed=42)
        node_colors = []
        for n in G.nodes():
            node_colors.append("#91c9f7" if n.startswith("rule::") else "#a8e6a1")

        weights = np.array([d.get("weight", 1.0) for _, _, d in G.edges(data=True)], dtype=float)
        if weights.size == 0:
            widths = 1.0
        else:
            w_norm = weights / (weights.max() if weights.max() > 0 else 1.0)
            widths = 1.0 + 3.0 * w_norm

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            edge_color="gray",
            width=widths.tolist() if isinstance(widths, np.ndarray) else widths,
            font_size=8,
        )
        plt.title(f"Rule -> Action Graph (Episode {ep_id})")
        plt.tight_layout()
        out3 = os.path.join(output_dir, f"causal_graph_ep{ep_id}.png")
        plt.savefig(out3, dpi=300)
        plt.close()
        print(f"[OK] Saved {os.path.basename(out3)}")


# =============================================================================
# Part 2: Rule Graph Visualization (from rule_val.py)
# =============================================================================

ALL_ACTIONS = ["IDLE", "FASTER", "SLOWER", "LANE_LEFT", "LANE_RIGHT"]


def _safe(x):
    if x is None:
        return ""
    return str(x).strip()


def _nid(kind: str, name: str):
    return f"{kind}:{name}"


def concentric_ring_layout_equal_density(
    nodes, seed=42, n_rings=7, r_min=0.25, r_max=1.0, angle_jitter=0.03
):
    """Concentric ring layout with equal angular density."""
    rng = random.Random(seed)
    nodes = list(nodes)
    rng.shuffle(nodes)

    n = len(nodes)

    if n_rings == 1:
        radii = [r_max]
    else:
        step = (r_max - r_min) / (n_rings - 1)
        radii = [r_min + i * step for i in range(n_rings)]

    weights = [max(r, 1e-6) for r in radii]
    wsum = sum(weights)
    counts = [max(1, int(n * w / wsum)) for w in weights]

    diff = n - sum(counts)
    i = 0
    while diff != 0:
        idx = i % n_rings
        if diff > 0:
            counts[idx] += 1
            diff -= 1
        else:
            if counts[idx] > 1:
                counts[idx] -= 1
                diff += 1
        i += 1

    pos = {}
    idx_node = 0
    for ring_i, cnt in enumerate(counts):
        r = radii[ring_i]
        for j in range(cnt):
            if idx_node >= n:
                break
            node = nodes[idx_node]
            idx_node += 1

            theta = 2 * math.pi * j / cnt
            theta += rng.uniform(-angle_jitter, angle_jitter)
            pos[node] = (r * math.cos(theta), r * math.sin(theta))

    return pos


def visualize_rule_graph(
    json_path: str,
    output_path: str = "sampled_rules_graph.pdf",
    sample_n: int = 200,
    seed: int = 42,
    figsize: Tuple[int, int] = (10, 10),
):
    """
    Visualize rule graph (variable -> rule -> action relationship graph)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "rules" in data:
        rules = data["rules"]
    elif isinstance(data, list):
        rules = data
    else:
        raise ValueError(f"Unrecognized JSON structure: {type(data)}")

    print(f"Loaded rules: {len(rules)}")

    idx_counter = Counter(_safe(r.get("rule_index", "")) or "UNKNOWN" for r in rules)
    print("\n[rule_index counts]")
    for k, v in idx_counter.most_common():
        print(f"{k}: {v}")

    random.seed(seed)
    sampled = random.sample(rules, k=min(sample_n, len(rules)))

    G = nx.DiGraph()
    node_kind = {}

    def add_node(kind: str, name: str):
        name = _safe(name)
        if not name:
            return None
        node_id = _nid(kind, name)
        if node_id not in G:
            G.add_node(node_id)
            node_kind[node_id] = kind
        return node_id

    existing_actions_in_sample = set()

    for r in sampled:
        cause = _safe(r.get("cause", ""))
        effect = _safe(r.get("effect", ""))
        rule_txt = _safe(r.get("rule", "")) or _safe(r.get("rule_text_clean", "")) or _safe(r.get("rule_text", ""))
        action = _safe(r.get("action", ""))

        if action:
            existing_actions_in_sample.add(action)

        c_id = add_node("variable", cause)
        e_id = add_node("variable", effect)
        r_id = add_node("rule", rule_txt)
        a_id = add_node("action", action)

        if c_id and r_id:
            G.add_edge(c_id, r_id)
        if e_id and r_id:
            G.add_edge(e_id, r_id)
        if r_id and a_id:
            G.add_edge(r_id, a_id)

    for a in ALL_ACTIONS:
        a_id = _nid("action", a)
        if a_id not in G:
            G.add_node(a_id)
            node_kind[a_id] = "action"

    color_map = {"variable": "gray", "rule": "royalblue", "action": "hotpink"}
    node_colors = [color_map.get(node_kind.get(n, "variable"), "gray") for n in G.nodes()]

    node_sizes = []
    for n in G.nodes():
        k = node_kind.get(n, "variable")
        if k == "action":
            node_sizes.append(100)
        elif k == "rule":
            node_sizes.append(100)
        else:
            node_sizes.append(100)

    pos = concentric_ring_layout_equal_density(
        G.nodes(), seed=seed, n_rings=7, r_min=0.25, r_max=1.0, angle_jitter=0.03
    )

    edge_colors = [color_map.get(node_kind.get(u, "variable"), "gray") for u, v in G.edges()]

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(
        G, pos, arrows=True, arrowstyle="-|>", arrowsize=9,
        width=1.15, alpha=0.55, edge_color=edge_colors,
    )
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        linewidths=0.9, edgecolors="#222222", alpha=0.95,
    )

    action_nodes = [n for n in G.nodes() if node_kind.get(n) == "action"]
    action_labels = {n: n.split("action:", 1)[-1] for n in action_nodes}
    nx.draw_networkx_labels(G, pos, labels=action_labels, font_size=17, font_color="black")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Variable",
               markerfacecolor="gray", markeredgecolor="#222222", markersize=11),
        Line2D([0], [0], marker="o", color="w", label="Rule",
               markerfacecolor="royalblue", markeredgecolor="#222222", markersize=11),
        Line2D([0], [0], marker="o", color="w", label="Action",
               markerfacecolor="hotpink", markeredgecolor="#222222", markersize=11),
    ]
    plt.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=20)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nSaved figure to: {output_path}")


# =============================================================================
# Part 3: Training Loss Visualization (from loss.py)
# =============================================================================

def visualize_training_loss(
    log_path: str,
    save_path: str = "result/figures/combined_step_loss_smooth.png",
    last_n_epochs: int = 5,
    window: int = 200,
):
    """
    Visualize training loss curve (with smoothing)
    """
    pattern = re.compile(r"Epoch\s+(\d+)\s+Step\s+(\d+)\s+Loss\s+([0-9.]+)")

    epoch_data = {}

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch, step, loss = map(float, match.groups())
                epoch = int(epoch)
                if epoch not in epoch_data:
                    epoch_data[epoch] = []
                epoch_data[epoch].append(loss)

    epochs = sorted(epoch_data.keys())
    if len(epochs) < last_n_epochs:
        print(f"[WARN] Log only contains {len(epochs)} epochs, fewer than {last_n_epochs}")
        last_n_epochs = len(epochs)

    selected_epochs = epochs[-last_n_epochs:]
    print(f"[INFO] Concatenated Epochs: {selected_epochs}")

    combined_losses = np.concatenate([epoch_data[e] for e in selected_epochs])
    steps = np.arange(1, len(combined_losses) + 1)

    def smooth(x, window_size=100):
        if len(x) < window_size:
            return x
        kernel = np.ones(window_size) / window_size
        return np.convolve(x, kernel, mode="valid")

    smoothed = smooth(combined_losses, window_size=window)

    plt.figure(figsize=(10, 5))
    plt.plot(steps[:len(smoothed)], smoothed, linewidth=2.2, color="tab:blue", label=f"Smoothed (window={window})")
    plt.plot(steps, combined_losses, color="gray", alpha=0.2, label="Raw Step Loss")

    plt.title(f"Combined Step-wise Training Loss (Last {last_n_epochs} Epochs, Smoothed)")
    plt.xlabel("Step (across last epochs)")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[OK] Saved smoothed curve plot -> {save_path}")


# =============================================================================
# Part 4: Variable Hierarchy Visualization (from plot_varibales.py)
# =============================================================================

MACROS = ["Environment", "Traffic", "Ego", "Action", "Outcome"]
MACRO_DATA = {
    "Environment": ["weather_type", "visibility", "road_friction", "road_type", "time_of_day", "speed_limit"],
    "Traffic": ["headway_distance", "lead_speed", "lead_rel_speed", "lead_ttc", "lead_thw",
                "rear_distance", "rear_speed", "rear_rel_speed"],
    "Ego": ["ego_x", "ego_y", "ego_vx", "ego_vy", "ego_speed"],
    "Action": ["FASTER", "SLOWER", "LANE_LEFT", "LANE_RIGHT", "IDLE"],
    "Outcome": ["risk_score", "comfort_score", "energy_consumption", "success_flag", "violation_flag"]
}
MACRO_COLOR_MAP = {
    "Environment": "#A8C8E8",
    "Traffic":     "#A9EAD2",
    "Ego":         "#FAD7A0",
    "Action":      "#F5B7B1",
    "Outcome":     "#D7BDE2",
}


def _soft_wrap(label: str, max_width: int = 12) -> str:
    """Prefer line breaks at underscores"""
    if "_" in label:
        parts = label.split("_")
        left, right, cur = [], [], 0
        total = len(label.replace("_", ""))
        for p in parts:
            (left if cur < total/2 else right).append(p)
            cur += len(p)
        base = "_".join(left) if right else label
        rest = "_".join(right)
        joined = " ".join([base, rest] if rest else [base])
    else:
        joined = label
    return "\n".join(textwrap.wrap(joined, width=max_width)[:2])


def _fit_fs(span_deg: float, radius: float, max_fs: int, min_fs: int = 7) -> int:
    """Automatically estimate font size based on sector arc length"""
    span_rad = math.radians(span_deg)
    arc_len = radius * span_rad
    est = arc_len / (0.002 * 12)
    return int(max(min_fs, min(max_fs, est)))


def _add_text(ax, txt, r, ang, fs, wedge, bold=False):
    """Draw text and constrain it within the sector"""
    t = ax.text(r*math.cos(math.radians(ang)),
                r*math.sin(math.radians(ang)),
                txt, ha="center", va="center",
                fontsize=fs, fontweight=("bold" if bold else "normal"),
                rotation=ang-90, rotation_mode="anchor",
                color="black", clip_on=True)
    t.set_clip_path(wedge)
    return t


def visualize_variable_hierarchy(
    output_path: str = "hierarchical_radial.png",
    macros: List[str] = None,
    data: Dict[str, List[str]] = None,
    color_map: Dict[str, str] = None,
):
    """
    Visualize variable hierarchy structure (radial layout)
    """
    macros = macros or MACROS
    data = data or MACRO_DATA
    color_map = color_map or MACRO_COLOR_MAP

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"aspect": "equal"})
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis("off")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    RINGS = [
        (0.00, 0.34),
        (0.345, 0.58),
        (0.585, 0.82),
        (0.825, 0.99),
    ]

    n = len(macros)
    macro_span = 360.0 / n
    PAD_MACRO = 0.12
    PAD_MICRO = 0.35
    SEAM_LW = 0.5

    for i, macro in enumerate(macros):
        color = color_map[macro]
        th1 = i * macro_span
        th2 = (i + 1) * macro_span
        m1, m2 = th1 + PAD_MACRO, th2 - PAD_MACRO
        m_mid = (m1 + m2) / 2

        r0, r1 = RINGS[0]
        w_macro = Wedge((0, 0), r1, th1, th2, width=r1-r0,
                        facecolor=color, edgecolor="white", linewidth=SEAM_LW)
        ax.add_patch(w_macro)
        fsM = _fit_fs(m2 - m1, (r0 + r1) / 2, max_fs=34)
        _add_text(ax, macro, (r0 + r1) / 2, m_mid, fsM, w_macro, bold=True)

        micros = data[macro]

        r0, r1 = RINGS[1]
        top2 = micros[:2]
        if top2:
            part = (m2 - m1) / len(top2)
            for j, var in enumerate(top2):
                a1, a2 = m1 + j * part + PAD_MICRO, m1 + (j + 1) * part - PAD_MICRO
                w2 = Wedge((0, 0), r1, a1, a2, width=r1 - r0,
                           facecolor=color, edgecolor="white", linewidth=SEAM_LW)
                ax.add_patch(w2)
                ang = (a1 + a2) / 2
                rr = (r0 + r1) / 2
                fs2 = _fit_fs(a2 - a1, rr, max_fs=28)
                _add_text(ax, _soft_wrap(var, 10), rr, ang, fs2, w2)

        r0, r1 = RINGS[2]
        last3 = micros[-3:]
        if last3:
            part = (m2 - m1) / len(last3)
            for j, var in enumerate(last3):
                a1, a2 = m1 + j * part + PAD_MICRO, m1 + (j + 1) * part - PAD_MICRO
                w3 = Wedge((0, 0), r1, a1, a2, width=r1 - r0,
                           facecolor=color, edgecolor="white", linewidth=SEAM_LW)
                ax.add_patch(w3)
                ang = (a1 + a2) / 2
                rr = (r0 + r1) / 2
                fs3 = _fit_fs(a2 - a1, rr, max_fs=28)
                _add_text(ax, _soft_wrap(var, 10), rr, ang, fs3, w3)

    r0, r1 = RINGS[3]
    for i, macro in enumerate(macros):
        color = color_map[macro]
        a1, a2 = i * macro_span + PAD_MACRO, (i + 1) * macro_span - PAD_MACRO
        w4 = Wedge((0, 0), r1, a1, a2, width=r1 - r0,
                   facecolor=color, edgecolor="white", linewidth=SEAM_LW)
        ax.add_patch(w4)
        mid = (a1 + a2) / 2
        rr = (r0 + r1) / 2
        fs4 = _fit_fs(a2 - a1, rr, max_fs=28)
        _add_text(ax, "...", rr, mid, fs4, w4)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close()
    print(f"[OK] Saved: {output_path}")
