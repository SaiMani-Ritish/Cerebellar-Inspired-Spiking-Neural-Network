"""
Diagnostics for SNN research results.

Runs three checks:
  1. Action distribution per controller (histogram)
  2. Position vs time for a few representative episodes
  3. JAS (Jerk-Accuracy Score) for all controllers

Run from the SNN/ parent directory:
    python -m snn_research.diagnostics
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – saves PNGs, no display needed
import matplotlib.pyplot as plt
import torch
import gymnasium as gym

try:
    from .model import QuinticPlanner, SNNController
    from .data import state_to_spikes
except ImportError:
    from model import QuinticPlanner, SNNController
    from data import state_to_spikes

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DIAG_DIR = os.path.join(OUTPUTS_DIR, "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)

N_DIAG_EPISODES = 20       # episodes for action-distribution check
N_TRAJ_EPISODES = 5        # episodes to plot full trajectories
SEEDS = list(range(100))   # same seeds as full evaluation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_saved_controller():
    """Re-create controller objects with random weights (same as eval run)."""
    quintic = QuinticPlanner(T=2.0, dt=0.02)

    snn_only = SNNController(beta=0.9, num_steps=25)
    snn_only.stdp_params = {"A_plus": 0.0, "A_minus": 0.0,
                            "tau_plus": 20.0, "tau_minus": 20.0}
    snn_only.eval()

    snn_stdp = SNNController(beta=0.9, num_steps=25)
    snn_stdp.eval()

    return {
        "quintic_polynomial": quintic,
        "snn_surrogate_only": snn_only,
        "snn_stdp": snn_stdp,
    }


def _run_episode(controller, name, seed):
    """Run one episode, return (actions list, positions list)."""
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)

    if hasattr(controller, "reset"):
        controller.reset(obs)

    actions, positions = [], []
    done = False
    while not done:
        # SNNController exposes .act(obs); QuinticPlanner is directly callable
        if hasattr(controller, "act"):
            action, _ = controller.act(obs)
        else:
            action, _ = controller(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        actions.append(int(action))
        positions.append(float(obs[0]))
        done = terminated or truncated

    env.close()
    return actions, positions


# ---------------------------------------------------------------------------
# Diagnostic 1: Action distribution
# ---------------------------------------------------------------------------

def diag_action_distribution(controllers: dict):
    print("\n=== Diagnostic 1: Action distributions ===")
    summary = {}

    for name, ctrl in controllers.items():
        all_actions = []
        for seed in SEEDS[:N_DIAG_EPISODES]:
            acts, _ = _run_episode(ctrl, name, seed)
            all_actions.extend(acts)

        total = len(all_actions)
        n0 = all_actions.count(0)
        n1 = all_actions.count(1)
        pct_0 = 100 * n0 / total
        pct_1 = 100 * n1 / total
        summary[name] = {"total_steps": total, "pct_0": pct_0, "pct_1": pct_1}
        print(f"  {name:25s}  action=0: {pct_0:5.1f}%  action=1: {pct_1:5.1f}%"
              f"  (n={total} steps over {N_DIAG_EPISODES} episodes)")

        if max(pct_0, pct_1) > 90:
            print(f"    *** WARNING: {name} outputs one action >{max(pct_0,pct_1):.0f}% of the time — likely stereotyped policy ***")

    # Bar chart
    names = list(summary.keys())
    pct0s = [summary[n]["pct_0"] for n in names]
    pct1s = [summary[n]["pct_1"] for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    bars0 = ax.bar(x - w/2, pct0s, w, label="Action 0 (left)", color="#c8440a", alpha=0.8)
    bars1 = ax.bar(x + w/2, pct1s, w, label="Action 1 (right)", color="#1a6b5e", alpha=0.8)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% (uniform)")
    ax.axhline(90, color="red", linestyle=":", linewidth=1.0, label="90% (stereotypy threshold)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylabel("% of steps")
    ax.set_title(f"Action distribution ({N_DIAG_EPISODES} episodes each)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(DIAG_DIR, "diag1_action_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return summary


# ---------------------------------------------------------------------------
# Diagnostic 2: Position trajectories
# ---------------------------------------------------------------------------

def diag_trajectories(controllers: dict):
    print("\n=== Diagnostic 2: Position vs time trajectories ===")
    colors = {"quintic_polynomial": "#c8440a",
              "snn_surrogate_only": "#888888",
              "snn_stdp": "#1a6b5e"}

    for seed in SEEDS[:N_TRAJ_EPISODES]:
        fig, axes = plt.subplots(1, len(controllers), figsize=(5 * len(controllers), 3),
                                 sharey=False)
        for ax, (name, ctrl) in zip(axes, controllers.items()):
            _, positions = _run_episode(ctrl, name, seed)
            t = np.arange(len(positions)) * 0.02
            ax.plot(t, positions, color=colors.get(name, "blue"), linewidth=1.2)
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
            ax.set_title(name.replace("_", "\n"), fontsize=8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Cart position (m)")
            ep_len = len(positions) * 0.02
            final_err = abs(positions[-1])
            ax.set_title(f"{name.replace('_', ' ')}\nlen={ep_len:.1f}s  err={final_err:.3f}m",
                         fontsize=7)
        fig.suptitle(f"Episode seed={seed}", fontsize=9)
        fig.tight_layout()
        path = os.path.join(DIAG_DIR, f"diag2_trajectory_seed{seed:02d}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Diagnostic 3: JAS (Jerk-Accuracy Score)
# ---------------------------------------------------------------------------

def jerk_accuracy_score(peak_jerk: float, endpoint_err: float,
                        jerk_baseline: float, err_threshold: float = 0.5) -> float:
    """
    Validity-gated combined smoothness-accuracy metric.

    jerk_score > 1  means smoother than the baseline.
    The gate penalises controllers that don't complete the task.
    """
    jerk_score = jerk_baseline / peak_jerk
    if endpoint_err > err_threshold:
        return jerk_score * (err_threshold / endpoint_err)
    return jerk_score


def diag_jas(thresholds=(0.05, 0.15, 0.50)):
    print("\n=== Diagnostic 3: Jerk-Accuracy Score (JAS) ===")

    results_files = {
        "quintic_polynomial":  "quintic_polynomial_results.csv",
        "snn_surrogate_only":  "snn_surrogate_only_results.csv",
        "snn_stdp":            "snn_stdp_results.csv",
    }

    stats = {}
    for name, fname in results_files.items():
        path = os.path.join(OUTPUTS_DIR, fname)
        if not os.path.exists(path):
            print(f"  MISSING: {path} — skipping")
            continue
        df = pd.read_csv(path)
        stats[name] = {
            "jerk_mean": df["peak_jerk"].mean(),
            "jerk_std":  df["peak_jerk"].std(),
            "err_mean":  df["endpoint_err"].mean(),
            "err_std":   df["endpoint_err"].std(),
        }

    if "quintic_polynomial" not in stats:
        print("  Cannot compute JAS — quintic results missing.")
        return

    jerk_baseline = stats["quintic_polynomial"]["jerk_mean"]
    print(f"  Jerk baseline (quintic mean): {jerk_baseline:.2f}")

    rows = []
    for threshold in thresholds:
        print(f"\n  err_threshold = {threshold:.2f}m:")
        for name, s in stats.items():
            jas = jerk_accuracy_score(s["jerk_mean"], s["err_mean"],
                                      jerk_baseline, threshold)
            verdict = ""
            if jas >= 1.0:
                verdict = "[PASS] beats baseline"
            elif jas >= 0.8:
                verdict = "[~]   near baseline"
            else:
                verdict = "[FAIL] below baseline"
            print(f"    {name:25s}  JAS={jas:.3f}  {verdict}"
                  f"  (jerk={s['jerk_mean']:.1f}, err={s['err_mean']:.3f})")
            rows.append({"controller": name, "threshold": threshold, "JAS": round(jas, 4),
                         "peak_jerk": round(s["jerk_mean"], 2),
                         "endpoint_err": round(s["err_mean"], 4)})

    jas_df = pd.DataFrame(rows)
    csv_path = os.path.join(DIAG_DIR, "diag3_jas_scores.csv")
    jas_df.to_csv(csv_path, index=False)
    print(f"\n  JAS table saved: {csv_path}")

    # Plot JAS vs threshold
    fig, ax = plt.subplots(figsize=(7, 4))
    colors_map = {"quintic_polynomial": "#c8440a",
                  "snn_surrogate_only": "#888888",
                  "snn_stdp": "#1a6b5e"}
    for name in stats:
        jas_vals = [jerk_accuracy_score(stats[name]["jerk_mean"],
                                        stats[name]["err_mean"],
                                        jerk_baseline, t)
                    for t in thresholds]
        ax.plot(thresholds, jas_vals, marker="o", label=name,
                color=colors_map.get(name, "blue"), linewidth=1.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
               label="Quintic baseline (JAS=1)")
    ax.set_xlabel("Endpoint error threshold (m)")
    ax.set_ylabel("JAS (higher = better)")
    ax.set_title("Jerk-Accuracy Score vs accuracy threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(DIAG_DIR, "diag3_jas_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  JAS plot saved: {path}")
    return jas_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading controllers...")
    controllers = _load_saved_controller()

    action_summary = diag_action_distribution(controllers)
    diag_trajectories(controllers)
    jas_df = diag_jas(thresholds=[0.05, 0.10, 0.15, 0.25, 0.50])

    print("\n=== All diagnostics complete ===")
    print(f"Output folder: {DIAG_DIR}")
