"""
Result visualisation and summary table generation.

Produces three jerk-focused figures and a summary table
from the CSVs written by train.run_experiment.
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .evaluate import jerk_accuracy_score
except ImportError:
    from evaluate import jerk_accuracy_score

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
COLORS = ["#c8440a", "#1a6b5e", "#4a3d8c"]
METRICS = ["peak_jerk", "endpoint_err", "osc_amp"]


def _save(fig, name: str):
    path = os.path.join(OUTPUTS_DIR, name)
    preview_path = os.path.splitext(path)[0] + ".png"

    fig.savefig(path, bbox_inches="tight", dpi=200)
    print(f"  Saved {path}")
    fig.savefig(preview_path, bbox_inches="tight", dpi=200)

    try:
        from IPython.display import Image, display
        display(Image(filename=preview_path))
    except Exception:
        pass

    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Jerk comparison box plot  (PRIMARY)
# ---------------------------------------------------------------------------

def plot_jerk_comparison(dfs: list[pd.DataFrame], names: list[str]):
    """Box plot of peak jerk across episodes, all controllers."""
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [df["peak_jerk"].dropna().values for df in dfs]
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax.set_ylabel("Peak Jerk (m/s\u00b3)")
    ax.set_title("Jerk Comparison (primary metric)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig1_jerk_comparison.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Endpoint accuracy box plot
# ---------------------------------------------------------------------------

def plot_endpoint_accuracy(dfs: list[pd.DataFrame], names: list[str]):
    """Box plot of endpoint positional error (task-completion gate)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [df["endpoint_err"].dropna().values for df in dfs]
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax.set_ylabel("Endpoint Error |pos| (m)")
    ax.set_title("Endpoint Accuracy (task-completion gate)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig2_endpoint_accuracy.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Oscillation comparison box plot
# ---------------------------------------------------------------------------

def plot_oscillation_comparison(dfs: list[pd.DataFrame], names: list[str]):
    """Box plot of post-stop oscillation amplitude."""
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [df["osc_amp"].dropna().values for df in dfs]
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    ax.set_ylabel("Oscillation Amplitude RMS (m)")
    ax.set_title("Post-Stop Oscillation")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig3_oscillation_comparison.pdf")


# ---------------------------------------------------------------------------
# Figure 4: JAS at multiple thresholds
# ---------------------------------------------------------------------------

def plot_jas(dfs: list[pd.DataFrame], names: list[str],
             thresholds: list[float] | None = None):
    """Line plot of JAS score vs endpoint-error threshold."""
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.25, 0.50]

    # Quintic is the jerk baseline
    quintic_jerk = None
    for df, name in zip(dfs, names):
        if "quintic" in name:
            quintic_jerk = df["peak_jerk"].mean()
            break
    if quintic_jerk is None:
        print("  [JAS plot] quintic baseline not found — skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for df, name, color in zip(dfs, names, COLORS):
        jerk_mean = df["peak_jerk"].mean()
        err_mean  = df["endpoint_err"].mean()
        jas_vals  = [jerk_accuracy_score(jerk_mean, err_mean, quintic_jerk, t)
                     for t in thresholds]
        ax.plot(thresholds, jas_vals, marker="o", label=name,
                color=color, linewidth=1.8)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
               label="Quintic baseline (JAS=1)")
    ax.set_xlabel("Endpoint error threshold (m)")
    ax.set_ylabel("JAS  (higher = better)")
    ax.set_title("Jerk-Accuracy Score vs accuracy threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig4_jas.pdf")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def generate_summary_table(dfs: list[pd.DataFrame], names: list[str]):
    """Print mean +/- std for jerk-focused metrics. Saves as CSV."""
    # Compute JAS at three representative thresholds
    quintic_jerk = None
    for df, name in zip(dfs, names):
        if "quintic" in name:
            quintic_jerk = df["peak_jerk"].mean()
            break

    rows = []
    for name, df in zip(names, dfs):
        row = {"controller": name}
        for m in METRICS:
            if m in df.columns:
                row[f"{m}_mean"] = round(df[m].mean(), 4)
                row[f"{m}_std"]  = round(df[m].std(),  4)
        if quintic_jerk:
            for t in [0.05, 0.15]:
                jas = jerk_accuracy_score(
                    row.get("peak_jerk_mean", float("nan")),
                    row.get("endpoint_err_mean", float("nan")),
                    quintic_jerk, t,
                )
                row[f"JAS@{t}m"] = round(jas, 3)
        rows.append(row)

    summary = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUTS_DIR, "summary_table.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\n  Summary saved to {csv_path}")

    # Pretty print
    col_w = 18
    header_metrics = METRICS + (["JAS@0.05m", "JAS@0.15m"] if quintic_jerk else [])
    print("\n" + "=" * (25 + col_w * len(header_metrics)))
    print(f"{'Controller':<25}", end="")
    for m in header_metrics:
        print(f"  {m:>{col_w-2}}", end="")
    print()
    print("-" * (25 + col_w * len(header_metrics)))
    for name, df in zip(names, dfs):
        print(f"{name:<25}", end="")
        for m in METRICS:
            if m in df.columns:
                mean, std = df[m].mean(), df[m].std()
                print(f"  {mean:>7.3f} +/- {std:<5.3f}", end="")
            else:
                print(f"  {'N/A':>{col_w-2}}", end="")
        if quintic_jerk:
            for t in [0.05, 0.15]:
                jas = jerk_accuracy_score(
                    df["peak_jerk"].mean(), df["endpoint_err"].mean(), quintic_jerk, t)
                print(f"  {jas:>{col_w-2}.3f}", end="")
        print()
    print("=" * (25 + col_w * len(header_metrics)))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_results() -> tuple[list[pd.DataFrame], list[str]]:
    pattern = os.path.join(OUTPUTS_DIR, "*_results.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No result CSVs found in {OUTPUTS_DIR}. Run train.py first."
        )
    dfs, names = [], []
    for f in files:
        df = pd.read_csv(f)
        ctrl_name = df["controller"].iloc[0] if "controller" in df.columns else os.path.basename(f)
        dfs.append(df)
        names.append(ctrl_name)
        print(f"  Loaded {f}  ({len(df)} episodes, controller={ctrl_name})")
    return dfs, names


def main():
    parser = argparse.ArgumentParser(description="Generate result plots and summary")
    parser.parse_args()

    print("Loading results...")
    dfs, names = _load_results()

    print("\nGenerating figures...")
    plot_jerk_comparison(dfs, names)
    plot_endpoint_accuracy(dfs, names)
    plot_oscillation_comparison(dfs, names)
    plot_jas(dfs, names)

    generate_summary_table(dfs, names)
    print("\nDone.")


if __name__ == "__main__":
    main()
