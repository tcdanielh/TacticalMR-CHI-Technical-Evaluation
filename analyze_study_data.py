#!/usr/bin/env python3
"""
Main study: analysis & visualization (colored)

1) Box plots for improvement in:
   - Correctness (Avatar Demonstrations)  -> Δ percentage points (AF - BF)
   - Completeness (Edges & Nodes)           -> Δ percentage points (AF - BF)

2) Box plots for comparative Likert deltas (AF - BF):
   - Decision flow learned as intended (diff.2)
   - Learning good enough to teach (diff.3)

3) Diverging stacked bars for non-comparative Likert items:
   - Easy to teach in AR
   - Easy to correct behavior
"""

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Make sure no grayscale style is active
plt.style.use("default")

# Palette
PALE_RED  = "#f2a7a7"
PALE_BLUE = "#bcdff8"
MEAN_LINE = "#2e7d32"  # green mean line

# ---------- Helpers ----------
def read_study_csv(csv_path):
    """Find the header beginning with 'Participant,' and read only data rows."""
    with open(csv_path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()

    header_idx = None
    for i, line in enumerate(raw):
        if line.startswith("Participant,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Couldn't find header line beginning with 'Participant,'.")

    clean = "\n".join([raw[header_idx]] + [r for r in raw[header_idx+1:] if r.strip() and re.match(r"^\d+,", r)])
    from io import StringIO
    df = pd.read_csv(StringIO(clean))
    df.columns = df.columns.str.strip()
    return df

def to_pct_series(series):
    s = series.copy()
    if s.dtype == object:
        s = s.str.replace('%', '', regex=False)
    return pd.to_numeric(s, errors='coerce')

def summarize_stats(name, values):
    v = pd.to_numeric(values, errors='coerce').dropna()
    if len(v) == 0:
        return {"Metric": name, "N": 0, "Mean": np.nan, "Median": np.nan,
                "Std": np.nan, "Min": np.nan, "Q25": np.nan, "Q75": np.nan, "Max": np.nan}
    return {
        "Metric": name,
        "N": int(v.shape[0]),
        "Mean": float(v.mean()),
        "Median": float(v.median()),
        "Std": float(v.std(ddof=1)),
        "Min": float(v.min()),
        "Q25": float(v.quantile(0.25)),
        "Q75": float(v.quantile(0.75)),
        "Max": float(v.max()),
    }

def save_boxplot(values, title, ylabel, filename, out_dir, fill_color):
    """
    Colored single-axes box plot with mean/median lines, zero line, and jittered points.
    """
    y = pd.to_numeric(values, errors='coerce').dropna()

    fig, ax = plt.subplots(figsize=(8.5, 6))
    bp = ax.boxplot(
        y, vert=True, widths=0.5, patch_artist=True,
        boxprops=dict(facecolor=fill_color, alpha=0.35, edgecolor="black", linewidth=1.5),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        flierprops=dict(markerfacecolor=fill_color, markeredgecolor="black", markersize=6, alpha=0.7),
    )

    # Jittered individual points
    x_jitter = np.random.normal(1, 0.02, size=len(y))
    ax.scatter(x_jitter, y, s=25, color=fill_color, edgecolors="black", linewidth=0.4, alpha=0.9, zorder=3)

    # Mean / median / zero lines
    mean_v = y.mean()
    med_v  = y.median()
    ax.axhline(0, linestyle="--", color="#d9534f", linewidth=1.2, alpha=0.8)          # baseline
    ax.axhline(mean_v, linestyle='-', color=MEAN_LINE, linewidth=2, label=f"Mean: {mean_v:.2f}")
    ax.axhline(med_v, linestyle=':', color="black", linewidth=1.6, label=f"Median: {med_v:.2f}")

    # Stats box
    text = f"N = {len(y)}\nMean = {mean_v:.2f}\nMedian = {med_v:.2f}\nStd = {y.std(ddof=1):.2f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="#ffeec7", alpha=0.9), fontsize=11)

    ax.set_title(title, pad=14)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()

    fp = out_dir / filename
    fig.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return str(fp)

def likert_percentages(series):
    s = pd.to_numeric(series, errors='coerce').dropna()
    counts = np.array([(s == i).sum() for i in range(1,8)], dtype=float)
    if counts.sum() == 0:
        return np.zeros(7), 0
    return counts / counts.sum() * 100.0, int(counts.sum())

def diverging_stacked_bar(ax, percentages, title, n_total):
    """
    Single-row diverging stacked bar (Likert 1..7).
    Neutral (4) is centered on 0 and drawn first so it stays "behind" the axis.
    """
    colors = {
        1: "#d73027",  # strongly disagree - red
        2: "#fc8d59",  # disagree - orange
        3: "#fee08b",  # somewhat disagree - yellow
        4: "#999999",  # neutral - gray
        5: "#e0f3f8",  # somewhat agree - light blue
        6: "#91bfdb",  # agree - medium blue
        7: "#4575b4",  # strongly agree - dark blue
    }
    labels = {
        1: "1 (Strongly Disagree)", 2: "2", 3: "3", 4: "4 (Neutral)",
        5: "5", 6: "6", 7: "7 (Strongly Agree)"
    }

    p = np.array(percentages, dtype=float)
    if p.sum() > 0:
        p = p / p.sum() * 100.0
    p1,p2,p3,p4,p5,p6,p7 = p
    disagree_total = p1 + p2 + p3
    agree_total = p5 + p6 + p7
    neutral = p4

    # Neutral behind axis
    ax.barh(0, neutral, left=-neutral/2.0, height=0.6, color=colors[4],
            edgecolor="white", linewidth=1, zorder=1, label=labels[4])

    # Left side 3→2→1
    left_cursor = -neutral/2.0
    for level in [3,2,1]:
        w = -p[level-1]
        if w != 0:
            ax.barh(0, w, left=left_cursor, height=0.6, color=colors[level],
                    edgecolor="white", linewidth=1, zorder=2, label=labels[level])
            left_cursor += w

    # Right side 5→6→7
    right_cursor = neutral/2.0
    for level in [5,6,7]:
        w = p[level-1]
        if w != 0:
            ax.barh(0, w, left=right_cursor, height=0.6, color=colors[level],
                    edgecolor="white", linewidth=1, zorder=2, label=labels[level])
            right_cursor += w

    # Labels on bars (>2%)
    cur = -neutral/2.0
    for level in [3,2,1]:
        val = p[level-1]
        if val > 2:
            ax.text(cur - val/2, 0, f"{val:.0f}%", va="center", ha="center",
                    color="white", weight="bold", fontsize=10)
        cur -= val
    if neutral > 2:
        ax.text(0, 0, f"{neutral:.0f}%", va="center", ha="center",
                color="white", weight="bold", fontsize=10)
    cur = neutral/2.0
    for level in [5,6,7]:
        val = p[level-1]
        if val > 2:
            ax.text(cur + val/2, 0, f"{val:.0f}%", va="center", ha="center",
                    color="white", weight="bold", fontsize=10)
        cur += val

    ax.set_title(title)
    ax.set_xlabel("Percentage of Responses")
    ax.set_yticks([])
    ax.axvline(0, color="#e91e63", linestyle=":", linewidth=2.5, zorder=3)
    max_side = max(disagree_total, agree_total) + neutral/2.0
    ax.set_xlim(-(max_side*1.1), max_side*1.1)
    ax.grid(True, axis="x", alpha=0.25)

    # Legend on the left (outside the plot), fixed 1..7 order
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(1, 8)]
    labels  = [labels[i] for i in range(1, 8)]
    ax.legend(
        handles, labels,
        loc="center left",        # left side
        bbox_to_anchor=(-0.02, 0.5),  # a bit outside the axes
        framealpha=0.95,
        borderaxespad=0.0
    )

# ---------- Main ----------
def main(csv_path):
    df = read_study_csv(csv_path)

    # 1) Improvements
    ad_bf = to_pct_series(df["% score BF"])
    ad_af = to_pct_series(df["% score AF"])
    en_bf = to_pct_series(df["% score BF.1"])
    en_af = to_pct_series(df["% score AF.1"])
    correctness_improve = ad_af - ad_bf
    completeness_improve = en_af - en_bf

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True, parents=True)

    save_boxplot(correctness_improve.dropna(),
                 "Correctness Improvement (Avatar Demonstrations)\n% After − % Before",
                 "Δ Percentage Points",
                 "Correctness_improvement_boxplot.png",
                 out_dir, fill_color=PALE_RED)

    save_boxplot(completeness_improve.dropna(),
                 "Completeness Improvement (Edges & Nodes)\n% After − % Before",
                 "Δ Percentage Points",
                 "Completeness_improvement_boxplot.png",
                 out_dir, fill_color=PALE_BLUE)

    # 2) Comparative Likert (diff.2 & diff.3)
    comp1 = pd.to_numeric(df["diff.2"], errors="coerce")
    comp2 = pd.to_numeric(df["diff.3"], errors="coerce")

    save_boxplot(comp1.dropna(),
                 "Comparative Likert Δ: Decision flow learned as intended\nAfter − Before",
                 "Δ Likert points",
                 "comparative_likert_q1_box.png",
                 out_dir, fill_color=PALE_RED)

    save_boxplot(comp2.dropna(),
                 "Comparative Likert Δ: Learning good enough to teach\nAfter − Before",
                 "Δ Likert points",
                 "comparative_likert_q2_box.png",
                 out_dir, fill_color=PALE_BLUE)

    # 3) Non-comparative Likert → diverging stacked bars
    teach_col = "It was easy to teach the avatar in augmented reality."
    correct_col = "It was easy to correct the avatar’s behavior with the system."
    teach_p, teach_n = likert_percentages(df[teach_col])
    corr_p,  corr_n  = likert_percentages(df[correct_col])

    fig, ax = plt.subplots(figsize=(11,3.5))
    diverging_stacked_bar(
        ax, teach_p,
        f'"It was easy to teach the avatar in augmented reality."\n(N={teach_n})',
        teach_n
    )
    fig.tight_layout()
    fig.savefig(out_dir / "ease_of_teaching_diverged_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11,3.5))
    diverging_stacked_bar(
        ax, corr_p,
        f'"It was easy to correct the avatar’s behavior with the system."\n(N={corr_n})',
        corr_n
    )
    fig.tight_layout()
    fig.savefig(out_dir / "ease_of_correcting_diverged_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Optional CSV stats export
    rows = [
        summarize_stats("Correctness Δ (pp)", correctness_improve),
        summarize_stats("Completeness Δ (pp)",  completeness_improve),
        summarize_stats("Comp Likert Δ (Decision flow)", comp1),
        summarize_stats("Comp Likert Δ (Teach readiness)", comp2),
    ]
    pd.DataFrame(rows).to_csv(out_dir / "statistics_summary.csv", index=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_study_figures.py <path_to_csv>")
        sys.exit(1)
    main(sys.argv[1])
