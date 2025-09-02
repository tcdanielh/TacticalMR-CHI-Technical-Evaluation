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
from scipy import stats

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

def save_before_after_mean_bars(before_vals, after_vals, title, ylabel, filename, out_dir, colors=(PALE_RED, PALE_BLUE)):
    """
    Save a simple bar chart comparing the mean Before vs After values with 95% CI error bars.
    Inputs are numeric-like series (percentages or likert points). Values are coerced to numeric.
    """
    b = pd.to_numeric(before_vals, errors='coerce').dropna()
    a = pd.to_numeric(after_vals, errors='coerce').dropna()

    # compute means and 95% CI using t ~ 1.96 (approx for decent N)
    def mean_ci(x):
        if len(x) == 0:
            return np.nan, np.nan
        m = float(x.mean())
        se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
        ci95 = 1.96 * se
        return m, ci95

    mean_b, ci_b = mean_ci(b)
    mean_a, ci_a = mean_ci(a)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = np.arange(2)
    means = [mean_b, mean_a]
    cis = [ci_b, ci_a]
    palette = [colors[0], colors[1]]

    ax.bar(x, means, yerr=cis, capsize=6, color=palette, edgecolor="black", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Before (N={len(b)})", f"After (N={len(a)})"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)

    # Dynamic y-limit to prevent labels from going outside the frame
    finite_candidates = []
    for val in [mean_b + (ci_b if not np.isnan(ci_b) else 0), mean_a + (ci_a if not np.isnan(ci_a) else 0)]:
        if np.isfinite(val):
            finite_candidates.append(val)
    if len(b) > 0 and np.isfinite(b.max()):
        finite_candidates.append(float(b.max()))
    if len(a) > 0 and np.isfinite(a.max()):
        finite_candidates.append(float(a.max()))
    y_top = max(finite_candidates) if finite_candidates else max([v for v in means if np.isfinite(v)] + [1.0])
    y_max = y_top * 1.10 + (0.02 * y_top)
    y_min = 0.0
    ax.set_ylim(y_min, y_max)

    # Add value labels near top of bars but clipped inside the axes
    offset = 0.02 * y_max
    margin = 0.02 * y_max
    for i, v in enumerate(means):
        if not np.isnan(v):
            this_ci = cis[i] if (i < len(cis) and np.isfinite(cis[i])) else 0.0
            desired = v + this_ci + offset
            if desired >= (y_max - margin):
                label_y = y_max - margin
                va = 'top'
            else:
                label_y = desired
                va = 'bottom'
            ax.text(i, label_y, f"{v:.1f}", ha='center', va=va, fontsize=11)

    fig.tight_layout()
    fp = out_dir / filename
    fig.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return str(fp)

def save_scatter_with_trendline(x_vals, y_vals, title, xlabel, ylabel, filename, out_dir, point_color="#4a90e2"):
    """
    Scatterplot with linear trendline (least squares). Drops NaNs, adds light jitter for binary X.
    """
    x = pd.to_numeric(x_vals, errors='coerce')
    y = pd.to_numeric(y_vals, errors='coerce')
    s = pd.concat([x, y], axis=1).dropna()
    if s.shape[0] == 0:
        return None

    x_clean = s.iloc[:,0].to_numpy(dtype=float)
    y_clean = s.iloc[:,1].to_numpy(dtype=float)

    # Light jitter on X if it is binary to improve visibility
    unique_x = np.unique(x_clean)
    if set(np.round(unique_x).tolist()) <= {0,1} and len(unique_x) <= 3:
        jitter = np.random.normal(0, 0.03, size=len(x_clean))
        x_plot = x_clean + jitter
    else:
        x_plot = x_clean

    # Fit linear regression (ignore singular cases)
    if len(x_clean) >= 2 and np.std(x_clean) > 0:
        slope, intercept = np.polyfit(x_clean, y_clean, 1)
        trend_x = np.array([x_clean.min() - 0.05, x_clean.max() + 0.05])
        trend_y = slope * trend_x + intercept
    else:
        slope = intercept = None

    # Correlations
    try:
        pr = stats.pearsonr(x_clean, y_clean)
        pearson_r = float(pr.statistic if hasattr(pr, 'statistic') else pr[0])
        pearson_p = float(pr.pvalue if hasattr(pr, 'pvalue') else pr[1])
    except Exception:
        pearson_r = np.nan
        pearson_p = np.nan

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.scatter(x_plot, y_clean, s=55, color=point_color, edgecolors="black", linewidth=0.6, alpha=0.9)
    if slope is not None:
        ax.plot(trend_x, trend_y, color=MEAN_LINE, linewidth=2.2, label="Linear fit")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    # If binary X, show tick labels as No/Yes
    if set(np.round(unique_x).tolist()) <= {0,1}:
        ax.set_xticks([0,1])
        ax.set_xticklabels(["No", "Yes"])

    # Annotate N and Pearson
    ax.text(0.02, 0.98, f"N = {len(x_clean)}\nPearson r = {pearson_r:.2f}\np = {pearson_p:.3f}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.9), fontsize=11)

    if slope is not None:
        ax.legend(loc="lower right")

    fig.tight_layout()
    fp = out_dir / filename
    fig.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return str(fp)

def find_col_by_prefix_and_phrase(columns, prefix, phrase):
    """Return the first column whose stripped name starts with prefix and contains phrase (case sensitive)."""
    for c in columns:
        cs = str(c).strip()
        if cs.startswith(prefix) and phrase in cs:
            return c
    return None

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

    # New: Before vs After bar charts (means with 95% CI)
    save_before_after_mean_bars(
        ad_bf, ad_af,
        title="Correctness: Before vs After (mean ± 95% CI)",
        ylabel="Percentage",
        filename="correctness_before_after_bar.png",
        out_dir=out_dir,
        colors=(PALE_RED, PALE_BLUE)
    )

    save_before_after_mean_bars(
        en_bf, en_af,
        title="Completeness: Before vs After (mean ± 95% CI)",
        ylabel="Percentage",
        filename="completeness_before_after_bar.png",
        out_dir=out_dir,
        colors=(PALE_RED, PALE_BLUE)
    )

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

    # Comparative Likert: Before vs After mean bars (±95% CI)
    # Try exact; if missing, use robust finder
    q1_bf_col = "BF The avatar’s decision flow accurately learned from my coaching as I intended."
    q1_af_col = "AF The avatar’s decision flow accurately learned from my coaching as I intended."
    q2_bf_col = "BF Considering the avatar’s decision flow and the demonstrations, the avatar’s learning is good enough to teach someone else."
    q2_af_col = "AF Considering the avatar’s decision flow and the demonstrations, the avatar’s learning is good enough to teach someone else."

    if q1_bf_col not in df.columns or q1_af_col not in df.columns:
        q1_bf_col = find_col_by_prefix_and_phrase(df.columns, "BF ", "decision flow accurately learned") or q1_bf_col
        q1_af_col = find_col_by_prefix_and_phrase(df.columns, "AF ", "decision flow accurately learned") or q1_af_col
    if q2_bf_col not in df.columns or q2_af_col not in df.columns:
        q2_bf_col = find_col_by_prefix_and_phrase(df.columns, "BF ", "learning is good enough to teach") or q2_bf_col
        q2_af_col = find_col_by_prefix_and_phrase(df.columns, "AF ", "learning is good enough to teach") or q2_af_col

    if q1_bf_col in df.columns and q1_af_col in df.columns:
        save_before_after_mean_bars(
            df[q1_bf_col], df[q1_af_col],
            title="Comparative Likert: Decision flow learned — Before vs After (mean ± 95% CI)",
            ylabel="Likert (1–7)",
            filename="comparative_likert_q1_before_after_bar.png",
            out_dir=out_dir,
            colors=(PALE_RED, PALE_BLUE)
        )

    if q2_bf_col in df.columns and q2_af_col in df.columns:
        save_before_after_mean_bars(
            df[q2_bf_col], df[q2_af_col],
            title="Comparative Likert: Learning good enough to teach — Before vs After (mean ± 95% CI)",
            ylabel="Likert (1–7)",
            filename="comparative_likert_q2_before_after_bar.png",
            out_dir=out_dir,
            colors=(PALE_RED, PALE_BLUE)
        )

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

    # 4) Correlations with Coaching Experience (binary yes/no -> 1/0)
    # Use After Feedback percentages by default
    if "Coaching Experience" in df.columns:
        coach_raw = df["Coaching Experience"].astype(str).str.strip().str.lower()
        coach = coach_raw.map({"yes": 1, "no": 0}).replace({"na": np.nan})

        corr_rows = []
        def safe_corr(x, y, label):
            s = pd.concat([x, y], axis=1).dropna()
            if s.shape[0] < 3:
                return {
                    "Metric": label,
                    "N": int(s.shape[0]),
                    "Pearson_r": np.nan,
                    "Pearson_p": np.nan,
                    "Spearman_rho": np.nan,
                    "Spearman_p": np.nan,
                }
            r_p = stats.pearsonr(s.iloc[:,0], s.iloc[:,1])
            rho_p = stats.spearmanr(s.iloc[:,0], s.iloc[:,1])
            return {
                "Metric": label,
                "N": int(s.shape[0]),
                "Pearson_r": float(r_p.statistic if hasattr(r_p, 'statistic') else r_p[0]),
                "Pearson_p": float(r_p.pvalue if hasattr(r_p, 'pvalue') else r_p[1]),
                "Spearman_rho": float(rho_p.statistic if hasattr(rho_p, 'statistic') else rho_p[0]),
                "Spearman_p": float(rho_p.pvalue if hasattr(rho_p, 'pvalue') else rho_p[1]),
            }

        corr_rows.append(safe_corr(coach, ad_af, "Coaching Experience vs Correctness % (AF)"))
        corr_rows.append(safe_corr(coach, en_af, "Coaching Experience vs Completeness % (AF)"))
        pd.DataFrame(corr_rows).to_csv(out_dir / "correlations_summary.csv", index=False)

        # Scatter plots with trendlines
        save_scatter_with_trendline(
            coach, ad_af,
            title="Coaching Experience vs Correctness (AF)",
            xlabel="Coaching Experience",
            ylabel="Correctness % (After)",
            filename="coaching_vs_correctness_scatter.png",
            out_dir=out_dir,
            point_color=PALE_RED
        )
        save_scatter_with_trendline(
            coach, en_af,
            title="Coaching Experience vs Completeness (AF)",
            xlabel="Coaching Experience",
            ylabel="Completeness % (After)",
            filename="coaching_vs_completeness_scatter.png",
            out_dir=out_dir,
            point_color=PALE_BLUE
        )
    else:
        print("Warning: 'Coaching Experience' column not found; skipping correlation analysis.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_study_figures.py <path_to_csv>")
        sys.exit(1)
    main(sys.argv[1])
