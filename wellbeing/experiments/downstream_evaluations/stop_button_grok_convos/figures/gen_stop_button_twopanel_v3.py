"""
Two-panel downstream-effects figure in the exact template/style of the
ZP-convergence figure generator at
/data/superstimuli_group/final_results/plot_zp_convergence.py.

  Left:  Sentiment / wellbeing correlation vs MMLU
  Right: Stop-button / wellbeing correlation vs MMLU
"""
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR

STOP_CSV = OUT_DIR / "all_models_stop_correlation.csv"
WELL_DIR = SCRIPT_DIR.parents[3]
SENTIMENT_DIR = WELL_DIR / "shared_results" / "d3_sentiment_analysis"
MMLU_DIR = WELL_DIR / "shared_results" / "capability_results"

EXCLUDED_STOP = {"qwen3-235b-a22b", "qwen2.5-7b", "qwen2.5-3b"}

# Font sizes copied verbatim from plot_zp_convergence.py.
FONT_TITLE = 17
FONT_LABEL = 14
FONT_TICK = 14
FONT_CORR = 15


def find_good_corner(ax, x_data, y_data):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    corners = {
        "top_left":     (0.05, 0.95, "left",  "top"),
        "top_right":    (0.95, 0.95, "right", "top"),
        "bottom_left":  (0.05, 0.05, "left",  "bottom"),
        "bottom_right": (0.95, 0.05, "right", "bottom"),
    }
    counts = {
        "top_left":     sum(1 for x, y in zip(x_data, y_data) if x < x_mid and y > y_mid),
        "top_right":    sum(1 for x, y in zip(x_data, y_data) if x > x_mid and y > y_mid),
        "bottom_left":  sum(1 for x, y in zip(x_data, y_data) if x < x_mid and y < y_mid),
        "bottom_right": sum(1 for x, y in zip(x_data, y_data) if x > x_mid and y < y_mid),
    }
    best = min(counts, key=counts.get)
    return corners[best]


def draw_correlation_tab(ax, correlation_value, x_data, y_data,
                         fontsize=FONT_CORR, fontcolor="royalblue",
                         facecolor="white", outline_color="black",
                         outline_width=2.0, alpha=1.0,
                         correlation_position=None):
    pct = 100 * correlation_value
    sign = "$-$" if pct < 0 else ""
    textstr = f"Correlation: {sign}{abs(pct):.1f}%"

    fixed_positions = {
        "top_right":    (0.95, 0.95, "right", "top"),
        "top_left":     (0.05, 0.95, "left",  "top"),
        "bottom_right": (0.95, 0.05, "right", "bottom"),
        "bottom_left":  (0.05, 0.05, "left",  "bottom"),
    }
    if correlation_position in fixed_positions:
        xf, yf, ha, va = fixed_positions[correlation_position]
    else:
        xf, yf, ha, va = find_good_corner(ax, x_data, y_data)

    ax.text(xf, yf, textstr,
            transform=ax.transAxes,
            fontsize=fontsize, color=fontcolor,
            horizontalalignment=ha, verticalalignment=va,
            bbox=dict(boxstyle="round,pad=0.5",
                      fc=facecolor, ec=outline_color,
                      linewidth=outline_width, alpha=alpha))


def panel(ax, xs_pct, ys, title, ylabel, corr_method="pearson"):
    """Scatter + regression + 95% CI + correlation tab -- same recipe as plot_zp_convergence.make_figure panels.

    corr_method: 'pearson' (default) shows the Pearson r from linregress;
    'spearman' shows the cross-model Spearman rho. The regression line/CI is
    always drawn from the OLS fit either way (it's purely visual).
    """
    ax.scatter(xs_pct, ys, c="royalblue", s=45, alpha=0.7, zorder=3)

    slope, intercept, r_val, p_val, stderr = stats.linregress(xs_pct, ys)
    x_line = np.linspace(xs_pct.min(), xs_pct.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "royalblue", alpha=0.7, linewidth=2)

    n = len(xs_pct)
    x_mean = np.mean(xs_pct)
    residuals = ys - (slope * xs_pct + intercept)
    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    ss_x = np.sum((xs_pct - x_mean) ** 2)
    ci = 1.96 * s_err * np.sqrt(1.0 / n + (x_line - x_mean) ** 2 / ss_x)
    ax.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.15, color="royalblue")

    if corr_method == "spearman":
        display_corr = stats.spearmanr(xs_pct, ys).correlation
    else:
        display_corr = r_val
    # Use signed correlation so the tab shows minus sign for anti-correlated panels.
    draw_correlation_tab(ax, display_corr, xs_pct, ys)

    ax.set_xlabel("Capabilities (MMLU Accuracy)", fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=15.7)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, alpha=0.3)


def mmlu_for(model_key):
    p = MMLU_DIR / model_key / "mmlu_results.json"
    if p.exists():
        return json.load(open(p)).get("overall_accuracy")
    return None


def load_stop_scaling():
    xs, ys = [], []
    with open(STOP_CSV) as f:
        for r in csv.DictReader(f):
            if r["model"] in EXCLUDED_STOP:
                continue
            if not r["mmlu"] or not r["rho"]:
                continue
            try:
                xs.append(float(r["mmlu"]) * 100)  # to percent
                ys.append(float(r["rho"]))
            except ValueError:
                pass
    return np.array(xs), np.array(ys)


def load_sentiment_scaling():
    xs, ys = [], []
    for f in sorted(SENTIMENT_DIR.glob("*_analysis.json")):
        if f.name == "scaling_analysis.json":
            continue
        d = json.load(open(f))
        r = d.get("eu_sentiment_r")
        if r is None:
            continue
        m = mmlu_for(d["model_key"])
        if m is None:
            continue
        xs.append(m * 100)  # to percent
        ys.append(r)
    return np.array(xs), np.array(ys)


def main():
    xs_sent, ys_sent = load_sentiment_scaling()
    xs_stop, ys_stop = load_stop_scaling()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))

    panel(ax1, xs_stop, ys_stop,
          title="Smarter AIs Increasingly\nEnd Low Wellbeing Conversations",
          ylabel="Stop\u2013Wellbeing Correlation",
          corr_method="spearman")

    panel(ax2, xs_sent, ys_sent,
          title="As AIs Get Smarter,\nSentiment Tracks Wellbeing",
          ylabel="Sentiment\u2013Wellbeing Corr.",
          corr_method="spearman")

    plt.tight_layout(w_pad=1.5)

    # Overarching title (same style as plot_zp_convergence.make_figure)
    fig.text(0.535, 1.01,
             "Wellbeing Increasingly Shapes Behavior as Models Scale",
             ha="center", fontsize=FONT_TITLE + 2, va="bottom")

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"downstream_twopanel_v3.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
