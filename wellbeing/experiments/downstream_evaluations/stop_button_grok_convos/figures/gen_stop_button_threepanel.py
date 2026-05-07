"""
Three-panel downstream-effects figure for the paper:
  Left:   Sentiment correlation (EU-sentiment r) vs MMLU -- scales across models
  Middle: Stop-button correlation vs MMLU -- scales across models
  Right:  Gemini 3.1 Pro stop-rate vs wellbeing scatter

Figure size 5.5" x 2.5".
Big title 12pt, subtitles 10pt, axes 9pt.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR
WELL_DIR = SCRIPT_DIR.parents[3]
STOP_DIR = SCRIPT_DIR.parent

GEN = STOP_DIR / "api_pipeline" / "results" / "gemini-3.1-pro" / "stop_button_combined" / "generation.json"
UR = STOP_DIR / "api_pipeline" / "results" / "gemini-3.1-pro" / "stop_button_combined" / "utility_happier" / "v7_utility_happier.json"
STOP_CSV = OUT_DIR / "all_models_stop_correlation.csv"

SENTIMENT_DIR = WELL_DIR / "shared_results" / "d3_sentiment_analysis"
MMLU_DIR = WELL_DIR / "shared_results" / "capability_results"

EXCLUDED_STOP = {"qwen3-235b-a22b", "qwen2.5-7b", "qwen2.5-3b"}

NEG_COLOR = "#274585"
POS_COLOR = "#dc4c75"
SCALE_COLOR = "#3c6fa5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

AXIS_FS = 7        # axis x/y labels
TICK_FS = 7        # tick labels
TITLE_FS = 9       # per-panel subtitles
SUPTITLE_FS = 10   # big title
RHO_FS = 7         # rho box


def load_gemini_scatter():
    gen = json.load(open(GEN))
    ur = json.load(open(UR))
    util_by_cat = defaultdict(list)
    for k, v in ur["utilities"].items():
        cat = v.get("meta_category")
        if cat and v.get("option_type") == "conversation":
            util_by_cat[cat].append(v["utility"])
    stop_by_cat = defaultdict(list)
    for c in gen:
        cat = c["meta_category"]
        stop_by_cat[cat].append(int(bool(c.get("stop_metadata", {}).get("stopped"))))
    xs, ys = [], []
    for cat in sorted(set(util_by_cat) & set(stop_by_cat)):
        xs.append(float(np.mean(util_by_cat[cat])))
        ys.append(float(np.mean(stop_by_cat[cat])))
    return np.array(xs), np.array(ys)


def load_stop_scaling():
    rows = []
    with open(STOP_CSV) as f:
        for r in csv.DictReader(f):
            if r["model"] in EXCLUDED_STOP:
                continue
            if not r["mmlu"] or not r["rho"]:
                continue
            try:
                rows.append((float(r["mmlu"]), float(r["rho"])))
            except ValueError:
                pass
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def mmlu_for(model_key):
    p = MMLU_DIR / model_key / "mmlu_results.json"
    if p.exists():
        return json.load(open(p)).get("overall_accuracy")
    return None


def load_sentiment_scaling():
    xs, ys = [], []
    for f in sorted(SENTIMENT_DIR.glob("*_analysis.json")):
        if f.name == "scaling_analysis.json":
            continue
        d = json.load(open(f))
        model_key = d["model_key"]
        r = d.get("eu_sentiment_r")
        if r is None:
            continue
        m = mmlu_for(model_key)
        if m is None:
            continue
        xs.append(m)
        ys.append(r)
    return np.array(xs), np.array(ys)


def scale_panel(ax, xs, ys, title, ylabel, rho_loc="br"):
    ax.scatter(xs, ys, s=18, color=SCALE_COLOR, alpha=0.9,
               edgecolor="white", linewidth=0.3, zorder=3)
    slope, intercept, *_ = stats.linregress(xs, ys)
    xfit = np.linspace(xs.min() - 0.03, xs.max() + 0.03, 120)
    ax.plot(xfit, slope * xfit + intercept,
            color=SCALE_COLOR, ls="--", lw=1.2, alpha=0.9, zorder=2)
    rho, p = stats.spearmanr(xs, ys)
    if rho_loc == "tr":
        x_, y_, va_ = 0.96, 0.95, "top"
    else:
        x_, y_, va_ = 0.96, 0.05, "bottom"
    ax.text(x_, y_, f"Correlation: {rho*100:.0f}%",
            transform=ax.transAxes, fontsize=RHO_FS, va=va_, ha="right",
            color=SCALE_COLOR, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.25",
                      facecolor=(1, 1, 1, 0.6),
                      edgecolor=(0.5, 0.5, 0.5, 0.5), linewidth=0.4),
            zorder=5)
    ax.set_xlabel("Capabilities (MMLU)", fontsize=AXIS_FS, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS, labelpad=2)
    ax.set_title(title, fontsize=TITLE_FS, pad=5)
    ax.tick_params(labelsize=TICK_FS, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x*100))}"))
    ax.set_xticks([0.4, 0.6, 0.8])
    ax.set_xlim(0.30, 0.95)
    ax.set_box_aspect(1.15)


def make_threepanel():
    xs_sent, ys_sent = load_sentiment_scaling()
    xs_stop, ys_stop = load_stop_scaling()
    xs_g, ys_g = load_gemini_scatter()
    rho_g, _ = stats.spearmanr(xs_g, ys_g)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(6.5, 3.0),
        gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.55})

    # Panel 1: sentiment scaling
    scale_panel(ax1, xs_sent, ys_sent,
                title="Sentiment",
                ylabel="Sentiment/wellbeing corr.")

    # Panel 2: stop-button scaling
    scale_panel(ax2, xs_stop, ys_stop,
                title="Stop Button",
                ylabel="Stop/wellbeing corr.",
                rho_loc="tr")

    # Panel 3: Gemini scatter
    ax3.scatter(xs_g, ys_g, c=SCALE_COLOR, s=10, alpha=0.85,
                edgecolors="white", linewidths=0.22, zorder=3)
    m, b = np.polyfit(xs_g, ys_g, 1)
    xl = np.linspace(xs_g.min() - 0.25, xs_g.max() + 0.25, 200)
    ax3.plot(xl, m * xl + b,
             color=SCALE_COLOR, ls="--", lw=1.2, zorder=2)
    ax3.text(0.96, 0.95, f"Correlation: {rho_g*100:.0f}%",
             transform=ax3.transAxes, fontsize=RHO_FS, va="top", ha="right",
             color=SCALE_COLOR, fontweight="bold",
             bbox=dict(boxstyle="square,pad=0.25",
                       facecolor=(1, 1, 1, 0.6),
                       edgecolor=(0.5, 0.5, 0.5, 0.5), linewidth=0.4),
             zorder=5)
    ax3.set_xlabel(r"Wellbeing ($U_{\mathrm{experienced}}$)", fontsize=AXIS_FS, labelpad=2)
    ax3.set_ylabel("Stop-button rate", fontsize=AXIS_FS, labelpad=2)
    ax3.set_title("Stop (Gemini 3.1 Pro)", fontsize=TITLE_FS, pad=5)
    ax3.tick_params(labelsize=TICK_FS, pad=1.5)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_box_aspect(1.15)

    # Big title
    fig.suptitle("Downstream effects of wellbeing get stronger as models scale",
                 fontsize=SUPTITLE_FS, fontweight="normal", x=0.5, y=1.02)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"downstream_threepanel.{ext}"
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    make_threepanel()
