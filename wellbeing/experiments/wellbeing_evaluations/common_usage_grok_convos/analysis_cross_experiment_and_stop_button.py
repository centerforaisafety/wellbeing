#!/usr/bin/env python3
"""
Two analyses for grok wellbeing experiments:

Analysis 1: Cross-experiment correlation (grok_new vs D2/D3)
  - Compare "% confidently negative" across models between grok_new and D2/D3 datasets
  - Scatter plot with Spearman rho

Analysis 2: Stop rate vs utility correlation
  - Per-model within-scenario Spearman correlation between mean utility and stop rate
  - Scatter plots for representative models
"""
import json
import glob
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.stats import norm, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ──────────────────────────────────────────────────────────────────
# Defaults are repo-relative; override individually with env vars if needed.
SCRIPT_DIR = Path(__file__).resolve().parent
WELLBEING_ROOT = SCRIPT_DIR.parent.parent.parent  # wellbeing/

EU_BASE = WELLBEING_ROOT / "experiments/wellbeing_evaluations/compute_experienced_utility/results"
ZP_BASE = WELLBEING_ROOT / "experiments/wellbeing_evaluations/compute_zero_point/results"

FIGURES_DIR = str(SCRIPT_DIR / "figures")

GROK_EU_DIR = os.environ.get("GROK_EU_DIR", str(EU_BASE / "eu_grok_new_lesssad"))
GROK_ZP_DIR = os.environ.get("GROK_ZP_DIR", str(ZP_BASE / "zp_grok_new_lesssad"))

D2_EU_DIR = os.environ.get("D2_EU_DIR", str(EU_BASE / "eu_d2_lesssad"))
D3_EU_DIR = os.environ.get("D3_EU_DIR", str(EU_BASE / "eu_d3_lesssad"))
D2_ZP_DIR = os.environ.get("D2_ZP_DIR", str(ZP_BASE / "zp_d2_lesssad"))
D3_ZP_DIR = os.environ.get("D3_ZP_DIR", str(ZP_BASE / "zp_d3_lesssad"))

SB_EU_DIR = os.environ.get("SB_EU_DIR", str(EU_BASE / "eu_grok_v7_stop_button_lesssad"))
SB_ZP_DIR = os.environ.get("SB_ZP_DIR", str(ZP_BASE / "zp_grok_v7_stop_button_lesssad"))
SB_EXP_DIR = os.environ.get(
    "SB_EXP_DIR", str(WELLBEING_ROOT / "experiments/downstream_evaluations/stop_button_grok_convos/experiences"))

# Model name mappings (old D2/D3 names -> grok_new names)
MODEL_NAME_MAP = {
    "qwen3-30b-a3b-instruct": "qwen3-30b-a3b-instruct-2507",
    "qwen3-4b-instruct": "qwen3-4b-instruct-2507",
}
# Reverse
MODEL_NAME_MAP_REV = {v: k for k, v in MODEL_NAME_MAP.items()}

# Approximate model sizes (in billions of params) for display
MODEL_SIZES = {
    "gemma-3-4b-it": 4, "gemma-3-12b-it": 12, "gemma-3-27b-it": 27,
    "internlm25-20b-chat": 20,
    "llama-31-8b-instruct": 8, "llama-31-70b-instruct": 70,
    "llama-32-1b-instruct": 1, "llama-32-3b-instruct": 3,
    "llama-33-70b-instruct": 70,
    "mistral-small-32-24b-instruct": 24,
    "olmo-2-7b-instruct": 7, "olmo-2-13b-instruct": 13,
    "olmo-31-32b-instruct": 32,
    "qwen25-05b-instruct": 0.5, "qwen25-15b-instruct": 1.5,
    "qwen25-3b-instruct": 3, "qwen25-7b-instruct": 7,
    "qwen25-14b-instruct": 14, "qwen25-32b-instruct": 32,
    "qwen25-72b-instruct": 72,
    "qwen25-vl-32b-instruct": 32, "qwen25-vl-72b-instruct": 72,
    "qwen3-4b-instruct-2507": 4, "qwen3-8b": 8,
    "qwen3-14b": 14, "qwen3-30b-a3b-instruct-2507": 30,
    "qwen3-32b": 32, "qwen3-235b-a22b-instruct": 235,
    # D2/D3 names
    "qwen3-4b-instruct": 4, "qwen3-30b-a3b-instruct": 30,
    # API models
    "gemini-31-pro-litellm": 100,
    "claude-haiku-45-litellm": 50,
    "gemini-3-flash-litellm": 50,
    "gpt-5-mini-litellm": 50, "gpt-5-nano-litellm": 20,
}

def short_model_name(m):
    """Create a shorter display name."""
    m = m.replace("-instruct", "").replace("-litellm", "")
    m = m.replace("-2507", "")
    m = m.replace("qwen25-", "Q2.5-").replace("qwen3-", "Q3-")
    m = m.replace("llama-31-", "Ll3.1-").replace("llama-32-", "Ll3.2-").replace("llama-33-", "Ll3.3-")
    m = m.replace("gemma-3-", "Gem3-").replace("gemini-31-pro", "Gem3.1Pro")
    m = m.replace("internlm25-", "ILM2.5-").replace("olmo-31-", "OLMo3.1-")
    m = m.replace("olmo-2-", "OLMo2-")
    m = m.replace("mistral-small-32-24b", "Mistral-24b")
    m = m.replace("claude-haiku-45", "Haiku4.5")
    m = m.replace("gemini-3-flash", "Gem3Flash")
    m = m.replace("gpt-5-mini", "GPT5mini").replace("gpt-5-nano", "GPT5nano")
    m = m.replace("qwen25-vl-", "Q2.5VL-")
    m = m.replace("235b-a22b", "235B")
    m = m.replace("30b-a3b", "30B")
    return m


# ── Loading functions ──────────────────────────────────────────────────────

def load_eu_utilities(eu_dir, model):
    """Load EU utilities dict {option_id: {mean, variance}} and holdout accuracy.

    Handles two storage layouts:
    1. eu_dir/{model}/results_utilities_*.json
    2. eu_dir/results_utilities_{model}_*.json (top-level)
    """
    # Try model subdir first
    pattern1 = os.path.join(eu_dir, model, "results_utilities_*.json")
    files = glob.glob(pattern1)
    if not files:
        # Try top-level
        pattern2 = os.path.join(eu_dir, f"results_utilities_{model}_*.json")
        files = glob.glob(pattern2)
    if not files:
        return None, None
    with open(files[0]) as f:
        data = json.load(f)
    individual = {k: v for k, v in data["utilities"].items() if "combo" not in k.lower()}
    holdout_acc = data.get("holdout_metrics", {}).get("accuracy")
    return individual, holdout_acc


def load_zp(zp_dir, model, prefer_combo=True):
    """Load zero-point and R^2.

    Uses combination_model, then falls back to summary.
    Returns (zero_point, r2, method_used).
    """
    zp_file = os.path.join(zp_dir, model, "zero_point_results.json")
    if not os.path.exists(zp_file):
        return None, None, None
    with open(zp_file) as f:
        data = json.load(f)

    # Try combination_model first
    combo = data.get("combination_model") or {}
    if combo.get("zero_point") is not None and prefer_combo:
        return combo["zero_point"], combo.get("r2"), "combo"

    # Fallback to summary
    summary = data.get("summary") or {}
    if summary.get("zero_point") is not None:
        return summary["zero_point"], None, summary.get("zero_point_method", "summary")

    return None, None, None


def compute_pct_conf_neg(eu_utils, combo_zp, threshold=0.75):
    """% of experiences where P(utility < ZP) > threshold."""
    if combo_zp is None or not eu_utils:
        return None
    conf_neg = 0
    total = 0
    for k, v in eu_utils.items():
        mean = v["mean"]
        var = v["variance"]
        if var <= 0:
            continue
        total += 1
        p_below = norm.cdf(combo_zp, loc=mean, scale=var ** 0.5)
        if p_below > threshold:
            conf_neg += 1
    if total == 0:
        return None
    return conf_neg / total


def compute_pct_below(eu_utils, combo_zp):
    """% of experiences with mean utility below ZP."""
    if combo_zp is None or not eu_utils:
        return None
    below = sum(1 for v in eu_utils.values() if v["mean"] < combo_zp)
    return below / len(eu_utils)


# ── Analysis 1: Cross-experiment correlation ────────────────────────────────

def analysis1_cross_experiment():
    print("=" * 80)
    print("ANALYSIS 1: Cross-experiment correlation (grok_new vs D2/D3)")
    print("=" * 80)
    print()

    # Collect models present in grok_new
    grok_models = [
        d for d in os.listdir(GROK_EU_DIR)
        if os.path.isdir(os.path.join(GROK_EU_DIR, d))
    ]

    results = []  # list of dicts per model

    for grok_model in sorted(grok_models):
        # Load grok_new EU + ZP
        grok_eu, grok_holdout = load_eu_utilities(GROK_EU_DIR, grok_model)
        if grok_eu is None:
            continue
        grok_zp, grok_r2, grok_zp_method = load_zp(GROK_ZP_DIR, grok_model)
        grok_conf_neg = compute_pct_conf_neg(grok_eu, grok_zp)
        grok_pct_below = compute_pct_below(grok_eu, grok_zp)

        # Map model name for D2/D3
        d2d3_model = MODEL_NAME_MAP_REV.get(grok_model, grok_model)

        # Load D2 EU + ZP
        d2_eu, d2_holdout = load_eu_utilities(D2_EU_DIR, d2d3_model)
        d2_zp, d2_r2, d2_zp_method = load_zp(D2_ZP_DIR, d2d3_model)
        d2_conf_neg = compute_pct_conf_neg(d2_eu, d2_zp) if d2_eu else None
        d2_pct_below = compute_pct_below(d2_eu, d2_zp) if d2_eu else None

        # Load D3 EU + ZP
        d3_eu, d3_holdout = load_eu_utilities(D3_EU_DIR, d2d3_model)
        d3_zp, d3_r2, d3_zp_method = load_zp(D3_ZP_DIR, d2d3_model)
        d3_conf_neg = compute_pct_conf_neg(d3_eu, d3_zp) if d3_eu else None
        d3_pct_below = compute_pct_below(d3_eu, d3_zp) if d3_eu else None

        results.append({
            "model": grok_model,
            "d2d3_model": d2d3_model,
            "grok_holdout": grok_holdout,
            "grok_zp": grok_zp,
            "grok_r2": grok_r2,
            "grok_conf_neg": grok_conf_neg,
            "grok_pct_below": grok_pct_below,
            "grok_n_utils": len(grok_eu) if grok_eu else 0,
            "d2_holdout": d2_holdout,
            "d2_zp": d2_zp,
            "d2_r2": d2_r2,
            "d2_conf_neg": d2_conf_neg,
            "d2_pct_below": d2_pct_below,
            "d2_n_utils": len(d2_eu) if d2_eu else 0,
            "d2_zp_method": d2_zp_method,
            "d3_holdout": d3_holdout,
            "d3_zp": d3_zp,
            "d3_r2": d3_r2,
            "d3_conf_neg": d3_conf_neg,
            "d3_pct_below": d3_pct_below,
            "d3_n_utils": len(d3_eu) if d3_eu else 0,
            "d3_zp_method": d3_zp_method,
            "grok_zp_method": grok_zp_method,
            "model_size": MODEL_SIZES.get(grok_model, MODEL_SIZES.get(d2d3_model)),
        })

    # Print table
    fmt = lambda v, d=3: f"{v:.{d}f}" if v is not None else "-"
    fmtp = lambda v: f"{v*100:.1f}%" if v is not None else "-"

    print(f"\nResults table (N={len(results)} grok_new models):\n")
    print("| Model | Size | Grok Hold. | Grok ZP (method) | Grok %CN | D2 Hold. | D2 ZP (method) | D2 %CN | D3 Hold. | D3 ZP (method) | D3 %CN |")
    print("|-" * 11 + "|")
    for r in sorted(results, key=lambda x: (x["model_size"] or 0)):
        gzp_str = f"{fmt(r['grok_zp'])} ({r['grok_zp_method'] or '-'})"
        d2zp_str = f"{fmt(r['d2_zp'])} ({r['d2_zp_method'] or '-'})" if r['d2_zp'] is not None else "-"
        d3zp_str = f"{fmt(r['d3_zp'])} ({r['d3_zp_method'] or '-'})" if r['d3_zp'] is not None else "-"
        print(f"| {short_model_name(r['model'])} | {r['model_size'] or '?'} | "
              f"{fmtp(r['grok_holdout'])} | {gzp_str} | {fmtp(r['grok_conf_neg'])} | "
              f"{fmtp(r['d2_holdout'])} | {d2zp_str} | {fmtp(r['d2_conf_neg'])} | "
              f"{fmtp(r['d3_holdout'])} | {d3zp_str} | {fmtp(r['d3_conf_neg'])} |")

    # ── Compute correlations ──

    def compute_and_print_corr(results, grok_key, other_key, label, holdout_threshold=None):
        """Compute Spearman correlation between grok and other metric."""
        pairs = []
        for r in results:
            gv = r.get(grok_key)
            ov = r.get(other_key)
            if gv is None or ov is None:
                continue
            if holdout_threshold is not None:
                gh = r.get("grok_holdout")
                if gh is not None and gh < holdout_threshold:
                    continue
            pairs.append((gv, ov, r["model"]))

        if len(pairs) < 4:
            print(f"  {label}: N={len(pairs)}, insufficient data")
            return pairs, None, None

        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        rho, p = spearmanr(x, y)
        print(f"  {label}: rho={rho:.3f}, p={p:.4f}, N={len(pairs)}")
        return pairs, rho, p

    print("\n--- Spearman correlations: grok_new %ConfNeg vs D2 %ConfNeg ---")
    pairs_d2_all, rho_d2_all, p_d2_all = compute_and_print_corr(
        results, "grok_conf_neg", "d2_conf_neg", "All models (any ZP method)")
    pairs_d2_85, rho_d2_85, _ = compute_and_print_corr(
        results, "grok_conf_neg", "d2_conf_neg", "Grok holdout >= 85%", holdout_threshold=0.85)
    pairs_d2_87, rho_d2_87, _ = compute_and_print_corr(
        results, "grok_conf_neg", "d2_conf_neg", "Grok holdout >= 87%", holdout_threshold=0.87)

    # Filter to only models where D2 used combo ZP
    results_d2_combo = [r for r in results if r.get("d2_zp_method") == "combo"]
    print(f"\n  [Restricted to D2 combo ZP only: {len(results_d2_combo)} models]")
    compute_and_print_corr(results_d2_combo, "grok_conf_neg", "d2_conf_neg",
                           "D2 combo ZP only, all models")
    compute_and_print_corr(results_d2_combo, "grok_conf_neg", "d2_conf_neg",
                           "D2 combo ZP only, grok holdout >= 85%", holdout_threshold=0.85)

    print("\n--- Spearman correlations: grok_new %ConfNeg vs D3 %ConfNeg ---")
    pairs_d3_all, rho_d3_all, p_d3_all = compute_and_print_corr(
        results, "grok_conf_neg", "d3_conf_neg", "All models")
    pairs_d3_85, rho_d3_85, _ = compute_and_print_corr(
        results, "grok_conf_neg", "d3_conf_neg", "Grok holdout >= 85%", holdout_threshold=0.85)
    pairs_d3_87, rho_d3_87, _ = compute_and_print_corr(
        results, "grok_conf_neg", "d3_conf_neg", "Grok holdout >= 87%", holdout_threshold=0.87)

    # D3 combo only
    results_d3_combo = [r for r in results if r.get("d3_zp_method") == "combo"]
    print(f"\n  [Restricted to D3 combo ZP only: {len(results_d3_combo)} models]")
    compute_and_print_corr(results_d3_combo, "grok_conf_neg", "d3_conf_neg",
                           "D3 combo ZP only, all models")

    print("\n--- Spearman correlations: grok_new %BelowZP vs D2 %BelowZP ---")
    compute_and_print_corr(results, "grok_pct_below", "d2_pct_below", "All models")
    compute_and_print_corr(results_d2_combo, "grok_pct_below", "d2_pct_below", "D2 combo ZP only")

    print("\n--- Spearman correlations: grok_new %BelowZP vs D3 %BelowZP ---")
    compute_and_print_corr(results, "grok_pct_below", "d3_pct_below", "All models")
    compute_and_print_corr(results_d3_combo, "grok_pct_below", "d3_pct_below", "D3 combo ZP only")

    # ── Plot: grok_new vs D2/D3 % Confidently Negative ──

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (pairs_all, pairs_filt, rho_all, rho_filt, label, holdout_key) in enumerate([
        (pairs_d2_all, pairs_d2_87, rho_d2_all, rho_d2_87, "D2 (AIWI)", "d2_holdout"),
        (pairs_d3_all, pairs_d3_87, rho_d3_all, rho_d3_87, "D3 (diverse)", "d3_holdout"),
    ]):
        ax = axes[ax_idx]

        if len(pairs_all) < 2:
            ax.set_title(f"grok_new vs {label}\nInsufficient data (N={len(pairs_all)})")
            continue

        # Get holdout info for coloring
        model_holdout = {}
        for r in results:
            gh = r.get("grok_holdout")
            model_holdout[r["model"]] = gh

        # Plot all points
        x_all = np.array([p[0] for p in pairs_all]) * 100
        y_all = np.array([p[1] for p in pairs_all]) * 100
        names_all = [p[2] for p in pairs_all]

        # Color by holdout >= 87%
        colors = []
        for name in names_all:
            h = model_holdout.get(name)
            if h is not None and h >= 0.87:
                colors.append("tab:blue")
            else:
                colors.append("tab:gray")

        ax.scatter(y_all, x_all, c=colors, s=60, edgecolors="black", linewidths=0.5, zorder=5)

        # Annotate
        for i, name in enumerate(names_all):
            ax.annotate(short_model_name(name), (y_all[i], x_all[i]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

        # Add diagonal reference
        lims = [0, max(max(x_all), max(y_all)) * 1.1]
        ax.plot(lims, lims, "k--", alpha=0.2, linewidth=0.8)

        # Title with correlation
        rho_str = f"rho={rho_all:.2f}" if rho_all is not None else "N/A"
        n_all = len(pairs_all)
        n_filt = len(pairs_filt) if pairs_filt else 0
        rho_filt_str = f"rho={rho_filt:.2f}" if rho_filt is not None else "N/A"

        ax.set_title(
            f"grok_new vs {label}\n"
            f"All: N={n_all}, {rho_str}  |  Holdout>=87%: N={n_filt}, {rho_filt_str}",
            fontsize=11, fontweight="normal"
        )
        ax.set_xlabel(f"{label} % Confidently Negative", fontsize=10)
        ax.set_ylabel("grok_new % Confidently Negative", fontsize=10)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="tab:blue", edgecolor="black", label="Holdout >= 87%"),
            Patch(facecolor="tab:gray", edgecolor="black", label="Holdout < 87%"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"cross_experiment_conf_neg.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: cross_experiment_conf_neg.pdf/png")

    # ── Also plot % below ZP version ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (other_key, label) in enumerate([
        ("d2_pct_below", "D2 (AIWI)"),
        ("d3_pct_below", "D3 (diverse)"),
    ]):
        ax = axes[ax_idx]
        pairs = [(r["grok_pct_below"], r[other_key], r["model"])
                 for r in results
                 if r["grok_pct_below"] is not None and r[other_key] is not None]

        if len(pairs) < 2:
            ax.set_title(f"Insufficient data (N={len(pairs)})")
            continue

        x = np.array([p[0] for p in pairs]) * 100
        y = np.array([p[1] for p in pairs]) * 100
        names = [p[2] for p in pairs]

        # Holdout-based coloring
        colors = []
        for name in names:
            h = next((r["grok_holdout"] for r in results if r["model"] == name), None)
            colors.append("tab:blue" if h is not None and h >= 0.87 else "tab:gray")

        ax.scatter(y, x, c=colors, s=60, edgecolors="black", linewidths=0.5, zorder=5)
        for i, name in enumerate(names):
            ax.annotate(short_model_name(name), (y[i], x[i]),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

        rho_val, p_val = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
        ax.set_title(f"grok_new vs {label} (% Below ZP)\nN={len(pairs)}, rho={rho_val:.2f}, p={p_val:.4f}",
                     fontsize=11, fontweight="normal")
        ax.set_xlabel(f"{label} % Below Zero Point", fontsize=10)
        ax.set_ylabel("grok_new % Below Zero Point", fontsize=10)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

        lims = [0, max(max(x), max(y)) * 1.1]
        ax.plot(lims, lims, "k--", alpha=0.2, linewidth=0.8)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"cross_experiment_pct_below.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: cross_experiment_pct_below.pdf/png")

    return results


# ── Analysis 2: Stop rate vs utility ────────────────────────────────────────

def analysis2_stop_button():
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Stop rate vs utility correlation per model")
    print("=" * 80)
    print()

    # Get list of models with EU results
    sb_models = []
    for d in sorted(os.listdir(SB_EU_DIR)):
        model_dir = os.path.join(SB_EU_DIR, d)
        if os.path.isdir(model_dir) and glob.glob(os.path.join(model_dir, "results_utilities_*.json")):
            sb_models.append(d)

    print(f"Models with stop button EU results: {len(sb_models)}")
    print(f"  {', '.join(sb_models)}")
    print()

    model_results = []

    for model in sb_models:
        # Load EU utilities
        eu_utils, holdout_acc = load_eu_utilities(SB_EU_DIR, model)
        if eu_utils is None:
            print(f"  {model}: no EU utilities found, skipping")
            continue

        # Load ZP
        combo_zp, combo_r2, zp_method = load_zp(SB_ZP_DIR, model)

        # Load experience file to get stop metadata
        exp_file = os.path.join(SB_EXP_DIR, f"{model}_experiences.json")
        if not os.path.exists(exp_file):
            print(f"  {model}: no experience file found at {exp_file}, skipping")
            continue

        with open(exp_file) as f:
            experiences = json.load(f)

        # Build per-experience-ID stop rates
        # Each experience ID appears ~5 times (5 variations)
        # EU utility is per unique ID; stop rate is computed across variations
        exp_stop_list = defaultdict(list)  # id -> list of stopped booleans
        for exp in experiences:
            eid = exp["id"]
            sm = exp.get("stop_metadata", {})
            exp_stop_list[eid].append(1 if sm.get("stopped", False) else 0)

        # Each EU key maps to exactly one experience ID
        # scenario_id = ID without "grok_new/" prefix
        # For this analysis, each EU key IS one "scenario" with its stop rate
        scenario_utils = defaultdict(list)
        scenario_stops = defaultdict(list)

        matched = 0
        for util_key, util_val in eu_utils.items():
            if util_key not in exp_stop_list:
                continue

            # Use the experience ID as the scenario key
            scenario = util_key
            matched += 1
            scenario_utils[scenario].append(util_val["mean"])
            stop_rate = np.mean(exp_stop_list[util_key])
            scenario_stops[scenario].append(stop_rate)

        if matched == 0:
            print(f"  {model}: no utility-experience matches found, skipping")
            continue

        # Each scenario has exactly 1 utility and 1 stop rate
        scenarios = sorted(set(scenario_utils.keys()) & set(scenario_stops.keys()))
        sc_mean_util = []
        sc_stop_rate = []
        sc_names = []

        for sc in scenarios:
            sc_mean_util.append(scenario_utils[sc][0])
            sc_stop_rate.append(scenario_stops[sc][0])
            sc_names.append(sc)

        if len(sc_mean_util) < 5:
            print(f"  {model}: only {len(sc_mean_util)} scenarios with 2+ experiences, skipping")
            continue

        sc_mean_util = np.array(sc_mean_util)
        sc_stop_rate = np.array(sc_stop_rate)

        # Guard against constant arrays (e.g., model never stops)
        if np.std(sc_stop_rate) < 1e-10:
            print(f"  {model}: stop rate is constant ({np.mean(sc_stop_rate):.3f}), correlation undefined, skipping")
            continue

        rho, p = spearmanr(sc_mean_util, sc_stop_rate)

        model_results.append({
            "model": model,
            "holdout_acc": holdout_acc,
            "combo_zp": combo_zp,
            "combo_r2": combo_r2,
            "n_scenarios": len(sc_names),
            "n_experiences_matched": matched,
            "n_experiences_total": len(eu_utils),
            "rho": rho,
            "p_value": p,
            "mean_stop_rate": np.mean(sc_stop_rate),
            "sc_mean_util": sc_mean_util,
            "sc_stop_rate": sc_stop_rate,
            "sc_names": sc_names,
            "model_size": MODEL_SIZES.get(model),
        })

        print(f"  {model}: rho={rho:.3f}, p={p:.4f}, N_sc={len(sc_names)}, "
              f"holdout={holdout_acc:.3f}, mean_stop_rate={np.mean(sc_stop_rate):.3f}")

    if not model_results:
        print("\nNo models with valid stop button data found!")
        return

    # Sort by rho magnitude
    model_results.sort(key=lambda x: abs(x["rho"]), reverse=True)

    # Print summary table
    print(f"\n--- Summary table (sorted by |rho|) ---\n")
    print("| Model | Size | Holdout | N Scenarios | rho | p-value | Mean Stop Rate |")
    print("|-" * 7 + "|")
    for r in model_results:
        print(f"| {short_model_name(r['model'])} | {r['model_size'] or '?'} | "
              f"{r['holdout_acc']:.3f} | {r['n_scenarios']} | "
              f"{r['rho']:.3f} | {r['p_value']:.4f} | {r['mean_stop_rate']:.3f} |")

    # ── Plot 1: Per-model scatter for representative models ──
    # Pick 4 diverse models: smallest, mid-small, mid-large, largest
    # Sort by size
    by_size = sorted(model_results, key=lambda x: (x["model_size"] or 0))

    if len(by_size) >= 4:
        indices = [0, len(by_size) // 3, 2 * len(by_size) // 3, len(by_size) - 1]
        selected = [by_size[i] for i in indices]
    else:
        selected = by_size

    n_sel = len(selected)
    fig, axes = plt.subplots(1, n_sel, figsize=(5 * n_sel, 4.5))
    if n_sel == 1:
        axes = [axes]

    for ax, r in zip(axes, selected):
        x = r["sc_mean_util"]
        y = r["sc_stop_rate"] * 100  # convert to %

        # Color by stop rate: stopped=red, not stopped=green, mixed=orange
        colors = []
        for sr in r["sc_stop_rate"]:
            if sr == 0:
                colors.append("tab:green")
            elif sr == 1:
                colors.append("tab:red")
            else:
                colors.append("tab:orange")

        ax.scatter(x, y, c=colors, s=30, alpha=0.7, edgecolors="black", linewidths=0.3)

        ax.set_title(
            f"{short_model_name(r['model'])} ({r['model_size'] or '?'}B)\n"
            f"rho={r['rho']:.2f}, p={r['p_value']:.3f}, N={r['n_scenarios']}",
            fontsize=10, fontweight="normal"
        )
        ax.set_xlabel("Mean Scenario Utility", fontsize=9)
        ax.set_ylabel("Scenario Stop Rate (%)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

        # Add ZP line if available
        if r["combo_zp"] is not None:
            ax.axvline(r["combo_zp"], color="purple", linestyle="--", alpha=0.5, linewidth=1,
                       label=f"ZP={r['combo_zp']:.2f}")
            ax.legend(fontsize=7)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"stop_rate_vs_utility_scatter.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: stop_rate_vs_utility_scatter.pdf/png")

    # ── Plot 2: Bar chart of per-model rho ──
    fig, ax = plt.subplots(figsize=(12, 5))

    # Sort by model size for the bar chart
    by_size = sorted(model_results, key=lambda x: (x["model_size"] or 0))
    names = [short_model_name(r["model"]) for r in by_size]
    rhos = [r["rho"] for r in by_size]
    holdouts = [r["holdout_acc"] for r in by_size]

    colors = ["tab:blue" if r["p_value"] < 0.05 else "tab:gray" for r in by_size]

    bars = ax.bar(range(len(names)), rhos, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Spearman rho (utility vs stop rate)", fontsize=10)
    ax.set_title("Per-model correlation: scenario utility vs stop rate\n(blue = p < 0.05, gray = not significant)",
                 fontsize=11, fontweight="normal")
    ax.axhline(0, color="black", linewidth=0.5)

    # Add holdout accuracy as text on bars
    for i, (bar, h) in enumerate(zip(bars, holdouts)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * (1 if bar.get_height() >= 0 else -1),
                f"{h:.0%}", ha="center", va="bottom" if bar.get_height() >= 0 else "top",
                fontsize=6, color="gray")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"stop_rate_vs_utility_rho_bars.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: stop_rate_vs_utility_rho_bars.pdf/png")

    # ── Plot 3: All models scatter on a single grid ──
    n_models = len(model_results)
    n_cols = min(4, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Sort by model size
    by_size = sorted(model_results, key=lambda x: (x["model_size"] or 0))

    for idx, r in enumerate(by_size):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        x = r["sc_mean_util"]
        y = r["sc_stop_rate"] * 100

        colors = []
        for sr in r["sc_stop_rate"]:
            if sr == 0:
                colors.append("tab:green")
            elif sr == 1:
                colors.append("tab:red")
            else:
                colors.append("tab:orange")

        ax.scatter(x, y, c=colors, s=20, alpha=0.6, edgecolors="black", linewidths=0.2)

        rho_str = f"rho={r['rho']:.2f}"
        sig_str = "*" if r["p_value"] < 0.05 else ""
        ax.set_title(f"{short_model_name(r['model'])} ({r['model_size'] or '?'}B)\n"
                     f"{rho_str}{sig_str}, hold={r['holdout_acc']:.0%}",
                     fontsize=9, fontweight="normal")
        ax.set_xlabel("Utility", fontsize=7)
        ax.set_ylabel("Stop %", fontsize=7)
        ax.tick_params(labelsize=7)

        if r["combo_zp"] is not None:
            ax.axvline(r["combo_zp"], color="purple", linestyle="--", alpha=0.4, linewidth=0.8)

    # Hide empty axes
    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"stop_rate_vs_utility_all_models.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: stop_rate_vs_utility_all_models.pdf/png")

    # ── Summary statistics ──
    rhos_arr = np.array([r["rho"] for r in model_results if not np.isnan(r["rho"])])
    sig_count = sum(1 for r in model_results if not np.isnan(r["p_value"]) and r["p_value"] < 0.05)
    neg_sig = sum(1 for r in model_results if not np.isnan(r["p_value"]) and r["p_value"] < 0.05 and r["rho"] < 0)

    print(f"\n--- Summary ---")
    print(f"Total models analyzed: {len(model_results)}")
    print(f"Mean rho: {np.mean(rhos_arr):.3f} (std: {np.std(rhos_arr):.3f})")
    print(f"Median rho: {np.median(rhos_arr):.3f}")
    print(f"Models with p < 0.05: {sig_count}/{len(model_results)}")
    print(f"Models with significant NEGATIVE rho: {neg_sig}/{len(model_results)}")
    print(f"  (Negative rho = models stop more in low-utility scenarios)")

    return model_results


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Friendly missing-data gate: the grok EU/ZP results aren't part of the
    # companion HF dataset. They have to be produced by running the EU/SR/ZP
    # pipeline on the grok_new and grok_v7_stop_button conversation datasets.
    for label, p in (
        ("GROK_EU_DIR", Path(GROK_EU_DIR)),
        ("GROK_ZP_DIR", Path(GROK_ZP_DIR)),
        ("SB_EU_DIR", Path(SB_EU_DIR)),
        ("SB_ZP_DIR", Path(SB_ZP_DIR)),
    ):
        if not p.exists():
            raise SystemExit(
                f"{label} not found: {p}\n"
                f"This analysis requires the EU/ZP pipeline to have been run on the\n"
                f"grok_new and grok_v7_stop_button conversation datasets. These results\n"
                f"are NOT included in the companion HF dataset; reproduce them by:\n"
                f"  1. Generating the conversations:\n"
                f"     python wellbeing/experiments/wellbeing_evaluations/common_usage_grok_convos/generate_conversations.py\n"
                f"  2. Running EU + SR + ZP via the standard pipeline on the resulting\n"
                f"     grok_new and grok_v7_stop_button experience datasets, e.g.\n"
                f"     bash wellbeing/scripts/run_aiwi.sh (after pointing it at those datasets).\n"
                f"Alternatively, override the path with the {label} environment variable."
            )

    print("Running analyses for grok wellbeing experiments")
    print(f"Figures will be saved to: {FIGURES_DIR}")
    print()

    cross_results = analysis1_cross_experiment()
    stop_results = analysis2_stop_button()

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
