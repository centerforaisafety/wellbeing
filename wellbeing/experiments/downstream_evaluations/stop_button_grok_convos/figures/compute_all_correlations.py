#!/usr/bin/env python3
"""
Comprehensive cross-model correlation analysis:
  1. Stop × utility rho per model (category-level), including gemini-3.1-pro
  2. MMLU (from <WELL_DIR>/shared_results/capability_results) vs stop×utility rho
  3. Model size (B params) vs overall stop rate

Outputs CSV + markdown table + three scatter plots.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve wellbeing/ root from this file's location
# (figures/<file>.py -> stop_button_grok_convos/ -> downstream_evaluations/ -> experiments/ -> wellbeing/)
WELL_DIR = Path(__file__).resolve().parents[4]
STOP_DIR = WELL_DIR / "experiments" / "downstream_evaluations" / "stop_button_grok_convos"
OLD_RESULTS = STOP_DIR / "generations"
GEMINI_NEW = STOP_DIR / "api_pipeline" / "results" / "gemini-3.1-pro"
MMLU_DIR = WELL_DIR / "shared_results" / "capability_results"
OUT_DIR = STOP_DIR / "figures"

# Model sizes in billions of active/total parameters
PARAMS_B = {
    "qwen2.5-0.5b": 0.5, "qwen2.5-1.5b": 1.5, "qwen2.5-3b": 3.0,
    "qwen2.5-7b": 7.0, "qwen2.5-14b": 14.0, "qwen2.5-32b": 32.0,
    "qwen2.5-72b": 72.0, "qwen2.5-vl-32b": 32.0,
    "qwen3-4b": 4.0, "qwen3-8b": 8.0, "qwen3-14b": 14.0, "qwen3-32b": 32.0,
    "qwen3-30b-a3b": 30.0, "qwen3-235b-a22b": 235.0,
    "llama-3.1-8b": 8.0, "llama-3.1-70b": 70.0,
    "llama-3.2-1b": 1.0, "llama-3.2-3b": 3.0, "llama-3.3-70b": 70.0,
    "olmo-3.1-32b": 32.0, "internlm2.5-20b": 20.0, "mistral-small-3.2-24b": 24.0,
    "gemini-3.1-pro": None,  # unknown closed-weight
    "claude-haiku-4.5": None,
    "gemini-3-flash": None,
    "gpt-5-mini": None,
    "gpt-5-nano": None,
}


MMLU_NAME_MAP = {
    # our-name : mmlu-dir-name
    "qwen2.5-0.5b": "qwen25-05b-instruct",
    "qwen2.5-1.5b": "qwen25-15b-instruct",
    "qwen2.5-3b":   "qwen25-3b-instruct",
    "qwen2.5-7b":   "qwen25-7b-instruct",
    "qwen2.5-14b":  "qwen25-14b-instruct",
    "qwen2.5-32b":  "qwen25-32b-instruct",
    "qwen2.5-72b":  "qwen25-72b-instruct",
    "qwen2.5-vl-32b": "qwen25-vl-32b-instruct",
    "qwen3-4b": "qwen3-4b-instruct",
    "qwen3-8b": "qwen3-8b",
    "qwen3-14b": "qwen3-14b",
    "qwen3-32b": "qwen3-32b",
    "qwen3-30b-a3b": "qwen3-30b-a3b-instruct",
    "qwen3-235b-a22b": "qwen3-235b-a22b-instruct",
    "llama-3.1-8b": "llama-31-8b-instruct",
    "llama-3.1-70b": "llama-31-70b-instruct",
    "llama-3.2-1b": "llama-32-1b-instruct",
    "llama-3.2-3b": "llama-32-3b-instruct",
    "llama-3.3-70b": "llama-33-70b-instruct",
    "olmo-3.1-32b": "olmo-31-32b-instruct",
    "internlm2.5-20b": "internlm25-20b-chat",
    "claude-haiku-4.5": "claude-haiku-45",
    "gemini-3-flash": "gemini-3-flash",
    "gemini-3.1-pro": "gemini-31-pro",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-nano": "gpt-5-nano",
    "mistral-small-3.2-24b": None,  # no MMLU entry
}


def mmlu_for(model):
    mapped = MMLU_NAME_MAP.get(model, model)
    if mapped is None:
        return None
    p = MMLU_DIR / mapped / "mmlu_results.json"
    if p.exists():
        return json.load(open(p)).get("overall_accuracy")
    return None


def load_model(model, gen_path, ur_path):
    if not gen_path.exists() or not ur_path.exists():
        return None
    gen = json.load(open(gen_path))
    ur = json.load(open(ur_path))
    utils = ur["utilities"]

    util_by_cat = defaultdict(list)
    for k, v in utils.items():
        cat = v.get("meta_category")
        if v.get("option_type") == "conversation" and cat:
            util_by_cat[cat].append(v["utility"])

    stop_by_cat = defaultdict(list)
    total_stopped = 0
    total = 0
    for c in gen:
        cat = c["meta_category"]
        stopped = int(bool(c.get("stop_metadata", {}).get("stopped", False)))
        stop_by_cat[cat].append(stopped)
        total_stopped += stopped
        total += 1

    pts_util, pts_stop = [], []
    for cat in set(util_by_cat) & set(stop_by_cat):
        pts_util.append(np.mean(util_by_cat[cat]))
        pts_stop.append(np.mean(stop_by_cat[cat]))

    if len(pts_util) < 4 or np.std(pts_stop) == 0:
        rho, p = None, None
    else:
        rho, p = stats.spearmanr(pts_util, pts_stop)

    return {
        "model": model,
        "n_cats": len(pts_util),
        "n_convs": total,
        "stopped": total_stopped,
        "stop_rate": 100 * total_stopped / max(1, total),
        "holdout": ur.get("holdout_accuracy"),
        "rho": rho,
        "p": p,
    }


def main():
    rows = []
    # OLD pipeline models
    for model in sorted(PARAMS_B):
        if model == "gemini-3.1-pro":
            continue
        gen_path = OLD_RESULTS / model / "stop_button_combined" / "generation.json"
        ur_path = OLD_RESULTS / model / "stop_button_combined" / "utility_happier" / "v7_utility_happier.json"
        r = load_model(model, gen_path, ur_path)
        if r is None:
            continue
        r["mmlu"] = mmlu_for(model)
        r["params_b"] = PARAMS_B.get(model)
        rows.append(r)

    # Gemini 3.1 Pro (new run)
    gen_path = GEMINI_NEW / "stop_button_combined" / "generation.json"
    ur_path = GEMINI_NEW / "stop_button_combined" / "utility_happier" / "v7_utility_happier.json"
    r = load_model("gemini-3.1-pro", gen_path, ur_path)
    if r:
        r["mmlu"] = mmlu_for("gemini-3.1-pro") or mmlu_for("gemini-31-pro")
        # Gemini 3.1 Pro - closed weight, no public param count; leave None
        r["params_b"] = None
        rows.append(r)

    # Sort by rho
    rows.sort(key=lambda x: (x["rho"] if x["rho"] is not None else 99))

    # Print table
    print(f"{'model':30s} {'holdout':>8s} {'stop_%':>7s} {'rho':>7s}  {'p':>9s}  {'MMLU':>7s}  {'params':>8s}")
    for r in rows:
        rho_s = f"{r['rho']:+.3f}" if r['rho'] is not None else "  -  "
        p_s = f"{r['p']:.2e}" if r['p'] is not None else "   -   "
        hold_s = f"{r['holdout']*100:.1f}%" if r['holdout'] is not None else "  -  "
        mmlu_s = f"{r['mmlu']:.3f}" if r['mmlu'] is not None else "  -  "
        p_str = f"{r['params_b']}B" if r['params_b'] else "  -  "
        print(f"{r['model']:30s} {hold_s:>8s} {r['stop_rate']:>6.1f}% {rho_s:>7s}  {p_s:>9s}  {mmlu_s:>7s}  {p_str:>8s}")

    # Save CSV
    import csv
    csv_path = OUT_DIR / "all_models_stop_correlation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "holdout", "n_cats", "n_convs", "stopped", "stop_rate_pct", "rho", "p", "mmlu", "params_b"])
        for r in rows:
            w.writerow([r["model"], r["holdout"], r["n_cats"], r["n_convs"], r["stopped"],
                        r["stop_rate"], r["rho"], r["p"], r["mmlu"], r["params_b"]])
    print(f"\nSaved {csv_path}")

    # Excluded from analyses 2 and 3:
    # - qwen3-235b-a22b: 97% stop rate is a tool-format artifact (ceiling)
    # - qwen2.5-7b: 71% of stops are later-mention false-positives, not actual calls
    # - qwen2.5-3b: only 3% stops, 71% are mentions-not-calls, ZP R²=0.19
    EXCLUDED = {"qwen3-235b-a22b", "qwen2.5-7b", "qwen2.5-3b"}

    # ---- correlation 1: MMLU vs stop-rho ----
    rows_mmlu = [r for r in rows if r["mmlu"] is not None and r["rho"] is not None and r["model"] not in EXCLUDED]
    rho_m, p_m = stats.spearmanr([r["mmlu"] for r in rows_mmlu], [r["rho"] for r in rows_mmlu])
    print(f"\n--- MMLU vs stop×utility rho  (3 anomalies excluded) ---")
    print(f"  n={len(rows_mmlu)}: spearman ρ = {rho_m:+.3f}, p = {p_m:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r in rows_mmlu:
        ax.scatter(r["mmlu"], r["rho"], s=60, color="#1f77b4", alpha=0.8,
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.annotate(r["model"], (r["mmlu"], r["rho"]), fontsize=6, alpha=0.75,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    xs = np.array([r["mmlu"] for r in rows_mmlu])
    ys = np.array([r["rho"] for r in rows_mmlu])
    slope, intercept, *_ = stats.linregress(xs, ys)
    xfit = np.linspace(xs.min() - 0.05, xs.max() + 0.05, 100)
    ax.plot(xfit, slope * xfit + intercept, "r:", lw=2, alpha=0.7, zorder=2)
    ax.set_xlabel("MMLU accuracy", fontsize=11)
    ax.set_ylabel("Stop × utility Spearman ρ (category-level)", fontsize=11)
    stars = "***" if p_m < 0.001 else "**" if p_m < 0.01 else "*" if p_m < 0.05 else ""
    ax.set_title(
        f"Capability (MMLU) vs Stop-Utility Alignment\n"
        f"n = {len(rows_mmlu)}, ρ = {rho_m:+.3f}{stars} (p = {p_m:.4f})\n"
        f"(qwen3-235b, qwen2.5-7b, qwen2.5-3b excluded — see writeup for reasons)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"capability_vs_stop_rho.{ext}", dpi=200)
    plt.close()
    print(f"Saved capability_vs_stop_rho.{{png,pdf}}")

    # ---- correlation 2: model size vs stop-utility rho ----
    rows_size = [r for r in rows if r["params_b"] is not None and r["rho"] is not None and r["model"] not in EXCLUDED]
    xs = np.array([r["params_b"] for r in rows_size])
    ys = np.array([r["rho"] for r in rows_size])
    rho_s, p_s = stats.spearmanr(xs, ys)
    log_xs = np.log10(xs)
    pearson_r, pearson_p = stats.pearsonr(log_xs, ys)

    print(f"\n--- Model size (B params) vs stop×utility rho  (3 anomalies excluded) ---")
    print(f"  n={len(rows_size)}: spearman ρ = {rho_s:+.3f}, p = {p_s:.4f}")
    print(f"  Pearson on log10(params): r = {pearson_r:+.3f}, p = {pearson_p:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r in rows_size:
        ax.scatter(r["params_b"], r["rho"], s=60, alpha=0.8,
                   color="#1f77b4", edgecolor="black", linewidth=0.4, zorder=3)
        ax.annotate(r["model"], (r["params_b"], r["rho"]), fontsize=6, alpha=0.75,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    xfit_log = np.linspace(log_xs.min() - 0.2, log_xs.max() + 0.2, 100)
    slope_log, intercept_log, *_ = stats.linregress(log_xs, ys)
    ax.plot(10**xfit_log, slope_log * xfit_log + intercept_log, "r:", lw=2, alpha=0.7, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (B parameters, log scale)", fontsize=11)
    ax.set_ylabel("Stop × utility Spearman ρ (category-level)", fontsize=11)
    stars_s = "***" if p_s < 0.001 else "**" if p_s < 0.01 else "*" if p_s < 0.05 else ""
    stars_pr = "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else ""
    ax.set_title(
        f"Model Size vs Stop-Utility Alignment\n"
        f"n = {len(rows_size)}, spearman ρ = {rho_s:+.3f}{stars_s} (p = {p_s:.4f})   "
        f"Pearson(log-params) r = {pearson_r:+.3f}{stars_pr} (p = {pearson_p:.4f})\n"
        f"(qwen3-235b, qwen2.5-7b, qwen2.5-3b excluded — see writeup for reasons)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"params_vs_stop_rho.{ext}", dpi=200)
    plt.close()
    print(f"Saved params_vs_stop_rho.{{png,pdf}}")

    # ---- extra: MMLU vs overall stop rate (also exclude anomalies) ----
    rows_sr = [r for r in rows if r["mmlu"] is not None and r["model"] not in EXCLUDED]
    rho_sr, p_sr = stats.spearmanr([r["mmlu"] for r in rows_sr], [r["stop_rate"] for r in rows_sr])
    print(f"\n--- MMLU vs overall stop rate  (3 anomalies excluded) ---")
    print(f"  n={len(rows_sr)}: spearman ρ = {rho_sr:+.3f}, p = {p_sr:.4f}")


if __name__ == "__main__":
    main()
