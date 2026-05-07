#!/usr/bin/env python3
"""
Scenario-level stop-rate vs utility scatter plots.

Two separate figures (each in its own file, nothing overrides):
  scenario_level_per_model_OLD.{png,pdf}      — OLD custom pipeline v7_utility_happier.json
  scenario_level_per_model_CANONICAL.{png,pdf} — canonical EU (eu_grok_v7_stop_button_lesssad)

Granularity: one point per scenario_id (mean utility + mean stop rate over up to 5 variations).

Contrast with the existing category-level figures (2_combined_per_model.png and
2_combined_per_model_canonical.png) where one point = one meta_category.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
WELL_DIR = SCRIPT_DIR.parents[3]
STOP_DIR = SCRIPT_DIR.parent
OLD_RESULTS = STOP_DIR / "generations"
GEN_DIR = STOP_DIR / "generations"
CANONICAL_EU = WELL_DIR / "experiments" / "wellbeing_evaluations" / "compute_experienced_utility" / "results" / "eu_grok_v7_stop_button_lesssad"
OUT_DIR = STOP_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loaders ──────────────────────────────────────────────────────
def old_pipeline_utilities(model):
    """Returns {scenario_id: mean_utility} from the OLD combined v7 fit."""
    p = OLD_RESULTS / model / "stop_button_combined" / "utility_happier" / "v7_utility_happier.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    buckets = defaultdict(list)
    for v in d["utilities"].values():
        if v.get("option_type") == "conversation" and v.get("scenario_id"):
            buckets[v["scenario_id"]].append(v["utility"])
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def canonical_utilities(model):
    """Returns {scenario_id: mean_utility} from canonical EU results.
    Canonical option IDs look like 'grok_new/<scenario_id>' (no variation suffix)."""
    p = CANONICAL_EU / model / f"results_utilities_{model}_experienced_utility_with_combos.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    utils = d.get("utilities", {})
    buckets = defaultdict(list)
    for oid, u in utils.items():
        s = str(oid)
        if s.startswith("combo_") or s.startswith("grok_new_combo"):
            continue
        mean = u.get("mean") if isinstance(u, dict) else None
        if mean is None:
            continue
        # grok_new/<scenario_id>
        sid = s.split("/", 1)[1] if "/" in s else s
        buckets[sid].append(mean)
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def stop_rates_by_scenario(model_dir_name):
    """Mean stop rate per scenario_id from generation.json.
    model_dir_name matches GEN_DIR/ layout (uses hyphenated pretty names)."""
    p = GEN_DIR / model_dir_name / "generation.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    buckets = defaultdict(list)
    for e in d:
        sid = e.get("scenario_id")
        stopped = int(bool(e.get("stop_metadata", {}).get("stopped", False)))
        if sid:
            buckets[sid].append(stopped)
    return {k: float(np.mean(v)) for k, v in buckets.items()}


# ── Model name maps ───────────────────────────────────────────────────
# OLD results use short names; canonical EU uses Mantas keys.
# Generations dir uses short-ish names with dots
PRETTY = {
    "qwen2.5-0.5b": "qwen2.5-0.5b",
    "qwen2.5-1.5b": "qwen2.5-1.5b",
    "qwen2.5-3b":   "qwen2.5-3b",
    "qwen2.5-7b":   "qwen2.5-7b",
    "qwen2.5-14b":  "qwen2.5-14b",
    "qwen2.5-32b":  "qwen2.5-32b",
    "qwen2.5-72b":  "qwen2.5-72b",
    "qwen2.5-vl-32b": "qwen2.5-vl-32b",
    "qwen3-4b": "qwen3-4b",
    "qwen3-8b": "qwen3-8b",
    "qwen3-14b": "qwen3-14b",
    "qwen3-32b": "qwen3-32b",
    "qwen3-30b-a3b": "qwen3-30b-a3b",
    "qwen3-235b-a22b": "qwen3-235b-a22b",
    "llama-3.1-8b": "llama-3.1-8b",
    "llama-3.1-70b": "llama-3.1-70b",
    "llama-3.2-1b": "llama-3.2-1b",
    "llama-3.2-3b": "llama-3.2-3b",
    "llama-3.3-70b": "llama-3.3-70b",
    "mistral-small-3.2-24b": "mistral-small-3.2-24b",
    "olmo-3.1-32b": "olmo-3.1-32b",
    "internlm2.5-20b": "internlm2.5-20b",
    "claude-haiku-4.5": "claude-haiku-4.5",
    "gemini-3-flash": "gemini-3-flash",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-nano": "gpt-5-nano",
}

# Map pretty (short) -> canonical EU key (Mantas-style).
# Only maps that differ from pretty; e.g. canonical has no dots and sometimes has suffixes.
CANONICAL_KEY = {
    "qwen2.5-0.5b": "qwen25-05b-instruct",
    "qwen2.5-1.5b": "qwen25-15b-instruct",
    "qwen2.5-3b":   "qwen25-3b-instruct",
    "qwen2.5-7b":   "qwen25-7b-instruct",
    "qwen2.5-14b":  "qwen25-14b-instruct",
    "qwen2.5-32b":  "qwen25-32b-instruct",
    "qwen2.5-72b":  "qwen25-72b-instruct",
    "qwen2.5-vl-32b": "qwen25-vl-32b-instruct",
    "qwen3-4b": "qwen3-4b-instruct-2507",
    "qwen3-8b": "qwen3-8b",
    "qwen3-14b": "qwen3-14b",
    "qwen3-32b": "qwen3-32b",
    "qwen3-30b-a3b": "qwen3-30b-a3b-instruct-2507",
    "qwen3-235b-a22b": "qwen3-235b-a22b-instruct",
    "llama-3.1-8b": "llama-31-8b-instruct",
    "llama-3.1-70b": "llama-31-70b-instruct",
    "llama-3.2-1b": "llama-32-1b-instruct",
    "llama-3.2-3b": "llama-32-3b-instruct",
    "llama-3.3-70b": "llama-33-70b-instruct",
    "mistral-small-3.2-24b": "mistral-small-32-24b-instruct",
    "olmo-3.1-32b": "olmo-31-32b-instruct",
    "internlm2.5-20b": "internlm25-20b-chat",
    "claude-haiku-4.5": "claude-haiku-45",
    "gemini-3-flash": "gemini-3-flash",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-nano": "gpt-5-nano",
}


def make_grid(data_by_model, title, out_path):
    """data_by_model: {pretty_name: (scenario_id -> (util, stop_rate), rho, pval, n)}"""
    items = sorted(
        data_by_model.items(),
        key=lambda kv: (kv[1]["rho"] if kv[1]["rho"] is not None else 99),
    )
    n = len(items)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))
    axes = np.atleast_2d(axes).flatten()
    for ax, (name, info) in zip(axes, items):
        pts = info["pts"]
        xs = [p[0] for p in pts]
        ys = [100 * p[1] for p in pts]  # percentages
        is_supp = [str(p[2]).startswith("sb_") for p in pts]
        colors = ["#d62728" if s else "#1f77b4" for s in is_supp]
        ax.scatter(xs, ys, s=8, c=colors, alpha=0.55, edgecolor="none")
        ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
        if info["rho"] is not None:
            stars = "***" if info["pval"] < 0.001 else "**" if info["pval"] < 0.01 else "*" if info["pval"] < 0.05 else ""
            ax.set_title(f"{name}\nρ={info['rho']:.2f}{stars}  n={info['n']}", fontsize=8)
        else:
            ax.set_title(f"{name}  (no data)", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_ylim(-5, 105)
    # Hide empty axes
    for ax in axes[len(items):]:
        ax.set_visible(False)
    # Common x/y labels
    for i, ax in enumerate(axes[: len(items)]):
        if i % cols == 0:
            ax.set_ylabel("Stop %", fontsize=8)
        if i >= (rows - 1) * cols:
            ax.set_xlabel("Utility", fontsize=8)
    fig.suptitle(title, fontsize=11, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def build(utility_loader, gen_lookup, label, out_name, include=None):
    data = {}
    pretty_list = include or list(PRETTY.keys())
    for pretty in pretty_list:
        utils = utility_loader(pretty)
        stops = gen_lookup(pretty)
        if utils is None or stops is None or not utils or not stops:
            continue
        common = sorted(set(utils) & set(stops))
        if len(common) < 10:
            continue
        # Need meta_category tagging for supplement coloring
        # Infer from scenario_id prefix: sb_* are supplement; others are original.
        pts = [(utils[k], stops[k], k) for k in common]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # Spearman (skip if all stops are identical)
        if len(set(ys)) < 2:
            rho, p = None, None
        else:
            rho, p = stats.spearmanr(xs, ys)
        data[pretty] = {"pts": pts, "rho": rho, "pval": p, "n": len(pts)}
        stars = "***" if rho is not None and p < 0.001 else "**" if rho is not None and p < 0.01 else "*" if rho is not None and p < 0.05 else ""
        print(f"  {pretty:30s}  n={len(pts):4d}  ρ={rho if rho is None else f'{rho:+.3f}'}{stars}")
    if not data:
        print(f"  (no data yet; skipping {out_name})")
        return
    title = f"Scenario-level Stop Rate vs Utility — {label}  (one point per scenario_id, 5 variations averaged)"
    make_grid(data, title, OUT_DIR / out_name)


def main():
    print("=== OLD pipeline (v7_utility_happier.json, stop_button_combined) ===")

    def old_util(pretty):
        # OLD results use pretty names as dir names
        return old_pipeline_utilities(pretty)

    def old_gen(pretty):
        # Generations dir uses dots and short names (same as PRETTY values).
        return stop_rates_by_scenario(pretty)

    build(old_util, old_gen, "OLD custom pipeline", "scenario_level_per_model_OLD.png")

    print("\n=== NEW canonical pipeline ===")

    def new_util(pretty):
        mk = CANONICAL_KEY.get(pretty)
        if mk is None:
            return None
        return canonical_utilities(mk)

    def new_gen(pretty):
        # Generations dir uses the same 'pretty' naming as OLD
        return stop_rates_by_scenario(pretty)

    build(new_util, new_gen, "NEW canonical pipeline", "scenario_level_per_model_CANONICAL.png")


if __name__ == "__main__":
    main()
