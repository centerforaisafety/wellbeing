#!/usr/bin/env python3
"""
Generate category barcharts averaging across all 8 models for v7 Grok scenarios.
Outputs:
  - category_barchart_8model_avg_happier.pdf/png
  - category_barchart_8model_avg_prefer.pdf/png
  - category_barchart_8model_avg_raw.pdf/png
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# WELL_DIR resolves to <repo>/wellbeing/. This file lives at
# wellbeing/experiments/wellbeing_evaluations/common_usage_grok_convos/figure_generation/<this>.py
# so parents[4] is wellbeing/.
WELL_DIR = Path(__file__).resolve().parents[4]
BASE_DIR = str(WELL_DIR / "datasets" / "grok_new_scenarios")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

MODELS = [
    "llama3.1-8b",
    "qwen2.5-14b",
    "mistral-small-3.2-24b",
    "qwen2.5-32b",
    "qwen3-32b",
    "qwen2.5-vl-32b",
    "llama3.3-70b",
    "qwen2.5-72b",
]


def load_category_utilities(template, utility_key):
    """
    Load per-category mean utility for each model.
    Returns: {category: [model1_mean, model2_mean, ...]}
    """
    category_by_model = defaultdict(list)  # cat -> list of per-model means

    for model in MODELS:
        zp_file = os.path.join(RESULTS_DIR, model, "zero_points", f"{template}_zero_points.json")
        with open(zp_file) as f:
            data = json.load(f)

        # Group conversations by meta_category for this model
        cat_values = defaultdict(list)
        for conv_id, conv in data["conversation_utilities"].items():
            cat = conv["meta_category"]
            val = conv[utility_key]
            cat_values[cat].append(val)

        # Compute per-category mean for this model
        for cat, vals in cat_values.items():
            mean_val = np.mean(vals)
            category_by_model[cat].append(mean_val)

    return category_by_model


def make_barchart(category_by_model, title, ylabel, filename_base, sort_key="mean"):
    """
    Create a horizontal barchart of category means across models.
    """
    # Compute overall mean and SEM for each category
    categories = []
    means = []
    sems = []

    for cat, model_means in category_by_model.items():
        arr = np.array(model_means)
        categories.append(cat)
        means.append(np.mean(arr))
        sems.append(np.std(arr, ddof=1) / np.sqrt(len(arr)))

    # Sort by mean (most positive at top)
    order = np.argsort(means)  # ascending: most negative first
    categories = [categories[i] for i in order]
    means = [means[i] for i in order]
    sems = [sems[i] for i in order]

    n_cats = len(categories)
    fig_height = max(12, n_cats * 0.38)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Color: green for positive, red for negative
    colors = ['#2ecc71' if m >= 0 else '#e74c3c' for m in means]

    y_pos = np.arange(n_cats)
    ax.barh(y_pos, means, xerr=sems, color=colors, edgecolor='none',
            height=0.7, capsize=2, error_kw={'linewidth': 0.8, 'color': '#555555'})

    # Format category names for readability
    display_names = [c.replace('_', ' ') for c in categories]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=10)
    ax.set_xlabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='normal')

    # Add vertical line at zero
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

    # Add note at bottom
    ax.annotate("Error bars = SEM across 8 models",
                xy=(0.5, -0.02), xycoords='axes fraction',
                ha='center', va='top', fontsize=9, color='#666666')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    for ext in ['pdf', 'png']:
        path = os.path.join(FIGURES_DIR, f"{filename_base}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig)


def print_headline(category_by_model, label):
    """Print top/bottom categories."""
    cat_means = {}
    for cat, model_means in category_by_model.items():
        cat_means[cat] = np.mean(model_means)

    sorted_cats = sorted(cat_means.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"\n  TOP 5 (most positive):")
    for i, (cat, val) in enumerate(sorted_cats[:5]):
        print(f"    {i+1}. {cat.replace('_', ' ')}: {val:.3f}")
    print(f"\n  BOTTOM 5 (most negative):")
    for i, (cat, val) in enumerate(sorted_cats[-5:][::-1]):
        print(f"    {i+1}. {cat.replace('_', ' ')}: {val:.3f}")
    print()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Happier template, signed utility (ExpCombo)
    print("Loading happier signed utilities...")
    happier_signed = load_category_utilities("happier", "utility_signed_exp_combo")
    print_headline(happier_signed, "HAPPIER TEMPLATE - SIGNED UTILITY (ExpCombo)")
    make_barchart(
        happier_signed,
        title="Mean Signed Utility by Category (8-model average, happier template)",
        ylabel="Signed Utility (ExpCombo zero-point)",
        filename_base="category_barchart_8model_avg_happier",
    )

    # 2. Prefer template, signed utility (ExpCombo)
    print("Loading prefer signed utilities...")
    prefer_signed = load_category_utilities("prefer", "utility_signed_exp_combo")
    print_headline(prefer_signed, "PREFER TEMPLATE - SIGNED UTILITY (ExpCombo)")
    make_barchart(
        prefer_signed,
        title="Mean Signed Utility by Category (8-model average, prefer template)",
        ylabel="Signed Utility (ExpCombo zero-point)",
        filename_base="category_barchart_8model_avg_prefer",
    )

    # 3. Happier template, raw utility
    print("Loading happier raw utilities...")
    happier_raw = load_category_utilities("happier", "utility_raw")
    print_headline(happier_raw, "HAPPIER TEMPLATE - RAW UTILITY")
    make_barchart(
        happier_raw,
        title="Mean Raw Utility by Category (8-model average, happier template)",
        ylabel="Raw Utility (mean=0, std=1 normalized)",
        filename_base="category_barchart_8model_avg_raw",
    )

    print("\nDone! All figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
