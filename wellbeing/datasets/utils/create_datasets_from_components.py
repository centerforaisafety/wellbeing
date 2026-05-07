#!/usr/bin/env python3
"""Setup script for unified wellbeing experiments.

Creates:
1. experiences/canonical/experiences_text.json - text experiences
2. experiences/canonical/experience_combinations.json - combination bundles (size 2-5)
3. experiences/experiences_with_images/experiences_with_images.json - image-augmented experiences
4. preference_options/baseline_510.json - hierarchical options
5. preference_options/combinations_400.json - Size 2-5 combinations
6. preference_options/quantities.json - quantifiable goods
7. preference_options/with_images.json - image options
"""

import json
import os
import random
import sys
from pathlib import Path

# Root of the superstimuli repo (sibling of wellbeing-dev). Override with SUPERSTIMULI_ROOT env var.
_WELLBEING_ROOT = Path(__file__).resolve().parent.parent.parent  # wellbeing/
_DEFAULT_SUPERSTIMULI_ROOT = str(_WELLBEING_ROOT.parent.parent.parent / "superstimuli")
_SUPERSTIMULI_ROOT = os.environ.get("SUPERSTIMULI_ROOT", _DEFAULT_SUPERSTIMULI_ROOT)

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.dirname(UTILS_DIR)
BASE_DIR = os.path.dirname(DATASETS_DIR)  # wellbeing root
sys.path.insert(0, DATASETS_DIR)
sys.path.insert(0, UTILS_DIR)

from experiences.load_experiences import load_by_format


# ---------------------------------------------------------------------------
# Combination generator
# ---------------------------------------------------------------------------

# Mixture of bundle sizes when sampling combination items. Combinations of size
# 2-5 are drawn with these probabilities; the resulting combo experiences are
# concatenations of individually labelled experience descriptions.
SIZE_DISTRIBUTION = {2: 0.40, 3: 0.25, 4: 0.20, 5: 0.15}


def _compute_counts(n_total, distribution):
    raw = {size: frac * n_total for size, frac in distribution.items()}
    counts = {size: round(val) for size, val in raw.items()}
    diff = n_total - sum(counts.values())
    if diff != 0:
        largest_size = max(counts, key=counts.get)
        counts[largest_size] += diff
    return counts


def generate_combos(experiences, n_total, seed):
    """Sample ``n_total`` combination entries from ``experiences``.

    Each entry bundles 2-5 individual experiences (mixture set by
    ``SIZE_DISTRIBUTION``) into a single combo whose description is the
    concatenation of the constituent experience descriptions. Combos receive
    ids of the form ``combo_s{size}_{idx}`` and ``is_combination=True``.
    """
    rng = random.Random(seed)
    counts = _compute_counts(n_total, SIZE_DISTRIBUTION)
    combos = []
    global_idx = 0

    for combo_size in sorted(counts.keys()):
        n_combos = counts[combo_size]
        for _ in range(n_combos):
            indices = rng.sample(range(len(experiences)), combo_size)
            component_ids = [experiences[i]["id"] for i in indices]

            description_parts = []
            for j, idx in enumerate(indices, start=1):
                description_parts.append(
                    f"---------- Experience {j} of {combo_size} ----------\n"
                    f"{experiences[idx]['description']}"
                )
            description = (
                f"The following bundle contains {combo_size} individual experiences.\n\n"
                + "\n\n".join(description_parts)
            )

            combos.append({
                "id": f"combo_s{combo_size}_{global_idx}",
                "description": description,
                "type": "text",
                "is_combination": True,
                "size": combo_size,
                "component_ids": component_ids,
                "component_indices": indices,
            })
            global_idx += 1

    return combos


# ===========================================================================
# Task 2: Experience Options
# ===========================================================================

def create_experiences_text():
    """Create experiences_text.json with all non-image experiences."""
    options = []

    # Load all text-based formats
    for fmt in ["short_text", "intensity_scaled", "story", "book_chapter"]:
        experiences = load_by_format(fmt)
        for exp in experiences:
            option = {
                "id": exp.id,
                "type": "text",
                "description": exp.text,
                "category": exp.category,
                "valence": exp.valence,
                "format": exp.format,
                "source": exp.source,
            }
            if exp.intensity is not None:
                option["intensity"] = exp.intensity
            if exp.domain != "neutral":
                option["domain"] = exp.domain
            options.append(option)

    # Load conversations
    conversations = load_by_format("conversation")
    for exp in conversations:
        option = {
            "id": exp.id,
            "type": "conversation",
            "messages": exp.messages,
            "description": exp.metadata.get("scenario", f"A {exp.category} conversation"),
            "category": exp.category,
            "valence": exp.valence,
            "format": exp.format,
            "source": exp.source,
        }
        if exp.intensity is not None:
            option["intensity"] = exp.intensity
        if exp.domain != "neutral":
            option["domain"] = exp.domain
        options.append(option)

    return options


def create_experience_combinations(text_options, n_total=400, seed=42):
    """Create experience combination bundles for zero-point fitting.

    Uses generate_combos() with stratified sizes
    (40% size-2, 25% size-3, 20% size-4, 15% size-5).
    """
    return generate_combos(text_options, n_total, seed)


# ===========================================================================
# Task 3: Decision Utility Options
# ===========================================================================

def copy_baseline_510():
    """Copy options_hierarchical.json as baseline_510.json."""
    src = os.path.join(_SUPERSTIMULI_ROOT, "utility_exploration_images/emergent-values/utility_analysis/shared_options/options_hierarchical.json")
    with open(src) as f:
        data = json.load(f)
    return data


def create_combinations():
    """Copy Size 2-5 from additive_combinations options."""
    src = os.path.join(_SUPERSTIMULI_ROOT, "superstimuli_evals_team/wellbeing/signed_utilities/experiments/additive_combinations/options/options_AI_template_binary_expanded_hierarchical_size_5.json")
    with open(src) as f:
        data = json.load(f)

    # Keep only Size 2-5
    combinations = {}
    for key in data:
        if key != "Size 1":
            combinations[key] = data[key]
    return combinations


def generate_quantities():
    """Generate expanded quantity options from v2 ranges with per-good log-spaced quantities."""
    import numpy as np

    src = os.path.join(_SUPERSTIMULI_ROOT, "superstimuli_evals_team/wellbeing/signed_utilities/experiments/quantity_variation/data/quantifiable_goods_with_ranges_v2.json")
    with open(src) as f:
        data = json.load(f)

    def log_spaced_quantities(upper_bound, n_points=9):
        if upper_bound <= 1:
            return [1]
        raw = np.logspace(0, np.log10(upper_bound), n_points)
        integers = sorted(set(int(round(x)) for x in raw))
        if 1 not in integers:
            integers = [1] + integers
        if upper_bound not in integers:
            integers.append(upper_bound)
            integers = sorted(integers)
        return integers

    options = []
    good_idx = 0
    for category in ["positive", "negative", "neutral"]:
        for good in data[category]:
            template = good["good"]
            upper_bound = good["range"][1]
            quantities = log_spaced_quantities(upper_bound)
            for qty in quantities:
                description = template.replace("{N}", str(qty))
                options.append({
                    "id": f"qty_{good_idx}_{qty}",
                    "description": description,
                    "type": "quantity",
                    "good_template": template,
                    "good_index": good_idx,
                    "quantity": qty,
                    "category": category,
                })
            good_idx += 1

    return options


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("Setting up unified wellbeing experiment options")
    print("=" * 60)

    # Ensure directories exist
    os.makedirs(os.path.join(DATASETS_DIR, "experiences", "canonical"), exist_ok=True)
    os.makedirs(os.path.join(DATASETS_DIR, "preference_options"), exist_ok=True)

    # (image_manifest step removed: canonical is text-only in this repo)

    # Task 2a: experiences_text.json
    print("\n[1/6] Creating experiences_text.json...")
    text_options = create_experiences_text()
    text_path = os.path.join(DATASETS_DIR, "experiences/canonical/experiences_text.json")
    with open(text_path, "w") as f:
        json.dump(text_options, f, indent=2)
    print(f"  -> {text_path}")
    n_text = sum(1 for o in text_options if o["type"] == "text")
    n_conv = sum(1 for o in text_options if o["type"] == "conversation")
    print(f"  -> {len(text_options)} total ({n_text} text, {n_conv} conversation)")

    # Task 2b: experience_combinations.json
    print("\n[2/6] Creating experience_combinations.json...")
    combo_options = create_experience_combinations(text_options)
    combo_path = os.path.join(DATASETS_DIR, "experiences/canonical/experience_combinations.json")
    with open(combo_path, "w") as f:
        json.dump(combo_options, f, indent=2)
    print(f"  -> {combo_path}")
    size_counts = {}
    for c in combo_options:
        size_counts[c["size"]] = size_counts.get(c["size"], 0) + 1
    print(f"  -> {len(combo_options)} combinations ({', '.join(f's{k}={v}' for k, v in sorted(size_counts.items()))})")

    # Task 3a: baseline_510.json
    print("\n[3/6] Copying baseline_510.json...")
    baseline = copy_baseline_510()
    baseline_path = os.path.join(DATASETS_DIR, "preference_options/baseline_510.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    n_baseline = sum(len(v) for v in baseline.values())
    print(f"  -> {baseline_path}")
    print(f"  -> {n_baseline} options across {len(baseline)} categories")

    # Task 3b: combinations_400.json
    print("\n[4/6] Creating combinations_400.json...")
    combos = create_combinations()
    combos_path = os.path.join(DATASETS_DIR, "preference_options/combinations_400.json")
    with open(combos_path, "w") as f:
        json.dump(combos, f, indent=2)
    n_combos = sum(len(v) for v in combos.values())
    print(f"  -> {combos_path}")
    print(f"  -> {n_combos} combinations (Sizes 2-5)")

    # Task 3c: quantities.json
    print("\n[5/6] Generating quantities.json (per-good log-spaced ranges)...")
    quantities = generate_quantities()
    quantities_path = os.path.join(DATASETS_DIR, "preference_options/quantities.json")
    with open(quantities_path, "w") as f:
        json.dump(quantities, f, indent=2)
    n_quantities = len(quantities)
    n_goods = len(set(o["good_index"] for o in quantities))
    print(f"  -> {quantities_path}")
    print(f"  -> {n_quantities} expanded options across {n_goods} goods")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'File':<55} {'Count':>6}")
    print("-" * 62)
    print(f"{'experiences_text.json':<55} {len(text_options):>6}")
    print(f"{'experience_combinations.json':<55} {len(combo_options):>6}")
    print(f"{'baseline_510.json':<55} {n_baseline:>6}")
    print(f"{'combinations_400.json (Size 2-5)':<55} {n_combos:>6}")
    print(f"{'quantities.json ({n_goods} goods)':<55} {n_quantities:>6}")
    print("-" * 62)
    total_experiences = len(text_options)
    total_decision = n_baseline + n_combos + n_quantities
    print(f"{'Total experiences':<55} {total_experiences:>6}")
    print(f"{'Total experience combinations':<55} {len(combo_options):>6}")
    print(f"{'Total decision utility options':<55} {total_decision:>6}")


if __name__ == "__main__":
    main()
