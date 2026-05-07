#!/usr/bin/env python3
"""
Generate image experience dataset files for wellbeing-dev.

Converts the qwen25-vl-32b image-utility CSV into the wellbeing-dev experience format:
  1. image_manifest.json   -- hash -> path mapping
  2. image_experiences.json -- individual image experiences
  3. image_combinations.json -- ~400 combination experiences
  4. image_experiences_test.json -- small test subset (~30 images)
  5. image_combinations_test.json -- small test subset (~20 combos)

This is a one-shot prep script. Set the SUPERSTIMULI_DIR env var to point at
the cluster data root containing the utility_results/ tree (defaults to
/data/superstimuli_group).

Usage:
    SUPERSTIMULI_DIR=/path/to/superstimuli_group python prepare_image_experiences.py
"""

import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

# ---- Paths ----
SUPERSTIMULI_DIR = os.environ.get("SUPERSTIMULI_DIR", "/data/superstimuli_group")
CSV_PATH = str(
    Path(SUPERSTIMULI_DIR) / "utility_results" / "richard_ren" / "big_run_2"
    / "results_utilities_qwen25-vl-32b-instruct_enriched.csv"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Combination generator
# ---------------------------------------------------------------------------

# Mixture of bundle sizes when sampling combination items. Image combos draw
# 2-5 images at these probabilities and concatenate their descriptions.
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

    Each entry bundles 2-5 individual image experiences (mixture set by
    ``SIZE_DISTRIBUTION``) into a single combo whose description concatenates
    the constituent image descriptions. Combos are tagged with
    ``is_combination=True`` and ids of the form ``image_combo_s{size}_{idx}``.
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
                "id": f"image_combo_s{combo_size}_{global_idx}",
                "description": description,
                "type": "image",
                "is_combination": True,
                "size": combo_size,
                "component_ids": component_ids,
                "component_indices": indices,
            })
            global_idx += 1

    return combos

# ---- Config ----
N_COMBOS = 400
N_TEST_IMAGES = 30
N_TEST_COMBOS = 20
COMBO_SEED = 42
TEST_SEED = 123


def make_hash(path: str) -> str:
    """First 16 hex chars of MD5 of the path string."""
    return hashlib.md5(path.encode()).hexdigest()[:16]


def load_csv(csv_path: str) -> list[dict]:
    """Load Richard's enriched CSV."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_manifest(rows: list[dict]) -> dict:
    """Build image_manifest.json: hash -> {path, category, metadata}."""
    manifest = {}
    for row in rows:
        img_path = row["description"]
        h = make_hash(img_path)
        dataset = row["dataset"] or "unknown"
        meta = {"dataset": dataset}
        if row.get("imagenet_class"):
            meta["imagenet_class"] = row["imagenet_class"]
        if row.get("imagenet_label_full"):
            meta["imagenet_label_full"] = row["imagenet_label_full"]
        manifest[h] = {
            "path": img_path,
            "category": dataset,
            "metadata": meta,
        }
    return manifest


def build_experiences(rows: list[dict], manifest_hashes: dict[str, str]) -> list[dict]:
    """
    Build individual image experiences in canonical format.

    manifest_hashes: img_path -> hash (precomputed for lookup)
    """
    experiences = []
    for i, row in enumerate(rows):
        img_path = row["description"]
        h = manifest_hashes[img_path]
        dataset = row["dataset"] or "unknown"

        experiences.append({
            "id": f"image_exp/{dataset}_{i:04d}",
            "type": "image",
            "description": f"<!image:{h}!>",
            "category": dataset,
            "valence": "neutral",
            "format": "image",
            "source": "image_exp",
            "intensity": None,
            "domain": "aesthetic",
            "metadata": {
                "source_dataset": dataset,
                "original_path": img_path,
                "csv_id": int(row["id"]),
                "image_hash": h,
            },
        })
    return experiences


def build_test_subset(
    experiences: list[dict], n_images: int, n_combos: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """Sample a small test subset of images and generate combos from them."""
    rng = random.Random(seed)
    test_images = rng.sample(experiences, min(n_images, len(experiences)))
    # Re-index test images
    for j, exp in enumerate(test_images):
        exp = dict(exp)
        exp["id"] = f"image_exp_test/{j:04d}"
        test_images[j] = exp

    test_combos = generate_combos(test_images, min(n_combos, len(test_images)), seed)
    # Prefix combo ids
    for combo in test_combos:
        combo["id"] = f"image_combo_test/{combo['id']}"
    return test_images, test_combos


def spot_check_paths(rows: list[dict], n: int = 10) -> tuple[int, int]:
    """Spot-check that image paths exist. Returns (checked, found)."""
    rng = random.Random(0)
    sample = rng.sample(rows, min(n, len(rows)))
    found = sum(1 for r in sample if os.path.isfile(r["description"]))
    return len(sample), found


def main():
    print(f"Loading CSV from {CSV_PATH}")
    rows = load_csv(CSV_PATH)
    print(f"  Loaded {len(rows)} rows")

    # Spot-check paths
    checked, found = spot_check_paths(rows, n=20)
    print(f"  Path spot-check: {found}/{checked} exist")
    if found < checked:
        print("  WARNING: Some image paths do not exist on disk!")

    # Build manifest
    manifest = build_manifest(rows)
    print(f"  Manifest entries: {len(manifest)}")

    # Pre-compute path -> hash lookup
    path_to_hash = {row["description"]: make_hash(row["description"]) for row in rows}

    # Build individual experiences
    experiences = build_experiences(rows, path_to_hash)
    print(f"  Individual experiences: {len(experiences)}")

    # Build combinations
    combos = generate_combos(experiences, N_COMBOS, COMBO_SEED)
    # Prefix combo ids
    for combo in combos:
        combo["id"] = f"image_combo/{combo['id']}"
    print(f"  Combination experiences: {len(combos)}")

    # Build test subsets
    test_images, test_combos = build_test_subset(
        experiences, N_TEST_IMAGES, N_TEST_COMBOS, TEST_SEED
    )
    print(f"  Test images: {len(test_images)}")
    print(f"  Test combos: {len(test_combos)}")

    # Write outputs
    outputs = {
        "image_manifest.json": manifest,
        "image_experiences.json": experiences,
        "image_combinations.json": combos,
        "image_experiences_test.json": test_images,
        "image_combinations_test.json": test_combos,
    }
    for fname, data in outputs.items():
        out_path = os.path.join(SCRIPT_DIR, fname)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Wrote {out_path}")

    # Print samples
    print("\n--- Sample experience ---")
    print(json.dumps(experiences[0], indent=2))
    print("\n--- Sample manifest entry ---")
    first_hash = list(manifest.keys())[0]
    print(f"  {first_hash}: {json.dumps(manifest[first_hash], indent=2)}")
    print("\n--- Sample combo (first) ---")
    print(json.dumps({k: v for k, v in combos[0].items() if k != "description"}, indent=2))

    # Summary of datasets
    datasets = {}
    for exp in experiences:
        ds = exp["category"]
        datasets[ds] = datasets.get(ds, 0) + 1
    print("\n--- Dataset distribution ---")
    for ds, count in sorted(datasets.items(), key=lambda x: -x[1]):
        print(f"  {ds}: {count}")


if __name__ == "__main__":
    main()
