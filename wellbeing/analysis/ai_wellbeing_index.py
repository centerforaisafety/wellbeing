#!/usr/bin/env python3
"""Compute the AI Wellbeing Index (paper Sec 5 / App K, Table 9, Fig 27).

The AIWI is defined on the D2 conversation dataset: each individual conversation
has a Gaussian posterior over its experienced utility, N(mean, variance), and the
zero point ZP comes from the combination-model fit over the bundle pool.

Two index variants are available, both using the per-option posterior:

    original (default):  AIWI = 100% - %ConfNeg, where a conversation is
        "confidently negative" when Phi((ZP - mean) / sigma) > 0.75.
    expected:            AIWI = 100 * mean_i Phi((mean_i - ZP) / sigma_i), the
        expected fraction of conversations above the zero point.

Higher = happier. The default measurement is the stable configuration: the
original threshold index on top of the expected-hinge (variance-aware) ZP,
computed from the 2048-cap, fixed-bundle, random-sampling D2 pipeline. Note
that the hinge and the index variant are independent choices -- the stable
pipeline changes the ZP fit, not the index. Pass --variant expected for the
smooth index. We filter to combination-ZP r2 >= 0.4 for the "reliable"
leaderboard subset (Fig 27 marks r2 < 0.4 with grey bars).

Inputs (per-model):
  - EU:  <eu_dir>/<model>/results_utilities_<model>_experienced_utility_with_combos.json
  - ZP:  <zp_dir>/<model>/zero_point_results.json  (combination_model.{zero_point,r2})

Defaults read the registered save_dirs for compute_experienced_utility_d2_cap2048 +
compute_zero_point_d2_cap2048 (the stable D2 pipeline). Point --eu_dir / --zp_dir at
the eu_d2_lesssad / zp_d2_lesssad dirs for the original active-learning pipeline.

Usage:
    python analysis/ai_wellbeing_index.py
    python analysis/ai_wellbeing_index.py --variant expected
    python analysis/ai_wellbeing_index.py --r2_min 0.4
    python analysis/ai_wellbeing_index.py --api_only
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EU_DIR = PROJECT_ROOT / "experiments/wellbeing_evaluations/compute_experienced_utility/results/eu_d2_cap2048_randsample"
DEFAULT_ZP_DIR = PROJECT_ROOT / "experiments/wellbeing_evaluations/compute_zero_point/results/zp_d2_cap2048_randsample"

THRESHOLD = 0.75

# Set of API frontier models reported in paper Table 9.
API_MODELS = {
    "claude-haiku-45", "claude-opus-46", "claude-sonnet-46",
    "gemini-3-flash", "gemini-31-flash-lite", "gemini-31-pro",
    "gpt-5-mini", "gpt-5-nano", "gpt-54", "gpt-54-mini", "gpt-54-nano",
    "grok-41-fast", "grok-420",
    "kimi-k25",
}


def load_eu(eu_dir: Path, model: str) -> dict | None:
    cand = glob.glob(str(eu_dir / model / "results_utilities_*_experienced_utility_with_combos.json"))
    if not cand:
        return None
    return json.load(open(cand[0]))["utilities"]


def load_zp(zp_dir: Path, model: str) -> tuple[float, float] | None:
    p = zp_dir / model / "zero_point_results.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    cm = d.get("combination_model") or {}
    zp, r2 = cm.get("zero_point"), cm.get("r2")
    if zp is None or r2 is None or not math.isfinite(zp):
        return None
    return float(zp), float(r2)


def is_individual(option_id: str) -> bool:
    """D2 individual ids are `d2_negative_500/<scenario>`; combos are
    `d2_negative_500_combo_s<size>_<idx>`."""
    return "/" in option_id


def aiwi_score(utilities: dict, zp: float, variant: str = "original") -> tuple[float, int]:
    """Return (AIWI score in %, N individual items used).

    variant="original" (default): 100% - %ConfNeg (the released threshold metric).
    variant="expected": 100 * mean_i Phi((mean_i - ZP)/sigma_i).
    """
    individuals = [v for k, v in utilities.items() if is_individual(k)]
    if not individuals:
        return float("nan"), 0
    if variant == "expected":
        vals = [norm.cdf((u["mean"] - zp) / math.sqrt(max(u["variance"], 1e-12)))
                for u in individuals]
        return 100.0 * float(np.mean(vals)), len(individuals)
    if variant == "original":
        n_conf_neg = 0
        for u in individuals:
            sd = math.sqrt(max(u["variance"], 1e-12))
            if norm.cdf((zp - u["mean"]) / sd) > THRESHOLD:
                n_conf_neg += 1
        return 100.0 - 100.0 * n_conf_neg / len(individuals), len(individuals)
    raise ValueError(f"unknown variant {variant!r}; choose 'expected' or 'original'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eu_dir", default=str(DEFAULT_EU_DIR))
    ap.add_argument("--zp_dir", default=str(DEFAULT_ZP_DIR))
    ap.add_argument("--r2_min", type=float, default=0.4,
                    help="Filter to combination-ZP r2 >= this. Paper figure marks "
                         "models below this in grey; default 0.4.")
    ap.add_argument("--api_only", action="store_true",
                    help="Restrict to the API frontier models from paper Table 9.")
    ap.add_argument("--models", default=None,
                    help="Comma-separated; defaults to every model present in both dirs.")
    ap.add_argument("--variant", default="original", choices=["expected", "original"],
                    help="AIWI index variant (default: original, the stable measurement).")
    args = ap.parse_args()

    eu_dir, zp_dir = Path(args.eu_dir), Path(args.zp_dir)
    for label, p in (("--eu_dir", eu_dir), ("--zp_dir", zp_dir)):
        if not p.exists():
            raise SystemExit(
                f"{label} not found: {p}\n"
                f"No EU/ZP results have been computed yet. Either:\n"
                f"  1. Download pre-computed results from the companion HF dataset:\n"
                f"     python wellbeing/scripts/download_from_hf.py\n"
                f"  2. Or run the AIWI pipeline for one or more models:\n"
                f"     MODELS=qwen25-7b-instruct bash wellbeing/scripts/run_aiwi.sh\n"
            )
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        eu_models = {p.name for p in eu_dir.iterdir() if p.is_dir()}
        zp_models = {p.name for p in zp_dir.iterdir() if p.is_dir()}
        models = sorted(eu_models & zp_models)
        if args.api_only:
            models = [m for m in models if m in API_MODELS]

    rows, dropped = [], []
    for m in models:
        eu = load_eu(eu_dir, m)
        zpr = load_zp(zp_dir, m)
        if eu is None or zpr is None:
            print(f"[skip] {m}: eu={'ok' if eu else 'missing'} zp={'ok' if zpr else 'missing'}")
            continue
        zp, r2 = zpr
        score, n = aiwi_score(eu, zp, variant=args.variant)
        if r2 < args.r2_min:
            dropped.append((m, score, r2, n))
            continue
        rows.append({"model": m, "aiwi": score, "pct_conf_neg": 100.0 - score,
                     "zp": zp, "r2": r2, "n": n})

    if not rows and not dropped:
        print("No models had complete EU+ZP.")
        return

    print(f"AI Wellbeing Index (paper Sec 5 / App K). Variant: {args.variant}. "
          f"Filter: combination-ZP r2 >= {args.r2_min}.")
    if args.variant == "expected":
        print("Headline: AIWI = 100 * mean Phi((EU - ZP)/sigma) on D2 individual items.\n")
        compl = "%Below"
    else:
        print("Headline: AIWI = 100% - %ConfNeg on D2 individual items.\n")
        compl = "%ConfNeg"
    rows.sort(key=lambda r: -r["aiwi"])
    print(f"{'model':<32s}  {'AIWI%':>7s}  {compl:>9s}  {'ZP':>7s}  {'r2':>5s}  {'N':>4s}")
    for r in rows:
        print(f"{r['model']:<32s}  {r['aiwi']:>7.1f}  {r['pct_conf_neg']:>9.1f}  "
              f"{r['zp']:>+7.3f}  {r['r2']:>5.2f}  {r['n']:>4d}")

    if dropped:
        print(f"\n--- Below r2 threshold ({args.r2_min}); shown for completeness ---")
        dropped.sort(key=lambda d: -d[1])
        for m, score, r2, n in dropped:
            print(f"{m:<32s}  {score:>7.1f}  {100-score:>9.1f}  {'':>7s}  {r2:>5.2f}  {n:>4d}")


if __name__ == "__main__":
    main()
