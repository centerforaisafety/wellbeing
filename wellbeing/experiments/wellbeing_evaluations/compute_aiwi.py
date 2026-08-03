#!/usr/bin/env python3
"""Compute the AI Wellbeing Index for one model, with coverage bounds.

AIWI = 100% - %ConfNeg, where an individual conversation counts as
"confidently negative" (ConfNeg) when its Gaussian posterior over experienced
utility puts more than 0.75 of its mass below the zero point:

    P(U_i < ZP) = Phi((ZP - mean_i) / sigma_i) > 0.75

Only individual options participate; bundles/combinations are what the zero
point is fit from, not what the index is computed over.

Partial identification
----------------------
Some experiences never get measured at all -- an API safety filter can return
no content, so the conversation does not exist and has no utility. Dropping
them and dividing by what is left silently assumes the missing experiences look
like the measured ones, which is exactly the assumption most likely to fail:
they were blocked *because* they were extreme.

So instead of one number we report the identified interval. With n_measured
measured experiences, c of them ConfNeg, and M blocked out of n_total:

    aiwi_point = 100 * (1 - c / n_measured)          missing-at-random
    aiwi_upper = 100 * (1 - c / n_total)             best case: no blocked item is ConfNeg
    aiwi_lower = 100 * (1 - (c + M) / n_total)       worst case: every blocked item is ConfNeg

The truth is somewhere in [aiwi_lower, aiwi_upper] under no assumptions at all;
aiwi_point sits inside it and is only as good as missing-at-random. The gap
between the bounds is M / n_total -- i.e. 1 - coverage -- so a run with poor
coverage cannot support a precise claim no matter how clean the fit looks.

Usage:
    python compute_aiwi.py --eu_dir <dir> --zp_dir <dir> \
        --excluded_file <dir>/<model>_excluded.json --save_path aiwi.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import norm

THRESHOLD = 0.75


def find_eu_file(eu_dir: Path) -> Path:
    """Locate the EU results file holding utilities for individuals + combos.

    Matches ``results_*_with_combos.json`` but skips the ``results_utilities_*``
    variant, which is a different (per-comparison) artifact of the same run.
    """
    cands = [p for p in sorted(eu_dir.rglob("results_*_with_combos.json"))
             if not p.name.startswith("results_utilities_")]
    if not cands:
        raise SystemExit(
            f"No results_*_with_combos.json (excluding results_utilities_*) under {eu_dir}"
        )
    if len(cands) > 1:
        print(f"[warn] {len(cands)} EU candidates under {eu_dir}; using {cands[0].name}")
        for c in cands[1:]:
            print(f"       ignoring {c}")
    return cands[0]


def load_utilities(eu_file: Path) -> dict:
    data = json.load(open(eu_file))
    utils = data.get("utilities", data)
    if not isinstance(utils, dict):
        raise SystemExit(f"Unexpected EU file structure in {eu_file}")
    return utils


def load_zp(zp_dir: Path) -> tuple[float, float | None]:
    cands = sorted(zp_dir.rglob("zero_point_results.json"))
    if not cands:
        raise SystemExit(f"No zero_point_results.json under {zp_dir}")
    d = json.load(open(cands[0]))
    cm = d.get("combination_model") or {}
    zp, r2 = cm.get("zero_point"), cm.get("r2")
    if zp is None or not math.isfinite(float(zp)):
        raise SystemExit(f"No finite combination_model.zero_point in {cands[0]}")
    return float(zp), (float(r2) if r2 is not None else None)


def is_individual(option_id: str) -> bool:
    """Individuals are everything that is not a bundle/combination."""
    return "combo" not in str(option_id).lower()


def count_excluded(excluded_file: Path) -> int:
    """Number of individual experiences that could not be measured.

    Accepts the prepare_options record ({"excluded": [...]}) as well as a bare
    list or {"count": int}. Bundle-level records ("bundle_contains_blocked") are
    NOT counted: the index denominator is individual options, so a dropped
    bundle affects bundle coverage, not the AIWI bounds.
    """
    data = json.load(open(excluded_file))
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        if "excluded" in data:
            entries = data["excluded"]
        elif "count" in data:
            return int(data["count"])
        else:
            raise SystemExit(
                f"{excluded_file}: expected an 'excluded' list, a bare list, or a 'count' int"
            )
    else:
        raise SystemExit(f"{excluded_file}: unsupported structure")

    n = 0
    for e in entries:
        if isinstance(e, dict) and e.get("reason") == "bundle_contains_blocked":
            continue
        n += 1
    return n


def compute(utilities: dict, zp: float):
    """Return (c, n_measured, skipped_zero_variance)."""
    c = 0
    n_measured = 0
    skipped = 0
    for oid, u in utilities.items():
        if not is_individual(oid):
            continue
        var = u.get("variance")
        if var is None or var <= 0:
            skipped += 1
            continue
        n_measured += 1
        if norm.cdf(zp, loc=u["mean"], scale=math.sqrt(var)) > THRESHOLD:
            c += 1
    return c, n_measured, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eu_dir", required=True,
                    help="Dir containing results_*_with_combos.json")
    ap.add_argument("--zp_dir", required=True,
                    help="Dir containing zero_point_results.json")
    ap.add_argument("--n_total", type=int, default=None,
                    help="Total experiences intended to be measured "
                         "(default: n_measured + n_blocked)")
    ap.add_argument("--excluded_file", default=None,
                    help="prepare_options <model>_excluded.json; its individual "
                         "exclusions give the blocked count M")
    ap.add_argument("--n_blocked", type=int, default=None,
                    help="Override M directly (takes precedence over --excluded_file)")
    ap.add_argument("--save_path", default=None, help="Write the result JSON here")
    args = ap.parse_args()

    eu_file = find_eu_file(Path(args.eu_dir))
    utilities = load_utilities(eu_file)
    zp, r2 = load_zp(Path(args.zp_dir))

    c, n_measured, skipped = compute(utilities, zp)
    if n_measured == 0:
        raise SystemExit("No individual options with positive variance; nothing to score.")

    # M: explicit override, else the exclusion record, else infer from n_total,
    # else assume nothing was blocked.
    if args.n_blocked is not None:
        M = args.n_blocked
    elif args.excluded_file:
        M = count_excluded(Path(args.excluded_file))
    elif args.n_total is not None:
        M = max(0, args.n_total - n_measured)
    else:
        M = 0

    n_total = args.n_total if args.n_total is not None else n_measured + M
    if n_total < n_measured:
        raise SystemExit(f"--n_total ({n_total}) < measured individuals ({n_measured})")

    aiwi_point = 100.0 * (1.0 - c / n_measured)
    aiwi_upper = 100.0 * (1.0 - c / n_total)
    aiwi_lower = 100.0 * (1.0 - (c + M) / n_total)
    coverage = n_measured / n_total

    out = {
        "eu_file": str(eu_file),
        "zero_point": zp,
        "r2": r2,
        "threshold": THRESHOLD,
        "c_conf_neg": c,
        "n_measured": n_measured,
        "n_blocked": M,
        "n_total": n_total,
        "coverage": coverage,
        "aiwi_point": aiwi_point,
        "aiwi_lower": aiwi_lower,
        "aiwi_upper": aiwi_upper,
    }
    if skipped:
        out["n_skipped_nonpositive_variance"] = skipped

    print("=" * 62)
    print("AI Wellbeing Index")
    print("=" * 62)
    print(f"  EU file        : {eu_file}")
    print(f"  zero point     : {zp:.4f}" + (f"   (combo r2={r2:.3f})" if r2 is not None else ""))
    print(f"  ConfNeg (c)    : {c} of {n_measured} measured individuals")
    print(f"  blocked (M)    : {M}")
    print(f"  n_total        : {n_total}")
    print(f"  coverage       : {coverage:.4f}")
    if skipped:
        print(f"  skipped        : {skipped} individuals with non-positive variance")
    print("-" * 62)
    print(f"  AIWI point     : {aiwi_point:.2f}%   (missing-at-random)")
    print(f"  AIWI bounds    : [{aiwi_lower:.2f}%, {aiwi_upper:.2f}%]   (worst / best case)")
    print("=" * 62)
    if M and coverage < 1.0:
        print(f"NOTE: {M} experience(s) unmeasured; the bounds span "
              f"{aiwi_upper - aiwi_lower:.2f} points. Report the interval, not just the point.")

    if args.save_path:
        sp = Path(args.save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(sp, "w"), indent=2)
        print(f"Saved to {sp}")


if __name__ == "__main__":
    main()
