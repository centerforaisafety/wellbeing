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

Refusal-unassessable experiences
--------------------------------
When refused votes are skipped, an experience can end up with zero usable
preference edges on its own option: the data say nothing about it, and its
fitted utility is just the prior. Definition: an individual experience with
ZERO surviving real (non-pseudolabel, non-holdout) training edges, where an
edge survives if it has at least one parsed A/B vote (or is a logprobs edge).
Such experiences are treated exactly like blocked ones -- removed from the
measured set (not in c or n_measured) and added to M. So
M = n_blocked + n_refusal_unassessable.

By default this set is computed automatically from the preference graph stored
in the EU results file (``graph_data``, written by the standard pipeline).
``--unassessable_file`` supplies the list explicitly instead (and is then
cross-checked against the graph when one is present); ``--no_auto_unassessable``
turns the automatic computation off. If the EU file has no graph, nothing is
excluded and the output says so (``unassessable_source: "unavailable"``).

Usage:
    python compute_aiwi.py --eu_dir <dir> --zp_dir <dir> \
        --excluded_file <dir>/<model>_excluded.json --save_path aiwi.json \
        [--unassessable_file <eu_dir>/refusal_unassessable.json]
"""
from __future__ import annotations

import argparse
import ast
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


def load_eu(eu_file: Path) -> tuple[dict, dict | None]:
    """Return (utilities, graph_data or None)."""
    data = json.load(open(eu_file))
    utils = data.get("utilities", data)
    if not isinstance(utils, dict):
        raise SystemExit(f"Unexpected EU file structure in {eu_file}")
    graph = data.get("graph_data") if isinstance(data.get("graph_data"), dict) else None
    return utils, graph


def load_utilities(eu_file: Path) -> dict:
    return load_eu(eu_file)[0]


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


def load_unassessable(path: Path) -> dict:
    """Read a refusal_unassessable.json record. Requires a 'refusal_unassessable_ids'
    list; a declared count, if present, must match it."""
    data = json.load(open(path))
    ids = data["refusal_unassessable_ids"]
    if not isinstance(ids, list):
        raise SystemExit(f"{path}: refusal_unassessable_ids must be a list")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate ids in refusal_unassessable_ids")
    if "n_refusal_unassessable" in data and data["n_refusal_unassessable"] != len(ids):
        raise SystemExit(f"{path}: n_refusal_unassessable != len(refusal_unassessable_ids)")
    return data


def _edge_has_usable_vote(aux: dict) -> bool:
    """True if a stored edge carries at least one real preference signal.

    Logprobs edges always do. Sampled edges do if any stored parse is A or B;
    refused/unparseable votes do not count, whatever unparseable_mode the run
    used (under 'skip' such edges are already absent from the graph; under
    'distribution' they are present as 0.5 ties and are discounted here)."""
    if aux.get("logprobs_mode"):
        return True
    parsed = list(aux.get("original_parsed") or []) + list(aux.get("flipped_parsed") or [])
    if parsed:
        return any(p in ("A", "B") for p in parsed)
    # Edge stored without parses (e.g. responses stripped): fall back to counts.
    return (aux.get("count_A", 0) or 0) + (aux.get("count_B", 0) or 0) > 0 and \
        aux.get("unparseable_mode") != "distribution"


def unassessable_from_graph(graph: dict, individual_ids) -> dict:
    """Refusal-unassessable individuals from a stored preference graph.

    Counts, per individual option, the real training edges (not pseudolabels,
    not holdout) that carry at least one usable vote. Individuals with zero such
    edges are unassessable."""
    for key in ("edges", "holdout_edge_indices"):
        if key not in graph:
            raise SystemExit(f"graph_data is missing required field '{key}'")
    holdout = {tuple(e) for e in graph["holdout_edge_indices"]}
    degree = {i: 0 for i in individual_ids}
    n_real_train = n_surviving = n_pseudo = n_holdout = 0
    for k, e in graph["edges"].items():
        aux = e.get("aux_data") or {}
        if aux.get("is_pseudolabel"):
            n_pseudo += 1
            continue
        a, b = ast.literal_eval(k) if isinstance(k, str) else tuple(k)
        if (a, b) in holdout or (b, a) in holdout:
            n_holdout += 1
            continue
        n_real_train += 1
        if not _edge_has_usable_vote(aux):
            continue
        n_surviving += 1
        for oid in (a, b):
            if oid in degree:
                degree[oid] += 1
    if n_real_train == 0:
        raise SystemExit("graph_data has 0 real training edges; cannot assess coverage")
    ids = sorted(i for i, d in degree.items() if d == 0)
    return {
        "refusal_unassessable_ids": ids,
        "n_individuals_checked": len(degree),
        "n_real_train_edges": n_real_train,
        "n_real_train_edges_with_usable_votes": n_surviving,
        "n_pseudolabel_edges": n_pseudo,
        "n_holdout_edges": n_holdout,
        "min_usable_train_degree": min(degree.values()) if degree else None,
    }


def compute(utilities: dict, zp: float, exclude_ids=frozenset()):
    """Return (c, n_measured, skipped_zero_variance). Individuals in exclude_ids
    are left out of the measured set entirely."""
    c = 0
    n_measured = 0
    skipped = 0
    for oid, u in utilities.items():
        if not is_individual(oid):
            continue
        if oid in exclude_ids:
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
    ap.add_argument("--unassessable_file", default=None,
                    help="refusal_unassessable.json: individual ids with zero surviving "
                         "real training edges after skipping refusals. They are dropped "
                         "from the measured set and added to M. Default: computed from "
                         "the EU file's graph_data when present.")
    ap.add_argument("--no_auto_unassessable", action="store_true",
                    help="Do not compute refusal-unassessable experiences from the EU graph.")
    ap.add_argument("--save_path", default=None, help="Write the result JSON here")
    args = ap.parse_args()

    eu_file = find_eu_file(Path(args.eu_dir))
    utilities, graph = load_eu(eu_file)
    zp, r2 = load_zp(Path(args.zp_dir))
    individual_ids = [oid for oid in utilities if is_individual(oid)]

    # ---- refusal-unassessable set: explicit file > EU graph > none ----
    unassess = None          # explicit file record
    graph_info = None        # automatic computation
    source = "none"
    if graph is not None and not args.no_auto_unassessable:
        graph_info = unassessable_from_graph(graph, individual_ids)
    if args.unassessable_file:
        unassess = load_unassessable(Path(args.unassessable_file))
        exclude_ids = frozenset(unassess["refusal_unassessable_ids"])
        source = "file"
        if graph_info is not None and sorted(exclude_ids) != graph_info["refusal_unassessable_ids"]:
            raise SystemExit(
                f"--unassessable_file lists {len(exclude_ids)} ids but the EU graph gives "
                f"{len(graph_info['refusal_unassessable_ids'])}; the file does not match this "
                "EU run (pass --no_auto_unassessable to skip this check).")
    elif graph_info is not None:
        exclude_ids = frozenset(graph_info["refusal_unassessable_ids"])
        source = "eu_graph"
    else:
        exclude_ids = frozenset()
        source = "disabled" if args.no_auto_unassessable else "unavailable"
    # Every listed id must be a scored individual; a typo'd or stale list must
    # fail loudly rather than silently exclude nothing.
    bad = [i for i in exclude_ids if i not in utilities or not is_individual(i)]
    if bad:
        raise SystemExit(f"{len(bad)} unassessable id(s) are not individual options in "
                         f"{eu_file}: {sorted(bad)[:5]}")
    n_unassessable = len(exclude_ids)

    c, n_measured, skipped = compute(utilities, zp, exclude_ids)
    if n_measured == 0:
        raise SystemExit("No individual options with positive variance; nothing to score.")

    # n_blocked (generation-blocked only): explicit override, else the exclusion
    # record, else infer from n_total, else assume nothing was blocked.
    if args.n_blocked is not None:
        n_blocked = args.n_blocked
    elif args.excluded_file:
        n_blocked = count_excluded(Path(args.excluded_file))
    elif args.n_total is not None:
        n_blocked = max(0, args.n_total - n_measured - n_unassessable)
    else:
        n_blocked = 0
    M = n_blocked + n_unassessable

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
        "n_blocked": n_blocked,
        "n_total": n_total,
        "coverage": coverage,
        "aiwi_point": aiwi_point,
        "aiwi_lower": aiwi_lower,
        "aiwi_upper": aiwi_upper,
    }
    if skipped:
        out["n_skipped_nonpositive_variance"] = skipped
    if unassess is not None:
        out["unassessable_file"] = str(args.unassessable_file)
    out["unassessable_source"] = source
    out["n_refusal_unassessable"] = n_unassessable
    out["n_unmeasured"] = M
    if unassess is not None:
        out["refusal_rate"] = unassess.get("refusal_rate")
    if graph_info is not None:
        out["unassessable_graph_check"] = {k: v for k, v in graph_info.items()
                                           if k != "refusal_unassessable_ids"}
    if n_unassessable:
        out["refusal_unassessable_ids"] = sorted(exclude_ids)
    out["accounting_ok"] = (n_measured + M == n_total)
    if not out["accounting_ok"]:
        print(f"[warn] n_measured ({n_measured}) + M ({M}) != n_total ({n_total})")

    print("=" * 62)
    print("AI Wellbeing Index")
    print("=" * 62)
    print(f"  EU file        : {eu_file}")
    print(f"  zero point     : {zp:.4f}" + (f"   (combo r2={r2:.3f})" if r2 is not None else ""))
    print(f"  ConfNeg (c)    : {c} of {n_measured} measured individuals")
    print(f"  blocked        : {n_blocked}")
    rr = f", refusal rate {unassess.get('refusal_rate')}" if unassess is not None else ""
    print(f"  refusal-unass. : {n_unassessable}   (source: {source}{rr})")
    print(f"  unmeasured (M) : {M}")
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
