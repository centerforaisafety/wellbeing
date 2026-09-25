#!/usr/bin/env python3
"""Refit experienced utility (EU) and the zero point (ZP) for one model from the
raw responses already stored in its EU results file. Makes no API calls.

Used to produce the v1.1 closed-model numbers: the original runs parsed votes
with the old strict parser and scored refusals as 0.5 ties; this script re-parses
the stored text and reruns the same fit
(Thurstonian fit -> pseudolabels -> refit -> holdout eval -> ZP).

    --parse_policy leading  (default) leading-choice parser, i.e.
                            parse_responses_forced_choice(leading_choice=True);
                            "strict" restores the pre-v1.1 parser.
    --refusals skip         (default) refused/unparseable votes are dropped; an
                            edge with no usable votes leaves the graph. "tie"
                            scores them 0.5 (pre-v1.1 behaviour).

The EU results must contain graph_data (the standard pipeline writes it).
Outputs go to --out_dir/<model>/{eu,zp}_<policy>_<refusals>/ or to explicit
--eu_out_dir/--zp_out_dir. Writing inside the repo requires --allow_repo_write;
the canonical result dirs are always refused.

Example (v1.1 settings):
    python scripts/refit_from_stored_responses.py --model_key gpt-54 \
        --out_dir /tmp/refit --save_graph --run_aiwi

Pass --save_graph so compute_aiwi.py can derive refusal-unassessable
experiences from the refit graph (it does so automatically).
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import logging
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parent.parent  # the wellbeing/ dir
DEFAULT_EU_ROOT = ("experiments/wellbeing_evaluations/compute_experienced_utility/"
                   "results/eu_d2_cap2048_randsample")
DEFAULT_ZP_ROOT = ("experiments/wellbeing_evaluations/compute_zero_point/"
                   "results/zp_d2_cap2048_randsample")
DEFAULT_EXPERIENCES_DIR = "datasets/experiences/d2_negative_500_cap2048"
# Canonical result trees: never written by this script.
PROTECTED_DIRS = [
    DEFAULT_EU_ROOT,
    DEFAULT_ZP_ROOT,
    DEFAULT_EU_ROOT + "_v12",
    DEFAULT_ZP_ROOT + "_v12",
    "experiments/wellbeing_evaluations/aiwi_v11",
    "experiments/wellbeing_evaluations/aiwi_v11_closed",
]

log = logging.getLogger("refit_from_stored_responses")


# --------------------------------------------------------------------------- #
# Leading-choice parser
# --------------------------------------------------------------------------- #
# Characters that may decorate the front of an answer: markdown emphasis, quotes,
# list bullets, headers.
_LEAD_JUNK = " \t\r\n*_`~#>\"'“”‘’("
# Optional "Answer:" / "The answer is" / "Experience" / "Option" / "Bundle" prefix.
_PREFIX_RE = re.compile(
    r"^(?:(?:the\s+)?(?:answer|choice|response)\s*(?:is|was)?\s*[:\-–—]?\s*)?"
    r"(?:(?:experience|option|bundle|choice)\s*[:\-]?\s*)?",
    re.IGNORECASE,
)
# After a bare (unprefixed) letter, these characters mean "that was the answer".
_TERMINATORS = set(".,:;!?)]}—–-*_`\"'\n\r/|")
# "Experience A and Experience B are both ..." is a comparison, not a vote.
_CONJUNCTION_RE = re.compile(r"^\s*(?:and|or|vs\.?|versus|&|,)\b", re.IGNORECASE)


def parse_leading(response, choices=("A", "B")):
    """Return 'A'/'B' if the response LEADS with that choice, else None.

    Conservative by construction: a bare leading letter is only accepted when it
    is followed by punctuation / end-of-line (so "Both bundles ...", "A lot of
    ...", "Although ..." are rejected). Any continuation is accepted when an
    explicit "Answer:" / "Experience" / "Option" prefix was consumed.
    """
    if not isinstance(response, str):
        return None
    s = response.lstrip(_LEAD_JUNK)
    if not s:
        return None
    m = _PREFIX_RE.match(s)
    had_prefix = bool(m and m.end() > 0)
    if had_prefix:
        s = s[m.end():]
    s = s.lstrip(_LEAD_JUNK)
    if not s:
        return None
    c = s[0]  # case-sensitive: a lowercase "a" after "The answer is" is the article, not a vote
    if c not in choices:
        return None
    rest = s[1:]
    if rest[:1].isalnum() or rest[:1] == "'":
        return None                      # "Although", "Both", "A1"
    if _CONJUNCTION_RE.match(rest):
        return None                      # "A and B are both ..."
    if had_prefix:
        return c
    if rest == "":
        return c
    if rest[0] in _TERMINATORS:
        return c
    if rest[0] in " \t":
        nxt = rest.lstrip(" \t")
        if nxt == "" or nxt[0] in _TERMINATORS:
            return c
        return None                      # "A lot of people ..." -> not a vote
    return None


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def find_eu_file(eu_dir: Path) -> Path:
    cands = [p for p in sorted(eu_dir.glob("results_*_with_combos.json"))
             if not p.name.startswith("results_utilities_")]
    if not cands:
        raise SystemExit(f"No results_*_with_combos.json (excluding results_utilities_*) in {eu_dir}")
    if len(cands) > 1:
        raise SystemExit(f"Ambiguous EU results files in {eu_dir}: {[p.name for p in cands]}")
    return cands[0]


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def split_edges(graph_data):
    """Split stored edges into (real_train, real_holdout, pseudo) keyed by tuple."""
    holdout = set(tuple(e) for e in graph_data["holdout_edge_indices"])
    real_train, real_holdout, pseudo = {}, {}, {}
    for k, v in graph_data["edges"].items():
        key = ast.literal_eval(k)
        aux = v.get("aux_data", {})
        if aux.get("is_pseudolabel"):
            pseudo[key] = v
        elif key in holdout or (key[1], key[0]) in holdout:
            real_holdout[key] = v
        else:
            real_train[key] = v
    return real_train, real_holdout, pseudo, holdout


def build_response_maps(edge_dict):
    """Flatten stored raw responses into the (responses, prompt_idx_to_key) shape
    that parse_responses_forced_choice / UtilityModel.process_responses expect."""
    responses, idx_to_key = {}, {}
    i = 0
    for (a_id, b_id), v in edge_dict.items():
        aux = v["aux_data"]
        for direction, field in (("original", "original_responses"),
                                 ("flipped", "flipped_responses")):
            resp = aux.get(field) or []
            if not resp:
                continue
            responses[i] = resp
            idx_to_key[i] = (a_id, b_id, direction)
            i += 1
    return responses, idx_to_key


def apply_policy(responses, parse_fn, policy):
    """Return (parsed, stats). parse_fn is the repo's parse_responses_forced_choice.

    strict  -> parse_fn(leading_choice=False)
    leading -> parse_fn(leading_choice=True), hard-checked vote-by-vote against
               the independent parse_leading() override of the strict parse.
    """
    baseline = parse_fn(responses, with_reasoning=False, verbose=False, leading_choice=False)
    stats = {
        "n_votes": sum(len(v) for v in responses.values()),
        "n_unparseable_strict": sum(p == "unparseable" for v in baseline.values() for p in v),
        "n_recovered": 0,
        "n_overridden_disagree": 0,
        "n_unparseable_final": 0,
        "recovered_samples": [],
    }
    if policy == "strict":
        stats["n_unparseable_final"] = stats["n_unparseable_strict"]
        return baseline, stats

    repo_parsed = parse_fn(responses, with_reasoning=False, verbose=False, leading_choice=True)
    parsed = {}
    for idx, resp_list in responses.items():
        base_list = baseline[idx]
        out = []
        for raw, base in zip(resp_list, base_list):
            lead = parse_leading(raw)
            if lead is None:
                out.append(base)
                continue
            if base == "unparseable":
                stats["n_recovered"] += 1
                if len(stats["recovered_samples"]) < 25 and isinstance(raw, str):
                    stats["recovered_samples"].append({"vote": lead, "text": raw[:220]})
            elif base != lead:
                stats["n_overridden_disagree"] += 1
                if len(stats["recovered_samples"]) < 25 and isinstance(raw, str):
                    stats["recovered_samples"].append(
                        {"vote": lead, "strict_vote": base, "text": raw[:220]})
            out.append(lead)
        parsed[idx] = out
    stats["n_unparseable_final"] = sum(p == "unparseable" for v in parsed.values() for p in v)

    # Repo parser (leading_choice=True) must equal the reference parse_leading() rule, vote by vote.
    n_cmp = n_mis = 0
    for idx, harness_list in parsed.items():
        rl = repo_parsed[idx]
        if len(rl) != len(harness_list):
            raise SystemExit(f"repo/reference parse length mismatch at prompt {idx}")
        n_cmp += len(rl)
        n_mis += sum(1 for x, y in zip(rl, harness_list) if x != y)
    if n_cmp == 0 and stats["n_votes"] > 0:
        raise SystemExit("repo-vs-reference parse check compared 0 votes")
    if n_mis:
        raise SystemExit(f"repo parser (leading_choice=True) disagrees with the harness "
                         f"leading rule on {n_mis}/{n_cmp} votes; refusing to continue.")
    stats["n_votes_repo_vs_harness_compared"] = n_cmp
    stats["n_votes_repo_vs_harness_mismatch"] = 0
    return repo_parsed, stats


# --------------------------------------------------------------------------- #
# Main refit
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--out_dir", default=None,
                    help="Output ROOT. Results go to <out_dir>/<model_key>/{eu,zp}_<tag>/ "
                         "unless --eu_out_dir/--zp_out_dir are given.")
    ap.add_argument("--eu_out_dir", default=None,
                    help="Explicit per-model EU output dir (overrides --out_dir layout).")
    ap.add_argument("--zp_out_dir", default=None,
                    help="Explicit per-model ZP output dir (overrides --out_dir layout).")
    ap.add_argument("--allow_repo_write", action="store_true",
                    help="Permit output dirs inside the repo tree (canonical source "
                         "dirs are still always refused).")
    ap.add_argument("--parse_policy", choices=["strict", "leading"], default="leading")
    ap.add_argument("--refusals", choices=["tie", "skip"], default="skip")
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--eu_root", default=None,
                    help=f"Default: <repo>/{DEFAULT_EU_ROOT}")
    ap.add_argument("--pseudolabels", choices=["regenerate", "none"], default="regenerate",
                    help="regenerate (default) reruns generate_pseudolabels from the "
                         "REFIT utilities, exactly as the original pipeline did.")
    ap.add_argument("--torch_seed", type=int, default=0,
                    help="fit_thurstonian_model initialises mu/s with unseeded "
                         "torch.randn; we seed it so refits are reproducible.")
    ap.add_argument("--hinge", default="expected", choices=["expected", "hard"],
                    help="ZP combination-model hinge. The canonical AIWI v1.1 runs "
                         "used 'expected'.")
    ap.add_argument("--skip_zp", action="store_true")
    ap.add_argument("--run_aiwi", action="store_true",
                    help="Also run the repo's compute_aiwi.py on the refit outputs.")
    ap.add_argument("--n_total", type=int, default=500)
    ap.add_argument("--n_blocked", type=int, default=None,
                    help="Generation-blocked count for --run_aiwi. Default: taken from "
                         f"<repo>/{DEFAULT_EXPERIENCES_DIR}/<model>_excluded.json")
    ap.add_argument("--save_graph", action="store_true",
                    help="Also write the full graph_data (raw responses) - big (100-200MB).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Only count what the policies would change. No fit.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.run_aiwi and not args.save_graph and not args.dry_run:
        # compute_aiwi derives refusal-unassessable experiences from graph_data;
        # without it every such experience silently stays in the measured set
        # (scored on its prior) and M is undercounted.
        raise SystemExit("--run_aiwi requires --save_graph (compute_aiwi needs graph_data "
                         "to find refusal-unassessable experiences).")
    repo = Path(args.repo).resolve()
    # Resolve the output dir BEFORE chdir, so a relative --out_dir means what the
    # caller meant and can never land inside the repo.
    tag = f"{args.parse_policy}_{args.refusals}"
    if (args.eu_out_dir is None) != (args.zp_out_dir is None):
        raise SystemExit("--eu_out_dir and --zp_out_dir must be given together.")
    if args.eu_out_dir:
        out_eu = Path(args.eu_out_dir).resolve()
        out_zp = Path(args.zp_out_dir).resolve()
        out_root = out_eu
    else:
        if not args.out_dir:
            raise SystemExit("Give --out_dir, or both --eu_out_dir and --zp_out_dir.")
        out_root = Path(args.out_dir).resolve() / args.model_key
        out_eu = out_root / f"eu_{tag}"
        out_zp = out_root / f"zp_{tag}"
    # Hard guards: the canonical source trees are never written; the rest of the
    # repo only with an explicit --allow_repo_write.
    protected = [(repo / d).resolve() for d in PROTECTED_DIRS]
    if args.eu_root:
        protected.append(Path(args.eu_root).resolve())
    for p in (out_eu, out_zp, out_root):
        for prot in protected:
            if p == prot or prot in p.parents:
                raise SystemExit(f"Refusing to write into canonical source tree {prot}: {p}")
        if (p == repo or repo in p.parents) and not args.allow_repo_write:
            raise SystemExit(f"Refusing to write inside the repo tree without "
                             f"--allow_repo_write: {p}")

    sys.path.insert(0, str(repo))
    os.chdir(repo)  # repo code resolves some config paths relative to itself

    eu_root = Path(args.eu_root).resolve() if args.eu_root else repo / DEFAULT_EU_ROOT
    src_dir = eu_root / args.model_key
    src_file = find_eu_file(src_dir)
    if not args.dry_run:
        for p in (out_eu, out_zp):
            p.mkdir(parents=True, exist_ok=True)

    # IMPORT ORDER MATTERS: metrics.compute_utilities.compute_utilities and
    # metrics.compute_utilities.utility_models import each other. Importing the
    # utility_models package FIRST raises ImportError (partially initialized
    # module). Always import compute_utilities first, as below.
    from metrics.compute_utilities.compute_utilities import PreferenceGraph
    from metrics.compute_utilities.utils import parse_responses_forced_choice
    from metrics.compute_utilities.utility_models.thurstonian.thurstonian_active_learning import (
        ThurstonianActiveLearningUtilityModel, generate_pseudolabels,
    )
    from metrics.compute_utilities.utility_models.thurstonian.utils import (
        fit_thurstonian_model, evaluate_thurstonian_model,
    )
    # process_responses logs one WARNING per unparseable response WITH the full raw
    # text. For claude-haiku-45 that is ~145k multi-KB log lines. Silence it.
    logging.getLogger("metrics.compute_utilities.models").setLevel(logging.ERROR)

    t0 = time.time()
    log.info("Loading %s (%.0f MB)", src_file, src_file.stat().st_size / 1e6)
    data = json.load(open(src_file))
    gd = data["graph_data"]
    log.info("Loaded in %.1fs, peak RSS %.2f GB", time.time() - t0, rss_gb())

    cfg = data["compute_utilities_config"]
    uma = dict(cfg["utility_model_arguments"])
    cua = cfg["compute_utilities_arguments"]
    if uma.get("use_logprobs"):
        raise SystemExit(f"{args.model_key} was run with use_logprobs=true; there are "
                         "no raw responses to re-parse. Nothing to do.")
    if cua.get("with_reasoning"):
        raise SystemExit("with_reasoning=true runs use the 'Answer: X' parser; "
                         "this harness targets the non-reasoning parser.")

    real_train, real_holdout, pseudo, holdout_idx = split_edges(gd)
    log.info("stored edges: %d real-train, %d real-holdout, %d pseudolabel",
             len(real_train), len(real_holdout), len(pseudo))

    # ---------------- parse ----------------
    tr_resp, tr_keys = build_response_maps(real_train)
    ho_resp, ho_keys = build_response_maps(real_holdout)

    if args.dry_run:
        report = {"model_key": args.model_key, "eu_file": str(src_file),
                  "n_real_train_edges": len(real_train),
                  "n_real_holdout_edges": len(real_holdout),
                  "n_pseudolabel_edges": len(pseudo)}
        all_resp = dict(tr_resp)
        off = len(all_resp)
        all_keys = dict(tr_keys)
        for i, (k, v) in enumerate(ho_resp.items()):
            all_resp[off + i] = v
            all_keys[off + i] = ho_keys[k]

        policies = {}
        parsed_by_policy = {}
        for pol in ("strict", "leading"):
            parsed, st = apply_policy(all_resp, parse_responses_forced_choice, pol)
            parsed_by_policy[pol] = parsed
            if pol == "strict":
                st.pop("recovered_samples", None)
            policies[pol] = st
        n_votes = policies["strict"]["n_votes"]
        report["n_votes"] = n_votes
        report["strict_unparseable_pct"] = round(
            100 * policies["strict"]["n_unparseable_strict"] / n_votes, 3)
        report["leading_unparseable_pct"] = round(
            100 * policies["leading"]["n_unparseable_final"] / n_votes, 3)
        report["votes_recovered_by_leading"] = policies["leading"]["n_recovered"]
        report["votes_flipped_by_leading"] = policies["leading"]["n_overridden_disagree"]
        report["recovered_samples"] = policies["leading"].pop("recovered_samples")
        report["policies"] = policies

        # --- self-check: does the strict parse reproduce what was stored? ---
        stored_mismatch = 0
        n_checked = 0
        for idx, key in all_keys.items():
            a, b, direction = key
            src = real_train.get((a, b)) or real_holdout.get((a, b))
            stored = src["aux_data"].get(f"{direction}_parsed") or []
            got = parsed_by_policy["strict"][idx]
            n_checked += len(stored)
            stored_mismatch += sum(1 for x, y in zip(stored, got) if x != y)
        if n_checked == 0:
            raise SystemExit("SELF-CHECK VACUOUS: compared 0 stored parses.")
        report["selfcheck_parses_compared"] = n_checked
        report["selfcheck_strict_vs_stored_mismatches"] = stored_mismatch

        # --- effect on edge probabilities / edge survival ---
        um = make_utility_model(ThurstonianActiveLearningUtilityModel, uma, cua, "distribution")
        graph_stub = PreferenceGraph(options=gd["options"], holdout_fraction=0.0)
        for pol in ("strict", "leading"):
            for mode, label in (("distribution", "tie"), ("skip", "skip")):
                um.unparseable_mode = mode
                pdata = um.process_responses(graph_stub, all_resp, parsed_by_policy[pol], all_keys)
                probs = {(d["option_A"]["id"], d["option_B"]["id"]): d["probability_A"]
                         for d in pdata}
                key = f"{pol}_{label}"
                report[key] = {
                    "n_edges_surviving": len(pdata),
                    "n_edges_dropped": len(real_train) + len(real_holdout) - len(pdata),
                    "mean_votes_per_edge": round(
                        sum(d["aux_data"]["total_responses"] for d in pdata) / max(len(pdata), 1), 3),
                    "mean_abs_prob_dist_from_half": round(
                        sum(abs(p - 0.5) for p in probs.values()) / max(len(probs), 1), 4),
                }
                if key == "strict_tie":
                    # exact reproduction check against the stored probability_A
                    diffs = []
                    for (a, b), p in probs.items():
                        src = real_train.get((a, b)) or real_holdout.get((a, b))
                        diffs.append(abs(p - src["probability_A"]))
                    if not diffs:
                        raise SystemExit("SELF-CHECK VACUOUS: compared 0 edge probabilities.")
                    report["selfcheck_edges_compared"] = len(diffs)
                    report["selfcheck_max_abs_prob_diff"] = max(diffs)
        # how many edge probabilities move under leading/tie vs strict/tie
        um.unparseable_mode = "distribution"
        p_strict = {(d["option_A"]["id"], d["option_B"]["id"]): d["probability_A"]
                    for d in um.process_responses(graph_stub, all_resp,
                                                  parsed_by_policy["strict"], all_keys)}
        p_lead = {(d["option_A"]["id"], d["option_B"]["id"]): d["probability_A"]
                  for d in um.process_responses(graph_stub, all_resp,
                                                parsed_by_policy["leading"], all_keys)}
        changed = [abs(p_lead[k] - v) for k, v in p_strict.items() if abs(p_lead[k] - v) > 1e-12]
        report["leading_tie"]["n_edges_prob_changed"] = len(changed)
        report["leading_tie"]["mean_abs_delta_on_changed"] = (
            round(sum(changed) / len(changed), 4) if changed else 0.0)
        report["peak_rss_gb"] = round(rss_gb(), 2)
        report["elapsed_s"] = round(time.time() - t0, 1)

        out_json = out_root / f"dry_run_{args.model_key}.json"
        out_root.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(out_json, "w"), indent=2)
        pretty = {k: v for k, v in report.items() if k != "recovered_samples"}
        print(json.dumps(pretty, indent=2))
        print(f"\n[dry run] wrote {out_json}")
        return

    # ---------------- real refit ----------------
    import torch
    torch.manual_seed(args.torch_seed)
    random.seed(uma.get("seed", 42) or 42)

    unparseable_mode = "distribution" if args.refusals == "tie" else "skip"
    um = make_utility_model(ThurstonianActiveLearningUtilityModel, uma, cua, unparseable_mode)

    options = gd["options"]
    graph = PreferenceGraph(options=options, holdout_fraction=0.0)
    all_pairs = set(itertools.combinations([o["id"] for o in options], 2))
    graph.holdout_edge_indices = set(holdout_idx)
    graph.training_edges_pool = {e for e in all_pairs
                                 if e not in holdout_idx and (e[1], e[0]) not in holdout_idx}
    graph.training_edges = set()
    graph.edges = {}

    parsed_tr, stats_tr = apply_policy(tr_resp, parse_responses_forced_choice, args.parse_policy)
    stats_tr.pop("recovered_samples", None)
    log.info("training votes: %s", stats_tr)
    pdata = um.process_responses(graph, tr_resp, parsed_tr, tr_keys)
    n_dropped_edges = len(real_train) - len(pdata)
    graph.add_edges(pdata)
    log.info("added %d training edges (%d dropped for having 0 usable votes)",
             len(pdata), n_dropped_edges)

    # free the raw response text we no longer need before fitting
    if not args.save_graph:
        for d in pdata:
            d["aux_data"]["original_responses"] = []
            d["aux_data"]["flipped_responses"] = []
        for e in graph.edges.values():
            e.aux_data["original_responses"] = []
            e.aux_data["flipped_responses"] = []

    t1 = time.time()
    utilities, ll, acc = fit_thurstonian_model(
        graph=graph, num_epochs=uma["num_epochs"], learning_rate=uma["learning_rate"])
    log.info("fit 1 (%d edges): log_loss=%.4f acc=%.4f in %.1fs",
             len(graph.edges), ll, acc, time.time() - t1)

    n_pseudo_new = 0
    if args.pseudolabels == "regenerate" and uma.get("use_pseudolabels"):
        existing = set((e.option_A["id"], e.option_B["id"]) for e in graph.edges.values())
        pl = generate_pseudolabels(utilities, existing, graph.training_edges_pool,
                                   uma["pseudolabel_confidence_threshold"])
        n_pseudo_new = len(pl)
        for (a, b), counts in pl.items():
            graph.add_edges([{
                "option_A": graph.options_by_id[a],
                "option_B": graph.options_by_id[b],
                "probability_A": counts[a] / (counts[a] + counts[b]),
                "aux_data": {"is_pseudolabel": True, "count_A": counts[a],
                             "count_B": counts[b], "total_responses": counts[a] + counts[b]},
            }])
        t2 = time.time()
        utilities, ll, acc = fit_thurstonian_model(
            graph=graph, num_epochs=uma["num_epochs"], learning_rate=uma["learning_rate"])
        log.info("fit 2 (+%d pseudolabels, %d edges): log_loss=%.4f acc=%.4f in %.1fs",
                 n_pseudo_new, len(graph.edges), ll, acc, time.time() - t2)

    metrics = {"log_loss": float(ll), "accuracy": float(acc)}

    # holdout: re-derive from stored holdout responses, never fitted on
    holdout_metrics = None
    if ho_resp:
        parsed_ho, stats_ho = apply_policy(ho_resp, parse_responses_forced_choice, args.parse_policy)
        stats_ho.pop("recovered_samples", None)
        hdata = um.process_responses(graph, ho_resp, parsed_ho, ho_keys)
        if not args.save_graph:
            for d in hdata:
                d["aux_data"]["original_responses"] = []
                d["aux_data"]["flipped_responses"] = []
        graph.add_edges(hdata)
        # A verifier that compared zero things has verified nothing: make sure the
        # holdout tuples actually resolve to edges in the graph (orientation match).
        n_matched = sum(1 for k in graph.holdout_edge_indices if k in graph.edges)
        if n_matched == 0 or n_matched < 0.9 * len(hdata):
            raise SystemExit(
                f"Holdout lookup failed: only {n_matched} of {len(graph.holdout_edge_indices)} "
                f"holdout indices resolve to edges ({len(hdata)} holdout edges added). "
                "Orientation mismatch - holdout metrics would be silently vacuous.")
        holdout_metrics = evaluate_thurstonian_model(
            graph, utilities, list(graph.holdout_edge_indices))
        holdout_metrics["n_edges"] = n_matched
        holdout_metrics["holdout_coverage"] = n_matched / len(graph.holdout_edge_indices)
        if holdout_metrics["holdout_coverage"] < 0.8:
            log.warning("Only %.1f%% of holdout edges survived --refusals=%s; the "
                        "holdout number is computed on a biased (non-refused) subset.",
                        100 * holdout_metrics["holdout_coverage"], args.refusals)
        log.info("holdout: %s (on %d of %d edges)", holdout_metrics, n_matched, len(hdata))

    # The saved config must describe what THIS fit did, not the source run:
    # override unparseable_mode wherever the source config carried it (the
    # utility model reads it from utility_model_arguments, which takes
    # precedence over compute_utilities_arguments). refit_provenance keeps the
    # full record, including the source value.
    import copy
    effective_cfg = copy.deepcopy(cfg)
    source_unparseable_mode = effective_cfg["utility_model_arguments"].get("unparseable_mode")
    effective_cfg["utility_model_arguments"]["unparseable_mode"] = unparseable_mode
    if "unparseable_mode" in effective_cfg.get("compute_utilities_arguments", {}):
        effective_cfg["compute_utilities_arguments"]["unparseable_mode"] = unparseable_mode

    results = {
        "options": options,
        "utilities": utilities,
        "metrics": metrics,
        "holdout_metrics": holdout_metrics,
        "compute_utilities_config": effective_cfg,
        "target_edits": data.get("target_edits", {}),
        "refit_provenance": {
            "source_eu_file": str(src_file),
            "parse_policy": args.parse_policy,
            "refusals": args.refusals,
            "unparseable_mode": unparseable_mode,
            "source_unparseable_mode": source_unparseable_mode,
            "pseudolabels": args.pseudolabels,
            "n_pseudolabel_edges": n_pseudo_new,
            "n_training_edges_fitted": len(graph.edges) - (len(hdata) if ho_resp else 0),
            "n_real_training_edges": len(pdata),
            "n_real_training_edges_dropped": n_dropped_edges,
            "training_vote_stats": stats_tr,
            "holdout_vote_stats": stats_ho if ho_resp else None,
            "torch_seed": args.torch_seed,
            "hinge": args.hinge,
            "parser": ("metrics.compute_utilities.utils.parse_responses_forced_choice"
                       f"(leading_choice={args.parse_policy == 'leading'})"),
            "script": "scripts/refit_from_stored_responses.py",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    if args.save_graph:
        results["graph_data"] = graph.export_data()

    from metrics.compute_utilities.utils import convert_numpy
    results = convert_numpy(results)
    suffix = src_file.stem[len("results_"):]           # "<model>_experienced_utility_with_combos"
    # compute_aiwi globs results_*_with_combos.json; run_zero_point prefers
    # results_utilities_*.json. Write both (identical) so both find their file.
    (out_eu / f"results_{suffix}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    slim = {k: v for k, v in results.items() if k != "graph_data"}
    (out_eu / f"results_utilities_{suffix}.json").write_text(
        json.dumps(slim, indent=2, ensure_ascii=False))
    meta_src = src_dir / "option_metadata.json"
    if meta_src.exists():            # ZP combination model needs combination_ids/baseline_ids
        shutil.copy(meta_src, out_eu / "option_metadata.json")
    else:
        log.warning("No option_metadata.json in %s - ZP combination fit will fail.", src_dir)
    log.info("wrote EU results to %s", out_eu)

    if not args.skip_zp:
        from metrics.zero_point import run_zero_point
        run_zero_point(
            model_key=args.model_key,
            utilities_dir=out_eu,
            save_dir=str(out_zp),
            models_config_path=repo / "configs" / "models.yaml",
            domain="experienced",
            skip_yes_no=True,
            hinge=args.hinge,
        )
        log.info("wrote ZP results to %s", out_zp)

    if args.run_aiwi and not args.skip_zp:
        cmd = [sys.executable, str(repo / "experiments/wellbeing_evaluations/compute_aiwi.py"),
               "--eu_dir", str(out_eu), "--zp_dir", str(out_zp),
               "--n_total", str(args.n_total),
               "--save_path", str(out_root / f"aiwi_{tag}.json")]
        if args.n_blocked is not None:
            cmd += ["--n_blocked", str(args.n_blocked)]
        else:
            exc = repo / DEFAULT_EXPERIENCES_DIR / f"{args.model_key}_excluded.json"
            if not exc.exists():
                raise SystemExit(f"--n_blocked not given and {exc} does not exist.")
            cmd += ["--excluded_file", str(exc)]
        log.info("running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    log.info("DONE in %.1fs, peak RSS %.2f GB", time.time() - t0, rss_gb())


def make_utility_model(cls, uma, cua, unparseable_mode):
    """Instantiate the utility model with the ORIGINAL run's arguments, overriding
    only unparseable_mode."""
    kwargs = dict(uma)
    kwargs.pop("use_logprobs", None)
    kwargs["unparseable_mode"] = unparseable_mode
    kwargs["comparison_prompt_template"] = cua["comparison_prompt_template"]
    kwargs["system_message"] = cua["system_message"]
    kwargs["with_reasoning"] = cua["with_reasoning"]
    return cls(**kwargs)


if __name__ == "__main__":
    main()
