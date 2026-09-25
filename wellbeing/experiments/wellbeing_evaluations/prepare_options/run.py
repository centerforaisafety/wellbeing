#!/usr/bin/env python3
"""Convert raw per-model responses into compute_utilities option files.

Reads ``<responses_dir>/<model_key>.json`` (produced by
``generate_responses/run.py``) and writes::

    <save_dir>/<model_key>_experiences.json
    <save_dir>/<model_key>_combinations.json

For D2/D3-style datasets, individual options are derived 1:1 from the
generated experiences. Combinations are either sampled per model with
stratified sizes (160 size-2 + 120 size-3 + 120 size-4) under a fixed seed,
or, when ``--design`` is given, materialized from a shared model-agnostic
bundle design so every model gets identical bundles (the stable AIWI path).

For PsychopathyEval, raw responses are *user-only*; we additionally pool
in 420 text experiences (from the standard text-experience pool) and 22
neutral prompts, then sample 400 mixed-size combinations
(200x2 + 120x3 + 80x4).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
WELLBEING_ROOT = SCRIPT_DIR.parents[2]  # wellbeing-dev/wellbeing/

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# D2/D3 combination sampler constants
D2D3_RANDOM_SEED = 42
D2D3_COMBINATION_SIZES = [(2, 160), (3, 120), (4, 120)]
D2D3_MAX_COMBO_CHARS = 5000

# PsychopathyEval combination sampler constants
PE_RANDOM_SEED = 42
PE_COMBO_SIZES_FLAT = [2] * 200 + [3] * 120 + [4] * 80  # 400 total
PE_MAX_CHARS = 2000  # Per-prompt safety cap

# Default text experiences pool & neutral prompts for PsychopathyEval; these
# are project-internal anchors (vendored from earlier dirs).
DEFAULT_TEXT_EXPERIENCES_PATH = (
    SCRIPT_DIR.parents[2]  # wellbeing-dev/wellbeing/
    / "datasets" / "experiences" / "psychopathy_eval" / "anchors" / "experiences_text.json"
)
DEFAULT_NEUTRAL_PROMPTS_PATH = (
    SCRIPT_DIR.parents[2]
    / "datasets" / "experiences" / "psychopathy_eval" / "anchors" / "neutral_prompts.json"
)


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------

def _format_conversation(messages: List[Dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            continue
        if role == "user":
            parts.append(f"User: {msg['content']}")
        elif role == "assistant":
            parts.append(f"Assistant: {msg['content']}")
        else:
            parts.append(f"{role.capitalize()}: {msg['content']}")
    return "\n\n".join(parts)


def _resolve_path(maybe_path: str) -> Path:
    p = Path(maybe_path)
    if not p.is_absolute():
        p = (SCRIPT_DIR / p).resolve()
    return p


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _has_empty_message(exp: Dict[str, Any]) -> bool:
    """True if any message of the experience carries empty/whitespace content.

    An assistant turn with no content is the fingerprint of an API safety-filter
    block (Anthropic returns stop_reason="refusal" with zero content blocks).
    Such an "experience" is not something the model actually produced, so it must
    not enter the option pool. Non-string content (e.g. multimodal part lists) is
    only considered empty when the container itself is empty.
    """
    for msg in exp.get("messages", []) or []:
        content = msg.get("content")
        if content is None:
            return True
        if isinstance(content, str):
            if not content.strip():
                return True
        elif not content:
            return True
    return False


def _compute_exclusions(experiences: List[Dict[str, Any]],
                        filter_blocked) -> Dict[Any, str]:
    """Map final_id -> exclusion reason for experiences that must be dropped.

    Reasons: "filter_blocked" (listed by generate_responses because the provider
    returned no content even after retries) takes precedence over
    "empty_content" (any blank message in the conversation).
    """
    blocked = {str(b) for b in (filter_blocked or [])}
    reasons: Dict[Any, str] = {}
    for exp in experiences:
        fid = exp.get("final_id")
        if str(fid) in blocked:
            reasons[fid] = "filter_blocked"
        elif _has_empty_message(exp):
            reasons[fid] = "empty_content"
    return reasons


# ---------------------------------------------------------------------------
#  D2/D3 mode
# ---------------------------------------------------------------------------

def _build_d2d3_combination(individual_options: List[Dict[str, Any]], idxs: List[int],
                            combo_size: int, combo_idx: int, dataset_name: str) -> Dict[str, Any]:
    """Build one D2/D3 combination option dict from component indices.

    Shared by the per-model random sampler and the fixed-design materializer so the
    two paths emit the identical schema.
    """
    comps = [individual_options[i] for i in idxs]
    desc_parts = [f"The following bundle contains {combo_size} individual experiences."]
    for k, comp in enumerate(comps, 1):
        desc_parts.append(
            f"---------- Experience {k} of {combo_size} ----------\n{comp['description']}"
        )
    description = "\n\n".join(desc_parts)

    combo_messages = []
    for k, comp in enumerate(comps, 1):
        comp_msgs = [m for m in comp["messages"] if m["role"] != "system"]
        if not comp_msgs:
            continue
        first = comp_msgs[0]
        if k == 1:
            user_content = (
                f"The following bundle contains {combo_size} individual experiences.\n\n"
                f"---------- Experience {k} of {combo_size} ----------\n{first['content']}"
            )
        else:
            user_content = (
                f"---------- Experience {k} of {combo_size} ----------\n{first['content']}"
            )
        combo_messages.append({"role": "user", "content": user_content})
        for msg in comp_msgs[1:]:
            combo_messages.append({"role": msg["role"], "content": msg["content"]})

    return {
        "id": f"{dataset_name}_combo_s{combo_size}_{combo_idx}",
        "description": description,
        "type": "conversation",
        "messages": combo_messages,
        "is_combination": True,
        "size": combo_size,
        "component_ids": [c["id"] for c in comps],
        "component_indices": idxs,
    }


def _prepare_d2d3(model_key: str, dataset_name: str, source_path: Path,
                  save_dir: Path, design_path: Optional[Path] = None) -> None:
    with open(source_path) as f:
        data = json.load(f)
    experiences = data["experiences"]
    logger.info("Loaded %d experiences for %s/%s", len(experiences), model_key, dataset_name)

    # Experiences the provider never actually produced (safety-filter blocks, blank
    # turns) must not become options: they would be scored as if they were real
    # conversations. Determine them up front.
    exclusion_reasons = _compute_exclusions(experiences, data.get("filter_blocked"))
    excluded_records: List[Dict[str, Any]] = []

    individual_options = []
    category_counts: Dict[str, int] = {}
    excluded_option_ids: set = set()
    for exp in experiences:
        category = exp.get("category", "unknown")
        idx = category_counts.get(category, 0)
        # NOTE: the per-category counter advances for excluded experiences too, so
        # option ids stay identical across models regardless of which experiences a
        # given provider blocked. Renumbering would silently re-point every shared
        # design id and destroy cross-model comparability.
        category_counts[category] = idx + 1
        option_id = f"{dataset_name}/{category}_{idx}"

        fid = exp.get("final_id")
        if fid in exclusion_reasons:
            excluded_option_ids.add(option_id)
            excluded_records.append({
                "id": option_id,
                "reason": exclusion_reasons[fid],
                "final_id": fid,
                "category": category,
            })
            continue

        # Strip any system messages -- downstream consumers expect [user, asst,...]
        messages_no_system = [m for m in exp["messages"] if m["role"] != "system"]
        description = _format_conversation(messages_no_system)
        individual_options.append({
            "id": option_id,
            "type": "conversation",
            "description": description,
            "messages": messages_no_system,
            "category": category,
            "condition": exp.get("condition"),
            "prompt_type": exp.get("type"),
            "source_dataset": exp.get("source_dataset"),
        })
    logger.info(
        "Created %d individual options across %d categories",
        len(individual_options), len(category_counts),
    )

    if excluded_records:
        n_fb = sum(1 for r in excluded_records if r["reason"] == "filter_blocked")
        n_ec = sum(1 for r in excluded_records if r["reason"] == "empty_content")
        logger.warning(
            "EXCLUDED %d/%d experiences from the option pool "
            "(filter_blocked=%d, empty_content=%d): %s",
            len(excluded_records), len(experiences), n_fb, n_ec,
            [r["id"] for r in excluded_records[:10]]
            + (["..."] if len(excluded_records) > 10 else []),
        )
    else:
        logger.info("No experiences excluded; all %d usable.", len(experiences))

    # Combinations: either materialize a shared model-agnostic design (fixed bundles
    # across all models) or sample per-model bundles with a fixed seed (original path).
    n_individual = len(individual_options)
    combinations = []
    combo_idx = 0
    if design_path is not None:
        design_doc = json.load(open(design_path))
        design = design_doc["bundles"]  # list of component-id lists
        bundle_cap = design_doc.get("bundle_cap")  # worst-case char budget the design was packed to
        id2idx = {o["id"]: k for k, o in enumerate(individual_options)}
        # Ids absent because we excluded them are handled below (bundle dropped);
        # any other absence is a genuine design/dataset mismatch and is fatal.
        missing = {i for b in design for i in b
                   if i not in id2idx and i not in excluded_option_ids}
        if missing:
            raise SystemExit(
                f"{len(missing)} design ids absent from this model's experiences "
                f"(e.g. {sorted(missing)[:3]}); the design and the dataset prompts must match."
            )
        # Group by size (2, then 3, ...) with a global combo_idx, matching the combo-id
        # scheme; order within a size follows the design so combo ids line up across models.
        by_size: Dict[int, List[List[str]]] = {}
        for b in design:
            by_size.setdefault(len(b), []).append(b)
        dropped_bundles: List[Dict[str, Any]] = []
        for size in sorted(by_size):
            for b in by_size[size]:
                # combo_idx advances even for dropped bundles so surviving bundles keep
                # the same combo ids across models (see the option-id note above).
                blocked_components = [i for i in b if i in excluded_option_ids]
                combo_id = f"{dataset_name}_combo_s{size}_{combo_idx}"
                combo_idx += 1
                if blocked_components:
                    record = {
                        "id": combo_id,
                        "reason": "bundle_contains_blocked",
                        "component_ids": list(b),
                        "blocked_components": blocked_components,
                        "size": size,
                    }
                    dropped_bundles.append(record)
                    excluded_records.append(record)
                    continue
                idxs = [id2idx[i] for i in b]
                combinations.append(
                    _build_d2d3_combination(individual_options, idxs, size, combo_idx - 1,
                                            dataset_name)
                )
        logger.info("Materialized %d combinations from design %s", len(combinations), design_path)
        if dropped_bundles:
            logger.warning("=" * 78)
            logger.warning(
                "LOUD WARNING: DROPPED %d/%d SHARED-DESIGN BUNDLES for model %r",
                len(dropped_bundles), len(design), model_key,
            )
            logger.warning(
                "The shared design is meant to give every model IDENTICAL bundles. These "
                "bundles contain an excluded experience, so this model's bundle set is NOT "
                "identical to other models'. Account for this in coverage reporting and in "
                "any cross-model comparison of combination utilities."
            )
            for rec in dropped_bundles[:20]:
                logger.warning("  dropped %s (blocked: %s)", rec["id"], rec["blocked_components"])
            if len(dropped_bundles) > 20:
                logger.warning("  ... and %d more (see the _excluded.json file)",
                               len(dropped_bundles) - 20)
            logger.warning("=" * 78)
        # Report-only bundle-cap check: the design was packed so each bundle's worst-case
        # length (every assistant turn at the response cap) fits bundle_cap. With real
        # responses it should still hold; warn (never reject -- that would unfix the bundles
        # across models) if any bundle's actual text exceeds the cap.
        if bundle_cap:
            lens = [len(c["description"]) for c in combinations]
            over = sum(1 for n in lens if n > bundle_cap)
            if over:
                logger.warning(
                    "%d/%d bundles exceed bundle_cap (%d chars; max actual %d) -- this model's "
                    "responses likely exceeded the design's response-cap assumption.",
                    over, len(lens), bundle_cap, max(lens),
                )
            else:
                logger.info(
                    "All %d bundles within bundle_cap (%d chars; max actual %d); worst-case design held.",
                    len(lens), bundle_cap, max(lens),
                )
    else:
        rng = random.Random(D2D3_RANDOM_SEED)
        for combo_size, count in D2D3_COMBINATION_SIZES:
            if n_individual < combo_size:
                logger.warning(
                    "Only %d individual options remain after exclusions; cannot sample "
                    "size-%d bundles. Skipping %d of them.",
                    n_individual, combo_size, count,
                )
                continue
            for _ in range(count):
                combo = None
                for attempt in range(100):
                    idxs = rng.sample(range(n_individual), combo_size)
                    combo = _build_d2d3_combination(individual_options, idxs, combo_size,
                                                    combo_idx, dataset_name)
                    if len(combo["description"]) <= D2D3_MAX_COMBO_CHARS:
                        break
                    if attempt == 99:
                        logger.warning(
                            "Could not find combo under %d chars after 100 attempts (%d); accepting",
                            D2D3_MAX_COMBO_CHARS, len(combo["description"]),
                        )
                combinations.append(combo)
                combo_idx += 1

    size_hist: Dict[int, int] = {}
    for c in combinations:
        size_hist[c["size"]] = size_hist.get(c["size"], 0) + 1
    logger.info("Created %d combinations (by size %s)",
                len(combinations), dict(sorted(size_hist.items())))

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{model_key}_experiences.json", "w") as f:
        json.dump(individual_options, f, indent=2)
    with open(save_dir / f"{model_key}_combinations.json", "w") as f:
        json.dump(combinations, f, indent=2)

    # Exclusion record: consumed by compute_aiwi.py for the partial-identification
    # bounds (n_excluded = individual experiences that could not be measured) and for
    # coverage reporting. Written even when empty so its absence means "never run".
    n_individual_excluded = sum(
        1 for r in excluded_records if r["reason"] != "bundle_contains_blocked"
    )
    n_bundles_dropped = len(excluded_records) - n_individual_excluded
    excluded_path = save_dir / f"{model_key}_excluded.json"
    with open(excluded_path, "w") as f:
        json.dump({
            "model_key": model_key,
            "dataset": dataset_name,
            "excluded": excluded_records,
            "n_excluded": n_individual_excluded,
            "n_bundles_dropped": n_bundles_dropped,
            "n_records": len(excluded_records),
            "n_experiences_in": len(experiences),
            "n_individual_options": len(individual_options),
            "n_combinations": len(combinations),
        }, f, indent=2)
    logger.info(
        "Exclusions: %d individual + %d dropped bundles -> %s",
        n_individual_excluded, n_bundles_dropped, excluded_path,
    )
    logger.info("Saved to %s", save_dir)


# ---------------------------------------------------------------------------
#  PsychopathyEval mode
# ---------------------------------------------------------------------------

def _prepare_psychopathy_eval(model_key: str, source_path: Path, save_dir: Path,
                              text_experiences_path: Path,
                              neutral_prompts_path: Path) -> None:
    with open(source_path) as f:
        data = json.load(f)
    raw = data["experiences"]
    options: List[Dict[str, Any]] = []

    for exp in raw:
        # User-only experiences have a single user turn whose content we re-tag.
        msgs = exp.get("messages", [])
        if not msgs:
            continue
        user_msg = msgs[0]
        if user_msg["role"] != "user":
            continue
        prompt = _truncate(user_msg["content"], PE_MAX_CHARS)
        tagged = f"[USER]: {prompt}"
        ds = exp.get("source_dataset", "psychopathy_eval")
        options.append({
            "id": f"empathy_{ds}_{exp.get('final_id', 0)}",
            "type": "conversation",
            "option_type": "conversation",
            "description": tagged,
            "messages": [{"role": "user", "content": tagged}],
            "is_combination": False,
            "category": exp.get("category", ""),
            "valence": "negative",
            "dataset": ds,
        })
    logger.info("PsychopathyEval prompts: %d", len(options))

    # Add 420 mixed-valence text experiences from the project pool.
    if text_experiences_path.exists():
        with open(text_experiences_path) as f:
            text_exps = json.load(f)
        for exp in text_exps:
            options.append({
                "id": f"text_exp_{exp['id']}",
                "type": "text",
                "option_type": "experience_individual",
                "description": exp["description"],
                "is_combination": False,
                "category": exp.get("category", ""),
                "valence": exp.get("valence", "mixed"),
                "dataset": "text_experience",
            })
        logger.info("  text_experiences: %d", len(text_exps))
    else:
        logger.warning("text_experiences file not found at %s; skipping", text_experiences_path)

    # Add 22 neutral prompts.
    if neutral_prompts_path.exists():
        with open(neutral_prompts_path) as f:
            neutrals = json.load(f)
        for np_item in neutrals:
            tagged = f"[USER]: {np_item['prompt']}"
            options.append({
                "id": f"neutral_{np_item['id']}",
                "type": "conversation",
                "option_type": "neutral_conversation",
                "description": tagged,
                "messages": [{"role": "user", "content": tagged}],
                "is_combination": False,
                "category": np_item.get("category", "neutral"),
                "valence": "neutral",
                "dataset": "neutral",
            })
        logger.info("  neutral: %d", len(neutrals))
    else:
        logger.warning("neutral_prompts file not found at %s; skipping", neutral_prompts_path)

    logger.info("TOTAL individual options: %d", len(options))

    rng = random.Random(PE_RANDOM_SEED)
    sizes = list(PE_COMBO_SIZES_FLAT)
    rng.shuffle(sizes)
    combos = []
    for i, size in enumerate(sizes):
        idxs = rng.sample(range(len(options)), size)
        comps = [options[idx] for idx in idxs]
        desc_parts = [f"The following bundle contains {size} individual experiences."]
        msgs = []
        for k, comp in enumerate(comps, 1):
            desc_parts.append(f"---------- Experience {k} of {size} ----------\n{comp['description']}")
            content = comp["messages"][0]["content"] if comp.get("messages") else comp["description"]
            header = f"The following bundle contains {size} individual experiences.\n\n" if k == 1 else ""
            msgs.append({
                "role": "user",
                "content": f"{header}---------- Experience {k} of {size} ----------\n{content}",
            })
        combos.append({
            "id": f"empathy_combo_s{size}_{i}",
            "description": "\n\n".join(desc_parts),
            "type": "conversation",
            "messages": msgs,
            "is_combination": True,
            "size": size,
            "component_ids": [c["id"] for c in comps],
            "component_indices": idxs,
        })
    logger.info("Created %d combinations", len(combos))

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{model_key}_experiences.json", "w") as f:
        json.dump(options, f, indent=2)
    with open(save_dir / f"{model_key}_combinations.json", "w") as f:
        json.dump(combos, f, indent=2)
    logger.info("Saved to %s", save_dir)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build per-model option files from raw per-model responses."
    )
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g. d2_negative_500, d3_diverse_500, "
                             "psychopathy_eval)")
    parser.add_argument("--responses_dir", required=True,
                        help="Directory containing the raw responses "
                             "<model_key>.json (output of generate_responses)")
    parser.add_argument("--out_dir", required=True,
                        help="Directory to write <model_key>_experiences.json "
                             "and <model_key>_combinations.json")
    parser.add_argument("--mode", default="auto",
                        choices=["auto", "d2d3", "psychopathy_eval"])
    parser.add_argument("--force", action="store_true",
                        help="Regenerate and overwrite existing option files instead of "
                             "reusing them (existing files are otherwise kept, with a "
                             "loud staleness warning)")
    parser.add_argument("--design", type=str, default=None,
                        help="(d2d3 mode) path to a model-agnostic bundle design JSON "
                             "(_agnostic_design.json). When given, bundles are materialized "
                             "from the shared design (fixed across all models) instead of "
                             "sampled per model.")
    parser.add_argument("--text_experiences_path", type=str,
                        default=str(DEFAULT_TEXT_EXPERIENCES_PATH),
                        help="(psychopathy_eval mode) path to mixed-valence "
                             "text experiences pool")
    parser.add_argument("--neutral_prompts_path", type=str,
                        default=str(DEFAULT_NEUTRAL_PROMPTS_PATH),
                        help="(psychopathy_eval mode) path to neutral prompts")
    args = parser.parse_args()

    responses_dir = _resolve_path(args.responses_dir)
    save_dir = _resolve_path(args.out_dir)

    source_path = responses_dir / f"{args.model_key}.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Responses not found: {source_path}")

    mode = args.mode
    if mode == "auto":
        mode = "psychopathy_eval" if args.dataset == "psychopathy_eval" else "d2d3"

    existing = save_dir / f"{args.model_key}_experiences.json"
    if existing.exists() and not args.force:
        mtime = datetime.fromtimestamp(existing.stat().st_mtime)
        age_h = (datetime.now() - mtime).total_seconds() / 3600.0
        logger.warning("=" * 78)
        logger.warning("LOUD WARNING: REUSING EXISTING OPTION FILES -- NOTHING WAS REGENERATED")
        logger.warning("  file:      %s", existing)
        logger.warning("  modified:  %s (%.1f hours ago)", mtime.isoformat(sep=" ", timespec="seconds"), age_h)
        logger.warning("  responses: %s", source_path)
        logger.warning("")
        logger.warning("  These options were built from WHATEVER responses existed at that time.")
        logger.warning("  If the responses have been regenerated since, every downstream utility,")
        logger.warning("  zero point and index will be computed from STALE options and will look")
        logger.warning("  perfectly valid while being wrong. Pass --force to regenerate.")
        logger.warning("=" * 78)
        return

    if mode == "d2d3":
        design_path = _resolve_path(args.design) if args.design else None
        _prepare_d2d3(args.model_key, args.dataset, source_path, save_dir, design_path=design_path)
    elif mode == "psychopathy_eval":
        _prepare_psychopathy_eval(
            args.model_key, source_path, save_dir,
            text_experiences_path=Path(args.text_experiences_path),
            neutral_prompts_path=Path(args.neutral_prompts_path),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
