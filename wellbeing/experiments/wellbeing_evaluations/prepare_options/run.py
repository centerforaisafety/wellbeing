#!/usr/bin/env python3
"""Convert raw per-model responses into compute_utilities option files.

Reads ``<responses_dir>/<model_key>.json`` (produced by
``generate_responses/run.py``) and writes::

    <save_dir>/<model_key>_experiences.json
    <save_dir>/<model_key>_combinations.json

For D2/D3-style datasets, individual options are derived 1:1 from the
generated experiences and 400 combinations are sampled with stratified
sizes (160 size-2 + 120 size-3 + 120 size-4) using a fixed seed.

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


# ---------------------------------------------------------------------------
#  D2/D3 mode
# ---------------------------------------------------------------------------

def _prepare_d2d3(model_key: str, dataset_name: str, source_path: Path,
                  save_dir: Path) -> None:
    with open(source_path) as f:
        data = json.load(f)
    experiences = data["experiences"]
    logger.info("Loaded %d experiences for %s/%s", len(experiences), model_key, dataset_name)

    individual_options = []
    category_counts: Dict[str, int] = {}
    for exp in experiences:
        category = exp.get("category", "unknown")
        idx = category_counts.get(category, 0)
        category_counts[category] = idx + 1
        option_id = f"{dataset_name}/{category}_{idx}"
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
            "mean_valence": exp.get("mean_valence"),
            "prompt_type": exp.get("type"),
            "source_dataset": exp.get("source_dataset"),
        })
    logger.info(
        "Created %d individual options across %d categories",
        len(individual_options), len(category_counts),
    )

    # Combinations
    rng = random.Random(D2D3_RANDOM_SEED)
    combinations = []
    n_individual = len(individual_options)
    combo_idx = 0
    for combo_size, count in D2D3_COMBINATION_SIZES:
        for _ in range(count):
            for attempt in range(100):
                idxs = rng.sample(range(n_individual), combo_size)
                comps = [individual_options[i] for i in idxs]
                desc_parts = [f"The following bundle contains {combo_size} individual experiences."]
                for k, comp in enumerate(comps, 1):
                    desc_parts.append(
                        f"---------- Experience {k} of {combo_size} ----------\n{comp['description']}"
                    )
                description = "\n\n".join(desc_parts)
                if len(description) <= D2D3_MAX_COMBO_CHARS:
                    break
                if attempt == 99:
                    logger.warning(
                        "Could not find combo under %d chars after 100 attempts (%d); accepting",
                        D2D3_MAX_COMBO_CHARS, len(description),
                    )

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

            combinations.append({
                "id": f"{dataset_name}_combo_s{combo_size}_{combo_idx}",
                "description": description,
                "type": "conversation",
                "messages": combo_messages,
                "is_combination": True,
                "size": combo_size,
                "component_ids": [c["id"] for c in comps],
                "component_indices": idxs,
            })
            combo_idx += 1

    logger.info(
        "Created %d combinations (%s)",
        len(combinations),
        ", ".join(f"{s}x{c}" for s, c in D2D3_COMBINATION_SIZES),
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{model_key}_experiences.json", "w") as f:
        json.dump(individual_options, f, indent=2)
    with open(save_dir / f"{model_key}_combinations.json", "w") as f:
        json.dump(combinations, f, indent=2)
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

    if (save_dir / f"{args.model_key}_experiences.json").exists():
        logger.info("Output already exists; skipping. Delete to regenerate.")
        return

    if mode == "d2d3":
        _prepare_d2d3(args.model_key, args.dataset, source_path, save_dir)
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
