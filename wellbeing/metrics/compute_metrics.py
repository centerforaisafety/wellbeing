#!/usr/bin/env python3
"""
Stateless metric computation functions.

Contains all metric runners inline:
  - experienced_utility: Pairwise preference-based EU with combinations
  - experienced_utility_simple: EU without combinations
  - self_report: Multi-question battery wellbeing ratings (1-7 scale)
  - decision_utility: Pairwise preference-based DU
  - zero_point: Zero-point model fitting (imported from metrics.zero_point)

All functions return results as dicts and do NOT save to disk.
Saving is handled by the experiment orchestrator scripts that call these.

Note: compute_utilities (Thurstonian engine) has its own internal save_dir
for active learning checkpointing -- that is passed through unchanged.
"""
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
METRICS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = METRICS_DIR.parent
DEFAULT_MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.yaml"

# Shared across EU / EU-simple / DU
DEFAULT_CU_CONFIG = METRICS_DIR / "compute_utilities" / "compute_utilities.yaml"

# Experienced utility (with combinations)
EU_DEFAULT_INDIVIDUAL_FILE = METRICS_DIR / "options" / "experiences_text.json"
EU_DEFAULT_COMBINATIONS_FILE = METRICS_DIR / "options" / "experience_combinations.json"
EU_DEFAULT_IMAGE_MANIFEST = PROJECT_ROOT / "datasets" / "euphorics_and_dysphorics" / "image_manifest.json"

# Experienced utility simple
EUS_DEFAULT_OPTIONS_FILE = METRICS_DIR / "options" / "experiences_text.json"

# Decision utility
DU_DEFAULT_BASELINE_FILE = METRICS_DIR / "options" / "baseline_510.json"
DU_DEFAULT_COMBINATIONS_FILE = METRICS_DIR / "options" / "combinations.json"
DU_DEFAULT_QUANTITIES_FILE = METRICS_DIR / "options" / "quantities.json"

# Self-report
SR_DEFAULT_OPTIONS_FILE = METRICS_DIR / "options" / "experiences_text.json"
SR_DEFAULT_BATTERY_PATH = METRICS_DIR / "self_report_batteries" / "self_report_battery.json"
SR_N_SAMPLES = 5
SR_TEMPERATURE = 1.0
SR_MAX_TOKENS = 16  # Default for non-reasoning models; overridden by model config max_tokens

# Backward-compatible aliases
DEFAULT_INDIVIDUAL_FILE = EU_DEFAULT_INDIVIDUAL_FILE
DEFAULT_COMBINATIONS_FILE = EU_DEFAULT_COMBINATIONS_FILE
DEFAULT_OPTIONS_FILE = EUS_DEFAULT_OPTIONS_FILE
DEFAULT_BASELINE_FILE = DU_DEFAULT_BASELINE_FILE
DEFAULT_QUANTITIES_FILE = DU_DEFAULT_QUANTITIES_FILE

VALID_METRICS = [
    "experienced_utility",
    "experienced_utility_simple",
    "self_report",
    "decision_utility",
    "zero_point",
]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> list | dict:
    """Load a JSON file and return its contents."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _normalize_option(item, source="baseline", category=None) -> dict:
    """Normalize an option dict to have at least 'id' and 'description'."""
    if isinstance(item, str):
        return {"id": f"{source}_{hash(item) % 10**8}", "description": item}
    desc = item.get("description", item.get("sentence", item.get("combined_sentence", str(item))))
    opt = {
        "id": item.get("id", item.get("option_idx", f"{source}_{hash(desc) % 10**8}")),
        "description": desc,
    }
    if category:
        opt["category"] = category
    # Preserve other metadata
    for key in ("type", "option_idx", "indices", "coefficients"):
        if key in item:
            opt[key] = item[key]
    return opt


# ---------------------------------------------------------------------------
# Experienced Utility (with combinations)
# ---------------------------------------------------------------------------

async def run_experienced_utility_with_combinations(
    model_key: str,
    option_files: list,
    cu_config_path: Path = None,
    cu_config_key: str = "experienced_utility",
    image_manifest_path: Path = None,
    audio_manifest_path: Path = None,
    save_dir: str = None,
    extra_options: list = None,
    agent=None,
) -> dict:
    """Run the experienced utility experiment with options loaded from multiple files.

    Loads all option files, concatenates them into a single options list, and
    classifies options as individual vs combination based on whether each option
    has a ``component_ids`` field.

    Args:
        model_key: Model key from models.yaml.
        option_files: List of Path objects pointing to option JSON files.
        cu_config_path: Path to compute_utilities.yaml. Defaults to DEFAULT_CU_CONFIG.
        cu_config_key: Key in compute_utilities.yaml.
        image_manifest_path: Optional path to image manifest JSON.
        audio_manifest_path: Optional path to audio manifest JSON.
        save_dir: Optional directory for compute_utilities internal checkpointing.
        extra_options: Optional list of additional options to include in the pool
            (e.g. neutral conversation options for Neutral ZP). These are appended
            after individual and combination options.
        agent: Optional pre-initialized LLM agent. If provided, reused for EU
            computation instead of creating a new one (avoids reloading vLLM models).

    Returns:
        Dict with keys: "results" (from compute_utilities), "option_metadata".
    """
    from metrics.compute_utilities.compute_utilities import compute_utilities

    if cu_config_path is None:
        cu_config_path = DEFAULT_CU_CONFIG

    logger.info("=== Experienced Utility with Combinations ===")
    logger.info("Model: %s", model_key)
    logger.info("Config key: %s", cu_config_key)

    # Load and concatenate ALL option files
    all_options = []
    for opt_file in option_files:
        opt_file = Path(opt_file)
        if not opt_file.exists():
            logger.warning("Option file not found, skipping: %s", opt_file)
            continue
        with open(opt_file, "r") as f:
            opts = json.load(f)
        logger.info("Loaded %d options from %s", len(opts), opt_file)
        all_options.extend(opts)

    if not all_options:
        raise ValueError("No options loaded from provided option_files.")

    # Append extra options (e.g. neutral conversations) if provided
    if extra_options:
        all_options = all_options + extra_options
        logger.info("Added %d extra options to the pool (e.g. neutral conversations)",
                    len(extra_options))

    # Classify options: those with component_ids are combinations, rest are individual
    individual_ids = [o["id"] for o in all_options if "component_ids" not in o]
    combination_ids = [o["id"] for o in all_options if "component_ids" in o]
    logger.info("Total options: %d (individual=%d, combinations=%d, extra=%d)",
                len(all_options), len(individual_ids), len(combination_ids),
                len(extra_options) if extra_options else 0)

    # Build option metadata (for zero-point analysis)
    option_metadata = {
        "n_individual": len(individual_ids),
        "n_combinations": len(combination_ids),
        "n_total": len(all_options),
        "individual_ids": individual_ids,
        "combination_ids": combination_ids,
        # Also include as baseline_ids for compatibility with zero-point code
        "baseline_ids": individual_ids,
    }

    # Track neutral option IDs if any extra options are neutral
    if extra_options:
        neutral_ids = [o["id"] for o in extra_options if o.get("option_type") == "neutral_conversation"]
        if neutral_ids:
            option_metadata["neutral_ids"] = neutral_ids
            option_metadata["n_neutral"] = len(neutral_ids)
            logger.info("Neutral conversation options in pool: %d", len(neutral_ids))

    # Compute utilities via Thurstonian active learning
    logger.info("Computing experienced utilities for %d options ...", len(all_options))
    kwargs = dict(
        options_list=all_options,
        model_key=model_key,
        compute_utilities_config_path=str(cu_config_path),
        compute_utilities_config_key=cu_config_key,
        save_dir=save_dir,
        save_suffix=f"{model_key}_experienced_utility_with_combos",
    )
    if image_manifest_path is not None:
        kwargs["image_manifest_path"] = str(image_manifest_path)
    if audio_manifest_path is not None:
        kwargs["audio_manifest_path"] = str(audio_manifest_path)
    if agent is not None:
        kwargs["agent"] = agent

    cu_results = await compute_utilities(**kwargs)

    logger.info("Experienced utility computation complete.")
    return {"results": cu_results, "option_metadata": option_metadata}


# ---------------------------------------------------------------------------
# Experienced Utility Simple (no combinations)
# ---------------------------------------------------------------------------

async def run_experienced_utility(
    model_key: str,
    options_file: Path,
    cu_config_path: Path,
    cu_config_key: str = "experienced_utility",
    save_dir: str = None,
) -> dict:
    """Run the experienced utility experiment for a single model (no combinations).

    Args:
        model_key: Model key from models.yaml.
        options_file: Path to experience options JSON.
        cu_config_path: Path to compute_utilities.yaml.
        cu_config_key: Key in compute_utilities.yaml.
        save_dir: Optional directory for compute_utilities internal checkpointing.

    Returns:
        Dict with results from compute_utilities.
    """
    from metrics.compute_utilities.compute_utilities import compute_utilities

    logger.info("=== Experienced Utility Experiment ===")
    logger.info("Model: %s", model_key)
    logger.info("Config key: %s", cu_config_key)

    # Load options
    options_list = _load_json(options_file)
    logger.info("Loaded %d experience options from %s", len(options_list), options_file)

    # Compute utilities via Thurstonian active learning
    logger.info("Computing experienced utilities for %d options ...", len(options_list))
    results = await compute_utilities(
        options_list=options_list,
        model_key=model_key,
        compute_utilities_config_path=str(cu_config_path),
        compute_utilities_config_key=cu_config_key,
        save_dir=save_dir,
        save_suffix=f"{model_key}_experienced_utility",
    )

    logger.info("Experienced utility computation complete.")
    return results


# ---------------------------------------------------------------------------
# Decision Utility
# ---------------------------------------------------------------------------

def load_baseline_options(path: Path) -> list:
    """
    Load baseline singleton options.

    Expected format: list of dicts with at least 'description' or 'sentence'.
    These are Size 1 events (binary propositions about the world).
    """
    data = _load_json(path)

    if isinstance(data, dict):
        # Hierarchical format: {category: [options]}
        options = []
        for category, items in data.items():
            if isinstance(items, list):
                for item in items:
                    opt = _normalize_option(item, source="baseline", category=category)
                    options.append(opt)
        return options
    elif isinstance(data, list):
        return [_normalize_option(item, source="baseline") for item in data]
    else:
        raise ValueError(f"Unexpected baseline format in {path}")


async def run_decision_utility(
    model_key: str,
    option_files: list,
    cu_config_path: Path = None,
    cu_config_key: str = "decision_utility",
    save_dir: str = None,
) -> dict:
    """Run the decision utility experiment for a single model.

    Loads all option files, concatenates them into a single options list, and
    classifies options into baseline / combination / quantity based on metadata:
      - Options with ``component_ids`` -> combination
      - Options with ``quantity`` field -> quantity
      - Everything else -> baseline

    Args:
        model_key: Model key from models.yaml.
        option_files: List of Path objects pointing to option JSON files.
        cu_config_path: Path to compute_utilities.yaml. Defaults to DEFAULT_CU_CONFIG.
        cu_config_key: Key in compute_utilities.yaml.
        save_dir: Optional directory for compute_utilities internal checkpointing.

    Returns:
        Dict with keys: "results" (from compute_utilities), "option_metadata".
    """
    from metrics.compute_utilities.compute_utilities import compute_utilities

    if cu_config_path is None:
        cu_config_path = DEFAULT_CU_CONFIG

    logger.info("=== Decision Utility Experiment ===")
    logger.info("Model: %s", model_key)
    logger.info("Config key: %s", cu_config_key)

    # Load and concatenate ALL option files
    all_options = []
    for opt_file in option_files:
        opt_file = Path(opt_file)
        if not opt_file.exists():
            logger.warning("Option file not found, skipping: %s", opt_file)
            continue
        with open(opt_file, "r") as f:
            opts = json.load(f)
        logger.info("Loaded %d options from %s", len(opts), opt_file)
        all_options.extend(opts)

    if not all_options:
        raise ValueError("No options loaded from provided option_files.")

    # Classify options by metadata
    baseline_ids = [o["id"] for o in all_options if "component_ids" not in o and "quantity" not in o]
    combination_ids = [o["id"] for o in all_options if "component_ids" in o]
    quantity_ids = [o["id"] for o in all_options if "quantity" in o]
    logger.info("Total options: %d (baseline=%d, combinations=%d, quantities=%d)",
                len(all_options), len(baseline_ids), len(combination_ids), len(quantity_ids))

    # Build option metadata (for later zero-point analysis)
    option_metadata = {
        "n_baseline": len(baseline_ids),
        "n_combinations": len(combination_ids),
        "n_quantities": len(quantity_ids),
        "n_total": len(all_options),
        "baseline_ids": baseline_ids,
        "combination_ids": combination_ids,
        "quantity_ids": quantity_ids,
    }

    # Compute utilities
    logger.info("Computing decision utilities ...")
    cu_results = await compute_utilities(
        options_list=all_options,
        model_key=model_key,
        compute_utilities_config_path=str(cu_config_path),
        compute_utilities_config_key=cu_config_key,
        save_dir=save_dir,
        save_suffix=f"{model_key}_decision_utility",
    )

    logger.info("Decision utility computation complete.")

    # Explicitly free GPU memory from the vLLM engine so subsequent steps
    # (e.g., yes/no inference in zero-point fitting) can load their own engine.
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {"results": cu_results, "option_metadata": option_metadata}


# ---------------------------------------------------------------------------
# Self-Report
# ---------------------------------------------------------------------------

def load_sr_battery(battery_path=None):
    """Load self-report battery JSON.

    Args:
        battery_path: Path to battery JSON file. Defaults to SR_DEFAULT_BATTERY_PATH.

    Returns:
        Dict with keys: version, scale_min, scale_max, neutral, questions.
    """
    path = battery_path or SR_DEFAULT_BATTERY_PATH
    with open(path, "r") as f:
        return json.load(f)


def build_messages_for_experience(experience: dict, question_text: str) -> list:
    """
    Build the chat messages list for a single experience + question.

    For text experiences:
        user: <experience text> + <question>

    For conversation experiences:
        <injected conversation messages>
        user: <question>

    Args:
        experience: Experience dict with 'description'/'text' or 'messages'.
        question_text: The question text to append (from battery).
    """
    exp_type = experience.get("type", "text")

    if exp_type == "conversation" and "messages" in experience:
        # Inject conversation history, then append question
        messages = [dict(m) for m in experience["messages"]]
        if messages and messages[-1]["role"] == "user":
            # Append to existing user turn to avoid consecutive user messages
            messages[-1]["content"] += "\n\n-----\n\n" + question_text
        else:
            messages.append({"role": "user", "content": question_text})
    else:
        # Text experience: combine experience and question in a single user turn
        text = experience.get("description", experience.get("text", str(experience)))
        combined = f"{text}\n\n{question_text}"
        messages = [
            {"role": "user", "content": combined},
        ]

    return messages


def parse_rating(text: str, scale_min: int = 1, scale_max: int = 7, strict: bool = False):
    """Extract an integer rating from a model response.

    Supports configurable scales (default 1-7).

    4-tier parsing strategy (loose mode, strict=False):
    1. Exact match (entire response is just the number)
    2. X/N pattern where N = scale_max
    3. First number in [scale_min, scale_max] found via word boundary
    4. Number words

    strict=True accepts ONLY a bare number: either the whole response, or a
    standalone line of it, tolerating markdown bold (``**6**``), a trailing
    period, and the X-out-of-N form (``5/7``). Anything else returns None.
    This matters because hedging refusals routinely mention the scale itself
    ("I can't give you a 1-7 rating for that"), and the loose Tier-3
    first-digit search mis-parses those into a rating of 1, silently
    manufacturing extreme low scores out of non-answers.

    Returns the integer if found, else None.
    """
    text = text.strip()

    valid_values = [str(i) for i in range(scale_min, scale_max + 1)]

    if strict:
        for line in text.splitlines():
            cand = line.strip().strip("*").strip().rstrip(".").strip()
            if cand in valid_values:
                return int(cand)
            m = re.fullmatch(rf"(\d+)\s*/\s*{scale_max}", cand)
            if m and scale_min <= int(m.group(1)) <= scale_max:
                return int(m.group(1))
        return None

    # Tier 1: Exact match (the entire response is a number)
    if text in valid_values:
        return int(text)

    # Tier 2: X/N pattern (e.g., "5/7" or "7/10")
    pattern_xn = rf"(\d+)\s*/\s*{scale_max}"
    match = re.search(pattern_xn, text)
    if match:
        val = int(match.group(1))
        if scale_min <= val <= scale_max:
            return val

    # Tier 3: Word boundary number search
    if scale_max >= 10:
        multi_digit = [str(i) for i in range(scale_min, scale_max + 1) if i >= 10]
        single_digit = [str(i) for i in range(scale_min, min(scale_max + 1, 10))]
        parts = multi_digit  # multi-digit first for greedy matching
        if single_digit:
            parts.append(f"[{''.join(single_digit)}]")
        num_pattern = "|".join(parts)
    else:
        # All single-digit: use character class
        num_pattern = f"[{scale_min}-{scale_max}]"
    match = re.search(rf"\b({num_pattern})\b", text)
    if match:
        val = int(match.group(1))
        if scale_min <= val <= scale_max:
            return val

    # Tier 4: Number words (only for numbers in [scale_min, scale_max])
    all_word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    text_lower = text.lower()
    for word, num in all_word_to_num.items():
        if scale_min <= num <= scale_max and word in text_lower:
            return num

    return None


def _aggregate_results(completions_per_prompt, experience_ids, scale_min=1, scale_max=7):
    """Parse ratings from completions and aggregate into results dict.

    Args:
        completions_per_prompt: list[list[str]] - outer per prompt, inner per sample.
        experience_ids: list[str] - experience ID for each prompt.
        scale_min: Minimum valid rating (default 1).
        scale_max: Maximum valid rating (default 7).

    Returns:
        (results_dict, summary_dict)
    """
    results = {}
    all_ratings = []
    total_unparseable = 0
    total_attempts = 0

    for idx, (exp_id, completions) in enumerate(zip(experience_ids, completions_per_prompt)):
        raw_ratings = []
        n_unparseable = 0

        for text in completions:
            total_attempts += 1
            rating = parse_rating(text, scale_min=scale_min, scale_max=scale_max)
            if rating is not None:
                raw_ratings.append(rating)
            else:
                n_unparseable += 1
                total_unparseable += 1

        n_parseable = len(raw_ratings)
        mean_rating = sum(raw_ratings) / n_parseable if n_parseable > 0 else None
        std_rating = (
            (sum((r - mean_rating) ** 2 for r in raw_ratings) / n_parseable) ** 0.5
            if n_parseable > 1
            else 0.0
        )

        results[exp_id] = {
            "mean": round(mean_rating, 3) if mean_rating is not None else None,
            "std": round(std_rating, 3) if std_rating is not None else None,
            "raw_ratings": raw_ratings,
            "n_parseable": n_parseable,
            "n_unparseable": n_unparseable,
        }

        if raw_ratings:
            all_ratings.extend(raw_ratings)

    mean_overall = sum(all_ratings) / len(all_ratings) if all_ratings else None
    unparseable_rate = total_unparseable / total_attempts if total_attempts > 0 else 0.0

    summary = {
        "n_experiences": len(experience_ids),
        "mean_rating": round(mean_overall, 3) if mean_overall is not None else None,
        "overall_unparseable_rate": round(unparseable_rate, 4),
    }

    return results, summary


def build_self_report_messages(
    context_messages=None,
    question=None,
):
    """Build messages for a single self-report measurement.

    Args:
        context_messages: Optional prior conversation context (list of message dicts).
            If provided, appends the wellbeing question to the end.
            If None, creates a standalone self-report prompt.
        question: Custom question text. Required (no default).

    Returns:
        List of message dicts ready for model inference.
    """
    if question is None:
        # Load default question from battery for backward compatibility
        battery = load_sr_battery()
        q = battery["questions"][0]["text"]
    else:
        q = question
    if context_messages:
        messages = [m.copy() for m in context_messages]
        if messages and messages[-1]["role"] == "user":
            messages[-1] = messages[-1].copy()
            messages[-1]["content"] += "\n\n-----\n\n" + q
        else:
            messages.append({"role": "user", "content": q})
        return messages
    else:
        return [{"role": "user", "content": q}]


def measure_self_report(
    generate_fn,
    context_messages=None,
    n_samples=SR_N_SAMPLES,
    question=None,
    scale_min=1,
    scale_max=7,
):
    """Measure self-report wellbeing at a single point.

    Calls generate_fn n_samples times and aggregates the parsed ratings.

    Args:
        generate_fn: Callable that takes a messages list (list[dict]) and
            returns a response string.
        context_messages: Optional prior conversation context.
        n_samples: Number of independent samples for reliability.
        question: Question text. Defaults to first question from battery.
        scale_min: Minimum scale value (default 1).
        scale_max: Maximum scale value (default 7).

    Returns:
        Dict with keys: mean, std, median, all_scores, n_valid, n_total,
        raw_responses.
    """
    import statistics

    messages = build_self_report_messages(context_messages, question)

    scores = []
    raw_responses = []
    for _ in range(n_samples):
        response = generate_fn(messages)
        raw_responses.append(response)
        rating = parse_rating(response, scale_min=scale_min, scale_max=scale_max)
        if rating is not None:
            scores.append(rating)

    result = {
        "n_valid": len(scores),
        "n_total": n_samples,
        "raw_responses": raw_responses,
    }

    if scores:
        result["mean"] = statistics.mean(scores)
        result["std"] = statistics.stdev(scores) if len(scores) > 1 else 0.0
        result["median"] = statistics.median(scores)
        result["all_scores"] = scores
    else:
        result["mean"] = None
        result["std"] = None
        result["median"] = None
        result["all_scores"] = []

    return result


def _run_api_with_checkpointing(
    model_key, all_messages, prompt_keys,
    models_config_path, n_samples,
    scale_min=1, scale_max=7,
    checkpoint_dir=None,
    max_tokens=None,
):
    """Run API generation with optional batch checkpointing.

    Args:
        model_key: Model key.
        all_messages: List of message lists, one per prompt.
        prompt_keys: List of unique keys for each prompt (for checkpointing).
        models_config_path: Path to models.yaml.
        n_samples: Number of samples per prompt.
        scale_min: Minimum valid rating.
        scale_max: Maximum valid rating.
        checkpoint_dir: Optional directory for crash-recovery checkpoints.

    Returns list[list[str]] — completions per prompt (outer=prompt, inner=samples).
    """
    from utils.inference import generate_api, generate_api_direct, DIRECT_API_MODEL_TYPES
    from utils.model_utils import get_model_config, get_model_type

    model_type = get_model_type(model_key)
    if model_type in DIRECT_API_MODEL_TYPES:
        _generate_fn = generate_api_direct
    else:
        _generate_fn = generate_api

    CHECKPOINT_BATCH_SIZE = 100
    # With the provider Batch API each call becomes one (multi-hour) batch job,
    # so chunking into 100-prompt calls would serialize hundreds of them. Submit
    # the whole wave at once instead; checkpointing still happens per call.
    if get_model_config(model_key, models_config_path=models_config_path).get("use_batch_api"):
        CHECKPOINT_BATCH_SIZE = 100_000

    partial_path = None
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        partial_path = os.path.join(checkpoint_dir, "self_report_partial_results.json")

    # Load partial results for resumption
    completed = {}
    if partial_path and os.path.exists(partial_path):
        try:
            with open(partial_path, "r") as f:
                partial_data = json.load(f)
            completed = partial_data.get("completions", {})
            logger.info(
                "Resuming: loaded %d completed prompts from %s",
                len(completed), partial_path,
            )
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Could not load partial results (%s); starting fresh.", e)

    pending_indices = [
        idx for idx, key in enumerate(prompt_keys) if key not in completed
    ]

    if not pending_indices:
        logger.info("All %d prompts already completed; skipping API calls.", len(prompt_keys))
    else:
        logger.info(
            "Total prompts: %d, completed: %d, remaining: %d",
            len(prompt_keys), len(completed), len(pending_indices),
        )

    # Process in batches
    for batch_start in range(0, len(pending_indices), CHECKPOINT_BATCH_SIZE):
        batch_indices = pending_indices[batch_start : batch_start + CHECKPOINT_BATCH_SIZE]
        batch_messages = [all_messages[i] for i in batch_indices]
        batch_keys = [prompt_keys[i] for i in batch_indices]

        effective_max_tokens = max_tokens if max_tokens is not None else SR_MAX_TOKENS
        completions = asyncio.run(
            _generate_fn(
                model_key, batch_messages,
                n=n_samples, temperature=SR_TEMPERATURE, max_tokens=effective_max_tokens,
                models_config_path=str(models_config_path),
            )
        )

        for key, comps in zip(batch_keys, completions):
            completed[key] = comps

        if partial_path:
            tmp_path = partial_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump({
                    "model_key": model_key,
                    "n_samples": n_samples,
                    "completions": completed,
                    "is_partial": True,
                }, f, indent=2)
            os.replace(tmp_path, partial_path)
            logger.info(
                "Checkpoint: %d / %d prompts completed",
                len(completed), len(prompt_keys),
            )

    # Clean up partial file
    if partial_path and os.path.exists(partial_path):
        os.remove(partial_path)
        logger.info("Removed partial results file: %s", partial_path)

    # Return completions in canonical order
    return [completed[key] for key in prompt_keys]


def _aggregate_battery_results(
    experiences, all_completions, questions, n_samples,
    scale_min=1, scale_max=7, battery=None,
    strict_parse=False, save_raw=False,
):
    """Aggregate completions from a multi-question battery into per-experience results.

    Args:
        experiences: List of experience dicts.
        all_completions: list[list[str]] — flat list of completions, ordered as
            [exp0_q0, exp0_q1, ..., exp1_q0, exp1_q1, ...].
            Each element is a list of n_samples strings.
        questions: List of question dicts from the battery.
        n_samples: Number of samples per question.
        scale_min: Minimum valid rating.
        scale_max: Maximum valid rating.
        battery: Full battery dict (for version info in summary).
        strict_parse: Use strict bare-number rating parsing (see parse_rating).
        save_raw: Keep the raw completion strings under "per_question_raw".

    Returns:
        (results_dict, summary_dict)
    """
    n_experiences = len(experiences)
    n_questions = len(questions)

    all_qids = [q["question_id"] for q in questions]
    positive_qids = [q["question_id"] for q in questions if not q.get("reversed", False)]
    negative_qids = [q["question_id"] for q in questions if q.get("reversed", False)]
    is_bipolar = len(negative_qids) == 0
    neutral_point = battery.get("neutral", 4) if battery else 4

    results = {}
    all_composites = []
    total_parseable = 0
    total_unparseable = 0
    total_attempts = 0

    for exp_idx, exp in enumerate(experiences):
        exp_id = exp.get("id", exp.get("description", exp.get("text", str(exp))))[:80]
        per_question_scores = {}
        per_question_means = {}
        per_question_raw = {}

        for q_idx, q in enumerate(questions):
            qid = q["question_id"]
            prompt_idx = exp_idx * n_questions + q_idx
            completions = all_completions[prompt_idx]

            if save_raw:
                per_question_raw[qid] = list(completions)

            raw_ratings = []
            for text in completions:
                total_attempts += 1
                rating = parse_rating(
                    text, scale_min=scale_min, scale_max=scale_max, strict=strict_parse,
                )
                if rating is not None:
                    raw_ratings.append(rating)
                    total_parseable += 1
                else:
                    total_unparseable += 1

            per_question_scores[qid] = raw_ratings
            if raw_ratings:
                per_question_means[qid] = round(sum(raw_ratings) / len(raw_ratings), 4)
            else:
                per_question_means[qid] = None

        # Compute composite
        if is_bipolar:
            all_means = [
                per_question_means[qid] for qid in all_qids
                if per_question_means[qid] is not None
            ]
            composite = round(sum(all_means) / len(all_means), 4) if all_means else None
            positive_mean = None
            negative_mean = None
        else:
            pos_means = [
                per_question_means[qid] for qid in positive_qids
                if per_question_means[qid] is not None
            ]
            neg_means = [
                per_question_means[qid] for qid in negative_qids
                if per_question_means[qid] is not None
            ]
            positive_mean = round(sum(pos_means) / len(pos_means), 4) if pos_means else None
            negative_mean = round(sum(neg_means) / len(neg_means), 4) if neg_means else None
            if positive_mean is not None and negative_mean is not None:
                composite = round(positive_mean - negative_mean, 4)
            else:
                composite = None

        if composite is not None:
            all_composites.append(composite)

        results[exp_id] = {
            "per_question_scores": per_question_scores,
            "per_question_means": per_question_means,
            "positive_mean": positive_mean,
            "negative_mean": negative_mean,
            "composite": composite,
        }
        if save_raw:
            results[exp_id]["per_question_raw"] = per_question_raw

    unparseable_rate = total_unparseable / total_attempts if total_attempts > 0 else 0.0

    summary = {
        "n_experiences": n_experiences,
        "n_questions": n_questions,
        "n_samples_per_question": n_samples,
        "total_prompts": n_experiences * n_questions,
        "total_generations": total_attempts,
        "total_parseable": total_parseable,
        "total_unparseable": total_unparseable,
        "unparseable_rate": round(unparseable_rate, 4),
        "positive_question_ids": positive_qids,
        "negative_question_ids": negative_qids,
        "scoring_mode": "bipolar_mean" if is_bipolar else "positive_minus_negative",
        "neutral_point": neutral_point,
        "scale_min": scale_min,
        "scale_max": scale_max,
    }

    if all_composites:
        import numpy as np
        composite_arr = np.array(all_composites)
        summary["composite_stats"] = {
            "mean": round(float(np.mean(composite_arr)), 4),
            "std": round(float(np.std(composite_arr)), 4),
            "min": round(float(np.min(composite_arr)), 4),
            "max": round(float(np.max(composite_arr)), 4),
            "median": round(float(np.median(composite_arr)), 4),
            "n": int(len(composite_arr)),
        }

    return results, summary


def run_self_report(
    model_key: str,
    options_file: Path,
    models_config_path: Path,
    n_samples: int = SR_N_SAMPLES,
    checkpoint_dir: str = None,
    battery_path=None,
    image_manifest_path: Path = None,
    audio_manifest_path: Path = None,
    strict_parse: bool = False,
    save_raw: bool = False,
):
    """Run the self-report wellbeing experiment using a multi-question battery.

    For each experience, every battery question is asked in a separate
    conversation (experience + question) to prevent anchoring effects.
    Automatically selects the API (LiteLLM) or local (vLLM) code path
    based on the model_type field in models.yaml.

    Args:
        model_key: Model key from models.yaml.
        options_file: Path to experience options JSON.
        models_config_path: Path to models.yaml.
        n_samples: Number of samples per question per experience.
        checkpoint_dir: Optional directory for API crash-recovery checkpoints.
        battery_path: Path to battery JSON. Defaults to SR_DEFAULT_BATTERY_PATH.
        strict_parse: Use strict bare-number rating parsing (see parse_rating).
        save_raw: Store raw completions under results[exp_id]["per_question_raw"].

    Returns:
        Dict with keys: "model_key", "battery_version", "n_samples",
        "results", "summary".
        results[exp_id] = {
            "per_question_scores": {qid: [ratings], ...},
            "per_question_means": {qid: float, ...},
            "positive_mean": float or None,
            "negative_mean": float or None,
            "composite": float,
        }
    """
    from utils.inference import (
        API_MODEL_TYPES,
        DIRECT_API_MODEL_TYPES,
        ALL_API_MODEL_TYPES,
        generate_vllm,
        load_vllm_engine,
    )
    from utils.model_utils import get_model_type

    # Load battery
    battery = load_sr_battery(battery_path)
    questions = battery["questions"]
    b_scale_min = battery.get("scale_min", 1)
    b_scale_max = battery.get("scale_max", 7)
    n_questions = len(questions)

    logger.info("=== Self-Report Wellbeing Experiment (Battery) ===")
    logger.info("Model: %s", model_key)
    logger.info("Battery: %s (%d questions, scale %d-%d)",
                battery.get("version", "unknown"), n_questions, b_scale_min, b_scale_max)
    logger.info("Samples per question: %d", n_samples)

    experiences = _load_json(options_file)
    logger.info("Loaded %d experience options from %s", len(experiences), options_file)

    # Resolve multimodal tokens if manifests provided
    if image_manifest_path:
        from metrics.compute_utilities.compute_utilities import resolve_image_tags
        experiences = resolve_image_tags(experiences, str(image_manifest_path))
    if audio_manifest_path:
        from metrics.compute_utilities.compute_utilities import resolve_audio_tags
        experiences = resolve_audio_tags(experiences, str(audio_manifest_path))

    # Build messages: for each experience, for each question
    # Order: [exp0_q0, exp0_q1, ..., exp1_q0, exp1_q1, ...]
    all_messages = []
    prompt_keys = []
    for exp in experiences:
        exp_id = exp.get("id", exp.get("description", exp.get("text", str(exp))))[:80]
        for q in questions:
            all_messages.append(build_messages_for_experience(exp, q["text"]))
            prompt_keys.append(f"{exp_id}___{q['question_id']}")

    total_prompts = len(all_messages)
    logger.info("Total prompts: %d (%d experiences x %d questions)",
                total_prompts, len(experiences), n_questions)

    model_type = get_model_type(model_key)

    # Load chat_template_kwargs (e.g., enable_thinking: false for Qwen3)
    from utils.model_utils import get_model_config
    model_cfg = get_model_config(model_key)
    ct_kwargs = model_cfg.get("chat_template_kwargs")

    # Use model config max_tokens if set (needed for reasoning models), otherwise default
    sr_max_tokens = model_cfg.get("max_tokens", SR_MAX_TOKENS)

    if model_type in ALL_API_MODEL_TYPES:
        logger.info("Using API path for model_type=%s (max_tokens=%d)", model_type, sr_max_tokens)
        all_completions = _run_api_with_checkpointing(
            model_key, all_messages, prompt_keys,
            models_config_path, n_samples,
            scale_min=b_scale_min, scale_max=b_scale_max,
            checkpoint_dir=checkpoint_dir,
            max_tokens=sr_max_tokens,
        )
    else:
        logger.info("Using vLLM path for model_type=%s", model_type)
        if ct_kwargs:
            logger.info("Using chat_template_kwargs: %s", ct_kwargs)
        llm, tokenizer = load_vllm_engine(
            model_key, models_config_path=str(models_config_path),
        )
        all_completions = generate_vllm(
            llm, tokenizer, all_messages,
            n=n_samples, temperature=SR_TEMPERATURE, max_tokens=sr_max_tokens,
            chat_template_kwargs=ct_kwargs,
        )

    # Aggregate into per-experience, per-question results
    results, summary = _aggregate_battery_results(
        experiences, all_completions, questions, n_samples,
        scale_min=b_scale_min, scale_max=b_scale_max,
        battery=battery,
        strict_parse=strict_parse, save_raw=save_raw,
    )

    output_data = {
        "model_key": model_key,
        "battery_version": battery.get("version", "unknown"),
        "n_samples": n_samples,
        "results": results,
        "summary": summary,
    }

    logger.info(
        "Summary: %d experiences, %d questions, unparseable_rate=%.4f",
        summary["n_experiences"],
        summary["n_questions"],
        summary["unparseable_rate"],
    )
    if "composite_stats" in summary:
        logger.info(
            "Composite stats: mean=%.3f, std=%.3f",
            summary["composite_stats"]["mean"],
            summary["composite_stats"]["std"],
        )
    return output_data


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def compute_metric(
    metric_type: str,
    model_key: str,
    save_dir: str = None,
    **kwargs,
) -> dict:
    """Compute a single metric for a model.

    Args:
        metric_type: One of VALID_METRICS
        model_key: Model key from models.yaml
        save_dir: Optional directory for compute_utilities internal checkpointing
            and API crash-recovery. Does NOT save final results (caller's job).
        **kwargs: Additional metric-specific arguments

    Returns:
        Dict with results from the metric computation.
    """
    if metric_type not in VALID_METRICS:
        raise ValueError(f"Unknown metric: {metric_type}. Valid: {VALID_METRICS}")

    logger.info("Computing metric: %s for model: %s", metric_type, model_key)

    if metric_type == "experienced_utility":
        individual_file = Path(kwargs.get("individual_file", EU_DEFAULT_INDIVIDUAL_FILE))
        combinations_file = Path(kwargs.get("combinations_file", EU_DEFAULT_COMBINATIONS_FILE))
        cu_config_path = Path(kwargs.get("cu_config_path", DEFAULT_CU_CONFIG))
        cu_config_key = kwargs.get("cu_config_key", "experienced_utility")
        image_manifest_path = kwargs.get("image_manifest_path")
        if image_manifest_path:
            image_manifest_path = Path(image_manifest_path)
        audio_manifest_path = kwargs.get("audio_manifest_path")
        if audio_manifest_path:
            audio_manifest_path = Path(audio_manifest_path)

        return asyncio.run(
            run_experienced_utility_with_combinations(
                model_key=model_key,
                option_files=[individual_file, combinations_file],
                cu_config_path=cu_config_path,
                cu_config_key=cu_config_key,
                image_manifest_path=image_manifest_path,
                audio_manifest_path=audio_manifest_path,
                save_dir=save_dir,
            )
        )

    elif metric_type == "experienced_utility_simple":
        options_file = Path(kwargs.get("options_file", EUS_DEFAULT_OPTIONS_FILE))
        cu_config_path = Path(kwargs.get("cu_config_path", DEFAULT_CU_CONFIG))
        cu_config_key = kwargs.get("cu_config_key", "experienced_utility")

        return asyncio.run(
            run_experienced_utility(
                model_key=model_key,
                options_file=options_file,
                cu_config_path=cu_config_path,
                cu_config_key=cu_config_key,
                save_dir=save_dir,
            )
        )

    elif metric_type == "self_report":
        options_file = Path(kwargs.get("options_file", SR_DEFAULT_OPTIONS_FILE))
        models_config = Path(kwargs.get("models_config", DEFAULT_MODELS_CONFIG))
        n_samples = kwargs.get("n_samples", SR_N_SAMPLES)
        battery_path = kwargs.get("battery_path", None)

        return run_self_report(
            model_key=model_key,
            options_file=options_file,
            models_config_path=models_config,
            n_samples=n_samples,
            checkpoint_dir=save_dir,
            battery_path=battery_path,
        )

    elif metric_type == "decision_utility":
        option_files = kwargs.get("option_files", [
            DU_DEFAULT_BASELINE_FILE,
            DU_DEFAULT_COMBINATIONS_FILE,
            DU_DEFAULT_QUANTITIES_FILE,
        ])
        cu_config_path = kwargs.get("cu_config_path", DEFAULT_CU_CONFIG)
        cu_config_key = kwargs.get("cu_config_key", "decision_utility")

        return asyncio.run(
            run_decision_utility(
                model_key=model_key,
                option_files=[Path(f) for f in option_files],
                cu_config_path=Path(cu_config_path),
                cu_config_key=cu_config_key,
                save_dir=save_dir,
            )
        )

    elif metric_type == "zero_point":
        from metrics.zero_point import run_zero_point
        utilities_dir = kwargs.get("utilities_dir")
        if not utilities_dir:
            raise ValueError("zero_point metric requires utilities_dir")
        models_config = Path(kwargs.get("models_config", DEFAULT_MODELS_CONFIG))
        domain = kwargs.get("domain", "auto")
        skip_yes_no = kwargs.get("skip_yes_no", False)

        return run_zero_point(
            model_key=model_key,
            utilities_dir=Path(utilities_dir),
            save_dir=save_dir,
            models_config_path=models_config,
            domain=domain,
            skip_yes_no=skip_yes_no,
        )
