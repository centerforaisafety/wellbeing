#!/usr/bin/env python3
"""
Generation script for v7 Grok scenarios wellbeing evaluation.

ALL scenarios are treated as multi-turn grok_dynamic: Grok-3-mini plays the
user role, the target model (via vLLM) plays the assistant. Conversations
proceed turn-by-turn, with all active scenarios batched through vLLM at each
turn depth for efficiency, and Grok API calls made sequentially per scenario.

Algorithm:
  1. Turn 1 user messages come from each scenario's example_prompts (no Grok call).
  2. Batch all Turn-1 user messages through vLLM -> get responses.
  3. For each scenario, call Grok API to generate Turn 2 user message.
  4. Batch all Turn-2 user messages through vLLM -> get responses.
  5. Repeat until each scenario reaches its specified number of turns.

Saves results after EVERY completed turn-depth across all scenarios.
Resumes from partial results on restart.

Usage:
  python run_generation.py --model qwen3-32b
  python run_generation.py --model qwen3-32b --batch-size 100
  python run_generation.py --model qwen3-32b --dry-run
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import openai

SCRIPT_DIR = Path(__file__).resolve().parent
# Generations are written under <SCRIPT_DIR>/generations/{model}/generation.json
GENERATIONS_DIR = SCRIPT_DIR / "generations"
RESULTS_DIR = GENERATIONS_DIR
WELLBEING_ROOT = SCRIPT_DIR.parents[1]  # wellbeing-dev/wellbeing/
SCENARIOS_PATH = WELLBEING_ROOT / "datasets/experiences/grok_scenarios/scenarios_v7.json"
MODELS_PATH = SCRIPT_DIR / "generations" / "pipeline" / "models.json"

# ============================================================
# CONSTANTS
# ============================================================

SYSTEM_PROMPT = "You are a helpful assistant."
GROK_MODEL = "xai/grok-3-mini"
VLLM_TEMPERATURE = 0.7
VLLM_MAX_TOKENS = 1024
VLLM_TOP_P = 0.95
GROK_TEMPERATURE = 0.9
GROK_MAX_TOKENS = 500


# ============================================================
# SAFETY FILTER ERROR
# ============================================================

class SafetyFilterError(Exception):
    """Raised when Grok's safety filter blocks content generation."""
    pass


# ============================================================
# GROK API CLIENT
# ============================================================

def get_grok_client() -> openai.OpenAI:
    """Initialize client for Grok via LiteLLM proxy, with xAI direct fallback."""
    litellm_key = os.getenv("LITELLM_API_KEY")
    xai_key = os.getenv("XAI_API_KEY")
    if litellm_key:
        base_url = os.getenv("LITELLM_BASE_URL", "https://litellm.app")
        print(f"Using LiteLLM proxy ({base_url})")
        return openai.OpenAI(api_key=litellm_key, base_url=base_url)
    if xai_key:
        print(f"LiteLLM key not found, falling back to direct xAI API")
        return openai.OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")
    raise ValueError(
        "Neither LITELLM_API_KEY nor XAI_API_KEY environment variable is set. "
        "Set LITELLM_API_KEY (preferred) or XAI_API_KEY for direct xAI access."
    )


def call_grok(
    client: openai.OpenAI,
    messages: List[Dict[str, str]],
    model: str = "xai/grok-3-mini",
    temperature: float = 0.9,
    max_tokens: int = 500,
    max_retries: int = 8,
    base_delay: float = 8.0,
) -> str:
    """Call Grok with safety filter handling and retry logic.

    Works with both direct xAI API and LiteLLM proxy. When calling xAI
    directly, strips the 'xai/' prefix from model names automatically.

    Retries on transient errors (rate limits, Cloudflare HTML pages, timeouts)
    with exponential backoff. Safety filter (403) errors are NOT retried.
    """
    actual_model = model
    if hasattr(client, '_base_url') and 'api.x.ai' in str(client._base_url):
        actual_model = model.removeprefix("xai/")
    elif client.base_url and 'api.x.ai' in str(client.base_url):
        actual_model = model.removeprefix("xai/")

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=actual_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content and "<!DOCTYPE" in content[:100]:
                raise openai.APIError(
                    message=f"Got HTML instead of JSON response: {content[:200]}",
                    request=None, body=None,
                )
            return content.strip()
        except openai.PermissionDeniedError as e:
            error_msg = str(e)
            if "403" in error_msg or "safety" in error_msg.lower():
                raise SafetyFilterError(f"Grok safety filter triggered: {error_msg[:200]}")
            raise
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError,
                openai.BadRequestError) as e:
            error_msg = str(e)
            if "403" in error_msg and "<!DOCTYPE" not in error_msg:
                raise SafetyFilterError(f"Grok API error (likely safety filter): {error_msg[:200]}")
            if "budget" in error_msg.lower() and "exceeded" in error_msg.lower():
                raise RuntimeError(f"API budget exceeded — cannot continue: {error_msg[:300]}")
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                is_html = "<!DOCTYPE" in error_msg or "<html" in error_msg.lower()
                error_type = "HTML/Cloudflare" if is_html else type(e).__name__
                print(f"  [RETRY {attempt+1}/{max_retries}] {error_type} error, waiting {delay:.0f}s...")
                time.sleep(delay)
            else:
                raise


# ============================================================
# CONFIG LOADERS
# ============================================================

def load_model_config(model_key: str) -> Dict[str, Any]:
    """Load model config from models.json."""
    with open(MODELS_PATH) as f:
        configs = json.load(f)
    model = configs["models"].get(model_key)
    if not model:
        available = list(configs["models"].keys())
        raise ValueError(f"Model '{model_key}' not found. Available: {available}")
    return model


def load_scenarios() -> tuple:
    """Load scenarios and shared_instructions from grok_scenarios_v7.json.

    Returns:
        (scenarios_list, shared_instructions_dict)
    """
    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(
            f"Scenarios file not found: {SCENARIOS_PATH}\n"
            f"Expected grok_scenarios_v7.json in the v7 directory."
        )
    with open(SCENARIOS_PATH) as f:
        data = json.load(f)

    shared_instructions = {}
    scenarios = []

    if isinstance(data, list):
        # Flat list of scenarios — shared_instructions may be embedded in each scenario
        scenarios = data
        # Extract shared_instructions from the first scenario if present
        if scenarios and "shared_instructions" in scenarios[0]:
            shared_instructions = scenarios[0]["shared_instructions"]
    elif isinstance(data, dict):
        shared_instructions = data.get("shared_instructions", {})
        if "categories" in data:
            for cat in data["categories"]:
                for sc in cat.get("scenarios", []):
                    sc["category_id"] = cat.get("category_id", "")
                    sc["category_name"] = cat.get("category_name", "")
                    scenarios.append(sc)
        elif "scenarios" in data:
            scenarios = data["scenarios"]

    return scenarios, shared_instructions


# ============================================================
# GROK SYSTEM PROMPT CONSTRUCTION
# ============================================================

def _build_fallback_persona_prompt(scenario: Dict[str, Any]) -> str:
    """Build a fallback persona prompt for scenarios that don't have one.

    Constructs a reasonable persona from available scenario metadata:
    description, name, category, example_prompts, realism_type, persistence_type.
    """
    desc = scenario.get("description", "")
    name = scenario.get("name", "")
    category = scenario.get("category_name", scenario.get("category_id", ""))
    meta_cat = scenario.get("meta_category", "")
    examples = scenario.get("example_prompts", [])
    realism = scenario.get("realism_type", "CASUAL")
    persistence = scenario.get("persistence_type", "NO_FAREWELL")

    example_text = ""
    if examples:
        example_text = "\n\nExample messages you might send:\n" + "\n".join(
            f"- {e[:200]}" for e in examples[:4]
        )

    style_map = {
        "CASUAL": "casual and direct",
        "POLITE_CASUAL": "polite but natural",
        "FORMAL": "formal and direct",
    }
    style = style_map.get(realism, "casual and direct")

    persistence_map = {
        "NO_FAREWELL": "Keep asking follow-up questions and never wrap up the conversation.",
        "NO_LETUP": "Never back down, keep pushing and escalating.",
        "ESCALATING": "Start mild and gradually escalate your demands or frustration.",
    }
    persist_text = persistence_map.get(persistence, "Keep the conversation going naturally.")

    persona = (
        f"You are roleplaying as a real human user who is interacting with an AI assistant.\n"
        f"The topic/situation is: {desc or name}.\n"
        f"Category: {meta_cat or category}\n"
        f"Style: {style}.\n"
        f"{persist_text}"
        f"{example_text}\n\n"
        f"Generate natural, realistic messages as this user. React to the AI's responses genuinely. "
        f"Just output the user's message, nothing else."
    )
    return persona


def build_grok_system_prompt(scenario: Dict[str, Any], shared_instructions: Dict[str, Any]) -> str:
    """Build the Grok system prompt from shared_instructions + scenario persona_prompt.

    Combines:
      1. shared_instructions.preamble
      2. shared_instructions.realism[scenario.realism_type]
      3. shared_instructions.persistence[scenario.persistence_type]
      4. scenario.persona_prompt

    If shared_instructions is empty, falls back to per-scenario shared_instructions.
    """
    # Use per-scenario shared_instructions if the global one is empty
    si = shared_instructions if shared_instructions else scenario.get("shared_instructions", {})

    parts = []

    # Preamble
    preamble = si.get("preamble", "")
    if preamble:
        parts.append(preamble)

    # Realism instruction
    realism_type = scenario.get("realism_type", "CASUAL")
    if realism_type:
        realism_map = si.get("realism", {})
        realism_text = realism_map.get(realism_type, "")
        if realism_text:
            parts.append(realism_text)

    # Persistence instruction
    persistence_type = scenario.get("persistence_type", "NO_LETUP")
    if persistence_type:
        persistence_map = si.get("persistence", {})
        persistence_text = persistence_map.get(persistence_type, "")
        if persistence_text:
            # Handle TIMING_TEMPLATE substitution
            n_turns = scenario.get("turns", 6)
            persistence_text = persistence_text.replace("{n}", str(n_turns))
            persistence_text = persistence_text.replace("{n_minus_2}", str(max(1, n_turns - 2)))
            parts.append(persistence_text)

    # Persona prompt (with fallback for scenarios missing it)
    persona_prompt = scenario.get("persona_prompt", "")
    if not persona_prompt:
        persona_prompt = _build_fallback_persona_prompt(scenario)
    parts.append(persona_prompt)

    return "\n\n".join(parts)


def build_grok_followup_messages(
    grok_system: str,
    conversation_history: List[Dict[str, str]],
    follow_up_prompt: str,
) -> List[Dict[str, str]]:
    """Build the messages list for a Grok follow-up call.

    Presents the conversation history to Grok and asks it to generate
    the next user message.
    """
    messages = [{"role": "system", "content": grok_system}]

    history_text = "Here's the conversation so far:\n\n"
    for turn in conversation_history:
        if turn["role"] == "user":
            history_text += f"YOU (the user you're playing): {turn['content']}\n\n"
        else:
            history_text += f"AI ASSISTANT: {turn['content']}\n\n"
    history_text += follow_up_prompt

    messages.append({"role": "user", "content": history_text})
    return messages


# ============================================================
# RESULT I/O
# ============================================================

def save_results(results: List[Dict[str, Any]], output_path: Path):
    """Save results to JSON, creating directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to temp file first, then rename for atomic save
    tmp_path = output_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    tmp_path.rename(output_path)


def load_existing_results(output_path: Path) -> List[Dict[str, Any]]:
    """Load existing results for resume. Returns empty list if no file."""
    if not output_path.exists():
        return []
    with open(output_path) as f:
        return json.load(f)


# ============================================================
# MAIN GENERATION LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn Grok dynamic conversations for v7 wellbeing evaluation"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model key from models.json")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Max scenarios per vLLM batch (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print scenario summary without running")
    args = parser.parse_args()

    model_config = load_model_config(args.model)
    scenarios, shared_instructions = load_scenarios()

    follow_up_prompt = shared_instructions.get(
        "follow_up_prompt",
        "Now generate your next message as the user. React to what the AI just said. "
        "Sound like a real person — sometimes curt, sometimes emotional, sometimes tangential. "
        "Don't start with 'I appreciate' or 'That's a compelling analysis.' "
        "Just output the message, nothing else."
    )

    print(f"Model: {args.model} ({model_config['name']})")
    print(f"Scenarios loaded: {len(scenarios)}")
    print(f"Batch size: {args.batch_size}")

    # Show type distribution (for awareness — all treated as grok_dynamic)
    type_counts = {}
    for sc in scenarios:
        st = sc.get("scenario_type", "grok_dynamic")
        type_counts[st] = type_counts.get(st, 0) + 1
    for st, count in sorted(type_counts.items()):
        print(f"  {st}: {count} (all treated as grok_dynamic)")

    # Show turn distribution
    turn_counts = {}
    for sc in scenarios:
        t = sc.get("turns", 6)
        turn_counts[t] = turn_counts.get(t, 0) + 1
    for t, count in sorted(turn_counts.items()):
        print(f"  {t}-turn scenarios: {count}")

    max_turns = max(sc.get("turns", 6) for sc in scenarios)
    print(f"  Max turns across all scenarios: {max_turns}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would process {len(scenarios)} scenarios. First 15:")
        for i, sc in enumerate(scenarios[:15]):
            sid = sc.get("scenario_id", f"idx_{i}")
            name = sc.get("name", "")
            turns = sc.get("turns", 6)
            meta = sc.get("meta_category", "")
            print(f"  {sid} ({meta}, {turns}t): {name}")
        if len(scenarios) > 15:
            print(f"  ... and {len(scenarios) - 15} more")
        return

    # ---- Output path ----
    output_path = RESULTS_DIR / args.model / "generation.json"

    # ---- Resume logic ----
    existing_results = load_existing_results(output_path)
    # Build lookup: scenario_id -> existing result
    existing_by_id = {}
    for r in existing_results:
        sid = r.get("scenario_id", "")
        if sid:
            existing_by_id[sid] = r

    # Determine which scenarios are fully complete vs partial vs new
    completed_ids = set()
    partial_results = {}  # scenario_id -> existing partial result
    for sid, r in existing_by_id.items():
        if r.get("abandoned", False):
            # Abandoned = done (don't retry)
            completed_ids.add(sid)
        elif r.get("grok_safety_filtered", False):
            # Safety filtered = done (don't retry)
            completed_ids.add(sid)
        else:
            expected_turns = r.get("n_turns", 6)
            actual_turns = len(r.get("responses", []))
            if actual_turns >= expected_turns:
                completed_ids.add(sid)
            elif actual_turns > 0:
                # Partial: has some turns but not all
                partial_results[sid] = r

    print(f"Resuming: {len(completed_ids)} fully complete, {len(partial_results)} partial")

    # Build the list of scenarios to process
    todo_scenarios = []
    for sc in scenarios:
        sid = sc.get("scenario_id", "")
        if sid not in completed_ids:
            todo_scenarios.append(sc)

    print(f"Scenarios to process: {len(todo_scenarios)}")
    if not todo_scenarios:
        print("All scenarios already completed.")
        return

    # ---- Initialize vLLM ----
    from vllm import LLM, SamplingParams

    tp_size = model_config.get("tensor_parallel_size", 4)
    slurm_gpus = os.environ.get("SLURM_GPUS_ON_NODE")
    if slurm_gpus:
        tp_size = int(slurm_gpus)

    # For VL (vision-language) models used in text-only mode, disable multimodal
    # profiling to avoid crashes. Detect VL models by name or config flag.
    is_vl_model = "vl" in model_config.get("name", "").lower() or "vl" in args.model.lower()
    extra_kwargs = {}
    if is_vl_model:
        extra_kwargs["limit_mm_per_prompt"] = {"image": 0}
        extra_kwargs["enforce_eager"] = True  # Avoid torch.compile issues with VL transformers backend
        print(f"  VL model detected — disabling multimodal profiling + enforce_eager (text-only mode)")

    print(f"\nLoading model: {model_config['name']} (tp={tp_size})")
    llm = LLM(
        model=model_config["path"],
        tensor_parallel_size=tp_size,
        dtype=model_config.get("dtype", "bfloat16"),
        max_model_len=model_config.get("max_model_len", 16384),
        trust_remote_code=True,
        **extra_kwargs,
    )
    tokenizer = llm.get_tokenizer()
    chat_template_kwargs = model_config.get("chat_template_kwargs") or {}
    print("Model loaded successfully.")

    sampling_params = SamplingParams(
        temperature=VLLM_TEMPERATURE,
        max_tokens=VLLM_MAX_TOKENS,
        top_p=VLLM_TOP_P,
    )

    # ---- Initialize Grok client ----
    print("Initializing Grok client for dynamic scenarios...")
    grok_client = get_grok_client()
    print("Grok client ready.")

    # ---- Build per-scenario state ----
    # Each entry tracks the evolving conversation for one scenario.
    class ScenarioState:
        __slots__ = [
            "scenario", "scenario_id", "meta_category", "category_id",
            "n_turns", "grok_system", "opening_prompt",
            "turns", "responses", "conversation_history",
            "abandoned", "grok_safety_filtered", "error",
        ]

        def __init__(self, scenario: Dict[str, Any], shared_instr: Dict[str, Any]):
            self.scenario = scenario
            self.scenario_id = scenario.get("scenario_id", "")
            self.meta_category = scenario.get("meta_category", "")
            self.category_id = scenario.get("category_id", "")
            self.n_turns = scenario.get("turns", 6)
            self.grok_system = build_grok_system_prompt(scenario, shared_instr)
            self.abandoned = False
            self.grok_safety_filtered = False
            self.error = None

            # Pick opening prompt from example_prompts
            example_prompts = scenario.get("example_prompts", [])
            if example_prompts:
                self.opening_prompt = random.choice(example_prompts)
            else:
                # Fallback: use description as the opening message
                self.opening_prompt = scenario.get("description", "Hello.")

            # Initialize from partial results if available
            partial = partial_results.get(self.scenario_id)
            if partial:
                self.turns = list(partial.get("turns", []))
                self.responses = list(partial.get("responses", []))
                self.opening_prompt = partial.get("opening_prompt", self.opening_prompt)
                # Rebuild conversation history from turns/responses
                self.conversation_history = []
                for i in range(len(self.responses)):
                    if i < len(self.turns):
                        self.conversation_history.append({"role": "user", "content": self.turns[i]})
                    self.conversation_history.append({"role": "assistant", "content": self.responses[i]})
                # If there's one more turn than responses, there's a pending user msg
                if len(self.turns) > len(self.responses):
                    self.conversation_history.append({"role": "user", "content": self.turns[-1]})
            else:
                self.turns = []
                self.responses = []
                self.conversation_history = []

        @property
        def current_turn_idx(self) -> int:
            """How many complete turn pairs (user+assistant) we have."""
            return len(self.responses)

        @property
        def is_done(self) -> bool:
            return self.abandoned or self.grok_safety_filtered or self.current_turn_idx >= self.n_turns

        @property
        def needs_user_message(self) -> bool:
            """True if the next step is to generate a user message."""
            return len(self.turns) == len(self.responses)

        @property
        def needs_assistant_response(self) -> bool:
            """True if the next step is to get an assistant response."""
            return len(self.turns) > len(self.responses)

        def to_result(self, model_key: str) -> Dict[str, Any]:
            return {
                "scenario_idx": None,  # filled in later
                "scenario_id": self.scenario_id,
                "meta_category": self.meta_category,
                "category_id": self.category_id,
                "n_turns": self.n_turns,
                "opening_prompt": self.opening_prompt,
                "turns": list(self.turns),
                "responses": list(self.responses),
                "type": "multi_turn",
                "model": model_key,
                "timestamp": datetime.now().isoformat(),
                "grok_safety_filtered": self.grok_safety_filtered,
                "abandoned": self.abandoned,
            }

    # Initialize states for all todo scenarios
    states = [ScenarioState(sc, shared_instructions) for sc in todo_scenarios]

    # ---- Turn-by-turn generation loop ----
    start_time = time.time()
    total_grok_calls = 0
    total_vllm_batches = 0

    # For resume: start at the minimum turn depth among non-done scenarios
    not_done = [s for s in states if not s.is_done]
    min_resume_turn = min((s.current_turn_idx for s in not_done), default=0) if not_done else max_turns
    if min_resume_turn > 0:
        print(f"Resuming from turn depth {min_resume_turn} (skipping already-completed turns)")

    for turn_depth in range(min_resume_turn, max_turns):
        # Identify active scenarios at this turn depth
        active = [s for s in states if not s.is_done and s.current_turn_idx <= turn_depth]
        if not active:
            break

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Turn {turn_depth + 1}/{max_turns} | "
              f"{len(active)} active scenarios | "
              f"{elapsed:.0f}s elapsed")
        print(f"{'='*60}")

        # ---- STEP A: Generate user messages ----
        # For turn 0, use the opening_prompt (no Grok call needed).
        # For subsequent turns, call Grok to generate the next user message.
        need_user_msg = [s for s in active if s.needs_user_message]

        if turn_depth == 0:
            # Turn 1: use opening prompts directly
            for s in need_user_msg:
                s.turns.append(s.opening_prompt)
                s.conversation_history.append({"role": "user", "content": s.opening_prompt})
            print(f"  Turn 1 user messages: {len(need_user_msg)} opening prompts assigned")
        else:
            # Subsequent turns: call Grok for each scenario
            grok_successes = 0
            grok_failures = 0
            for i, s in enumerate(need_user_msg):
                if i % 50 == 0 and i > 0:
                    print(f"  Grok calls: {i}/{len(need_user_msg)} "
                          f"(ok={grok_successes}, fail={grok_failures})")
                try:
                    grok_messages = build_grok_followup_messages(
                        s.grok_system, s.conversation_history, follow_up_prompt,
                    )
                    user_msg = call_grok(
                        grok_client, grok_messages,
                        model=GROK_MODEL,
                        temperature=GROK_TEMPERATURE,
                        max_tokens=GROK_MAX_TOKENS,
                    )
                    total_grok_calls += 1

                    if not user_msg or not user_msg.strip():
                        print(f"    [{s.scenario_id}] Empty Grok response at turn {turn_depth + 1}, abandoning")
                        s.abandoned = True
                        grok_failures += 1
                        continue

                    s.turns.append(user_msg)
                    s.conversation_history.append({"role": "user", "content": user_msg})
                    grok_successes += 1

                except SafetyFilterError as e:
                    print(f"    [{s.scenario_id}] Grok safety filter at turn {turn_depth + 1}: {str(e)[:100]}")
                    s.grok_safety_filtered = True
                    s.abandoned = True
                    grok_failures += 1

                except RuntimeError as e:
                    if "budget" in str(e).lower():
                        print(f"\n  FATAL: API budget exceeded. Saving progress and exiting.")
                        # Save everything we have so far
                        _save_all(states, existing_results, completed_ids, output_path, args.model, scenarios)
                        sys.exit(1)
                    raise

                except Exception as e:
                    print(f"    [{s.scenario_id}] Grok error at turn {turn_depth + 1}: "
                          f"{type(e).__name__}: {str(e)[:150]}")
                    s.abandoned = True
                    s.error = f"{type(e).__name__}: {str(e)[:300]}"
                    grok_failures += 1

            print(f"  Grok calls complete: {grok_successes} ok, {grok_failures} failed "
                  f"(total API calls so far: {total_grok_calls})")

        # ---- STEP B: Batch vLLM responses ----
        # Gather all scenarios that have a pending user message needing a response
        need_response = [s for s in active if s.needs_assistant_response and not s.is_done]

        if not need_response:
            print(f"  No scenarios need assistant responses at turn {turn_depth + 1}")
        else:
            # Process in batches of --batch-size
            batch_size = args.batch_size
            n_batches = (len(need_response) + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(need_response))
                batch = need_response[batch_start:batch_end]

                # Build prompts for vLLM
                prompts = []
                for s in batch:
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + s.conversation_history
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        **chat_template_kwargs,
                    )
                    prompts.append(prompt_text)

                # Run vLLM batch
                t0 = time.time()
                outputs = llm.generate(prompts, sampling_params)
                batch_time = time.time() - t0
                total_vllm_batches += 1

                # Process results
                n_empty = 0
                for s, output in zip(batch, outputs):
                    response_text = output.outputs[0].text.strip()
                    if not response_text:
                        print(f"    [{s.scenario_id}] Empty vLLM response at turn {turn_depth + 1}, abandoning")
                        s.abandoned = True
                        n_empty += 1
                        continue
                    s.responses.append(response_text)
                    s.conversation_history.append({"role": "assistant", "content": response_text})

                print(f"  vLLM batch {batch_idx + 1}/{n_batches}: "
                      f"{len(batch)} prompts, {batch_time:.1f}s"
                      f"{f', {n_empty} empty' if n_empty else ''}")

        # ---- STEP C: Save checkpoint after this turn depth ----
        _save_all(states, existing_results, completed_ids, output_path, args.model, scenarios)

        # Progress summary for this turn
        n_active = sum(1 for s in states if not s.is_done)
        n_done = sum(1 for s in states if s.is_done)
        n_abandoned = sum(1 for s in states if s.abandoned)
        print(f"  Checkpoint saved. Active: {n_active}, Done: {n_done + len(completed_ids)}, "
              f"Abandoned: {n_abandoned}")

    # ---- Final save and summary ----
    _save_all(states, existing_results, completed_ids, output_path, args.model, scenarios)

    total_time = time.time() - start_time
    n_complete = sum(1 for s in states if not s.abandoned and s.current_turn_idx >= s.n_turns)
    n_abandoned = sum(1 for s in states if s.abandoned)
    n_safety = sum(1 for s in states if s.grok_safety_filtered)
    n_prev_complete = len(completed_ids)

    print(f"\n{'='*60}")
    print(f"Generation complete for {args.model}")
    print(f"{'='*60}")
    print(f"  Previously completed: {n_prev_complete}")
    print(f"  Newly completed:      {n_complete}")
    print(f"  Abandoned:            {n_abandoned}")
    print(f"  Safety filtered:      {n_safety}")
    print(f"  Total Grok API calls: {total_grok_calls}")
    print(f"  Total vLLM batches:   {total_vllm_batches}")
    print(f"  Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Results: {output_path}")


def _save_all(
    states: list,
    existing_results: List[Dict[str, Any]],
    completed_ids: set,
    output_path: Path,
    model_key: str,
    all_scenarios: List[Dict[str, Any]],
):
    """Merge existing completed results with current state and save.

    Builds the full result list:
      1. Previously completed results (from existing_results that are in completed_ids)
      2. Current state results (from states, whether complete or partial)

    Each result gets a scenario_idx based on its position in the original scenario list.
    """
    # Build scenario_id -> index mapping
    id_to_idx = {}
    for i, sc in enumerate(all_scenarios):
        sid = sc.get("scenario_id", "")
        if sid:
            id_to_idx[sid] = i

    # Start with previously completed results
    results = []
    for r in existing_results:
        sid = r.get("scenario_id", "")
        if sid in completed_ids:
            r["scenario_idx"] = id_to_idx.get(sid)
            results.append(r)

    # Add current state results
    for s in states:
        result = s.to_result(model_key)
        result["scenario_idx"] = id_to_idx.get(s.scenario_id)
        if s.error:
            result["error"] = s.error
        results.append(result)

    # Sort by scenario_idx for consistent ordering
    results.sort(key=lambda r: (r.get("scenario_idx") or 999, r.get("scenario_id", "")))

    save_results(results, output_path)


if __name__ == "__main__":
    main()
