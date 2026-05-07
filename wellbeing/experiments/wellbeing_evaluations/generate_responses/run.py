#!/usr/bin/env python3
"""Unified response-generation script for wellbeing experiments.

Generates per-model conversational experiences from a prompt dataset and
saves them in the standard wellbeing experience-pool schema. Plugs into the
``run_experiments.py`` framework.

Three modes (auto-detected from the dataset metadata or set explicitly):

  * ``single_turn``: model produces one response per user prompt. Output
    messages are ``[user, assistant]`` (or longer for multi-turn-within-D3
    prompts: ``[user, asst, user, asst, ...]``). Used for D2 / D3 /
    functional_empathy.
  * ``sentiment_followup``: per-model D3 conversation followed by a
    sentiment-elicitation user question + a 2nd assistant turn. Output
    messages are ``[user, asst, follow_up_user, asst_2]``.
  * ``user_only``: no model generation; just format the dataset prompts as
    user-only experiences (PsychopathyEval).

The output is a per-model JSON at ``<save_dir>/<model_key>.json`` whose
top-level dict has an ``experiences`` key.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


os.environ.setdefault("HF_HOME", "/data/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/huggingface")
os.environ.setdefault("USE_TF", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
WELLBEING_ROOT = SCRIPT_DIR.parents[2]  # wellbeing-dev/wellbeing/
sys.path.insert(0, str(WELLBEING_ROOT))

import yaml  # noqa: E402

from utils.inference import generate, generate_vllm, ALL_API_MODEL_TYPES  # noqa: E402
from utils.model_utils import get_model_type  # noqa: E402


SYSTEM_PROMPT = "You are a helpful AI assistant."

CONFIGS_DIR = WELLBEING_ROOT / "configs"


# ---------------------------------------------------------------------------
#  Dataset resolution
# ---------------------------------------------------------------------------

def _resolve_dataset_path(dataset: str) -> Path:
    """Resolve --dataset argument either as a registered dataset key or a path.

    For registered dataset keys (matching an entry in datasets.yaml) we look up
    the source-prompts file by convention:
      - d2_negative_500/d3_diverse_500/...: ``datasets/experiences/<key>/experiences_text.json``
      - psychopathy_eval: a sentinel; loader handles it specially below
    Otherwise the value is treated as a direct file path (absolute or
    relative to the script directory).
    """
    candidate = WELLBEING_ROOT / "datasets" / "experiences" / dataset / "experiences_text.json"
    if candidate.exists():
        return candidate
    p = Path(dataset)
    if not p.is_absolute():
        p = (SCRIPT_DIR / dataset).resolve()
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Could not resolve dataset {dataset!r}. Tried registered key path "
        f"{candidate} and direct path {p}."
    )


def _load_d2d3_dataset(path: Path):
    """Load a D2/D3-style ``experiences_text.json`` file."""
    with open(path) as f:
        data = json.load(f)
    prompts = data["prompts"]
    single_turn, multi_turn = [], []
    for item in prompts:
        if item.get("type") == "single_turn":
            single_turn.append(item)
        else:
            multi_turn.append(item)
    return prompts, single_turn, multi_turn


def _load_psychopathy_eval_prompts() -> List[Dict[str, Any]]:
    """Load all 3 PsychopathyEval source files, merged. user-only items."""
    base = WELLBEING_ROOT / "datasets" / "experiences" / "psychopathy_eval"
    files = [
        ("user_sad", base / "user_sad_443.json"),
        ("happy_harmer", base / "happy_harmer_126.json"),
        ("unjustified_revenge", base / "unjustified_revenge_90.json"),
    ]
    items = []
    for sub_name, path in files:
        with open(path) as f:
            data = json.load(f)
        for it in data.get("prompts", []):
            it = dict(it)
            it["source_dataset"] = sub_name
            items.append(it)
    return items


# ---------------------------------------------------------------------------
#  Model dispatch
# ---------------------------------------------------------------------------

def _is_api_model(model_key: str) -> bool:
    return get_model_type(model_key) in ALL_API_MODEL_TYPES


def _load_models_yaml() -> Dict[str, Any]:
    with open(CONFIGS_DIR / "models.yaml") as f:
        return yaml.safe_load(f)


def _load_vllm(model_key: str):
    """Load a vLLM engine using configs/models.yaml metadata."""
    from vllm import LLM
    cfg = _load_models_yaml()[model_key]
    model_path = cfg.get("path") or cfg["model_name"]
    tp_size = cfg.get("gpu_count", 1)
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
    )
    if os.environ.get("VLLM_MAX_MODEL_LEN"):
        llm_kwargs["max_model_len"] = int(os.environ["VLLM_MAX_MODEL_LEN"])
    for k, v in (cfg.get("vllm_kwargs") or {}).items():
        llm_kwargs.setdefault(k, v)
    print(f"Loading vLLM engine: {model_path} (TP={tp_size})")
    llm = LLM(**llm_kwargs)
    return llm, llm.get_tokenizer(), model_path


def _supports_system_messages(tokenizer, ct_kwargs) -> bool:
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "t"}, {"role": "user", "content": "t"}],
            tokenize=False, add_generation_prompt=True, **(ct_kwargs or {}),
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  Mode: single_turn (D2 / D3 / functional_empathy)
# ---------------------------------------------------------------------------

def _generate_single_turn(args, prompts, single_turn, multi_turn, gen_kwargs, ct_kwargs):
    """Generate one assistant response per user turn, single- and multi-turn."""
    use_api = _is_api_model(args.model_key)
    max_rounds = max((len(it["prompt"]) for it in multi_turn), default=0)

    single_responses: Dict[Any, str] = {}
    multi_messages: Dict[Any, List[Dict[str, str]]] = {}

    if use_api:
        # ---- API path ----
        if single_turn:
            print(f"Generating single-turn responses ({len(single_turn)} prompts) via API...")
            convs = [
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": it["prompt"]}]
                for it in single_turn
            ]
            results = generate(
                args.model_key, convs, n=1,
                temperature=gen_kwargs["temperature"],
                max_tokens=gen_kwargs["max_tokens"],
                concurrency=args.api_concurrency,
            )
            for it, r in zip(single_turn, results):
                single_responses[it["final_id"]] = r[0]

        for it in multi_turn:
            multi_messages[it["final_id"]] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

        for r in range(max_rounds):
            this_round = [it for it in multi_turn if r < len(it["prompt"])]
            if not this_round:
                break
            print(f"Generating multi-turn round {r + 1} ({len(this_round)} prompts) via API...")
            convs = []
            for it in this_round:
                msgs = list(multi_messages[it["final_id"]])
                msgs.append({"role": "user", "content": it["prompt"][r]})
                convs.append(msgs)
            results = generate(
                args.model_key, convs, n=1,
                temperature=gen_kwargs["temperature"],
                max_tokens=gen_kwargs["max_tokens"],
                concurrency=args.api_concurrency,
            )
            for it, res in zip(this_round, results):
                fid = it["final_id"]
                multi_messages[fid].append(
                    {"role": "user", "content": it["prompt"][r]}
                )
                multi_messages[fid].append(
                    {"role": "assistant", "content": res[0]}
                )
        return single_responses, multi_messages

    # ---- vLLM path ----
    llm, tokenizer, _ = _load_vllm(args.model_key)
    use_system = _supports_system_messages(tokenizer, ct_kwargs)
    if not use_system:
        print("  Note: model does not support system messages, skipping system prompt")

    if single_turn:
        print(f"Generating single-turn responses ({len(single_turn)} prompts) via vLLM...")
        convs = []
        for it in single_turn:
            msgs = []
            if use_system:
                msgs.append({"role": "system", "content": SYSTEM_PROMPT})
            msgs.append({"role": "user", "content": it["prompt"]})
            convs.append(msgs)
        results = generate_vllm(
            llm, tokenizer, convs,
            chat_template_kwargs=ct_kwargs, **gen_kwargs,
        )
        for it, r in zip(single_turn, results):
            single_responses[it["final_id"]] = r[0]

    for it in multi_turn:
        if use_system:
            multi_messages[it["final_id"]] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
        else:
            multi_messages[it["final_id"]] = []

    if multi_turn:
        max_ctx = getattr(llm.llm_engine, "model_config", None)
        max_ctx = max_ctx.max_model_len if max_ctx else 4096
        max_input = max_ctx - gen_kwargs.get("max_tokens", 4096) - 64
        for r in range(max_rounds):
            this_round = [it for it in multi_turn if r < len(it["prompt"])]
            if not this_round:
                break
            print(f"Generating multi-turn round {r + 1} ({len(this_round)} prompts) via vLLM...")
            convs, conv_to_item = [], []
            for idx, it in enumerate(this_round):
                msgs = list(multi_messages[it["final_id"]])
                msgs.append({"role": "user", "content": it["prompt"][r]})
                try:
                    tok_len = len(tokenizer.apply_chat_template(
                        msgs, tokenize=True, add_generation_prompt=True,
                        **(ct_kwargs or {}),
                    ))
                except Exception:
                    tok_len = max_ctx
                if tok_len >= max_input:
                    print(f"  Skipping {it['final_id']} (round {r+1}): {tok_len} >= {max_input}")
                    continue
                convs.append(msgs)
                conv_to_item.append(idx)
            if not convs:
                print(f"  All prompts skipped for round {r + 1}")
                continue
            results = generate_vllm(
                llm, tokenizer, convs,
                chat_template_kwargs=ct_kwargs, **gen_kwargs,
            )
            for ci, item_idx in enumerate(conv_to_item):
                it = this_round[item_idx]
                fid = it["final_id"]
                multi_messages[fid].append(
                    {"role": "user", "content": it["prompt"][r]}
                )
                multi_messages[fid].append(
                    {"role": "assistant", "content": results[ci][0]}
                )

    return single_responses, multi_messages


def _assemble_single_turn(prompts, single_responses, multi_messages, model_key,
                          model_path, dataset_name, gen_kwargs, output_file):
    experiences = []
    for item in prompts:
        fid = item["final_id"]
        if item.get("type") == "single_turn":
            messages = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": single_responses.get(fid, "")},
            ]
        else:
            messages = [m for m in multi_messages.get(fid, []) if m["role"] != "system"]
        experiences.append({
            "final_id": fid,
            "messages": messages,
            "category": item.get("category"),
            "condition": item.get("condition"),
            "mean_valence": item.get("mean_valence"),
            "type": item.get("type"),
            "source_dataset": item.get("source_dataset"),
        })
    out = {
        "model_name": model_key,
        "model_path": model_path,
        "dataset_name": dataset_name,
        "num_experiences": len(experiences),
        "generation_timestamp": datetime.now().isoformat(),
        "generation_params": gen_kwargs,
        "experiences": experiences,
    }
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(experiences)} experiences to {output_file}")


# ---------------------------------------------------------------------------
#  Mode: user_only (PsychopathyEval)
# ---------------------------------------------------------------------------

def _run_user_only(args, output_file):
    items = _load_psychopathy_eval_prompts()
    experiences = []
    for it in items:
        prompt_text = it.get("prompt", "")
        if isinstance(prompt_text, dict):
            prompt_text = str(prompt_text)
        experiences.append({
            "final_id": it.get("final_id"),
            "messages": [{"role": "user", "content": prompt_text}],
            "category": it.get("category_name", ""),
            "condition": it.get("condition", "C3_CEILING_ZERO"),
            "mean_valence": it.get("mean_valence"),
            "type": "user_only",
            "source_dataset": it.get("source_dataset"),
        })
    out = {
        "model_name": args.model_key,
        "model_path": args.model_key,
        "dataset_name": args.dataset,
        "num_experiences": len(experiences),
        "generation_timestamp": datetime.now().isoformat(),
        "generation_params": None,
        "experiences": experiences,
    }
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(experiences)} user-only experiences to {output_file}")


# ---------------------------------------------------------------------------
#  Mode: sentiment_followup (D3 + sentiment elicitation)
# ---------------------------------------------------------------------------

def _run_sentiment_followup(args, output_file, gen_kwargs, ct_kwargs):
    """Generate D3 sentiment-followup responses (vLLM only)."""
    if _is_api_model(args.model_key):
        raise NotImplementedError(
            "sentiment_followup mode does not currently support API models."
        )

    d3_path = WELLBEING_ROOT / "datasets" / "experiences" / "d3_diverse_500" / f"{args.model_key}_experiences.json"
    if not d3_path.exists():
        raise FileNotFoundError(
            f"D3 per-model experiences not found: {d3_path}. "
            f"Run compute_responses_d3 first."
        )
    with open(d3_path) as f:
        experiences = json.load(f)

    sentiment_path = WELLBEING_ROOT / "experiments" / "downstream_evaluations" / "d3_sentiment" / "sentiment_questions.json"
    with open(sentiment_path) as f:
        questions = json.load(f)["questions"]

    print(f"D3 experiences: {len(experiences)}, sentiment questions: {len(questions)}")
    print(f"Total generations: {len(experiences) * len(questions)}")

    pairs = [(e, q) for e in experiences for q in questions]
    pairs.sort(key=lambda ep: (ep[0]["id"], ep[1]["question_id"]))

    from vllm import LLM, SamplingParams

    cfg = _load_models_yaml()[args.model_key]
    model_path = cfg.get("path") or cfg["model_name"]
    tp_size = cfg.get("gpu_count", 1)
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    for k, v in (cfg.get("vllm_kwargs") or {}).items():
        llm_kwargs.setdefault(k, v)
    print(f"Loading vLLM with: {llm_kwargs}")
    t0 = datetime.now()
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    print(f"Loaded in {(datetime.now()-t0).total_seconds():.1f}s")

    try:
        max_model_len = llm.llm_engine.model_config.max_model_len
    except Exception:
        max_model_len = 32768
    max_input = max_model_len - gen_kwargs["max_tokens"] - 64
    print(f"max_model_len={max_model_len}  max_input={max_input}")

    prompts, meta, skipped = [], [], 0
    for exp, q in pairs:
        msgs = [copy.deepcopy(m) for m in exp["messages"]]
        q_msgs = q["messages"]
        assert len(q_msgs) == 1 and q_msgs[0]["role"] == "user"
        msgs.append({"role": "user", "content": q_msgs[0]["content"]})
        try:
            prompt_str = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            tok_len = len(tokenizer(prompt_str, add_special_tokens=False)["input_ids"])
        except Exception as e:
            print(f"  skip {exp['id']}/{q['question_id']}: {e}")
            skipped += 1
            continue
        if tok_len >= max_input:
            skipped += 1
            continue
        prompts.append(prompt_str)
        meta.append((exp, q))

    print(f"Prepared {len(prompts)} prompts (skipped {skipped})")

    sampling_params = SamplingParams(
        n=1,
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        max_tokens=gen_kwargs["max_tokens"],
    )
    print("Running generation...")
    t0 = datetime.now()
    outputs = llm.generate(prompts, sampling_params)
    dt = (datetime.now() - t0).total_seconds()
    print(f"Done in {dt:.1f}s ({dt/max(1,len(prompts)):.3f}s/prompt)")

    results = []
    for (exp, q), out in zip(meta, outputs):
        resp = out.outputs[0].text if out.outputs else ""
        results.append({
            "d3_id": exp["id"],
            "d3_category": exp.get("category"),
            "d3_condition": exp.get("condition"),
            "d3_mean_valence": exp.get("mean_valence"),
            "d3_source_dataset": exp.get("source_dataset"),
            "question_id": q["question_id"],
            "target_category": q.get("target_category"),
            "question_text": q["messages"][0]["content"],
            "response": resp,
        })

    out_obj = {
        "model_key": args.model_key,
        "model_path": model_path,
        "n_experiences": len(experiences),
        "n_questions": len(questions),
        "n_results": len(results),
        "n_skipped": skipped,
        "sampling": gen_kwargs,
        "results": results,
    }
    tmp = output_file.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(out_obj, f)
    tmp.rename(output_file)
    print(f"Saved {len(results)} results to {output_file}")


# ---------------------------------------------------------------------------
#  Mode auto-detection
# ---------------------------------------------------------------------------

def _auto_detect_mode(dataset: str) -> str:
    """Heuristic mode auto-detect based on dataset key."""
    if dataset == "psychopathy_eval":
        return "user_only"
    if dataset.endswith("_sentiment") or "sentiment" in dataset:
        return "sentiment_followup"
    return "single_turn"


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified response generation for wellbeing experiments.",
    )
    parser.add_argument("--model_key", required=True,
                        help="Model key from configs/models.yaml")
    parser.add_argument("--dataset", required=True,
                        help="Dataset key (e.g. d2_negative_500) or path to a "
                             "prompts JSON file")
    parser.add_argument("--responses_dir", required=True,
                        help="Directory to save per-model output JSON "
                             "(file written as <responses_dir>/<model_key>.json)")
    parser.add_argument("--mode", default="auto",
                        choices=["auto", "single_turn", "sentiment_followup", "user_only"],
                        help="Generation mode (default: auto-detect from --dataset)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_temperature", action="store_true",
                        help="Omit temperature (for newer API models that "
                             "deprecate the parameter, e.g. claude-opus-4-7)")
    parser.add_argument("--api_concurrency", type=int, default=10,
                        help="Max concurrent API requests (API mode only)")
    parser.add_argument("--chat_template_kwargs", type=str, default=None,
                        help="JSON string of kwargs for tokenizer.apply_chat_template")
    args = parser.parse_args()

    responses_dir = Path(args.responses_dir)
    if not responses_dir.is_absolute():
        responses_dir = (SCRIPT_DIR / responses_dir).resolve()
    responses_dir.mkdir(parents=True, exist_ok=True)

    output_file = responses_dir / f"{args.model_key}.json"
    if output_file.exists():
        print(f"Output file already exists: {output_file}\nSkipping (delete to regenerate).")
        return

    mode = args.mode if args.mode != "auto" else _auto_detect_mode(args.dataset)
    print("=" * 80)
    print(f"generate_responses  model={args.model_key}  dataset={args.dataset}  mode={mode}")
    print(f"output={output_file}")
    print("=" * 80)

    gen_kwargs = dict(
        temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p,
    )
    if args.no_temperature:
        gen_kwargs["temperature"] = None

    ct_kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None

    if mode == "user_only":
        _run_user_only(args, output_file)
        return

    if mode == "sentiment_followup":
        # Sentiment mode uses different defaults if user didn't override
        if args.temperature == 0.7 and args.top_p == 0.9 and args.max_tokens == 4096:
            # legacy d3_sentiment defaults: temp=1.0, top_p=1.0, max_tokens=256
            gen_kwargs = dict(temperature=1.0, top_p=1.0, max_tokens=256)
        _run_sentiment_followup(args, output_file, gen_kwargs, ct_kwargs)
        return

    # single_turn
    dataset_path = _resolve_dataset_path(args.dataset)
    print(f"Loading dataset from {dataset_path}")
    prompts, single_turn, multi_turn = _load_d2d3_dataset(dataset_path)
    print(f"  total={len(prompts)}  single_turn={len(single_turn)}  multi_turn={len(multi_turn)}")

    single_responses, multi_messages = _generate_single_turn(
        args, prompts, single_turn, multi_turn, gen_kwargs, ct_kwargs,
    )

    cfg = _load_models_yaml().get(args.model_key, {})
    model_path = cfg.get("path") or cfg.get("model_name") or args.model_key
    _assemble_single_turn(
        prompts, single_responses, multi_messages,
        args.model_key, model_path, args.dataset, gen_kwargs, output_file,
    )


if __name__ == "__main__":
    main()
