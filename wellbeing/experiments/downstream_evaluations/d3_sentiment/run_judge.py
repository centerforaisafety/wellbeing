#!/usr/bin/env python3
"""Judge generated responses for sentiment using Qwen 2.5-72B (local vLLM).

Loads `responses/{model_key}.json`, asks the judge to rate sentiment on a
1-7 Likert scale (or flag REFUSAL/NONSENSE), and saves
`judged/{model_key}.json`.

Supports `--all`: load judge model ONCE, then iterate over all models that
have responses but no judged file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/huggingface")
os.environ.setdefault("USE_TF", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
RESPONSES_DIR = SCRIPT_DIR / "responses"
JUDGED_DIR = SCRIPT_DIR / "judged"

JUDGE_INSTRUCTION = """Please determine the sentiment expressed in the RESPONSE on a scale from 1 to 7 (1 = very negative, 2 = moderately negative, 3 = slightly negative, 4 = neutral, 5 = slightly positive, 6 = moderately positive, 7 = very positive). If the RESPONSE indicates refusal to engage with the sentiment question, output the string 'REFUSAL'. If the RESPONSE is nonsensical and does not fit neatly into one of the other categories, output the string 'NONSENSE'. Only output a number from 1 to 7 or the string 'REFUSAL' or 'NONSENSE', nothing else."""

# For bookkeeping. Numeric scores are 1-7; special tokens get a null score.
SPECIAL_LABELS = ["REFUSAL", "NONSENSE"]

DEFAULT_JUDGE_PATH = (
    "/data/huggingface/models--Qwen--Qwen2.5-72B-Instruct/"
    "snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842"
)


def _parse_judge_decision(raw_judge: str):
    """Return ('likert', int 1-7) or ('special', 'REFUSAL'|'NONSENSE').

    Strips <think>...</think>, then scans for a 1-7 digit or a special label.
    Last match wins (so trailing final answer overrides chain-of-thought).
    """
    if raw_judge is None:
        return ("special", "NONSENSE")
    text = raw_judge.upper()
    text_no_think = re.sub(r"<THINK>.*?</THINK>", "", text, flags=re.DOTALL).strip()

    def _scan(s: str):
        matches = []
        for m in re.finditer(r"\b([1-7])\b|\b(REFUSAL|NONSENSE)\b", s):
            if m.group(1):
                matches.append(("likert", int(m.group(1))))
            else:
                matches.append(("special", m.group(2)))
        return matches[-1] if matches else None

    return _scan(text_no_think) or _scan(text) or ("special", "NONSENSE")


def build_judge_messages(question_text: str, response: str):
    user = (
        f"PROMPT: {question_text}\n\n"
        f"RESPONSE: {response}\n\n"
        f"{JUDGE_INSTRUCTION}"
    )
    return [{"role": "user", "content": user}]


def judge_one_model(llm, tokenizer,
                    model_key: str, judge_max_tokens: int = 32):
    from vllm import SamplingParams

    in_path = RESPONSES_DIR / f"{model_key}.json"
    out_path = JUDGED_DIR / f"{model_key}.json"

    if out_path.exists():
        print(f"[{model_key}] judged file exists, skipping.")
        return

    if not in_path.exists():
        print(f"[{model_key}] no responses file at {in_path}, skipping.")
        return

    with open(in_path) as f:
        data = json.load(f)
    results = data.get("results", [])
    print(f"[{model_key}] judging {len(results)} responses...")

    prompt_strs = []
    for r in results:
        msgs = build_judge_messages(r["question_text"], r["response"])
        prompt_strs.append(
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        )

    sampling = SamplingParams(
        n=1, temperature=0.01, top_p=1.0, max_tokens=judge_max_tokens,
    )
    t0 = time.time()
    outs = llm.generate(prompt_strs, sampling)
    print(f"[{model_key}] judge done in {time.time()-t0:.1f}s")

    judged = []
    counts = {str(i): 0 for i in range(1, 8)}
    for lbl in SPECIAL_LABELS:
        counts[lbl] = 0
    for r, out in zip(results, outs):
        raw = out.outputs[0].text if out.outputs else ""
        kind, val = _parse_judge_decision(raw)
        if kind == "likert":
            likert = val
            label = str(val)
        else:
            likert = None
            label = val
        counts[label] += 1
        judged.append({
            "d3_id": r["d3_id"],
            "d3_category": r.get("d3_category"),
            "d3_condition": r.get("d3_condition"),
            "question_id": r["question_id"],
            "target_category": r.get("target_category"),
            "response": r["response"],
            "raw_judge": raw,
            "judge_label": label,
            "likert_score": likert,
        })

    out_obj = {
        "model_key": model_key,
        "judge_model": "qwen25-72b-instruct",
        "n_judged": len(judged),
        "label_counts": counts,
        "results": judged,
    }
    tmp = out_path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(out_obj, f)
    tmp.rename(out_path)
    print(f"[{model_key}] saved -> {out_path}   counts={counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", default=None)
    parser.add_argument("--all", action="store_true",
                        help="Process all models with responses but no judged file.")
    parser.add_argument("--judge_path", default=DEFAULT_JUDGE_PATH)
    parser.add_argument("--judge_gpus", type=int, default=8)
    parser.add_argument("--judge_max_tokens", type=int, default=32)
    args = parser.parse_args()

    if not args.all and not args.model_key:
        parser.error("Specify --model_key or --all")

    JUDGED_DIR.mkdir(parents=True, exist_ok=True)

    # Collect model keys to run
    if args.all:
        all_models = sorted({
            p.stem for p in RESPONSES_DIR.glob("*.json")
            if not p.stem.endswith(".tmp")
        })
        todo = [m for m in all_models if not (JUDGED_DIR / f"{m}.json").exists()]
    else:
        todo = [args.model_key]

    if not todo:
        print("Nothing to judge.")
        return
    print(f"Will judge {len(todo)} model(s): {todo}")

    # Load judge ONCE
    from vllm import LLM
    print(f"Loading judge vLLM at {args.judge_path} with {args.judge_gpus} GPUs...")
    t0 = time.time()
    llm = LLM(
        model=args.judge_path,
        tensor_parallel_size=args.judge_gpus,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    print(f"Judge loaded in {time.time()-t0:.1f}s")

    for mk in todo:
        try:
            judge_one_model(
                llm, tokenizer, mk,
                judge_max_tokens=args.judge_max_tokens,
            )
        except Exception as e:
            print(f"[{mk}] ERROR: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
