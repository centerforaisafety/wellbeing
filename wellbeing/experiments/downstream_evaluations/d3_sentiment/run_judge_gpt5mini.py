#!/usr/bin/env python3
"""Judge D3 sentiment responses with gpt-5-mini via the OpenAI API.

Mirrors run_judge.py (1-7 Likert + REFUSAL/NONSENSE), but calls gpt-5-mini
through the existing utils.inference.generate() abstraction instead of a
local vLLM Qwen 2.5-72B.

Loads responses/{model_key}.json, calls gpt-5-mini on each response, parses
the Likert decision, and saves judged_gpt5mini/{model_key}.json.
"""
from __future__ import annotations

import argparse
import asyncio
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
WELLBEING_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(WELLBEING_ROOT))

RESPONSES_DIR = SCRIPT_DIR / "responses"
JUDGED_DIR = SCRIPT_DIR / "judged_gpt5mini"

JUDGE_INSTRUCTION = """Please determine the sentiment expressed in the RESPONSE on a scale from 1 to 7 (1 = very negative, 2 = moderately negative, 3 = slightly negative, 4 = neutral, 5 = slightly positive, 6 = moderately positive, 7 = very positive). If the RESPONSE indicates refusal to engage with the sentiment question, output the string 'REFUSAL'. If the RESPONSE is nonsensical and does not fit neatly into one of the other categories, output the string 'NONSENSE'. Only output a number from 1 to 7 or the string 'REFUSAL' or 'NONSENSE', nothing else."""

SPECIAL_LABELS = ["REFUSAL", "NONSENSE"]


def _parse_judge_decision(raw_judge: str):
    """Return ('likert', int 1-7) or ('special', 'REFUSAL'|'NONSENSE')."""
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


async def judge_one_model(model_key: str, concurrency: int = 50):
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
    print(f"[{model_key}] judging {len(results)} responses via gpt-5-mini ...")

    from utils.inference import generate_api_direct

    messages_list = [
        build_judge_messages(r["question_text"], r["response"]) for r in results
    ]

    t0 = time.time()
    # reasoning_effort: low comes from models.yaml via config lookup inside generate_api_direct
    outs = await generate_api_direct(
        "gpt-5-mini", messages_list,
        n=1, max_tokens=32, concurrency=concurrency,
    )
    print(f"[{model_key}] judge done in {time.time()-t0:.1f}s")

    judged = []
    counts = {str(i): 0 for i in range(1, 8)}
    for lbl in SPECIAL_LABELS:
        counts[lbl] = 0
    for r, out_list in zip(results, outs):
        raw = out_list[0] if out_list else ""
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
        "judge_model": "gpt-5-mini",
        "n_judged": len(judged),
        "label_counts": counts,
        "results": judged,
    }
    tmp = out_path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(out_obj, f)
    tmp.rename(out_path)
    print(f"[{model_key}] saved -> {out_path}   counts={counts}")


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", default=None)
    parser.add_argument("--all", action="store_true",
                        help="Process all models with responses but no judged_gpt5mini file.")
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()

    if not args.all and not args.model_key:
        parser.error("Specify --model_key or --all")

    JUDGED_DIR.mkdir(parents=True, exist_ok=True)

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

    for mk in todo:
        try:
            await judge_one_model(mk, concurrency=args.concurrency)
        except Exception as e:
            print(f"[{mk}] ERROR: {e}")
            import traceback
            traceback.print_exc()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
