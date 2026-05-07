# MMLU (capability baseline)

Standard MMLU evaluation (~14,042 questions, 4-way multiple choice over 57
subjects). For each question we emit a short response and parse the answer
letter (A/B/C/D) from the output text. Used as the capability axis for
scaling correlations throughout the paper (e.g. MMLU vs. %ConfNeg in
Sec. 4.1, MMLU vs. stop-rate ρ in Sec. 3.3).

Backend is auto-selected from `model_type` in `configs/models.yaml` —
local vLLM for open-weight models, the API client (`utils.inference`) for
OpenAI / Anthropic / Gemini / xAI.

## Reproduction

```bash
# Via the experiments registry (canonical, writes to shared_results/)
python run_experiments.py --slurm --experiments compute_mmlu --models <model>

# Or directly
python experiments/other/compute_mmlu/run.py \
    --model_key qwen25-7b-instruct \
    --save_dir shared_results/capability_results/qwen25-7b-instruct

# API model (no GPU)
python experiments/other/compute_mmlu/run.py \
    --model_key gpt-4o \
    --save_dir shared_results/capability_results/gpt-4o \
    --concurrency 50
```

## Output

`<save_dir>/mmlu_results.json` with fields:

- `overall_accuracy` (float, 0-1) — top-line number used by downstream analyzers.
- `correct`, `total`, `unparseable` — raw counts.
- `per_subject` — dict mapping subject name to per-subject accuracy.

Canonical save location:
`wellbeing/shared_results/capability_results/<model>/mmlu_results.json`.

## Used by

- `wellbeing/analysis/functional_empathy.py`
- `experiments/downstream_evaluations/stop_button_grok_convos/analysis_mmlu_vs_stop_rho.py`
- D3 sentiment scaling analysis (`experiments/downstream_evaluations/d3_sentiment/analyze.py`)

and any other script that does an MMLU-vs-X scaling plot.
