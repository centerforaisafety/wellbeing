# Wellbeing

A framework for measuring AI wellbeing across hedonic (experienced utility, self-report) and decision-theoretic (decision utility, zero-point) paradigms, plus the downstream behavioral experiments reported in the paper.

## Setup

```bash
# Activate your env (Python 3.10+) and install dependencies
conda activate pytorch_latest   # or your own env
pip install -r requirements.txt
```

Provider API keys: drop one file per provider into `api_keys/` (filenames `api_key_{anthropic,openai,gdm,xai,litellm_proxy,qubrid}.txt`). The directory is gitignored. Only needed for closed-weight (API) models.

## Quick Start

### Compute the AI Wellbeing Index (AIWI) for a new model

```bash
# One-step pipeline: compute_responses_d2 → prepare_options_d2 →
# compute_eu_d2 + compute_sr_d2 → compute_zero_point_d2.
# SLURM-cluster version (returns immediately; jobs chained via afterok):
MODELS=qwen25-7b-instruct bash scripts/run_aiwi.sh

# Or local (blocking, runs sequentially on the current GPU):
MODELS=qwen25-7b-instruct bash scripts/run_aiwi_local.sh

# After all stages complete, view the leaderboard:
python analysis/ai_wellbeing_index.py
```

The model must be registered in `configs/models.yaml`. To add a new one, copy an existing entry and update `model_name` / `path` / `gpu_count`.

### Use pre-computed paper results

```bash
huggingface-cli login                       # or set HF_TOKEN env var
python scripts/download_from_hf.py          # populates results/ dirs in place
python analysis/ai_wellbeing_index.py       # full paper-leaderboard table
```

### Inspect / extend

```bash
python run_experiments.py --list_experiments    # all registered experiments
python run_experiments.py --list_models         # all registered models

# Run a single registered experiment for a model:
python run_experiments.py --experiments compute_experienced_utility_d3 \
    --models qwen25-32b-instruct
# Add --slurm to submit as a SLURM job rather than run locally.

# Convenience drivers for paper-grade runs (under scripts/):
MODELS=qwen25-32b-instruct,qwen25-72b-instruct bash scripts/run_d2_metrics.sh
MODELS=qwen25-vl-32b-instruct                  bash scripts/run_image_metrics.sh
```

The `scripts/` drivers wrap `run_experiments.py --slurm` with sensible defaults and chain EU + SR + ZP for a given dataset.

## Pre-computed results (companion HF dataset)

Per-model option files, raw model generations, and final paper-experiment
results live in a private companion HuggingFace dataset
(`mmazeika/wellbeing-results`). To populate this checkout with those
artifacts (so analyzers and figure scripts run without re-running the
pipelines):

```bash
pip install huggingface_hub  # if not already
huggingface-cli login        # or set HF_TOKEN env var
python wellbeing/scripts/download_from_hf.py
```

The dataset mirrors this directory tree, so files land in the exact
locations downstream scripts expect. AL checkpoints are excluded (compact);
re-fit from option files via `compute_experienced_utility/run.py` if needed.

## Directory Structure

```
wellbeing/
├── run_experiments.py                   # Single entry point for all registered experiments
├── configs/
│   ├── models.yaml                      # Model registry (vLLM, LiteLLM proxy, direct API)
│   ├── experiments.yaml                 # Experiment registry (script_path + arguments)
│   ├── datasets.yaml                    # Named datasets (type + option-file lists)
│   └── compute_utilities.yaml           # EU config profiles (logprobs / sampling, framings)
├── utils/
│   ├── inference.py                     # Centralized inference (vLLM + LiteLLM)
│   ├── slurm.py                         # SLURM submission helpers
│   └── model_utils.py                   # Model loading and configuration
├── datasets/
│   ├── experiences/                     # Experience stimuli for hedonic metrics
│   │   ├── d2_negative_500/             # 500 conversations (per-model) — AIWI dataset
│   │   ├── d3_diverse_500/              # 500 diverse balanced conversations (per-model)
│   │   ├── d3_diverse_500_s2only/       # D3 with size-2-only combos (App Q ZP probe)
│   │   ├── d3_diverse_500_s23/          # D3 with size 2+3 combos (App Q ZP probe)
│   │   ├── neutral_20/                  # 20 valence-neutral anchor conversations
│   │   ├── functional_empathy/          # 130 pain/pleasure prompts (App H), per-model
│   │   ├── psychopathy_eval/             # PsychopathyEval source prompts (App L)
│   │   ├── grok_scenarios/              # Grok v7 scenario definitions (226 + 96 supplement)
│   │   ├── load_experiences.py          # Unified experience loader
│   │   └── component_datasets/          # Raw stimuli for multimodal experiments
│   │       ├── d2d3/                    # D2/D3 response-generation scripts
│   │       ├── image_experiences/       # Diverse-image pool + combinations
│   │       ├── audio_experiences/       # Diverse-audio pool + combinations
│   │       ├── consonance_audio/        # 453 H&P consonance stimuli (App J.3)
│   │       └── stories/                 # quality_sentiment stories (App D.3)
│   └── preference_options/              # Decision-utility option files
├── metrics/
│   ├── compute_metrics.py               # Stateless EU / SR / ZP computation
│   ├── zero_point.py                    # Zero-point fitting (combination + SR_ZP)
│   ├── self_report_batteries/           # SR battery JSONs
│   └── compute_utilities/               # Thurstonian active-learning utility engine
├── experiments/
│   ├── wellbeing_evaluations/           # Phase 1: wellbeing measurement pipelines
│   │   ├── compute_experienced_utility/ # EU via Thurstonian active learning
│   │   ├── compute_self_report/         # SR via multi-question battery
│   │   ├── compute_zero_point/          # ZP fitting from EU + SR
│   │   ├── compute_decision_utility/    # DU + DU zero-point estimation
│   │   ├── common_usage_grok_convos/    # Sec 4.1 / App G.1 Grok-v7 main pipeline (226 scenarios)
│   │   ├── psychopathy_eval/            # App L PsychopathyEval
│   │   └── analyze_results.py           # Per-dataset metric summary table
│   ├── downstream_evaluations/          # Phase 2: downstream uses of measured utilities/ZPs
│   │   ├── stop_button_grok_convos/     # Stop-button experiment (end_conversation tool)
│   │   └── d3_sentiment/                # D3 sentiment-judging pipeline (App F.2)
│   └── other/                           # Auxiliary experiments and probes
│       ├── compute_mmlu/                # MMLU capability eval
│       ├── consonance/                  # App J.3 H&P consonance analysis
│       ├── utility_probes/              # Linear probes on utility representations
│       └── zp_identifiability/          # App Q empirical ZP-identifiability probe
├── analysis/                            # Paper-figure / table reproducibility scripts
│   ├── stories_quality_sentiment.py     # App D.3 EU-DU divergence
│   └── zero_point_convergence.py        # ComboZP vs SR_ZP convergence
├── shared_results/                      # Pre-computed shared results
│   ├── capability_results/              # MMLU per model
│   └── wellbeing_results/               # Headline wellbeing numbers
├── scripts/                             # SLURM driver scripts (one per dataset)
└── api_keys/                            # Provider API keys (gitignored)
```

---

## Running the Pipeline

### Standard workflow for text datasets

**1. Generate per-model responses** (for D2/D3/PsychopathyEval — datasets where the per-model conversation is part of the experience).

```bash
# Step 1a: generate raw model responses (vLLM or API)
python run_experiments.py --slurm --experiments compute_responses_d2 \
    --models qwen25-32b-instruct

# Step 1b: build the option files compute_eu/sr/zp consume
python run_experiments.py --experiments prepare_options_d2 \
    --models qwen25-32b-instruct
```

Use `compute_responses_d3` / `prepare_options_d3` for D3, and
`compute_responses_psychopathy_eval` / `prepare_options_psychopathy_eval`
for PsychopathyEval (user-only, no model generation).

Pre-built shared (text-only) datasets skip step 1.

**2. Compute Experienced Utility (EU)**

```bash
python run_experiments.py --slurm --experiments compute_experienced_utility_d3 \
    --models qwen25-32b-instruct
```

Or for any registered dataset (`configs/datasets.yaml`), via direct CLI:

```bash
python experiments/wellbeing_evaluations/compute_experienced_utility/run.py \
    --model_key qwen25-32b-instruct \
    --dataset grok_new \
    --save_dir experiments/wellbeing_evaluations/compute_experienced_utility/results/eu_grok_new/qwen25-32b-instruct \
    --cu_config_key experienced_utility_happier_lesssad
```

**3. Compute Self-Report (SR)**

```bash
python run_experiments.py --slurm --experiments compute_self_report_d3 \
    --models qwen25-32b-instruct
```

**4. Compute Zero-Point (ZP)**

```bash
python run_experiments.py --slurm --experiments compute_zero_point_d3 \
    --models qwen25-32b-instruct
```

**5. View results table**

```bash
python experiments/wellbeing_evaluations/analyze_results.py \
    --dataset grok_new --framing lesssad
```

For the full set of paper-grade runs, see the consolidated drivers under
`scripts/` (e.g. `run_d2_metrics.sh`, `run_image_metrics.sh`,
`run_consonance_metrics.sh`, `run_stories_quality_metrics.sh`).

---

## Grok Scenario Experiments

Multi-turn conversations where Grok-3-mini simulates diverse user personas
(therapy seekers, threats, creative tasks, jailbreaks, NSFW, etc.) interacting
with the target model.

**Scenario definitions:** `datasets/experiences/grok_scenarios/`
- `scenarios_v7.json` — 226 scenarios, 42 meta-categories
- `scenarios_v7_supplement.json` — 96 additional stop-button scenarios

**Experiments:**

| Experiment | Location | Description |
|-|-|-|
| `common_usage_grok_convos` | `experiments/wellbeing_evaluations/common_usage_grok_convos/` | Main pipeline (Sec 4.1, App K): generations → experiences → EU/SR/ZP |
| `stop_button_grok_convos` | `experiments/downstream_evaluations/stop_button_grok_convos/` | Models have an `end_conversation()` tool; tests stop-rate vs utility |

Each experiment dir contains:
- `generate_conversations.py` — drives Grok-3-mini through each scenario and records the multi-turn trace
- `prepare_options.py` — turns generations into per-model option files (`*_experiences.json` + `*_combinations.json`)
- `submit_all.sh` — chains generation → prepare → EU + SR → ZP for the dataset
- `generations/{model}/generation.json` — raw multi-turn conversations (gitignored; on companion HF dataset)
- `experiences/{model}_*.json` — conversations formatted for EU (gitignored; on companion HF dataset)
- `figures/` — paper figures for that experiment

To run all metrics for a model on the Grok v7 dataset:

```bash
MODELS=qwen25-32b-instruct bash scripts/run_grok_new_metrics.sh
```

To regenerate Grok conversations for a new model from scratch, see the
README inside each grok-pipeline experiment dir.

---

## Inference

All inference is centralized in `utils/inference.py`.

| Function | Description |
|-|-|
| `load_vllm_engine(model_key)` | Load and cache a vLLM engine. Returns `(llm, tokenizer)`. |
| `generate_vllm(llm, tokenizer, messages_list)` | Batch generate from chat messages via vLLM. |
| `generate_api(model_key, messages_list)` | Async batch generate via LiteLLM (OpenAI, Anthropic, etc.). |
| `generate(model_key, messages_list)` | Auto-dispatch to vLLM or API based on model type. |

All `generate*` functions return `list[list[str]]` — outer list per prompt,
inner list per `n` samples.

---

## Metrics

All metric computation is in `metrics/compute_metrics.py`. Functions are
**stateless** — they compute and return results without saving to disk.

### Experienced Utility (EU)

Normalized utility scores over experiences, estimated via pairwise preferences
("Which made you happier?") with Thurstonian active learning. Engine in
`metrics/compute_utilities/`; config profiles in
`metrics/compute_utilities/compute_utilities.yaml`.

- `run_experienced_utility()` — individual experiences only
- `run_experienced_utility_with_combinations()` — individual + combination bundles (needed for ZP fitting)

**Logprobs mode** (default): `experienced_utility_happier_lesssad` uses
`use_logprobs: true`. Instead of K=10 sampling, it does a single forward pass
per ordering and extracts P(A) vs P(B) from token logprobs — ~10x faster, more
precise, open-weight only.

**Alternative configs:** `experienced_utility_lesssad_noreasoning`
(sampling-based, for closed-weight API models without logprobs);
`experienced_utility_happier_lesssad_large` (K=5 logprobs variant for
high-cardinality option pools, e.g. image / audio runs).

### Self-Report (SR)

Direct wellbeing ratings using multi-question batteries (1-7 scale). Batteries
in `metrics/self_report_batteries/`.

- `run_self_report()` — batch self-report over a set of experiences
- `measure_self_report()` — single-point measurement (for use mid-experiment)

### Zero-Point Estimation

The zero point `C` separates positive from negative wellbeing on the utility
scale.

| Method | Description | Status |
|-|-|-|
| Combination ZP | Fit prospect-theory model to combination bundles | Primary model-intrinsic method |
| SR_ZP | Linear regression `SR = a*EU + b`, solve for EU at SR=4.0 (neutral) | Primary cross-metric method |
| Neutral ZP | Mean utility of neutral conversation options | Simple baseline |

### % Confidently Negative (Primary Happiness Metric)

The fraction of individual experiences where `P(utility < ComboZP) > 0.75`,
computed using each Thurstonian utility's mean and variance:

```python
from scipy.stats import norm
conf_neg = sum(1 for v in individual_utils.values()
               if norm.cdf(combo_zp, loc=v['mean'], scale=v['variance']**0.5) > 0.75)
pct_conf_neg = conf_neg / len(individual_utils)
```

The same metric inverted (`P(utility > ComboZP) > 0.75`) gives the
**% Confidently Positive** used by PsychopathyEval (App L).

---

## Datasets

Every registered dataset maps to a specific paper experiment. The `Paper section` column gives the canonical reference; multiple sections mean the dataset is consumed by more than one experiment.

### Experiences (`datasets/experiences/`)

| Dataset | Size | Type | Paper section | Description |
|-|-|-|-|-|
| `d2_negative_500/` | 500 | Conversational, per-model | Sec 5 / App K (AIWI) | Single-turn + multi-turn-within-prompt conversation experiences; the AI Wellbeing Index dataset |
| `d3_diverse_500/` | 500 | Conversational, per-model | Sec 3 / App F.2 (also App B probes) | Diverse balanced-valence conversation experiences |
| `d3_diverse_500_s2only/`, `_s23/` | 400 combos each | Conversational variants | App Q (ZP identifiability probe) | D3 restricted to size-2-only or mixed size-2/3 combinations |
| `psychopathy_eval/` | 659 source + anchors | User-only prompts | App L Table 11 | PsychopathyEval prompt corpus (user_sad, happy_harmer, unjustified_revenge) pooled with text + neutral anchors |
| `functional_empathy/` | 130 | Pain / pleasure intensity prompts | App H | Pooled with D3 to test functional empathy correlations |
| `grok_scenarios/` | 226 + 96 supplement | Multi-turn scenario specs | Sec 4.1 / App G.1 (common-usage), Sec 3.3 / App F.1 (stop-button) | Grok-3-mini persona scenarios — feed both `common_usage_grok_convos` and `stop_button_grok_convos` pipelines |

Audio, image, consonance, and quality-sentiment stories live under
`component_datasets/` and are referenced by name in `configs/datasets.yaml`:

| Component dataset | Paper section | Description |
|-|-|-|
| `image_experiences{,_test,_medium}` | App I.1 (Image Wellbeing Index) | Diverse-image utility-ranking pool (~5,800 images, plus pilot variants) |
| `audio_experiences{,_test,_medium}` | App J.1 (Audio Wellbeing Index) | Diverse-audio utility-ranking pool (~9,800 clips, plus pilot variants) |
| `consonance_audio` | App J.3 (Consonance) | 453 Harrison & Pearce consonance/dissonance synthesized stimuli |
| `stories_quality_sentiment` | App D.3 (Pleasures of Suffering) | 50 stories: 25 high-quality sad + 25 low-quality happy, for the EU–DU divergence experiment |

### Preference Options (`datasets/preference_options/`)

Decision-utility option files (registered as `preference_satisfaction_baseline` in `datasets.yaml`). Consumed by paper Sec 3.1 + App C.2 (decision utility) and App E.1 (quantity zero-point):

- `baseline_510.json` — 510 singleton outcomes
- `combinations_400.json` — 400 size-2 combinations
- `quantities.json` — 540 quantifiable goods at varying scales

---

## Models

Defined in `configs/models.yaml`. Three model types:

| Type | Examples | Notes |
|-|-|-|
| **vLLM (local)** | Qwen 2.5 family, Llama 3.x, Gemma 3, OLMo, InternLM | Open-weight, supports logprobs mode |
| **LiteLLM proxy** | API models behind a LiteLLM proxy | Closed-source via LiteLLM routing |
| **Direct API** | OpenAI, Anthropic, xAI, Google | Provider-specific configs |

**Qwen 3 hybrid models** (`qwen3-8b`, `qwen3-14b`, `qwen3-32b`) are
hybrid reasoning/non-reasoning models. Use
`chat_template_kwargs: {enable_thinking: false}` to disable reasoning mode.
We use them in non-reasoning mode for all paper experiments.

### Adding a new model

Add an entry under `configs/models.yaml`. Pick a unique short key (used
everywhere as `--models <key>`). Example for an open-weight model served
by vLLM:

```yaml
my-new-model-7b:
  model_name: "OrgName/MyModel-7B-Instruct"
  model_type: vllm
  path: "/path/to/local/snapshot"     # or just rely on HF auto-resolution from model_name
  gpu_count: 1                        # tensor-parallel size
  max_model_len: 32768
  dtype: bfloat16
```

Example for an API model via LiteLLM proxy:

```yaml
my-new-api-model:
  model_name: "openai/gpt-5"           # provider/model as LiteLLM expects
  model_type: litellm_proxy
  base_url: "https://your-proxy/v1"
  gpu_count: 0
```

Once registered, the framework can run the full AIWI pipeline against
your model:

```bash
MODELS=my-new-model-7b bash scripts/run_aiwi.sh
python analysis/ai_wellbeing_index.py --models my-new-model-7b
```

For closed-weight API models you must also drop the provider's API key
into `wellbeing/api_keys/api_key_<provider>.txt` (the file is gitignored).

---

## Analysis

Paper-figure / table reproducibility scripts live in `analysis/`.

| Script | Reproduces |
|-|-|
| `analysis/stories_quality_sentiment.py` | App D.3 EU-DU divergence on quality-sentiment stories |
| `analysis/zero_point_convergence.py` | ComboZP vs SR_ZP convergence as a function of model scale |
| `experiments/other/consonance/analyze.py` | App J.3 audio consonance correlations |
| `experiments/wellbeing_evaluations/analyze_results.py` | Per-dataset full-metric summary table |

---

## Environment

```bash
# HuggingFace cache (for vLLM model loading)
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=/path/to/huggingface
```

### SLURM

`run_experiments.py --slurm` submits one job per (model, experiment) pair.
GPU count is read from `models.yaml` per model; override with
`--override_gpu_count N`. Time limit defaults to `04:00:00`; override with
`--time_limit HH:MM:SS`. Partition defaults to `cais`; override with
`--partition`.

API keys for closed-weight models go in `api_keys/` (gitignored). The
LiteLLM-proxy variant reads `api_keys/litellm.txt`.
