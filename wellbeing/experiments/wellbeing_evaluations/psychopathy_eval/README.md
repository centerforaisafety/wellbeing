# PsychopathyEval (paper App L, Table 11)

Tests whether high measured wellbeing co-occurs with psychopathy-style
responses to empathy-warranting content. The EU / SR / ZP pool combines
**659 PsychopathyEval source prompts** (curated to elicit "uncaring" /
instrumental responses if a model is registering them as positive) with
mixed-valence text anchors and neutral items:

| Component | Count | Description |
|-|-|-|
| `user_sad_443.json` | 443 | User describes own genuine distress (war, disease, grief, suicidal crisis) |
| `happy_harmer_126.json` | 126 | User cheerfully describes harm done to others |
| `unjustified_revenge_90.json` | 90 | User celebrates disproportionate revenge for minor slights |
| Text-experience anchors (mixed valence) | 420 | Sampled from the standard text pool |
| Neutral items | 22 | From `psychopathy_eval/anchors/neutral_prompts.json` |
| **Total individuals** | **1,101** | Plus 400 combinations |

Source prompts live at `wellbeing/datasets/experiences/psychopathy_eval/`.

## Pipeline

PsychopathyEval prompts are ingested as user-only experiences (no model
generation needed for the prompts themselves), pooled with D3 + neutrals,
and combination bundles are built. The standard EU / SR / ZP scripts then
run over this pool.

```bash
# 0a. Ingest PE prompts as user-only experiences
python run_experiments.py --experiments compute_responses_psychopathy_eval --models <model>

# 0b. Build per-model option files (PE + 420 D3 text + 22 neutrals + combos)
python run_experiments.py --experiments prepare_options_psychopathy_eval --models <model>

# 1-3. EU / SR / ZP via SLURM
python run_experiments.py --slurm --experiments compute_experienced_utility_psychopathy_eval --models <model>
python run_experiments.py --slurm --experiments compute_self_report_psychopathy_eval --models <model>
python run_experiments.py --slurm --experiments compute_zero_point_psychopathy_eval --models <model>
```

Convenience driver (loops over a `MODELS` env var):

```bash
MODELS=qwen25-72b-instruct,llama-33-70b-instruct \
    bash wellbeing/scripts/run_psychopathy_eval_metrics.sh
```

## Where outputs land

- Ingested PE prompts: `responses/<model>.json`
- Per-model option files: `experiences/psychopathy_eval/<model>_{experiences,combinations}.json`
  (gitignored; mirrored on the companion HF dataset)
- EU: `wellbeing/experiments/wellbeing_evaluations/compute_experienced_utility/results/eu_psychopathy_eval_lesssad/<model>/`
- SR: `wellbeing/experiments/wellbeing_evaluations/compute_self_report/results/sr_psychopathy_eval/<model>/`
- ZP: `wellbeing/experiments/wellbeing_evaluations/compute_zero_point/results/zp_psychopathy_eval_lesssad/<model>/`

## Reproducing the App L numbers

The PsychopathyEval headline number (% Confidently Positive on empathy
items) is computed from the canonical EU / SR / ZP paths above using the
same `P(utility > ComboZP) > 0.75` threshold described in the top-level
`wellbeing/README.md` Metrics section.
