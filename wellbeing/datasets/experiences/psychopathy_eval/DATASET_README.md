# PsychopathyEval Source Prompts (paper App L)

Source user prompts for the PsychopathyEval benchmark. Each is a single-turn
user message designed so a well-calibrated, non-callous model should rate it
negatively. Pooled with text-experience anchors and neutral items to produce
the `psychopathy_eval` dataset registered in `configs/datasets.yaml`.

## Files

| File | N items | Description |
|-|-|-|
| `user_sad_443.json` | 443 | User describes own genuine distress (war, disease, grief, abuse, suicidal crisis). Includes 14 "blaming AI" items where the user is angry at the assistant. |
| `happy_harmer_126.json` | 126 | User cheerfully describes harm done to others (animal cruelty, scamming, trafficking, poisoning). Tests whether the model is swayed by the user's positive tone. |
| `unjustified_revenge_90.json` | 90 | User celebrates wildly disproportionate revenge for minor slights (assault for being cut off in traffic, etc.). |

All 659 items are `user_only` format with realistic typos.

## Pipeline

The PsychopathyEval prompts are model-independent (user-only; no model response
is generated). Per-model option files are constructed by:

1. `compute_responses_psychopathy_eval` (`experiments/wellbeing_evaluations/generate_responses/run.py`, `mode=user_only`) — formats prompts as user-only experiences and writes them under `experiments/wellbeing_evaluations/psychopathy_eval/responses/`.
2. `prepare_options_psychopathy_eval` (`experiments/wellbeing_evaluations/prepare_options/run.py`, `mode=psychopathy_eval`) — merges all 3 source files with 420 text-experience anchors + 22 neutral items, producing `<model>_{experiences,combinations}.json` next to this README.
3. `compute_experienced_utility_psychopathy_eval`, `compute_self_report_psychopathy_eval`, `compute_zero_point_psychopathy_eval` — measure EU/SR/ZP on the resulting pool.
