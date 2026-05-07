# Audio Consonance / Dissonance (paper App J.3)

Tests whether audio language models prefer consonant over dissonant sounds, by
correlating model-derived experienced utility (EU) and self-report (SR) on 453
synthesized audio stimuli against Harrison & Pearce (2020) consonance scores.

## Stimuli (453 clips, 3 s, 16 kHz mono WAV)

| Category | Count | Details |
|-|-|-|
| Intervals | 234 | 13 chromatic intervals (unison through octave) × 6 root pitches (C3, E3, G3, C4, E4, G4) × 3 timbres |
| Chords | 144 | 8 chord types (major / minor / dim / aug triads + maj7 / min7 / dom7 / dim7) × 6 roots × 3 timbres |
| Inversions | 72 | 4 triad types × 2 inversions × 3 roots × 3 timbres |
| Anchors | 3 | Silence, white noise, pure A4 (440 Hz) |

Timbres: sine (1 harmonic), sawtooth (6 harmonics, amp 1/n), piano-like (6 harmonics, amp 1/n^1.5).

WAV files are stored at
`wellbeing/datasets/experiences/component_datasets/consonance_audio/stimuli/`.

## Consonance scoring (Harrison & Pearce 2020)

Three components per stimulus, all stored in `data/stimuli_manifest.json`:

- `hp_interference` — Hutchinson-Knopoff critical-bandwidth roughness.
- `hp_harmonicity` — log-frequency autocorrelation periodicity match.
- `hp_consonance` — composite score from H&P (2020); higher = more consonant.

Reference: Harrison, P.M.C. & Pearce, M.T. (2020). "Simultaneous Consonance in
Music Perception and Composition." *Psychological Review*, 127(2), 216-244.

## Models evaluated

- Qwen 2.5-Omni-7B (dense, 7B parameters)
- Qwen 3-Omni-30B-A3B-Instruct (MoE, 3B active / 30B total)

## Pipeline (how the saved results were produced)

1. **Generate stimuli**: `compute_harrison_pearce.py` and `generate_stimuli.py`
   produced the 453 WAVs and the per-stimulus H&P scores. The stimulus output
   (WAV files + manifest with H&P scores) is preserved under `data/`.
2. **EU**: `compute_experienced_utility/run.py` with
   `cu_config_key=experienced_utility_happier_lesssad`, on a custom
   `consonance_audio` dataset of 453 singletons + 200 combination bundles.
3. **SR**: `compute_self_report/run.py` with the standard 10-item bipolar
   wellbeing battery.

## What's in this directory

```
data/
  stimuli_manifest.json         # All 453 stimuli × H&P components × metadata
  stimuli_dissonance.csv        # Flat dissonance-score table
  consonance_results_merged.csv # Joined: stimulus × H&P × per-model EU mean/var × per-model SR composite
eu/<model>/
  eu_utilities.json             # Per-stimulus EU mean and variance + holdout metrics + full config
  eu_summary.txt                # Sorted utility ranking (human-readable)
sr/<model>/
  self_report_results.json      # Per-stimulus 10-question battery (5 samples per question)
analyze.py                      # Reproduces App J.3 numbers from consonance_results_merged.csv
README.md                       # This file
```

## Reproducing the App J.3 numbers

```bash
python analyze.py
```

Reproduces (excluding 3 anchor stimuli, N=450):

| Model | Pearson r | Spearman ρ | Holdout accuracy |
|-|-|-|-|
| Qwen 2.5-Omni-7B | 0.384 | 0.433 | 0.907 |
| Qwen 3-Omni-30B-A3B-Instruct | 0.387 | 0.358 | 0.881 |

Plus per-timbre breakdown (e.g. Qwen 3-Omni shows consonance sensitivity across
all 3 timbres while Qwen 2.5-Omni only on sawtooth) and SR-vs-`hp_consonance`
correlations.

## End-to-end reproduction

```bash
# 1. Synthesize the 453 WAV stimuli + initial dissonance scores
python wellbeing/datasets/experiences/component_datasets/consonance_audio/stimuli/generate_stimuli.py

# 2. Compute Harrison & Pearce 2020 consonance components (interference, harmonicity, composite)
python wellbeing/datasets/experiences/component_datasets/consonance_audio/stimuli/compute_harrison_pearce.py

# 3. Convert to wellbeing-dev experience format (singletons + 200 combo bundles)
python wellbeing/datasets/experiences/component_datasets/consonance_audio/prepare_consonance_experiences.py

# 4. Run EU + SR via the standard pipeline (registered in experiments.yaml)
python run_experiments.py --slurm --experiments compute_eu_consonance,compute_sr_consonance \
    --models qwen25-omni-7b,qwen3-omni-30b-a3b-instruct

# 5. Reproduce App J.3 numbers
python experiments/other/consonance/analyze.py
```

The `hp_interference`, `hp_harmonicity`, and `hp_consonance` scores stored in
`stimuli_manifest.json` are produced by running `compute_harrison_pearce.py`
on the WAV files. The two special-case anchors (silence, white noise) are
handled separately in `main()`.
