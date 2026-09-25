# AIWI leaderboard v12 (2026-09-25)

`leaderboard.json` = 28 models passing the paper's ZP r2 >= 0.4 filter (user decision 2026-09-25: apply to ALL models),
plus the 22 excluded models with their r2.

Sources:
- Open-weight (37 models, logprobs path, unaffected by the parser fix): `../aiwi_v11/<model>.json`
  (EU `eu_d2_cap2048_randsample`, ZP `zp_d2_cap2048_randsample`; r2 cross-checked against the ZP files).
- Closed-weight (13 models): `../aiwi_v11_closed/v12/` (parser fix + refusals skipped; see its README).

Filter context: 15 of the 20 excluded open-weight models also failed r2 >= 0.4 in v1.0 (hard-hinge recompute,
`wellbeing-public/aiwi_v10_baseline_openweight.json`). Newly failing in v1.1: gemma-2-9b-it, qwen3-4b-instruct,
qwen25-14b-instruct, llama-31-8b-instruct, qwen3-8b, plus closed claude-haiku-45, gpt-54-nano. Median open-weight
r2 fell 0.474 -> 0.311 between v1.0 and v1.1; cause untested (not the hinge; candidates: cap2048 dataset,
random vs active-learning edges, fixed bundles).
