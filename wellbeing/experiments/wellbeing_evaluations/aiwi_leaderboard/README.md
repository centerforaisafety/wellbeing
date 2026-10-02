# AI Wellbeing Index leaderboard

`leaderboard.json` lists the AI Wellbeing Index (AIWI) for every evaluated model.

- `leaderboard`: models whose zero-point fit passes the reliability filter (combination-model r² ≥ 0.4), sorted by AIWI.
- `excluded`: models that fail the filter, with their r².

Per-model fields: `aiwi` (point estimate over measured experiences), `lower` / `upper` (partial-identification
bounds when some experiences could not be measured, e.g. because generation was blocked), `coverage`
(fraction of the 500 experiences measured), `r2` (zero-point fit), and for API models `refusal_rate`
(fraction of comparison votes skipped as refusals).

Results are produced by `scripts/run_aiwi.sh` (see the main README).
