# Data

This folder holds the numerical results of the study and is the place to add the
raw inputs and per-seed logs before the repository is made public.

## Included (confirmed results)

- **`results_per_metric.csv`** — mean ± SD (and 95% CI for waiting time) of every
  metric (waiting time, queue, speed, CO₂) for all 16 algorithm × condition cells,
  over five seeds. Mirrors Table 5 of the paper.
- **`results_stats.csv`** — paired Wilcoxon signed-rank contrasts (PURE-vs-LLM,
  LLM-vs-RAG) per algorithm, with mean difference, Cohen's *d*, raw *p*,
  Holm–Bonferroni-corrected *p*, and the evidence verdict. Mirrors Table 6.

Units: waiting time in seconds, queue in vehicles, speed in km/h, CO₂ as an
emission rate in g/km.

## To add (authors)

- `network/` — the SUMO files (`*.net.xml`, `*.rou.xml`, `*.add.xml`) and the
  OpenStreetMap export of the Halil Sezai Erkut Avenue intersection.
- `demand/` — the on-site vehicle counts and the scaled per-interval demand.
- `raw_logs/` — per-seed, per-episode logs for every condition (the source for
  the aggregated CSVs above) so the statistics can be recomputed end-to-end.
