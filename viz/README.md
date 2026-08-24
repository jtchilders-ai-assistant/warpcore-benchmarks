# Figures

Plots-as-code for the benchmark results. Every figure is generated from
**committed raw artifacts only** — nothing here is hand-drawn, hand-edited, or
typed in from a model card.

## Regenerating

```sh
pip install -r viz/requirements.txt
make figs          # from the repo root
```

`make check` regenerates and fails if the committed figures no longer match the
code — use it before merging a results change.

Output is deterministic: `svg.hashsalt` is fixed and no timestamps are embedded,
so a no-op `make figs` produces a byte-identical file and therefore an empty
`git diff`. (Guaranteed within a matplotlib minor release; see
`requirements.txt`.)

## What each figure claims

| file | claim |
|---|---|
| `fig1_pareto` | The serving envelope. Peak throughput and what a single user actually feels are different questions with different winners. |
| `fig2_swebench` | Ornith's SWE-bench lead is real and survives pairing; Lightning vs Qwen3.6 does not. A third of Qwen3.6's misses never produced a patch at all. |
| `fig3_discrimination` | pi-30 is saturated and no longer separates models. Benchmark rank does not transfer across task families. |

Panel titles state the finding rather than naming the method — if the title
doesn't tell you the point, the panel has failed.

## Files

```
viz/
  common.py               palette, plot style, Wilson/McNemar/binomial-SE helpers
  parse_sweeps.py         results/*/raw/throughput_sweep/* -> data/throughput_all.csv
  collect_matrix.py       quality + pi-30 + SWE-bench artifacts -> data/bench_matrix.json
  fig1_pareto.py          serving envelope
  fig2_swebench.py        SWE-bench Verified, paired analysis
  fig3_discrimination.py  benchmark discrimination + rank transfer
  data/                   derived intermediates (regenerable; committed for diffability)
  out/                    generated PNG + SVG
```

## Reading the statistics

- **Wilson intervals** for a single proportion; at n=100 that is roughly ±9 pp,
  which is why SWE-bench scores are never drawn as bare bars.
- **Paired (McNemar) intervals** for model *differences*. All three models saw
  the identical seed-42 100-instance set, so only the discordant pairs carry
  information — this interval is much tighter than comparing two independent
  Wilson intervals, and it is the correct one for "is A better than B here".
- **spread ÷ binomial SE** for benchmark discrimination. A raw max−min range
  rewards small samples and grows mechanically with the number of models
  compared; normalizing by noise is what demotes pi-30 (1.6×) and promotes
  GSM8K (25.9×).

## Caveats the figures deliberately preserve

These are visible on the figures themselves, not buried here:

- **Ornith GPQA (69.70%) is an underestimate** — 21.2% of items returned empty
  content (ISSUES #15). It is a floor, not an estimate.
- **Qwen3.6 GPQA is non-thinking mode**; Lightning's GPQA is a 64k-budget
  composite (32k run + 64k replay of truncated items, 151/198).
- **Empty patches are not wrong answers.** They are runs that never finished, so
  they partly convert throughput into benchmark score. Only Qwen3.6 has a
  complete `exit_statuses` artifact; the other two are not fully attributable.
- **The SWE-bench sample is 56% django.** `fig3` shows how much the headline
  moves under repo-balanced reweighting.
- **fig1 mixes sweeps with different `num_prompts`** (Laguna 3/concurrency,
  others 32). Curve shape is comparable; per-point noise is not.

## Known gaps

- Per-repo reweighting in `fig3` uses the 6 repos with n≥4. Equal-weighting all
  11 repos gives a smaller Ornith−Lightning gap (7.6 pp vs 16.6 pp). The choice
  is arbitrary and the figure does not yet show both.
- `results/*/raw/launch_*.sh` show that serving config (`--max-model-len`,
  `--max-num-seqs`, `--gpu-memory-utilization`) **drifted between phases** for at
  least Lightning, Ornith and Laguna. Nothing currently records the config a
  given score was produced under, so cross-phase comparisons carry an
  unquantified confound.
