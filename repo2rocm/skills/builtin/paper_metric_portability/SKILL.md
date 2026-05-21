---
name: paper_metric_portability
description: How to classify a paper metric and pick a sane default tolerance when comparing AMD reproduction against the paper's CUDA numbers
when_to_use: When constructing MetricSpec entries for PaperVerify or shortlisting reproducible experiments
---

# Metric portability classes

When the paper was run on different hardware (A100/H100 vs AMD MI250X/MI300X),
absolute throughput/latency aren't directly comparable. Ratios and accuracy-like
metrics travel much better. Always prefer the most portable metric available.

## Classes (most portable → least)

1. **ratio_speedup** — unitless ratios: `2.5x`, `+35%` speedup, compression ratios.
   Default tolerance: ≤15% relative.
2. **accuracy** — accuracy %, F1, BLEU, EM, top-k, pass@k.
   Default tolerance: ≤3 absolute percentage points.
3. **quality** — perplexity, NLL, loss, reward, BPC/BPB.
   Default tolerance: ≤5% relative.
4. **absolute_perf** — throughput (tokens/s), latency (ms), QPS, wall time.
   Default tolerance: ≤25% relative (different GPU → different numbers).
5. **other** — anything else.
   Default tolerance: ≤15% relative.

## Classification keywords

| Class | Name tokens | Unit tokens |
|---|---|---|
| ratio_speedup | speedup, speed-up, improvement, acceleration, compression, reduction, ratio, relative | `x`, `×` |
| accuracy | accuracy, acc, top-k, F1, EM, exact match, BLEU, ROUGE, METEOR, chrF, pass@k, hit@, precision, recall, mAP, NDCG, win rate, success rate | `%`, `pp` |
| quality | perplexity, ppl, NLL, loss, reward, BPC, BPB, cross-entropy | (numeric) |
| absolute_perf | throughput, latency, QPS, RPS, tokens/s, tok/s, samples/s, frames/s, time per, wall time, runtime, TTFT, TPOT, elapsed | `ms`, `s`, `µs`, `tokens/s`, `samples/s` |

## Choosing experiments

When shortlisting paper experiments to reproduce:

1. Prefer experiments whose headline metric is `ratio_speedup` or `accuracy` —
   they're robust to GPU swap.
2. Avoid `absolute_perf`-only experiments unless the paper provides hardware-matched
   baselines AND a relative speedup metric.
3. Always pair "is this a baseline measurement?" with the metric. Baselines reproduce
   trivially; the interesting reproduction is the paper's METHOD vs ITS baseline.

## Detecting baselines

A row is a baseline if its name contains: `baseline`, `no-cache`, `no_cache`,
`vanilla`, `without`, `w/o`, `origin`, `naive`, `reference`, `unmodified`, `plain`,
`default`, `standard`, `fp16 baseline`, `fp32 baseline`, `untuned`.

Negators (phrase is comparing TO a baseline, not measuring one): `vs baseline`,
`vs. baseline`, `over baseline`, `relative to baseline`, `against baseline`,
`speedup over`.
