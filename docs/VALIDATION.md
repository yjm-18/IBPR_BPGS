# Implementation Validation

The repository contains the method implementation, selected A/B configurations and
checkpoint download information. Dataset performance results are reported in the paper.

Implementation checks covered proposal scores, region targets, source and evaluation
labels, and metric computation. Source tests remain available under `tests/`.
The Ofental zero-threshold configuration was checked by replaying the original A/B
models and recomputing full evaluation metrics. No model weights were refitted for
this execution-rule update.

Evaluation uses the ten IoU thresholds generated from integer percentages
`50, 55, ..., 95`, avoiding accumulated floating-point step error. The fixed
baseline-to-reference matching table covers the full source-instance label domain;
instances without a baseline match remain unmatched. This also applies to BPGS
marginal targets and does not rematch instances after a modification.

Regression tests cover exact threshold boundaries, labels that reappear after
5NN propagation, SoftGroup count-prefixed rulebooks, and tied boundary scores.
A real auxiliary training batch completed forward, backward and validation checks.
These implementation corrections retain the distributed weights and selected A/B
configurations.
