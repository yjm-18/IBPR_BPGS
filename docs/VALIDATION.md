# Implementation Validation

The repository contains the method implementation, selected A/B configurations and
checkpoint download information. Dataset performance results are reported in the paper.

Implementation checks covered proposal scores, region targets, source and evaluation
labels, and metric computation. Source tests remain available under `tests/`.
The Ofental zero-threshold configuration was checked by replaying the original A/B
models and recomputing full evaluation metrics. No model weights were refitted for
this execution-rule update.
