# Auxiliary Boundary Validation AP

The shared auxiliary boundary predictor is fitted on the 13 FOR-instance training
plots. The distributed checkpoint was selected using the eight validation plots.
A/B experiments use this fixed predictor and its precomputed outputs.

## AP uses sampled valid points

For each validation batch:

1. Retain points selected by the valid reference-tree mask.
2. Set `sample_count = min(5000, number_of_valid_points)`.
3. Select indices with `np.linspace(0, number_of_valid_points - 1, sample_count, dtype=np.int64)`.
4. Append the sampled boundary probabilities and binary reference labels to pooled arrays.

Compute one AP from these pooled arrays after the validation batches. It is neither
AP over every validation point nor a macro-average of per-plot or per-domain AP.
With batch size one, eight fully processed plots and at least 5,000 valid points per
plot, the pooled AP contains 40,000 points. Sampling is deterministic in the batch's
point order; it is not spatially balanced or randomly sampled.

The current training implementation computes `average_precision` by grouping equal
prediction scores before accumulating precision and recall. This makes AP invariant
to the input order within a tied-score group. The sampling and pooling rules above
are unchanged.

The distributed frozen checkpoint is epoch 97 of the recorded 100-epoch run and
remains the input reference for A/B reproduction. The grouped-score implementation
applies to new training runs; the distributed checkpoint has not been reselected.
Labels are binary with a configured hard-label threshold of 0.5. They are not
expanded soft labels in this configuration.

## Other counters are separate

TP, FP, TN and FN at a specified decision threshold are accumulated over the valid
points passed to each batch update before AP subsampling. Their full-point accumulation
does not make AP a full-point metric. The boundary AP here is also distinct from the
instance mAP used in the main experiment results.

See `BoundaryMetrics.update` and `compute_average_precision` in
[train_boundary_only.py](../auxiliary/boundary/train_boundary_only.py), and
[the component environment table](ENVIRONMENTS.md).
