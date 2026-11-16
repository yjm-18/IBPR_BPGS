# Auxiliary boundary predictor

The IBPR/BPGS pipeline consumes precomputed boundary probabilities. Use the supplied
predictor in the separate TL/boundary environment described in
[ENVIRONMENTS.md](../../docs/ENVIRONMENTS.md).

## Inference

```bash
python auxiliary/boundary/infer.py --points points.npz \
  --checkpoint checkpoints/boundary/best_boundary_model.pth --output probabilities.npy
```

`points.npz` contains `points` with shape (N, 3), in metres. Inference uses three zero
placeholder channels and three normalized voxel-coordinate channels, 0.1 m voxels,
cores of at most 800,000 points and a 2 m halo. Each point receives one probability.
Map probabilities to TL source points using the fixed coordinate correspondence;
SAT uses probabilities on the aligned evaluation points.

## Optional training

Prepare the 13 training and 8 validation plots listed in
[the split manifest](../../docs/protocols/forinstance_manifest.json).

```bash
python auxiliary/boundary/prepare_labels.py --las YOUR_TRAIN.las --domain NIBIO \
  --output data/boundary/train/plot_merged_inst_nostuff.pth
python auxiliary/boundary/train_boundary_only.py --config auxiliary/boundary/config.yaml --epochs 100
```

Place validation files under `data/boundary/val`; `data.test` in the configuration
is the validation loader. Install the original SoftGroup `voxelization_idx` extension.
The loader reads each rulebook row as a point count followed by valid point indices,
ignores padding and verifies the point-to-voxel correspondence.

Validation pools up to 5,000 valid reference-tree points per batch, selected with
`np.linspace(0, n - 1, min(5000, n), dtype=np.int64)`. Boundary labels are binarized at
0.5. The current training code groups tied scores when computing AP. This rule applies
to new training runs; the supplied frozen checkpoint is used directly for A/B reproduction.
