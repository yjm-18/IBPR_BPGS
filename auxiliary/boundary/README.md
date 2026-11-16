# Auxiliary boundary predictor

The IBPR/BPGS pipeline consumes precomputed boundary probabilities. Use the supplied
predictor in a separate environment: Python 3.12.9, PyTorch 2.5.1 with CUDA 11.8,
spconv-cu118 2.3.8 and cumm-cu118 0.7.11. The remaining Python package versions are
listed in [requirements-recorded.txt](requirements-recorded.txt). Build the SoftGroup
voxelization extension against this environment.

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

Known domains (CULS, NIBIO, RMIT, SCION and TUWIEN) retain their configured K values.
For another domain, provide `--domain YOUR_DATASET --default-k 128`, replacing 128
with your chosen positive neighbor count. `--default-k` only applies to unknown
domains; it does not override existing settings. Domain names are case-insensitive.

Place validation files under `data/boundary/val`; `data.test` in the configuration
is the validation loader. Install the original SoftGroup `voxelization_idx` extension.
The loader reads each rulebook row as a point count followed by valid point indices,
ignores padding and verifies the point-to-voxel correspondence.

Validation pools up to 5,000 valid reference-tree points per batch, selected with
`np.linspace(0, n - 1, min(5000, n), dtype=np.int64)`. Boundary labels are binarized at
0.5. The current training code groups tied scores when computing AP. This rule applies
to new training runs; the supplied frozen checkpoint is used directly for A/B reproduction.
