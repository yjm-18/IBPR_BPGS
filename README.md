# IBPR + BPGS

Point-to-instance reassignment (IBPR) and region selection (BPGS) for tree instance
segmentation, using TreeLearn (TL) or SegmentAnyTree (SAT) predictions.
The package includes training, inference and evaluation on FOR-instance, Ofental
and LAUTx. IBPR uses 39 features; BPGS uses 38 features.

## Installation

Use Python 3.11.7 and the pinned dependencies for the supplied scikit-learn models.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The auxiliary boundary network uses a separate CUDA environment. See
[component environments](docs/ENVIRONMENTS.md).

## Data preparation

Prepare aligned baseline predictions, point features and boundary probabilities
as described in [the input format](docs/DATA_FORMAT.md). Obtain the baseline models
from [the upstream checkpoint links](docs/BASELINES.md).

```bash
python scripts/prepare_scene.py --model TL --input raw_tl_export.npz --output scene.npz
```

Use `--model SAT` for SAT exports. TL requires a frozen 5NN mapping; SAT uses directly
aligned points. Training and evaluation file lists are under `configs/manifests/`.
Datasets and upstream baseline models are downloaded separately.

## Pretrained models

Download the [checkpoint archive](https://github.com/yjm-18/IBPR_BPGS/releases/download/v0.1.0/ibpr-bpgs-0.1.0-checkpoints.zip)
and [SHA256 file](https://github.com/yjm-18/IBPR_BPGS/releases/download/v0.1.0/ibpr-bpgs-0.1.0-checkpoints.zip.sha256).
Extract the archive in the repository root to create `checkpoints/`.
It contains four IBPR models, sixteen BPGS models and one shared boundary predictor.
LAUTx uses the corresponding FOR-instance IBPR model and one BPGS model per held-out plot.

```bash
ibpr-bpgs verify-weights --weights checkpoints
```

Boundary probability generation is described in
[auxiliary/boundary/README.md](auxiliary/boundary/README.md).

## Inference and evaluation

```bash
ibpr-bpgs infer --scene scene.npz --config configs/FOR-instance_TL.json \
  --weights checkpoints --output outputs/example --evaluate
```

Omit `--evaluate` for inputs without reference labels. Predictions are saved for
baseline, IBPR and IBPR+BPGS, together with proposal and region scores.

To evaluate an ordered dataset manifest:

```bash
python scripts/benchmark.py --manifest configs/manifests/LAUTx_SAT_test.json \
  --weights checkpoints --output outputs/LAUTx_SAT
```

Metric definitions and data splits are in [PROTOCOL.md](docs/PROTOCOL.md).

## Training

```bash
ibpr-bpgs-train reproduce-fixed \
  --manifest configs/manifests/FOR-instance_TL_train.json \
  --config configs/FOR-instance_TL.json --output outputs/train_FOR_TL
```

See [TRAINING.md](docs/TRAINING.md) for Ofental training and LAUTx leave-one-plot-out
BPGS training. Supplied configurations specify the model parameters and execution thresholds.

## Dependencies and third-party components

See [NOTICE.md](NOTICE.md) for licensing status and third-party components.
