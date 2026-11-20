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

The auxiliary boundary network uses a separate CUDA environment; see
[its setup instructions](auxiliary/boundary/README.md).

## Data preparation

Prepare aligned baseline predictions, point features and boundary probabilities
as described in [the input format](docs/DATA_FORMAT.md).
For FOR-instance and LAUTx, obtain the official baseline weights:

- TreeLearn: [model_weights_with_small_20241213.pth](https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/VPMPID/TYZJ4E), with [upstream setup](https://github.com/ecker-lab/TreeLearn/blob/main/docs/segmentation_pipeline.md).
- SegmentAnyTree: [PointGroup-PAPER.pt](https://media.githubusercontent.com/media/SmartForest-no/SegmentAnyTree/a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9/model_file/PointGroup-PAPER.pt), with [upstream setup](https://github.com/SmartForest-no/SegmentAnyTree).

Ofental baseline models use standard training on its designated split.

```bash
python scripts/prepare_scene.py --model TL --input raw_tl_export.npz --output scene.npz
```

Use `--model SAT` for SAT exports. TL requires a frozen 5NN mapping; SAT uses directly
aligned points. Training and evaluation file lists are under `configs/manifests/`.
Datasets and upstream baseline models are downloaded separately.

## Model weights

Pretrained IBPR (A) and BPGS (B) checkpoints are not distributed.
Train these components using [TRAINING.md](docs/TRAINING.md) and place your
checkpoints at the paths specified by the selected configuration.
LAUTx uses the corresponding FOR-instance IBPR model and one BPGS model per held-out plot.

Only the auxiliary boundary predictor is available as a
[boundary-only archive](https://github.com/yjm-18/IBPR_BPGS/releases/download/v0.1.1/ibpr-bpgs-boundary-only.zip)
with a [SHA256 file](https://github.com/yjm-18/IBPR_BPGS/releases/download/v0.1.1/ibpr-bpgs-boundary-only.zip.sha256).
Extract it in the repository root to create `checkpoints/boundary/`.
It does not contain IBPR or BPGS weights.

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
