# Baseline Models and Checkpoints

## FOR-instance and LAUTx

The two datasets use the corresponding official pretrained checkpoint below.
TL and SAT are evaluated independently. Obtaining a checkpoint alone does not reproduce the complete
input preparation, point alignment and evaluation pipeline.

| Model | Checkpoint | Download | Official reference |
|---|---|---|---|
| TreeLearn (TL) | `model_weights_with_small_20241213.pth` | [Official weight file](https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/VPMPID/TYZJ4E) | [Pipeline documentation](https://github.com/ecker-lab/TreeLearn/blob/main/docs/segmentation_pipeline.md) |
| SegmentAnyTree (SAT) | `PointGroup-PAPER.pt`, alias `latest` | [Official Git LFS weight file](https://media.githubusercontent.com/media/SmartForest-no/SegmentAnyTree/a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9/model_file/PointGroup-PAPER.pt) | [Pinned checkpoint page](https://github.com/SmartForest-no/SegmentAnyTree/blob/a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9/model_file/PointGroup-PAPER.pt) |

The SAT link points to the binary Git LFS object, rather than its small text pointer.
The FOR-instance SAT local copy changes stored data-root metadata only; its model
tensors match the official checkpoint. File SHA256 values are provided in
[baseline_weights.json](../configs/baseline_weights.json) to distinguish metadata
serialization from model-weight changes.

Both models use the same evaluation point set within each dataset. FOR-instance
removes class=3 before 0.1m voxelization. Evaluation definitions are documented in
[PROTOCOL.md](PROTOCOL.md); dataset performance results are reported in the paper.

## Ofental

The Ofental baselines are reproduced through standard model training and evaluation
on the designated data split. No separate Ofental pretrained-weight download is
provided here.
