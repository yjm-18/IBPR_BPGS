# Software Environments by Component

Use separate environments for the core package, the auxiliary boundary network and
the upstream baseline models. The versions below specify the component environments.

| Component | Python | PyTorch | CUDA (PyTorch build) | cuDNN | NumPy | SciPy | scikit-learn | laspy |
|---|---|---|---|---|---|---|---|---|
| TreeLearn baseline and auxiliary boundary predictor | 3.12.9 | 2.5.1 | 11.8 | 9.1.0 | 2.2.4 | 1.15.2 | 1.6.1 | 2.6.1 |
| SegmentAnyTree baseline | 3.8.20 | 1.9.0 | 11.1 | 8.0.5 | 1.19.5 | 1.7.3 | 1.1.2 | 2.0.3 |
| IBPR and BPGS fitting, inference and evaluation | 3.11.7 | Not required by the core package | Not required | Not required | 1.26.4 | 1.13.1 | 1.5.1 | 2.5.4 for data preparation |

The core package also uses Numba 0.60.0 and threadpoolctl 3.5.0. Its dependencies
are pinned in `requirements.txt` and `environment.yml`. Load the A/B checkpoints
in this environment.

The auxiliary environment uses spconv-cu118 2.3.8, cumm-cu118 0.7.11 and PyYAML 6.0.2,
plus the original SoftGroup voxelization extension. Install it separately from the
root A/B requirements. The root `environment.yml` defines the A/B environment only.
