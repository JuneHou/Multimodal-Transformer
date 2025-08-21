
# Multimodal-Transformer: Fuse-MoE Replication on MIMIC-IV


This repository provides the official implementation for replicating the Fuse-MoE framework on the multimodal MIMIC-IV dataset, including our extensions for the BTW model.

## Set Up Environment

Run the following commands to create a conda environment:
```bash
conda create -n multimodal-transformer python=3.8
source activate multimodal-transformer
pip install -r requirements.txt
conda activate multimodal-transformer
```

## Repository Structure

- `src/`: Source code
## Fuse-MoE Replication on MIMIC-IV and BTW Extensions

- **Fuse-MoE**: A Mixture-of-Experts approach for multimodal clinical data.
- **Dataset**: MIMIC-IV (multimodal: Text, TS (tabular), CXR (imaging), ECG (signal).)
- **Preprocessing**: Follows the Fuse-MoE procedure (see below).

### Key Files for BTW Model

- `src/core/model_btw.py`: BTW model implementation.
- `src/scripts/main_mimiciv_btw.py`: Main script for running BTW on MIMIC-IV.
- `src/scripts/run_mimiciv_btw.sh`: Shell script to launch experiments.
- `src/core/train_btw.py`: Training logic for BTW.
- `src/preprocessing/data_mimiciv_btw.py`: Data preprocessing for MIMIC-IV (Fuse-MoE style).

### Data Preprocessing

- Data preprocessing follows the Fuse-MoE pipeline [https://github.com/aaronhan223/FuseMoE].

## Run Experiments

Under `src/scripts/`:

MIMIC-IV experiments
```
sh run_mimiciv.sh
```

## Load Results
First change the `filepath` in `load_result.py`, then run
```
python load_result.py
```


## Acknowledgement

This repository extends the Fuse-MoE framework. Data preprocessing strictly follows the Fuse-MoE procedure and codebase:
- [FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion](https://arxiv.org/pdf/2402.03226.pdf), Han et al., 2024

## Citation

If you use this code, please cite our paper:

```
@inproceedings{
hou2025btw,
title={{BTW}: A Non-Parametric Variance Stabilization Framework for Multimodal Model Integration},
author={Jun Hou and Le Wang and Xuan Wang},
booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
year={2025},
url={https://openreview.net/forum?id=EXp1qDqhCk}
}
```