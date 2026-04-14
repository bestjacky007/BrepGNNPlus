# BrepGNN+

Official PyTorch implementation for the paper:

Modernizing Graph Neural Networks for Native B-Rep Learning

BrepGNN+ revisits native B-Rep learning by keeping the original UV-Net geometric encoders and modernizing only the graph-learning stage with residual connections, LayerNorm, node-wise FFNs, and stochastic depth.

## Overview

This repository contains the training and evaluation code used for the experiments in the paper.

Supported tasks and datasets in this release:

- Part classification: `solidletters`, `tmcad`
- Machining feature recognition / segmentation: `mfcad`, `mfcad2`, `fusiongallery`

In the paper, `mfcad2` corresponds to MFCAD++ and `fusiongallery` corresponds to Fusion 360 Gallery segmentation.

## Repository Structure

```text
.
|-- classification.py
|-- segmentation.py
|-- datasets/
|   |-- base.py
|   |-- classification_datasets.py
|   |-- fusiongallery.py
|   |-- mfcad.py
|   |-- mfcad2.py
|   |-- solidletters.py
|   `-- util.py
`-- uvnet/
    |-- encoders.py
    `-- models.py
```

## Environment Setup

The virtual environment can be prepared by directly following the official
[UV-Net setup](https://github.com/AutodeskAILab/UV-Net).

In particular, the UV-Net repository already provides a tested `environment.yml` and installation instructions for DGL.

Typical setup:

```bash
conda env create -f environment.yml
conda activate uv_net
```

If you use a CPU-only environment, please follow the additional DGL installation note in the UV-Net README.

This project depends on the same core stack as UV-Net, including:

- PyTorch
- PyTorch Lightning
- DGL
- scikit-learn
- TensorBoard

## Data Preparation

### Important

This repository assumes that each dataset has already been converted into the DGL `.bin` graph format used by UV-Net.

This repository does not redistribute any raw datasets, processed dataset archives, or third-party CAD assets. Users must obtain all datasets from their original sources and are responsible for ensuring that their use, preprocessing, storage, and redistribution comply with the corresponding dataset licenses, terms of use, and any institutional or commercial restrictions.

The data processing scripts and conversion pipeline can be directly referenced
from the official UV-Net repository:

- [UV-Net main repository](https://github.com/AutodeskAILab/UV-Net)
- [UV-Net processing guide](https://github.com/AutodeskAILab/UV-Net/blob/main/process/README.md)

Please follow the UV-Net pipeline to:

1. Convert raw STEP / B-Rep files into face-adjacency graphs.
2. Store face UV-grids in `graph.ndata["x"]`.
3. Store edge UV-grids in `graph.edata["x"]`.
4. Export the processed graphs as DGL `.bin` files.

If you generate derived `.bin` files from datasets obtained elsewhere, please verify whether those derived files can be shared publicly before uploading them to any repository or release package.

### Expected Dataset Layout

For classification datasets such as TMCAD, the repository expects the following layout:

```text
dataset_root/
|-- train.txt
|-- val.txt
|-- test.txt
|-- 000_example.bin
|-- 001_example.bin
`-- ...
```

For SolidLetters, the loader follows the UV-Net-style bin data organization and internally constructs a 70/15/15 split from the available file lists.

## Training

All results in the paper use BrepGNN+ with a modernized GNN backbone. The default backbone is GCN, and the paper uses 12 layers as the main configuration.

### 1. Part Classification

#### SolidLetters

```bash
python classification.py train \
  --dataset solidletters \
  --dataset_path /path/to/solidletters \
  --experiment_name brepgnn_solidletters \
  --gnn_type gcn \
  --num_layers 12 \
  --lr 5e-3 \
  --weight_decay 1e-2 \
  --batch_size 64 \
  --max_epochs 350 \
```

#### TMCAD

```bash
python classification.py train \
  --dataset tmcad \
  --dataset_path /path/to/tmcad \
  --experiment_name brepgnn_tmcad \
  --gnn_type gcn \
  --num_layers 12 \
  --lr 5e-4 \
  --weight_decay 1e-2 \
  --batch_size 64 \
  --max_epochs 350 \
```

### 2. Machining Feature Recognition / Segmentation

#### MFCAD++ (`mfcad2`)

```bash
python segmentation.py train \
  --dataset mfcad2 \
  --dataset_path /path/to/mfcad2 \
  --experiment_name brepgnn_mfcad2 \
  --gnn_type gcn \
  --num_layers 12 \
  --lr 5e-3 \
  --weight_decay 1e-2 \
  --batch_size 64 \
  --max_epochs 350 \
```

#### Fusion 360 Gallery

```bash
python segmentation.py train \
  --dataset fusiongallery \
  --dataset_path /path/to/fusiongallery \
  --experiment_name brepgnn_fusion360 \
  --gnn_type gcn \
  --num_layers 12 \
  --lr 5e-3 \
  --weight_decay 1e-2 \
  --batch_size 64 \
  --max_epochs 350 \
```

### 3. Reproducing Ablation Studies

The main ablation flags are available in both entry scripts:

- `--gnn_type {gcn,sage,gin,gatedgcn}`
- `--num_layers <int>`
- `--no_residual`
- `--no_ffn`
- `--no_layernorm`
- `--no_stochastic_depth`
- `--no_edge`

Example:

```bash
python segmentation.py train \
  --dataset mfcad2 \
  --dataset_path /path/to/mfcad2 \
  --experiment_name gcn_no_ffn_mfcad2 \
  --gnn_type gcn \
  --num_layers 12 \
  --no_ffn \
  --lr 5e-3 \
  --max_epochs 350 \
```

## Testing

The best checkpoints are written under `results/<experiment_name>/...`.

### Classification Test

```bash
python classification.py test \
  --dataset tmcad \
  --dataset_path /path/to/tmcad \
  --checkpoint /path/to/checkpoint.ckpt \
  --batch_size 64
```

### Segmentation / Feature Recognition Test

```bash
python segmentation.py test \
  --dataset fusiongallery \
  --dataset_path /path/to/fusiongallery \
  --checkpoint /path/to/checkpoint.ckpt \
  --batch_size 64
```

## Logging

Training logs and checkpoints are stored in:

```text
results/<experiment_name>/<month_day>/<time>/
```

You can monitor training with TensorBoard:

```bash
tensorboard --logdir results
```


## Citation

If you find this repository useful, please cite the paper once the bibliographic information is available.

```bibtex
@article{brepgnnplus,
  title   = {Modernizing Graph Neural Networks for Native B-Rep Learning},
  author  = {To be added},
  journal = {To be added},
  year    = {To be added}
}
```

## Acknowledgment

This project builds directly on the data representation and codebase structure introduced by UV-Net:

Pradeep Kumar Jayaraman, Aditya Sanghi, Joseph G. Lambourne, Karl D.D. Willis, Thomas Davies, Hooman Shayani, Nigel Morris. UV-Net: Learning from Boundary Representations. CVPR 2021.

The repository also includes adapted components derived from the official UV-Net implementation. Those upstream components are used under the MIT License; please retain the original attribution context when reusing or redistributing modified versions of this code.
