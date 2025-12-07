# GDN

Code implementation for: [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series (AAAI'21)](https://arxiv.org/pdf/2106.06947.pdf)

## Installation

### Requirements
* Python >= 3.6 (tested with Python 3.11)
* PyTorch >= 2.0.0
* PyTorch Geometric >= 2.0.0
* CUDA (optional, for GPU acceleration)

### Install Packages

**Option 1: Using installation script (Windows)**
```bash
install.bat
```

**Option 2: Using pip (Cross-platform)**

Step 1: Install PyTorch
```bash
pip install torch torchvision torchaudio
```

Step 2: Install PyTorch Geometric
```bash
pip install torch-geometric
```

Step 3: Install PyG extension libraries

For CPU:
```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```

For GPU (CUDA 12.1):
```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

For GPU (CUDA 11.8):
```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
```

Step 4: Install additional dependencies
```bash
pip install numpy scipy pandas scikit-learn matplotlib
```

**Note:** Replace `torch-2.3.0` in the URLs with your installed PyTorch version if different. Check your version with `python -c "import torch; print(torch.__version__)"`.

### Quick Start

Run to check if the environment is ready:

**CPU:**
```bash
python main.py -dataset msl -save_path_pattern msl -slide_stride 1 -slide_win 5 -batch 32 -epoch 30 -comment msl -random_seed 5 -decay 0 -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 -val_ratio 0.2 -report best -topk 5 -device cpu
```

**GPU (Windows PowerShell):**
```powershell
$env:CUDA_VISIBLE_DEVICES=0; python main.py -dataset msl -save_path_pattern msl -slide_stride 1 -slide_win 5 -batch 32 -epoch 30 -comment msl -random_seed 5 -decay 0 -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 -val_ratio 0.2 -report best -topk 5
```

**GPU (Linux/Mac):**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py -dataset msl -save_path_pattern msl -slide_stride 1 -slide_win 5 -batch 32 -epoch 30 -comment msl -random_seed 5 -decay 0 -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 -val_ratio 0.2 -report best -topk 5
```

Replace `0` with your desired GPU ID (0, 1, 2, etc.).

## Usage

We use part of the MSL dataset (refer to [telemanom](https://github.com/khundman/telemanom)) as a demo example.

### Data Preparation

Put your dataset under the `data/` directory with the same structure shown in `data/msl/`:

```
data
 |-msl
 | |-list.txt    # the feature names, one feature per line
 | |-train.csv   # training data
 | |-test.csv    # test data
 |-your_dataset
 | |-list.txt
 | |-train.csv
 | |-test.csv
 | ...
```

### Important Notes:
* The first column in .csv will be regarded as the index column
* The column sequence in .csv doesn't need to match the sequence in list.txt; the data columns will be rearranged according to the sequence in list.txt
* test.csv should have a column named "attack" which contains ground truth labels (0/1) indicating whether data is attacked or not (0: normal, 1: attacked)

## Run Commands

### Basic Usage

**CPU:**
```bash
python main.py -dataset <dataset_name> -device cpu
```

**GPU:**
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python main.py -dataset <dataset_name>
```

### Command Line Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-dataset` | Dataset name (e.g., msl, swat, wadi) | Required |
| `-batch` | Batch size | 128 |
| `-epoch` | Number of training epochs | 100 |
| `-slide_win` | Sliding window size | 15 |
| `-slide_stride` | Sliding window stride | 5 |
| `-dim` | Embedding dimension | 64 |
| `-out_layer_num` | Number of output layers | 1 |
| `-out_layer_inter_dim` | Output layer intermediate dimension | 256 |
| `-topk` | Top-k for graph construction | 20 |
| `-val_ratio` | Validation data ratio | 0.1 |
| `-decay` | Weight decay for optimizer | 0 |
| `-device` | Device to use (cpu or cuda) | cuda (if available) |
| `-random_seed` | Random seed for reproducibility | 0 |
| `-comment` | Comment for this run | '' |
| `-save_path_pattern` | Save path pattern | '' |
| `-report` | Report mode (best/val) | 'best' |

### Example Commands

**Train on MSL dataset with default parameters:**
```bash
python main.py -dataset msl -save_path_pattern msl -device cpu
```

**Train with custom parameters:**
```bash
python main.py -dataset msl -batch 64 -epoch 50 -dim 128 -topk 10 -device cpu
```

**Train on GPU:**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py -dataset msl -batch 64 -epoch 50
```

## Model Output

Training results are saved in:
* **Model checkpoint:** `./pretrained/<dataset>/best_<timestamp>.pt`
* **Results CSV:** `./results/<dataset>/<timestamp>.csv`

## Additional Datasets

SWaT and WADI datasets can be requested from [iTrust](https://itrust.sutd.edu.sg/)

## Compatibility Notes

This repository has been updated to work with:
* PyTorch 2.x
* PyTorch Geometric 2.x
* Python 3.11+
* Windows/Linux/Mac

The original implementation used PyTorch 1.5.1 and PyTorch Geometric 1.5.0. The code has been updated for compatibility with newer versions.

## Citation

If you find this repo or our work useful for your research, please consider citing the paper:

```bibtex
@inproceedings{deng2021graph,
  title={Graph neural network-based anomaly detection in multivariate time series},
  author={Deng, Ailin and Hooi, Bryan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={5},
  pages={4027--4035},
  year={2021}
}
```
