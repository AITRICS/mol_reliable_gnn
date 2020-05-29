# Reliable GNNs for molecular machine learning

Source codes for "A baseline for reliable molecular prediction models via Bayesian learning"


## Setup Python environment
for GPU usage,
DGL requires CUDA **10.0**.

```
# Install python environment
# For CPU usage,
conda env create -f environment_cpu.yml   
# For GPU usage,
conda env create -f environment_gpu.yml

# Activate environment
conda activate reliable_gnn
```

# Download datasets

```
# At the root of the project
cd data/ 
bash script_download_molecules.sh
```

# Reproducibility

## 1. Usage

### 1.1 In terminal

```
# Run the main file (at the root of the project)
# To run GIN on BACE dataset,
python main_classification.py --config configs/gin_bace.json # for CPU
python main_classification.py --gpu_id 0 --config configs/gin_bace.json # for GPU

# To run GCN on BACE dataset,
python main_classification.py --config configs/gcn_bace.json # for CPU
python main_classification.py --gpu_id 0 --config configs/gcn_bace.json # for GPU

# To run GIN on BBBP dataset,
python main_classification.py --config configs/gin_bbbp.json # for CPU
python main_classification.py --gpu_id 0 --config configs/gin_bbbp.json # for GPU

# To run GIN on HIV dataset,
python main_classification.py --config configs/gin_hiv.json # for CPU
python main_classification.py --gpu_id 0 --config configs/gin_hiv.json # for GPU

# To run GIN on TOX21 dataset,
python main_tox21.py --config configs/gin_tox21.json # for CPU
python main_tox21.py --gpu_id 0 --config configs/gin_tox21.json # for GPU

# To run GIN on BACE dataset with MCDropout,
python main_classification.py --config configs/gin_bace.json --mcdropout True # for CPU
python main_classification.py --gpu_id 0 --config configs/gin_bace.json --mcdropout True # for GPU

# To run GIN on BACE dataset with SWA,
python main_classification.py --config configs/gin_bace.json --swa True # for CPU
python main_classification.py --gpu_id 0 --config configs/gin_bace.json --swa True # for GPU

# To run GIN on BACE dataset with SWAG,
python main_classification.py --config configs/gin_bace.json --swag True # for CPU
python main_classification.py --gpu_id 0 --config configs/gin_bace.json --swag True # for GPU

# To run GIN on BACE dataset with pSGLD,
python main_classification.py --config configs/gin_bace.json --psgld True # for CPU
python main_classification.py --gpu_id 0 --config configs/gin_bace.json --psgld True # for GPU

# To run GIN on BACE dataset with Bayes By Backprop,
python main_classification.py --config configs/gin_bace.json --bbp True # for CPU
python main_classification.py --gpu_id 0 --config configs/gin_bace.json --bbp True # for GPU

# To run GIN on BACE dataset with Checkpoint saved,
python main_classification.py --config configs/gin_bace.json --save_params True # for CPU
python main_classification.py --gpu_id 0 --config configs/gin_bace.json --save_params True # for GPU

```
The training and network parameters for each dataset and network is stored in a json file in the [`configs/`](../configs) directory.

## 2. Output, checkpoints

Output results are located in the folder defined by the variable `out_dir` in the corresponding config file (eg. [`configs/molecules_graph_regression_GatedGCN_ZINC.json`](../configs/gin_bace.json) file).  
If `out_dir = 'out/bace_classification/'`, then 

To see checkpoints and results,
1. Go to`out/bace_classification/results` to view all result text files.
2. Directory `out/bace_classification/checkpoints` contains model checkpoints.
