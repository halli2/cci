#!/bin/bash

# ./job.sh cnn as | mlp full

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="$1_$2"
#SBATCH --error=logs/%x.%A_%a.err
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
uenv verbose miniconda3-py311 cudnn-12.x-9.0.0 cuda-12.3.2

# --- NOTE: Uncomment for setup ---
# pip install pdm
# pdm install -L pdm.cuda.lock
# --- NOTE: Uncomment for setup ---

# Timeout of 24H
# Trial name, modelname, dataset name
pdm run cci tune-model "$1_$2_fixed" $1 $2  /home/prosjekt/BMDLab/data/oohca --timeout 86400 --n-splits $3

echo "Finished"

EOT
