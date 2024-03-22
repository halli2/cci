#!/bin/bash

# ./job.sh cnn|mlp

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="$1"
#SBATCH --error=%x.%A_%a.err
#SBATCH --output=%x.%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
uenv verbose miniconda3-py311 cudnn-12.x-9.0.0 cuda-12.3.2

# Timeout of 24H
pdm run cci tune-model "$1" /home/stud/hnkvamme/BMDLab/data/oocha --timeout 86400

echo "Finished"

EOT