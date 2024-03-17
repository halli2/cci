#!/bin/bash

#SBATCH --job-name=hp_cnn
#SBATCH --error=%x.%A_%a.err
#SBATCH --output=%x.%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100

uenv verbose miniconda3-py311 cudnn-12.x-9.0.0 cuda-12.3.2

# Timeout of 24H
pdm run cci tune-model "CNN" /home/stud/hnkvamme/BMDLab/data/oocha --timeout 86400

echo "Finished"
