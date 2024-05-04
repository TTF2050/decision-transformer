#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:2

# Request 1 CPU core
#SBATCH -n 12
#SBATCH -t 24:00:00

# Load a CUDA module
module load cuda

# Run program
singularity run --nv --bind dectransform:${HOME}/dectransform dectransform.simg run_grid.sh