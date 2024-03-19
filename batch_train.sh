#!/bin/bash
#SBATCH -J alexnet_int_8_relu_layer
#SBATCH --gpus 8
#SBATCH -t 20:00:00
#SBATCH -A berzelius-2024-8
cd /proj/berzelius-2023-29/users/x_hammo/NetAug/FADER
torchpack dist-run -np 8 python train.py 