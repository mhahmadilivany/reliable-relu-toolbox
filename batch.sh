#!/bin/bash
#SBATCH -J alexnet_int_8_relu_layer
#SBATCH --gpus 5
#SBATCH -t 07:00:00
#SBATCH -A berzelius-2023-29
cd /proj/berzelius-2023-29/users/x_hammo/NetAug/FADER
torchpack dist-run -np 5 python search_bounded_model.py 