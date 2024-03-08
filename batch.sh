#!/bin/bash
#SBATCH -J alexnet_int_8_relu_layer
#SBATCH --gpus 2
#SBATCH -t 20:00:00
#SBATCH -A berzelius-2024-8
cd /proj/berzelius-2023-29/users/x_hammo/NetAug/FADER
torchpack dist-run -np 2 python search_bounded_model.py --dataset cifar10 --data_path ./dataset/cifar10/cifar10 --model vgg16 --teacher_model vgg16 --init_from ./pretrained_models/vgg16_cifar10_c/checkpoint/best.pt \
                    --init_teacher_from ./pretrained_models/vgg16_cifar10_c/checkpoint/best.pt  --name_relu_bound fader --name_serach_bound fader --bounds_type layer --bitflip fixed --init_teacher_bounds_neuron False --init_teacher_bounds_layer False