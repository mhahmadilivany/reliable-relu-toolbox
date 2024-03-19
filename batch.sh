#!/bin/bash
#SBATCH -J alexnet_int_8_relu_layer
#SBATCH --gpus 1
#SBATCH -t 20:00:00
#SBATCH -A berzelius-2024-8
cd /proj/berzelius-2023-29/users/x_hammo/NetAug/FADER
torchpack dist-run -np 1 python search_bounded_model.py --dataset cifar100 --data_path ./dataset/cifar100/cifar100 --model resnet50_cifar100 --teacher_model vgg16 --init_from ./pretrained_models/resnet50_cifar100/checkpoint/best.pt \
                    --init_teacher_from ./pretrained_models/vgg16_cifar10_c/checkpoint/best.pt  --name_relu_bound none --name_serach_bound ftclip --bounds_type layer --bitflip fixed --init_teacher_bounds_neuron False --init_teacher_bounds_layer False