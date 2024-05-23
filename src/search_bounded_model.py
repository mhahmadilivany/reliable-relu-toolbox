import argparse
import os
import torch.backends.cudnn
import torch.nn as nn
from fxpmath import Fxp
from torchpack import distributed as dist
import copy
from setup import build_data_loader, build_model ,replace_act,replace_act_all
from utils import load_state_dict_from_file
from train import eval_fault,eval
import random
import numpy as np 
parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu", type=str, default=None
)  # used in single machine experiments
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_worker", type=int, default=8)
parser.add_argument("--iterations", type=int, default=100)
parser.add_argument("--n_word", type=int, default=32)
parser.add_argument("--n_frac", type=int, default=16)
parser.add_argument("--n_int", type=int, default=15)
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    choices=[
        "mnist",
        "cifar10",
        "cifar100"
    ],
)
parser.add_argument("--data_path", type=str, default="./dataset/cifar10/cifar10")
parser.add_argument("--image_size", type=int, default=32)
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument(
    "--model",
    type=str,
    default="vgg16",
    choices=[
        "lenet",
        "lenet_cifar10",
        "vgg16",
        "resnet50",
        "alexnet",
        "lenet_cifar10",
        "alexnet_cifar100" ,
        "resnet50_cifar100" ,
        "vgg16_cifar100" 
    ],
)

parser.add_argument("--init_from", type=str, default="./pretrained_models/vgg16_cifar10_c/checkpoint/best.pt")
parser.add_argument("--save_path", type=str, default=None)
parser.add_argument("--name_relu_bound",type=str,default="zero",
    choices=[
        "zero",
        "tresh",
        "fitact",
        "proact",
    ])
parser.add_argument("--name_serach_bound",type=str, default="ranger", choices=[
        "ranger",
        "ftclip",
        "fitact",
        "proact",
    ])
parser.add_argument("--bounds_type",type=str, default="layer", choices=[
        "layer",
        "neuron",
    ])
parser.add_argument("--bitflip",type=str, default="fixed", choices=[
        "fixed",
        "float",
    ])
parser.add_argument("--fault_rates",type=list, default=[1e-7,1e-6,3e-6,1e-5,3e-5])
if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parser.parse_args()
    # setup gpu and distributed training
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dist.init()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    # build data loader
    data_loader_dict, n_classes = build_data_loader(
        args.dataset,
        args.image_size,
        args.batch_size,
        args.n_worker,
        args.data_path,
        dist.size(),
        dist.rank(),
    )

    model = build_model(args.model, n_classes,0.0).cuda()
    checkpoint = load_state_dict_from_file(args.init_from)
    model.load_state_dict(checkpoint) 
    
    if isinstance(list(model.children())[-1],nn.ReLU):
        print("The model with ReLU in the last layer")
    else:
        print("The model without ReLU in the last layer")       
    print(args.dataset,args.model,args.name_relu_bound,args.name_serach_bound,args.bounds_type,args.bitflip,args.iterations)
   
    
    if args.bitflip=='fixed':
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param!=None:
                    param.copy_(torch.tensor(Fxp(param.clone().cpu().numpy(), True, n_word=args.n_word,n_frac=args.n_frac,n_int=args.n_int).get_val()))
    print("model accuracy in {} format = {} before replace ReLU activcation functions".format(args.bitflip,eval(model,data_loader_dict)))  
        
    if args.name_relu_bound=="none":
        for fault_rate in args.fault_rates:
            val_results_fault = eval_fault(model,data_loader_dict,fault_rate,args.iterations,args.bitflip,args.n_word , args.n_frac, args.n_int)
            print("top1 = {} ,  top5 = {} , Val_loss = {} , fault_rate = {}" .format(val_results_fault['val_top1'],val_results_fault['val_top1'],val_results_fault['val_loss'],val_results_fault['fault_rate']))   
    
    else:             
        model = replace_act(model,args.name_relu_bound,args.name_serach_bound,data_loader_dict,args.bounds_type,args.bitflip)
        print("model accuracy in {} format = {} after replace ReLU activcation functions".format(args.bitflip,eval(model,data_loader_dict)))
        for fault_rate in args.fault_rates:
            val_results_fault = eval_fault(model,data_loader_dict,fault_rate,args.iterations,args.bitflip,args.n_word , args.n_frac, args.n_int)
            print("top1 = {} ,  top5 = {} , Val_loss = {} , fault_rate = {}" .format(val_results_fault['val_top1'],val_results_fault['val_top1'],val_results_fault['val_loss'],val_results_fault['fault_rate']))   













