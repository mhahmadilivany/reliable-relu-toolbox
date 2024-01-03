import argparse
import os
import torch.backends.cudnn
import torch.nn as nn
from fxpmath import Fxp
from torchpack import distributed as dist
import copy
from q_models.quantization import quan_Conv2d,quan_Linear
from setup import build_data_loader, build_model ,replace_act,change_quan_bitwidth
from utils import load_state_dict_from_file
from train import eval_fault,eval
import random
import numpy as np 
parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu", type=str, default=None
)  # used in single machine experiments
parser.add_argument("--batch_size", type=int, default=500)
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
        "imagenet",
        "imagenet21k_winter_p",
        "car",
        "flowers102",
        "food101",
        "cub200",
        "pets",
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
    default="alexnet",
    choices=[
        "lenet",
        "lenet_cifar10",
        "vgg16",
        "resnet50",
        "alexnet",
        "lenet_cifar10",
        "lenet_q", 
        "lenet_cifar10_q", 
        "vgg16_q",
        "resnet50_q", 
        "alexnet_q", 
    ],
)

parser.add_argument("--init_from", type=str, default="./pretrained_models/alexnet_cifar10/checkpoint/best.pt")
parser.add_argument("--save_path", type=str, default=None)
parser.add_argument("--name_relu_bound",type=str,default="fader")
parser.add_argument("--name_serach_bound",type=str, default="fader")
parser.add_argument("--bounds_type",type=str, default="layer")
parser.add_argument("--bitflip",type=str, default="fixed")
parser.add_argument("--fault_rates",type=list, default=[1e-7,1e-6,3e-6,1e-5,3e-5])



if __name__ == "__main__":
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

    # build model
    model = build_model(args.model, n_classes,0.0).cuda()
    if isinstance(list(model.children())[-1],nn.ReLU):
        print("The model with ReLU in the last layer")
    else:
        print("The model without ReLU in the last layer")    
    # exit()    
    print(args.dataset,args.model,args.name_relu_bound,args.name_serach_bound,args.bounds_type,args.bitflip,args.iterations)
    # print(model)
    #load checkpoint
    checkpoint = load_state_dict_from_file(args.init_from)
    model.load_state_dict(checkpoint) 
    if args.bitflip=='fixed':
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param!=None:
                    param.copy_(torch.tensor(Fxp(param.clone().cpu().numpy(), True, n_word=args.n_word,n_frac=args.n_frac,n_int=args.n_int).get_val()))
    elif args.bitflip == "int":
        change_quan_bitwidth(model,args.n_word)
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            # simple step size update based on the pretrained model or weight init
                m.__reset_stepsize__()    
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                m.__reset_weight__()
    print("model accuracy in {} format = {} before replace ReLU activcation functions".format(args.bitflip,eval(model,data_loader_dict)))  
    if args.name_relu_bound=="none":
        for fault_rate in args.fault_rates:
            val_results_fault = eval_fault(model,data_loader_dict,fault_rate,args.iterations,args.bitflip,args.n_word , args.n_frac, args.n_int)
            print("top1 = {} ,  top5 = {} , Val_loss = {} , fault_rate = {}" .format(val_results_fault['val_top1'],val_results_fault['val_top1'],val_results_fault['val_loss'],val_results_fault['fault_rate']))   
    else:             
        model = replace_act(model,args.name_relu_bound,args.name_serach_bound,data_loader_dict,args.bounds_type,args.bitflip)
        # print(model)
        print("model accuracy in {} format = {} after replace ReLU activcation functions".format(args.bitflip,eval(model,data_loader_dict)))
        for fault_rate in args.fault_rates:
            val_results_fault = eval_fault(model,data_loader_dict,fault_rate,args.iterations,args.bitflip,args.n_word , args.n_frac, args.n_int)
            print("top1 = {} ,  top5 = {} , Val_loss = {} , fault_rate = {}" .format(val_results_fault['val_top1'],val_results_fault['val_top1'],val_results_fault['val_loss'],val_results_fault['fault_rate']))   













