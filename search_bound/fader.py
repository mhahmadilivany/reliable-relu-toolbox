import sys
import torch.nn as nn
import torch
import copy
import numpy as np
import warnings
import sys;
sys.path.append("/proj/berzelius-2023-29/users/x_hammo/NetAug/FADER") 
from search_bound.ranger import Ranger_bounds
import setup 
from relu_bound.bound_alpha import bounded_relu_alpha
from relu_bound.bound_zero import bounded_relu_zero
import os
import argparse
from typing import Dict, Optional
from relu_bound.bound_relu import Relu_bound
from q_models.quantization import quan_Conv2d,quan_Linear
from pytorchfi.weight_error_models import multi_weight_inj_float,multi_weight_inj_fixed,multi_weight_inj_int
from utils.metric import accuracy,AverageMeter
from utils.lr_scheduler import CosineLRwithWarmup
from utils.distributed import DistributedMetric
import random
from pytorchfi.core import FaultInjection
import torch.nn.functional as F
import time
from tqdm import tqdm
from torchpack import distributed as dist
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, metavar="DIR", help="run directory",default="pretrained_models/lenet_mnist/fader")
parser.add_argument("--base_batch_size", type=int, default=128)
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument(
    "--gpu", type=str, default=None
)  # used in single machine experiments


def eval(model: nn.Module, data_loader_dict) :

    test_criterion = nn.CrossEntropyLoss().cuda()

    val_loss = DistributedMetric()
    val_top1 = DistributedMetric()
    val_top5 = DistributedMetric()

    model.eval()
    with torch.no_grad():
        with tqdm(
            total=len(data_loader_dict["val"]),
            desc="Eval",
            disable=not dist.is_master(),
        ) as t:
            for images, labels in data_loader_dict["val"]:
                images, labels = images.cuda(), labels.cuda()
                # compute output
                output = model(images)
                loss = test_criterion(output, labels)
                val_loss.update(loss, images.shape[0])
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                val_top5.update(acc5[0], images.shape[0])
                val_top1.update(acc1[0], images.shape[0])

                t.set_postfix(
                    {
                        "loss": val_loss.avg.item(),
                        "top1": val_top1.avg.item(),
                        "top5": val_top5.avg.item(),
                        "#samples": val_top1.count.item(),
                        "batch_size": images.shape[0],
                        "img_size": images.shape[2],
                    }
                )
                t.update()

    val_results = {
        "val_top1": val_top1.avg.item(),
        "val_top5": val_top5.avg.item(),
        "val_loss": val_loss.avg.item(),
    }
    return val_results


activation={}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

def relu_hooks(model:nn.Module,name=''):
    for name1,layer in model.named_children():
        if list(layer.children()) == []:
            if isinstance(layer,nn.ReLU):
                name_ = name1 + name
                layer.register_forward_hook(get_activation(name_)) 
        else:
            name+=name1
            relu_hooks(layer,name)

def replace_act_all(model:nn.Module,bounds,tresh,name='')->nn.Module:
    for name1,layer in model.named_children():
        if list(layer.children()) == []:
            if isinstance(layer,nn.ReLU):
                name_ = name1 + name
                if tresh==None:
                    model._modules[name1] = bounded_relu_alpha(bounds[name_].detach(),tresh)   
                else:    
                    model._modules[name1] = bounded_relu_alpha(bounds[name_].detach(),tresh[name_].detach())    
        else:
            name+=name1
            replace_act_all(layer,bounds,tresh,name)               
    return model  
  

def eval_fault(model:nn.Module,data_loader_dict, fault_rate,iterations=2000,bitflip=None,total_bits = 32 , n_frac = 16 , n_int = 15 )-> Dict:
    inputs, classes = next(iter(data_loader_dict['val'])) 
    pfi_model = FaultInjection(model, 
                            inputs.shape[0],
                            input_shape=[inputs.shape[1],inputs.shape[2],inputs.shape[3]],
                            layer_types=[torch.nn.Conv2d, torch.nn.Linear ,Relu_bound,quan_Conv2d,quan_Linear],
                            total_bits= total_bits,
                            n_frac = n_frac, 
                            n_int = n_int, 
                            use_cuda=True,
                            )
    print(pfi_model.print_pytorchfi_layer_summary())
    test_criterion = nn.CrossEntropyLoss().cuda()

    val_loss = DistributedMetric()
    val_top1 = DistributedMetric()
    val_top5 = DistributedMetric()

    pfi_model.original_model.eval()
    with torch.no_grad():
        with tqdm(
            total= iterations,
            desc="Eval",
            disable=not dist.is_master(),
        ) as t:
            for i in range(iterations):
                if bitflip=='float':
                    corrupted_model = multi_weight_inj_float(pfi_model,fault_rate)
                elif bitflip=='fixed':    
                    corrupted_model = multi_weight_inj_fixed(pfi_model,fault_rate)
                elif bitflip =="int":
                    corrupted_model = multi_weight_inj_int (pfi_model,fault_rate)
                    # corrupted_model = multi_weight_inj_int(pfi_model,fault_rate)    
                for images, labels in data_loader_dict["val"]:
                    images, labels = images.cuda(), labels.cuda()
                    output = corrupted_model(images)
                    # print(output)
                    loss = test_criterion(output, labels)
                    val_loss.update(loss, images.shape[0])
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    val_top5.update(acc5[0], images.shape[0])
                    val_top1.update(acc1[0], images.shape[0])
                    
                t.set_postfix(
                    {
                        "loss": val_loss.avg.item(),
                        "top1": val_top1.avg.item(),
                        "top5": val_top5.avg.item(),
                        "#samples": val_top1.count.item(),
                        "batch_size": images.shape[0],
                        "img_size": images.shape[2],
                        "fault_rate": fault_rate,
                    }
                )
                t.update()
                # pfi_model.original_model = corrupted_model    
        val_results = {
            "val_top1": val_top1.avg.item(),
            "val_top5": val_top5.avg.item(),
            "val_loss": val_loss.avg.item(),
            "fault_rate": fault_rate,
        }
    return val_results


def fader_bounds(model:nn.Module, train_loader, device="cuda", bound_type='layer', bitflip='float'):
    model.eval()
    original_model  = copy.deepcopy(model)
    results,tresh,_ = Ranger_bounds(copy.deepcopy(model),train_loader,device,bound_type,bitflip)
    model = replace_act_all(model,results,tresh)
    eval(model,train_loader)
    warnings.filterwarnings("ignore")
    args, opt = parser.parse_known_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.distributed.is_initialized():
        dist.init()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())
    os.makedirs(args.path, exist_ok=True)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    for name, param in model.named_parameters():
        if np.any([key in name for key in ["weight", "norm","bias"]]):
            param.requires_grad=False
        else:
            print(name)
            param.requires_grad=True      
    model = nn.parallel.DistributedDataParallel(
        model.cuda(), device_ids=[dist.local_rank()]
    )
   
    model = train(original_model , model, train_loader, args.path,bitflip)
    alpha = {}
    keys=[]
    i=0
    for key,val in results.items():
        keys.append(key)
    # print(keys)    
    for name, param in model.module.named_parameters():
        if param.requires_grad:
            if np.any([key in name for key in ["alpha"]]):
                # print(param)
                alpha[keys[i]]=param
                i+=1
    
    print(alpha)
    bounds_dict = {}
    keys=[]
    i=0
   
    for key,val in results.items():
        keys.append(key)
    # print(keys)    
    for name, param in model.module.named_parameters():
        if param.requires_grad:
            if np.any([key in name for key in ["bounds_param"]]):
                # print(param)
                bounds_dict[keys[i]]=param
                i+=1
    print(bounds_dict)            
    return bounds_dict,tresh,alpha

def train(
    original_model:nn.Module,
    model: nn.Module,
    data_provider,
    path: str,
    bitflip='float',
    base_lr=0.1,
    warmup_epochs = 0 ,
    n_epochs = 20,
    weight_decay=4e-8
):
    params_without_wd = []
    params_with_wd = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if np.any([key in name for key in ["bias", "norm"]]):
                params_without_wd.append(param)
            else:
                # print(name)
                params_with_wd.append(param)
    net_params = [
        {"params": params_without_wd, "weight_decay": 0},
        {
            "params": params_with_wd,
            "weight_decay": weight_decay,
        },
    ]
    optimizer = torch.optim.SGD(
        net_params,
        lr=base_lr * dist.size(),
        weight_decay=weight_decay
    )
    # build lr scheduler
    lr_scheduler = CosineLRwithWarmup(
        optimizer,
        warmup_epochs * len(data_provider['train']),
        base_lr,
        n_epochs * len(data_provider['train']),
    )
    # train criterion
    train_criterion = nn.CrossEntropyLoss()
    # init
    best_val = 0.0
    start_epoch = 0
    checkpoint_path = os.path.join(path, "checkpoint")
    log_path = os.path.join(path, "logs")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    logs_writer = open(os.path.join(log_path, "exp.log"), "a") 
    # fault_rates = [1e-7,1e-6,3e-6,1e-5,3e-5]
    # images,lables = next(iter(data_provider['val']))
    # pfi_model = FaultInjection(model = copy.deepcopy(model),batch_size=images.shape[0],input_shape=[images.shape[1],images.shape[2],images.shape[3]],layer_types=[torch.nn.Conv2d, torch.nn.Linear ,Relu_bound,quan_Conv2d,quan_Linear],use_cuda=True)
    # for iteration in range(10):
        # print("iteration = {}".format(iteration))
        # for fault in fault_rates:
        #     print("fault_rate = {}".format(fault))
        #     if bitflip=='float':
        #         corrupted_model = multi_weight_inj_float(pfi_model,fault)
        #     elif bitflip=='fixed':    
        #         corrupted_model = multi_weight_inj_fixed(pfi_model,fault)
        #     elif bitflip =="int":
        #         corrupted_model = multi_weight_inj_int (pfi_model,fault)  

    for epoch in range(
        start_epoch,
        n_epochs
        + warmup_epochs,
    ):
        train_info_dict = train_one_epoch(
            original_model,
            original_model,
            model,
            data_provider,
            epoch,
            optimizer,
            train_criterion,
            lr_scheduler,
            bitflip,
        )

        val_info_dict = eval(model, data_provider)
# print(val_info_dict)
    return model
def distillation_loss(student_hidden_representation, teacher_hidden_representation,inputs,device):
    # loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    # loss = loss * ((source > target) | (target > 0)).float()
    cosine_loss = nn.CosineEmbeddingLoss()
    return cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(inputs.size(0)).to(device))
def kl_loss(a,b):
    loss = -a*b + torch.log(b+1e-5)*b
    return loss
def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))


def train_one_epoch(
    corrupted_model : nn.Module,
    original_model:nn.Module,
    model: nn.Module,
    data_provider,
    epoch: int,
    optimizer,
    criterion,
    lr_scheduler,
    bitflip,

):
    train_loss = DistributedMetric()
    train_top1 = DistributedMetric()
    model.train()
    data_provider['train'].sampler.set_epoch(epoch)

    data_time = AverageMeter()
    with tqdm(
        total=len(data_provider["train"]) ,
        desc="Train Epoch #{}".format(epoch + 1),
        disable=not dist.is_master(),
    ) as t:
        end = time.time()         
        for _, (images, labels) in enumerate(data_provider['train']):
            data_time.update(time.time() - end)
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                t_feats, t_out = original_model.extract_feature(images)
                teacher_output = corrupted_model(images).detach()
                teacher_logits = F.softmax(teacher_output, dim=1)
            s_feats, s_out = model.module.extract_feature(images)
            feat_num = len(t_feats)
            loss_distill = 0
            for i in range(feat_num):
                loss_distill += distillation_loss(s_feats[i], t_feats[i],images,'cuda') / 2 ** (feat_num - i - 1)
            
                # loss_distill.sum()
            optimizer.zero_grad()
            nat_logits = model(images)
            kd_loss = cross_entropy_loss_with_soft_target(
                        nat_logits,teacher_logits
                    )
            loss =   loss_distill  + kd_loss +  criterion(nat_logits, labels) #+ torch.mean(kd_loss) +   #+ 
            loss.backward()
            # for name, par in model.named_parameters():
            #     if par.requires_grad ==True :
            #         print(par.grad)
            top1 = accuracy(nat_logits, labels, topk=(1,))[0][0]
            optimizer.step()
            lr_scheduler.step()

            train_loss.update(loss, images.shape[0])
            train_top1.update(top1, images.shape[0])

            t.set_postfix(
                {
                    "loss": train_loss.avg.item(),
                    "top1": train_top1.avg.item(),
                    "batch_size": images.shape[0],
                    "img_size": images.shape[2],
                    "lr": optimizer.param_groups[0]["lr"],
                    "data_time": data_time.avg,
                }
            )
            t.update()

            end = time.time()
    return {
        "train_top1": train_top1.avg.item(),
        "train_loss": train_loss.avg.item(),
    }







if __name__ == "__main__":
    data,_ = setup.build_data_loader('mnist',28,16)
    model = setup.build_model('lenet')
    # print(model)
