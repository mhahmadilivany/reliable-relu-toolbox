import sys
import torch.nn as nn
import torch
import copy
import numpy as np
import warnings
import sys;
from rrelu.relu_bound.bound_proact import bounded_hyrelu_proact
from rrelu.relu_bound.bound_zero import bounded_relu_zero
import os
import argparse
from typing import Dict, Optional
from rrelu.relu_bound.bound_relu import Relu_bound
from rrelu.pytorchfi.weight_error_models import multi_weight_inj_float,multi_weight_inj_fixed,multi_weight_inj_int
from rrelu.utils.metric import accuracy,AverageMeter
from rrelu.utils.lr_scheduler import CosineLRwithWarmup
from rrelu.utils.distributed import DistributedMetric
import random
from rrelu.pytorchfi.core import FaultInjection
import torch.nn.functional as F
import time
from tqdm import tqdm
from torchpack import distributed as dist
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, metavar="DIR", help="run directory",default="pretrained_models/lenet_mnist")
parser.add_argument("--base_batch_size", type=int, default=128)
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument(
    "--gpu", type=str, default=None
)  # used in single machine experiments

def Ranger_bounds_proact(model:nn.Module, train_loader, device="cuda", bound_type='layer',bitflip = 'float'):
    model.eval()
    iteration = True 
    results={}
    tresh={}
    relu_hooks(model,name='')      
    for data, label in train_loader['sub_train']:
        data = data.to(device)
        label = label.to(device)
        model = model.to(device)
        output = model(data)
        if iteration:
            for key, val in activation.items():
                results[key] = val
                tresh[key] = val
            iteration = False

        for key, val in activation.items():
            prev_max = torch.max(results[key],dim=0)[0]
            prev_mean = torch.mean(tresh[key],dim=0)
            curr_max = torch.max(activation[key],dim=0)[0]
            curr_mean = torch.mean(activation[key],dim=0)
            results[key] = torch.maximum(prev_max,curr_max)
            tresh[key] = torch.minimum(prev_mean,curr_mean)   
    
    return results,tresh,None
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
            if isinstance(layer,nn.ReLU) and 'last' not in name1:
                name_ = name1 + name
                if tresh==None:
                    model._modules[name1] = bounded_hyrelu_proact(bounds[name_].detach(),tresh)   
                else:    
                    model._modules[name1] = bounded_hyrelu_proact(bounds[name_].detach(),tresh[name_].detach())  
            elif isinstance(layer,nn.ReLU) and 'last'  in name1:
                name_ = name1 + name
                if tresh==None:
                    model._modules[name1] = bounded_hyrelu_proact(bounds[name_].detach(),tresh,k=-20)   
                else:    
                    model._modules[name1] = bounded_hyrelu_proact(bounds[name_].detach(),tresh[name_].detach(),k=-20.0)  

        else:
            name+=name1
            replace_act_all(layer,bounds,tresh,name)               
    return model  
  

def eval_fault(model:nn.Module,data_loader_dict, fault_rate,iterations=2000,bitflip=None,total_bits = 32 , n_frac = 16 , n_int = 15 )-> Dict:
    inputs, classes = next(iter(data_loader_dict['val'])) 
    pfi_model = FaultInjection(model, 
                            inputs.shape[0],
                            input_shape=[inputs.shape[1],inputs.shape[2],inputs.shape[3]],
                            layer_types=[torch.nn.Conv2d, torch.nn.Linear ,Relu_bound],
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
from rrelu.models import (

   ResNet50,
)
def build_model(
    name: str,
    n_classes=10,
    dropout_rate=0.0,
    **kwargs,
) -> nn.Module:

    model_dict = {

        "resnet50": ResNet50,
    }

    name = name.split("-")
    if len(name) > 1:
        kwargs["width_mult"] = float(name[1])
    name = name[0]

    return model_dict[name](n_classes=n_classes, dropout_rate=dropout_rate, **kwargs)
def load_state_dict_from_file(file: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(file, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint
def proact_bounds(model:nn.Module, train_loader, device="cuda", bound_type='layer', bitflip='float'):
    model.eval()
    original_model  = copy.deepcopy(model)
    results,tresh,_ =  Ranger_bounds_proact(copy.deepcopy(model),train_loader,device,bound_type,bitflip) # FtClipAct_bounds(copy.deepcopy(model),teacher_model,train_loader,device,bound_type,bitflip)
    len_relu = len(results)
    if bound_type =="layer":
        for i,(key, val) in enumerate(results.items()):
                if i<len_relu - 1:
                    results[key] = torch.max(val)  
                    tresh[key] = torch.min(tresh[key]) 
    print(results)        
    model = replace_act_all(model,results,tresh)
    torch.save(model.state_dict(), "temp_{}_{}_{}.pth".format(bound_type,bitflip,original_model.__class__.__name__))      
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
            param.requires_grad=False      
    # model = nn.parallel.DistributedDataParallel(
    #     model.cuda(), device_ids=[dist.local_rank()]
    # )
    # for name, param in model.named_parameters():
    #     if np.any([key in name for key in ["weight", "norm","bias"]]):
    #         param.requires_grad=False
    #     else:
    #         print(name)
    #         param.requires_grad=False 
     
    weight_decay_list =[4e-1,4e-2,4e-3,4e-4,4e-5,4e-6,4e-7,4e-8,4e-9,4e-10,4e-11]
    model = train(model,original_model,train_loader,weight_decay_list,bound_type,bitflip)
    for name, param in model.named_parameters():
        if np.any([key in name for key in ["weight", "norm","bias"]]):
            param.requires_grad=False
        else:
            print(name)
            param.requires_grad=True   
    bounds_dict = {}
    keys=[]
    i=0
   
    for key,val in results.items():
        keys.append(key)
    # print(keys)    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if np.any([key in name for key in ["bounds_param"]]):
                # print(param)
                bounds_dict[keys[i]]=param
                i+=1
    print(bounds_dict)            
    return bounds_dict,tresh,None



def distillation_loss(feat_s, feat_t,inputs,device):
    # loss = loss * ((teacher_hidden_representation > student_hidden_representation) | (teacher_hidden_representation > 0)).float()
    cosine_loss =  nn.CosineSimilarity()
    # KL_loss = nn.KLDivLoss()
    return torch.mean(cosine_loss(feat_s.view(feat_s.size(0),-1), feat_t.view(feat_t.size(0),-1))) #oss.sum()
    # return KL_loss(feat_t,feat_s)
def kl_loss(a,b):
    loss = -a*b + torch.log(b+1e-5)*b
    return loss
def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))
def L1_reg(model):
    nweights = 0
    for name,param in model.named_parameters():
        if param.requires_grad==True:
            if 'bounds'  in name:
                nweights = nweights + param.numel()
    L1_term = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        if param.requires_grad==True:
            if 'bounds'  in name:
                weights_sum = torch.sum(torch.abs(param))
                L1_term = L1_term + weights_sum
    L1_term = L1_term / nweights
    return L1_term


def train(model,original_model,data_provider,weight_decay_list,base_lr=0.01,warmup_epochs=5,n_epochs=5 , treshold=torch.tensor(1.5),bound_type = "layer" , bitflip = "fixed"):
    val_info_dict = eval(model, data_provider)
    best_acc =torch.tensor(val_info_dict["val_top1"])
    for name, param in reversed(list(model.named_parameters())):
        if np.any([key in name for key in ["weight", "norm","bias"]]):
            param.requires_grad=False
            continue
        else:
            print(param.nelement() ,name)
            param.requires_grad=True 
            if param.nelement() > 3 : 
                base_lr = 0.001
                n_epochs = 100
                warmup_epochs=5
            else:
                base_lr = 0.01   
                n_epochs = 10
                warmup_epochs=5
        for wd in weight_decay_list:
            params_with_wd=[]
            for name, param in model.named_parameters():
                if param.requires_grad:
                    params_with_wd.append(param)
            net_params = [
                {
                    "params": params_with_wd,
                    "weight_decay": wd,
                },
            ]
            optimizer = torch.optim.AdamW(
                net_params,
                lr=base_lr * dist.size(),
                weight_decay=wd
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
            # test criterion 
            test_criterion = nn.CrossEntropyLoss()
            # init
            
            start_epoch = 0
            for epoch in range(
                start_epoch,
                n_epochs
                + warmup_epochs,
            ):
                train_info_dict = train_one_epoch(
                    model,
                    original_model,
                    data_provider,
                    epoch,
                    optimizer,
                    train_criterion,
                    lr_scheduler,
                )

            val_info_dict = eval(model, data_provider)
            if torch.abs(best_acc - val_info_dict["val_top1"]) <=treshold:
                print(wd)
                torch.save(model.state_dict(), "temp_{}_{}_{}.pth".format(bound_type,bitflip,original_model.__class__.__name__))
                param.requires_grad = False
                break
            else:
                model.load_state_dict(torch.load("temp_{}_{}_{}.pth".format(bound_type,bitflip,original_model.__class__.__name__)))  
    return model



def train_one_epoch(
    model: nn.Module,
    original_model,
    data_provider,
    epoch: int,
    optimizer,
    criterion,
    lr_scheduler,
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
            l2_bounds = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad==True:
                    if "bounds" in name:
                        l2_bounds += torch.mean(torch.pow(param, 2))
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_output = original_model(images).detach()
                # feat_t,_ = original_model.extract_feature(images)
                teacher_logits = F.softmax(teacher_output, dim=1)
            # feat_s,_ = model.extract_feature(images)    
            nat_logits = model(images)
            kd_loss = cross_entropy_loss_with_soft_target(
                        nat_logits,teacher_logits
                    )
            # k_loss = distillation_loss(feat_s,feat_t,images,'cuda')
            nat_logits = model(images)
            loss =  0.8 * criterion(nat_logits,labels) +0.1 * kd_loss  +  4e-8 * l2_bounds #+ 0.1 * k_loss #
            loss.backward()
            # for name,par in model.named_parameters():
            #     if par.requires_grad == True:
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
