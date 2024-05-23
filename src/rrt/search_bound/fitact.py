import sys
import torch.nn as nn
import torch
import copy
import numpy as np
import warnings
import sys; 
import setup 
from relu_bound.bound_fitact import bounded_relu_fitact
import os
import argparse
from utils.metric import accuracy,AverageMeter
from utils.lr_scheduler import CosineLRwithWarmup
from utils.distributed import DistributedMetric
from search_bound.ranger import Ranger_bounds
import random
import time
from tqdm import tqdm
from torchpack import distributed as dist
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, metavar="DIR", help="run directory",default="./pretrained_models/teachers/vgg16_bound_layer_relu_float/checkpoint")
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

def replace_act(model:nn.Module,bounds,tresh,name='')->nn.Module:
    for name1,layer in model.named_children():
        if list(layer.children()) == []:
            if isinstance(layer,nn.ReLU) and 'last' not in name1:
                name_ = name1 + name
                model._modules[name1] = bounded_relu_fitact(bounds[name_].detach(),tresh[name_],-20)
            elif  isinstance(layer,nn.ReLU) and 'last' in name1:
                print("last relu layer")
                name_ = name1 + name
                model._modules[name1] = bounded_relu_fitact(bounds[name_].detach(),tresh[name_],-20)
        else:
            name+=name1
            replace_act(layer,bounds,tresh,name)               
    return model  
  




def fitact_bounds(model:nn.Module,train_loader, device="cuda", bound_type='layer',bitflip='float'):
    model.eval()
    results,tresh,_ = Ranger_bounds(copy.deepcopy(model),train_loader,device,bound_type,bitflip)
    # print(results['relu1'])
    model = replace_act(model,results,tresh)
    # print(model)
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
            param.requires_grad=True
    model = nn.parallel.DistributedDataParallel(
        model.cuda(), device_ids=[dist.local_rank()]
    )
    for name, param in model.named_parameters():
        if np.any([key in name for key in ["weight", "norm","bias"]]):
            param.requires_grad=False
        else:
            param.requires_grad=True

    model = train(model, train_loader, args.path)
    # print(eval(model,train_loader))
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
    return bounds_dict,bounds_dict,None

def train(
    model: nn.Module,
    data_provider,
    path: str,
    base_lr=0.001,
    warmup_epochs = 0 ,
    n_epochs = 150,
    weight_decay = 4e-9

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
    # build optimizer
    optimizer = torch.optim.Adam(
        net_params,
        lr=base_lr * dist.size(),
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


    # start training
    for epoch in range(
        start_epoch,
        n_epochs
        + warmup_epochs,
    ):
        train_info_dict = train_one_epoch(
            model,
            data_provider,
            epoch,
            optimizer,
            train_criterion,
            lr_scheduler,
        )
    
        val_info_dict = eval(model, data_provider)
        is_best = val_info_dict["val_top1"] > best_val
        best_val = max(best_val, val_info_dict["val_top1"])
        # log
        epoch_log = f"[{epoch + 1 - warmup_epochs}/{n_epochs}]"
        epoch_log += f"\tval_top1={val_info_dict['val_top1']:.2f} ({best_val:.2f})"
        epoch_log += f"\ttrain_top1={train_info_dict['train_top1']:.2f}\tlr={optimizer.param_groups[0]['lr']:.2E}"
        if dist.is_master():
            logs_writer.write(epoch_log + "\n")
            logs_writer.flush()
        # save checkpoint
        checkpoint = {
            "state_dict": model.module.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        if dist.is_master():
            torch.save(
                checkpoint,
                os.path.join(checkpoint_path, "checkpoint.pt"),
                _use_new_zipfile_serialization=False,
            )
            if is_best:
                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_path, "best.pt"),
                    _use_new_zipfile_serialization=False,
                )
    
         
    return model




def train_one_epoch(
    model: nn.Module,
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
        total=len(data_provider["train"]),
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
            output = model(images)
            loss = criterion(output, labels) + 4e-9 * l2_bounds
            loss.backward()
            # for par in model.parameters():
            #     print(par.grad)
            top1 = accuracy(output, labels, topk=(1,))[0][0]
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


