import torch.nn as nn 
import torch.optim as optim 
import torch
from torch import autograd
from typing import Dict
from utils.distributed import DistributedMetric
import argparse
import copy
import os
import time
import warnings
from typing import Dict, Optional,List
from setup import build_data_loader,build_model
import numpy as np
import torch
import torch.nn as nn
import yaml
from torchpack import distributed as dist
from tqdm import tqdm
from utils.metric import accuracy,AverageMeter
from utils.lr_scheduler import CosineLRwithWarmup
from utils.init import load_state_dict,init_modules
from utils import load_state_dict_from_file
from pytorchfi.weight_error_models import multi_weight_inj
from pytorchfi.core import FaultInjection
import random 
from setup import *
import torch.nn.functional as F
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, metavar="DIR", help="run directory",default="pretrained_models/faultaware/alexnet_cifar10")
parser.add_argument("--dataset", type=str, help="dataset",default="cifar10")
parser.add_argument("--n_worker", type=int, default=8)
parser.add_argument("--data_path", type=str, help="data_path",default="./dataset/cifar10/cifar10")
parser.add_argument("--base_batch_size", type=int, default=128)
parser.add_argument(
    "--gpu", type=str, default=None
)  # used in single machine experiments
parser.add_argument("--name", type=str, help="model name",default="alexnet")
parser.add_argument("--init_type", type=str, help="init_type",default="he_fout")
parser.add_argument("--dropout_rate", type=float, default=0.0)
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--image_size", type=int, default=28)
# initialization
parser.add_argument("--init_from", type=str, default="pretrained_models/alexnet_cifar10/checkpoint/best.pt") #"pretrained_models/lenet_mnist/checkpoint/best.pt"



def eval_fault(model:nn.Module,data_loader_dict, fault_rate,iterations=10)-> Dict:
    inputs, classes = next(iter(data_loader_dict['val'])) 
    pfi_model = FaultInjection(model, 
                            inputs.shape[0],
                            input_shape=[inputs.shape[1],inputs.shape[2],inputs.shape[3]],
                            layer_types=[torch.nn.Conv2d, torch.nn.Linear],
                            use_cuda=True,
                            )
    
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
            for i in range(iterations):
                corrupted_model = multi_weight_inj(pfi_model,fault_rate,seed=i)
                for images, labels in data_loader_dict["val"]:
                    images, labels = images.cuda(), labels.cuda()
                    # compute output

                    output = corrupted_model(images)
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
        # print( val_top1.avg.item())
        val_results = {
            "val_top1": val_top1.avg.item(),
            "val_top5": val_top5.avg.item(),
            "val_loss": val_loss.avg.item(),
            "fault_rate": fault_rate,
        }
    return val_results

def eval(model: nn.Module, data_loader_dict) -> Dict:

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

def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))

def train_one_epoch(
    pfi_model: FaultInjection,
    fault_rates:List,
    data_provider,
    epoch: int,
    optimizer,
    criterion,
    lr_scheduler,

) -> Dict:
    train_loss = DistributedMetric()
    train_top1 = DistributedMetric()
    pfi_model.original_model.train()
    data_provider['train'].sampler.set_epoch(epoch)

    data_time = AverageMeter()
    with tqdm(
        total=len(data_provider["train"]),
        desc="Train Epoch #{}".format(epoch + 1),
        disable=not dist.is_master(),
    ) as t:
        end = time.time()
        for fault_rate in fault_rates:
            for _, (images, labels) in enumerate(data_provider['train']):
                corrupted_model = multi_weight_inj(pfi_model,fault_rate,seed=_)
                data_time.update(time.time() - end)
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    output_corrupt = corrupted_model(images).detach()
                    output_corrupt_soft_label = F.softmax(output_corrupt, dim=1)
                optimizer.zero_grad()
                output_original = pfi_model.original_model(images)
                kd_loss = cross_entropy_loss_with_soft_target(
                            output_original, output_corrupt_soft_label
                        )
                loss =  kd_loss + criterion(output_original, labels)
                loss.backward()
                # for param in pfi_model.original_model.parameters():
                #     if torch.any(param > 1e5):
                #         print("yes")
                top1 = accuracy(output_original, labels, topk=(1,))[0][0]
                optimizer.step()
                lr_scheduler.step()

                train_loss.update(loss, images.shape[0])
                train_top1.update(top1, images.shape[0])

                t.set_postfix(
                    {
                        "loss": train_loss.avg.item(),
                        "fault_rate": fault_rate,
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


def train(
    pfi_model: FaultInjection,
    fault_rates : List,
    data_provider:Dict,
    path: str,
    resume=False,
    base_lr=0.01,
    warmup_epochs = 5 ,
    n_epochs = 100,
    weight_decay=4.0e-5
):
    
    params_without_wd = []
    params_with_wd = []
    for name, param in pfi_model.original_model.named_parameters():
        if param.requires_grad:
            if np.any([key in name for key in ["bias", "norm"]]):
                params_without_wd.append(param)
            else:
                params_with_wd.append(param)
    net_params = [
        {"params": params_without_wd, "weight_decay": 0},
        {
            "params": params_with_wd,
            "weight_decay": weight_decay,
        },
    ]
    # print(net_params)
    # build optimizer
    optimizer = torch.optim.SGD(
        net_params,
        lr=base_lr * dist.size(),
        momentum=0.9,
        nesterov=True,
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

    if resume and os.path.isfile(os.path.join(checkpoint_path, "checkpoint.pt")):
        checkpoint = torch.load(
            os.path.join(checkpoint_path, "checkpoint.pt"), map_location="cpu"
        )
        pfi_model.original_model.module.load_state_dict(checkpoint["state_dict"])
        if "best_val" in checkpoint:
            best_val = checkpoint["best_val"]
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # start training
    for epoch in range(
        start_epoch,
        n_epochs
        + warmup_epochs,
    ):

        train_info_dict = train_one_epoch(
            pfi_model,
            fault_rates,
            data_provider,
            epoch,
            optimizer,
            train_criterion,
            lr_scheduler,
        )
        
        val_info_dict = eval_fault(pfi_model.original_model,data_provider,1e-3)
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
            "state_dict": pfi_model.original_model.module.state_dict(),
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

def main():
    warnings.filterwarnings("ignore")
    args, opt = parser.parse_known_args()

    # setup gpu and distributed training
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.distributed.is_initialized():
        dist.init()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    # setup path
    os.makedirs(args.path, exist_ok=True)

    # setup random seed
    if args.resume:
        args.manual_seed = int(time.time())
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)

    # build data_loader
    data_provider, n_classes= build_data_loader(
        args.dataset,
        args.image_size,
        args.base_batch_size,
        args.n_worker,
        args.data_path,
        dist.size(),
        dist.rank(),
    )
    images,lables = next(iter(data_provider['val']))
    # build model
    model = build_model(
        args.name,
        n_classes,
        args.dropout_rate,
    )
    # fault injector init
    pfi_model = FaultInjection(model = model,batch_size=images.shape[0],input_shape=[images.shape[1],images.shape[2],images.shape[3]],layer_types=[torch.nn.Conv2d,torch.nn.Linear],use_cuda=True)
    # load init
    if args.init_from is not None:
        init = load_state_dict_from_file(args.init_from)
        load_state_dict(pfi_model.original_model, init, strict=False)
        print("Loaded init from %s" % args.init_from)
    else:
        if  np.all( [key  not in args.name for key in ['vgg','nin']]) :
            init_modules(pfi_model.original_model, init_type=args.init_type)
            print("Random Init")
    pfi_model.original_model = replace_act(copy.deepcopy(model),'tresh','ranger',data_provider)
    # for name,param in pfi_model.original_model.named_parameters():
    #     if 'bounds' in name:
    #         print("call")
    #         param.requires_grad = False
    eval(pfi_model.original_model,data_provider)
    eval_fault(pfi_model.original_model,data_provider,1e-3)
    # faultrates
    fault_rates=[1e-3] 
    # train
    generator = torch.Generator()
    generator.manual_seed(args.manual_seed)
    pfi_model.original_model = nn.parallel.DistributedDataParallel(
        pfi_model.original_model.cuda(), device_ids=[dist.local_rank()]
    )
    train(pfi_model,fault_rates,data_provider, args.path, args.resume)
if __name__ == "__main__":
    main()