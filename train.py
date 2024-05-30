import torch.nn as nn 
import torch.optim as optim 
import torch
from torch import autograd
from typing import Dict
from rrelu.utils.distributed import DistributedMetric
import argparse
import copy
import os
import time
import warnings
from typing import Dict, Optional
from rrelu.setup import build_data_loader,build_model
import numpy as np
import torch
import torch.nn as nn
import yaml
from torchpack import distributed as dist
from tqdm import tqdm
from rrelu.utils.metric import accuracy,AverageMeter
from rrelu.utils.lr_scheduler import CosineLRwithWarmup
from rrelu.utils.init import load_state_dict,init_modules
from rrelu.utils import load_state_dict_from_file
from rrelu.pytorchfi.weight_error_models import multi_weight_inj_fixed,multi_weight_inj_float,multi_weight_inj_int
from rrelu.relu_bound.bound_relu import Relu_bound
from rrelu.pytorchfi.core import FaultInjection
import random 
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, metavar="DIR", help="run directory",default="pretrained_models/resnet50_cifar100")
parser.add_argument("--dataset", type=str, help="dataset",default="cifar100")
parser.add_argument("--n_worker", type=int, default=8)
parser.add_argument("--data_path", type=str, help="data_path",default="./dataset/cifar100/CIFAR100")
parser.add_argument("--base_batch_size", type=int, default=128)
parser.add_argument(
    "--gpu", type=str, default=None
)  # used in single machine experiments
parser.add_argument("--name", type=str, help="model name",default="resnet50_cifar100")
parser.add_argument("--init_type", type=str, help="init_type",default="he_fout")
parser.add_argument("--dropout_rate", type=float, default=0.0)
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument("--image_size", type=int, default=32)
parser.add_argument("--resume", action="store_true")

# initialization
parser.add_argument("--init_from", type=str, default=None) #"pretrained_models/lenet_mnist/checkpoint/best.pt"

def eval_fault(model:nn.Module,data_loader_dict, fault_rate,iterations=500,bitflip=None,total_bits = 32 , n_frac = 16 , n_int = 15 )-> Dict:
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
                    loss = test_criterion(output, labels)
                    val_loss.update(loss, images.shape[0])
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    val_top5.update(acc5[0], images.shape[0])
                    val_top1.update(acc1[0], images.shape[0])
                    
                ####        
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

def train_one_epoch(
    model: nn.Module,
    data_provider,
    epoch: int,
    optimizer,
    criterion,
    lr_scheduler,

) -> Dict:
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

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
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


def train(
    model: nn.Module,
    data_provider:Dict,
    path: str,
    resume=False,
    base_lr=0.01,
    warmup_epochs = 5 ,
    n_epochs = 300,
    weight_decay=5e-4
):
    
    params_without_wd = []
    params_with_wd = []
    for name, param in model.named_parameters():
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
        model.module.load_state_dict(checkpoint["state_dict"])
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

    # build model
    model = build_model(
        args.name,
        n_classes,
        args.dropout_rate,
    )
    print(model)

    # load init
    if args.init_from is not None:
        init = load_state_dict_from_file(args.init_from)
        load_state_dict(model, init, strict=False)
        print("Loaded init from %s" % args.init_from)
    else:
        if  np.all( [key  not in args.name for key in ['vgg','nin']]) :
            init_modules(model, init_type=args.init_type)
            print("Random Init")



    # train
    generator = torch.Generator()
    generator.manual_seed(args.manual_seed)
    model = nn.parallel.DistributedDataParallel(
        model.cuda(), device_ids=[dist.local_rank()]
    )
    train(model, data_provider, args.path, args.resume)
    # eval_fault(model,data_provider,1e-3)
if __name__ == "__main__":
    main()
