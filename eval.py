import argparse
import os
from fxpmath import Fxp
import torch.backends.cudnn
import torch.nn as nn
from torchpack import distributed as dist

from setup import build_data_loader, build_model
from train import eval,eval_fault
from utils import load_state_dict_from_file

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu", type=str, default=None
)  # used in single machine experiments
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--n_worker", type=int, default=8)
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
parser.add_argument("--fault_rate", type=float, default=1e-4)
parser.add_argument("--q_bit", type=int, default=8)
parser.add_argument("--n_word", type=int, default=32)
parser.add_argument("--n_frac", type=int, default=16)
parser.add_argument("--n_int", type=int, default=15)
parser.add_argument("--bitflip",type=str, default="int")
parser.add_argument("--fault_rates",type=list, default=[3e-5])
parser.add_argument("--iterations", type=int, default=10)
parser.add_argument(
    "--model",
    type=str,
    default="resnet50_q",
    choices=[
        "lenet",
        "lenet_cifar10",
        "vgg16",
        "resnet50",
        "nin",
        "alexnet",
    ],
)

parser.add_argument("--init_from", type=str, default="./pretrained_models/resnet50_cifar10/checkpoint/best.pt")
parser.add_argument("--save_path", type=str, default=None)
parser.add_argument("--n_word", type=int, default=32)
if __name__ == "__main__":
    args = parser.parse_args()
    # setup gpu and distributed training
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dist.init()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

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
    model = build_model(args.model, n_classes, 0).cuda()

    # load checkpoint
    checkpoint = load_state_dict_from_file(args.init_from)
    model.load_state_dict(checkpoint)
    if args.bitflip=='fixed':
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param!=None:
                    param.copy_(torch.tensor(Fxp(param.clone().cpu().numpy(), True, n_word=args.n_word,n_frac=args.n_frac,n_int=args.n_int).get_val()))
    elif args.bitflip == "int":
        change_quan_bitwidth(model,args.q_bit)
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            # simple step size update based on the pretrained model or weight init
                m.__reset_stepsize__()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[dist.local_rank()])
    val_results = eval(model, data_loader_dict)
    val_results_fault = eval_fault(model,data_loader_dict,args.fault_rate,args.iterations,args.bitflip,args.n_word,args.n_frac,args.n_int,args.q_bit)

    for key, val in val_results.items():
        print(key, ": ", val)
    print("###########################################\n")
    for key, val in val_results_fault.items():
        print(key, ": ", val)