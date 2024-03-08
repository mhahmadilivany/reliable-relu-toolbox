import argparse
import os
import torch.backends.cudnn
import torch.nn as nn
from fxpmath import Fxp
from torchpack import distributed as dist
import copy
from pytorchfi.core import FaultInjection
from q_models.quantization import quan_Conv2d,quan_Linear
from setup import build_data_loader, build_model ,replace_act,change_quan_bitwidth,replace_act_all
from utils import load_state_dict_from_file
from pytorchfi.weight_error_models import multi_weight_inj_fixed,multi_weight_inj_float,multi_weight_inj_int
from train import eval_fault,eval
import random
from relu_bound.bound_relu import Relu_bound
import matplotlib.pyplot as plt
import numpy as np 
from distutils.util import strtobool
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
    default="vgg16",
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
parser.add_argument(
    "--teacher_model",
    type=str,
    default="vgg16",
    choices=[
        "vgg16",
        "resnet50",
    ],
)
parser.add_argument("--init_from", type=str, default="./pretrained_models/vgg16_cifar10_c/checkpoint/best.pt")
parser.add_argument("--init_teacher_from", type=str, default="./pretrained_models/vgg16_cifar10_c/checkpoint/best.pt") #"./pretrained_models/teachers/vgg16_bound_layer/checkpoint/checkpoint/checkpoint.pt"
parser.add_argument("--init_teacher_bounds_neuron", type=lambda x: bool(strtobool(x)),default=False)#"./pretrained_models/teachers/vgg16_bound_neuron/checkpoint/checkpoint/checkpoint.pt"
parser.add_argument("--init_teacher_bounds_layer", type=lambda x: bool(strtobool(x)),default=False)
parser.add_argument("--save_path", type=str, default=None)
parser.add_argument("--name_relu_bound",type=str,default="fitact")
parser.add_argument("--name_serach_bound",type=str, default="fitact")
parser.add_argument("--bounds_type",type=str, default="layer")
parser.add_argument("--bitflip",type=str, default="fixed")
parser.add_argument("--fault_rates",type=list, default=[1e-7,1e-6,3e-6,1e-5,3e-5])

from search_bound.ranger import Ranger_bounds
from relu_bound.bound_fitact import bounded_relu_fitact
from relu_bound.bound_zero import bounded_relu_zero
activation={}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook


def relu_hooks(model:nn.Module,name=''):
    for name1,layer in model.named_children():
        if list(layer.children()) == []:
            if isinstance(layer,nn.ReLU) or isinstance(layer,Relu_bound):
                name_ = name1 + name
                layer.register_forward_hook(get_activation(name_)) 
        else:
            name+=name1
            relu_hooks(layer,name)
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

    # build model
    model = build_model(args.model, n_classes,0.0).cuda()
    checkpoint = load_state_dict_from_file(args.init_from)
    model.load_state_dict(checkpoint) 
    # teacher_model  = build_model(args.teacher_model,n_classes,0.0).cuda()
    # checkpoint_teacher = load_state_dict_from_file(args.init_teacher_from)
    # teacher_model.load_state_dict(checkpoint_teacher)
    # for param in model.parameters():
    #     print(param.shape)
    # if args.init_teacher_bounds_neuron:
    #     temp=[]
    #     relu_hooks(teacher_model,name='')      
    #     for data, label in data_loader_dict['sub_train']:
    #         data = data.to('cuda')
    #         label = label.to('cuda')
    #         teacher_model = teacher_model.to('cuda')
    #         output = teacher_model(data)
    #         for key, val in activation.items():
    #                 print(key)
    #                 temp.append(key)
    #         break
    #     bounds={}
    #     checkpoint_bounds = load_state_dict_from_file("./pretrained_models/teachers/vgg16_bound_neuron_relu_fixed/checkpoint/checkpoint/checkpoint.pt")
    #     i=0
    #     for key,val in checkpoint_bounds.items():
    #         if "bounds_param" in key:
    #             bounds[temp[i]] = val
    #             i+=1
    #     teacher_model = replace_act_all(teacher_model,bounded_relu_fitact,bounds,bounds,None)
    #     print("the accuracy of teacher model")
    #     eval(teacher_model,data_loader_dict)
    # elif  args.init_teacher_bounds_layer:
    #     temp=[]
    #     relu_hooks(teacher_model,name='')      
    #     for data, label in data_loader_dict['sub_train']:
    #         data = data.to('cuda')
    #         label = label.to('cuda')
    #         teacher_model = teacher_model.to('cuda')
    #         output = teacher_model(data)
    #         for key, val in activation.items():
    #                 temp.append(key)
    #         break
    #     bounds={}
    #     i=0
    #     for key,val in activation.items():
    #        val = torch.load("./ftclip_teachers_bounds/{}_{}_{}_{}.pt".format(teacher_model.__class__.__name__,args.bounds_type,args.bitflip,key))
    #        bounds[temp[i]] = val
    #        i+=1
    #     teacher_model = replace_act_all(teacher_model,bounded_relu_zero,bounds,bounds,None)
    #     print("the accuracy of teacher model")
    #     eval(teacher_model,data_loader_dict)
        





    if isinstance(list(model.children())[-1],nn.ReLU):
        print("The model with ReLU in the last layer")
    else:
        print("The model without ReLU in the last layer")    
    # exit()    
    print(args.dataset,args.model,args.name_relu_bound,args.name_serach_bound,args.bounds_type,args.bitflip,args.iterations)
    # print(model)
    #load checkpoint
   
    
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
    ######################################### analyse the distribution of the values ####################################################
    # model.eval()
    # results = {}
    # values={}
    # values_01 ={}
    # values_hi1 ={}
    # relu_hooks(model,name='')      
    # for data, label in data_loader_dict['sub_train']:
    #     data = data.to('cuda')
    #     label = label.to('cuda')
    #     model = model.to('cuda')
    #     output = model(data)
    #     for key,val in activation.items():
    #         values[key] = []
    #         results[key] = []
    #         values_01[key]=[]
    #         values_hi1[key]=[]
    #     break       
    # for data, label in data_loader_dict['sub_train']:
    #     data = data.to('cuda')
    #     label = label.to('cuda')
    #     model = model.to('cuda')
    #     output = model(data)
    #     for key,val in activation.items():
    #         results[key].append(val)
    # ################## for neuron ######################    
    # # for i,(key,val) in enumerate(results.items()):
    # #     values[key].append(torch.max(torch.cat(results[key]).flatten(start_dim=1),dim=0))

    # #     print(values[key][0].values.shape)
    # # fig, axs = plt.subplots(len(values),1,figsize=(20,5))        
    # # for i,key in enumerate(values.keys()):
    # #     axs[i].hist(values[key][0].values.detach().cpu().numpy(),histtype='stepfilled', alpha=0.3, color = 'b')
    # # plt.savefig("dist_"+args.name_serach_bound)    
    # # ##################################################  
    # del results    
    # for key in values.keys():
    #     mask = torch.where(values[key][0]<=1.0)
    #     mask1 = torch.where(values[key][0]>1.0)
    #     values_01[key] =  values[key][0][mask]  
    #     values_hi1[key] = values[key][0][mask1]  
    # del values 
    # del mask
    # del mask1   
    # model = replace_act(model,args.name_relu_bound,args.name_serach_bound,data_loader_dict,args.bounds_type,args.bitflip)
    # model.eval()
    # results_b = {}
    # values_b={}
    # values_01_b ={}
    # values_hi1_b ={}
    # relu_hooks(model,name='')      
    # for data, label in data_loader_dict['sub_train']:
    #     data = data.to('cuda')
    #     label = label.to('cuda')
    #     model = model.to('cuda')
    #     output = model(data)
    #     for key,val in activation.items():
    #         values_b[key] = []
    #         results_b[key] = []
    #         values_01_b[key]=[]
    #         values_hi1_b[key]=[]
    #     break       
    # for data, label in data_loader_dict['sub_train']:
    #     data = data.to('cuda')
    #     label = label.to('cuda')
    #     model = model.to('cuda')
    #     output = model(data)
    #     for key,val in activation.items():
    #         results_b[key].append(val)
        
    # for i,(key,val) in enumerate(results_b.items()):
    #     values_b[key].append(torch.cat(results_b[key]).flatten())
    # del results_b    
    # for key in values_b.keys():
    #     mask = torch.where(values_b[key][0]<=1.0)
    #     mask1 = torch.where(values_b[key][0]>1.0)
    #     values_01_b[key] =  values_b[key][0][mask]  
    #     values_hi1_b[key] = values_b[key][0][mask1]  
    # del values_b  
    # del mask
    # del mask1  
    # fault_rate = 3e-5
    # inputs, classes = next(iter(data_loader_dict['sub_train'])) 
    # pfi_model = FaultInjection(model, 
    #                         inputs.shape[0],
    #                         input_shape=[inputs.shape[1],inputs.shape[2],inputs.shape[3]],
    #                         layer_types=[torch.nn.Conv2d, torch.nn.Linear ,Relu_bound],
    #                         total_bits= args.n_word,
    #                         n_frac = args.n_frac, 
    #                         n_int = args.n_int, 
    #                         use_cuda=True,
    #                         )
    # if args.bitflip=='float':
    #     corrupted_model = multi_weight_inj_float(pfi_model,fault_rate)
    # elif args.bitflip=='fixed':    
    #     corrupted_model = multi_weight_inj_fixed(pfi_model,fault_rate)
    # elif args.bitflip =="int":
    #     corrupted_model = multi_weight_inj_int (pfi_model,fault_rate)
    # corrupted_model.eval()
    # results_f={}
    # values_f ={}
    # values_01_f ={}
    # values_hi1_f ={}
    # relu_hooks(corrupted_model,name='')      
    # for data, label in data_loader_dict['sub_train']:
    #     data = data.to('cuda')
    #     label = label.to('cuda')
    #     corrupted_model = corrupted_model.to('cuda')
    #     output = corrupted_model(data)
    #     for key,val in activation.items():
    #         results_f[key] = []
    #         values_f[key]=[]
    #         values_01_f [key] =[]
    #         values_hi1_f[key] =[]
    #     break 
      
    # for data, label in data_loader_dict['sub_train']:
    #     data = data.to('cuda')
    #     label = label.to('cuda')
    #     corrupted_model = corrupted_model.to('cuda')
    #     output = corrupted_model(data)
    #     for key,val in activation.items():
    #         results_f[key].append(val)
        
    # for i,(key,val) in enumerate(results_f.items()):
    #     values_f[key].append(torch.cat(results_f[key]).flatten())    
    # del results_f        
    # for key in values_f.keys():
    #     mask = torch.where(values_f[key][0]<=1.0)
    #     mask1 = torch.where(values_f[key][0]>1.0)
    #     values_01_f[key] =  values_f[key][0][mask]  
    #     values_hi1_f[key] = values_f[key][0][mask1]  
    # del values_f   
    # del mask
    # del mask1 
    # fig, axs = plt.subplots(len(values_01_f),1,figsize=(20,5))    
    # axs = axs.ravel()
    # for i,key in enumerate(values_01_f.keys()):
    #     # axs[i].hist(values_01[key].detach().cpu().numpy(),histtype='stepfilled', alpha=0.3,color = 'g')
    #     axs[i].hist(values_01_b[key].detach().cpu().numpy(),histtype='stepfilled', alpha=0.3, color = 'b')
    #     axs[i].hist(values_01_f[key].detach().cpu().numpy(),histtype='stepfilled', alpha=0.3, color = 'r')
    # plt.savefig("dist01_"+args.name_serach_bound)   
    # fig1, axs1 = plt.subplots(len(values_01_f),1,figsize=(20,5))    
    # axs1 = axs1.ravel()
    # for i,key in enumerate(values_01_f.keys()):
    #     # axs1[i].hist(values_hi1[key].detach().cpu().numpy(),histtype='stepfilled', alpha=0.3 ,color = 'g')
    #     axs1[i].hist(values_hi1_b[key].detach().cpu().numpy(),histtype='stepfilled', alpha=0.3 , color = 'b')
    #     axs1[i].hist(values_hi1_f[key].detach().cpu().numpy(),histtype='stepfilled', alpha=0.3 , color = 'r')
    # plt.savefig("disthi1_"+args.name_serach_bound)    
          
         
    # exit()
    ###########################################################################################################################################################       
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













