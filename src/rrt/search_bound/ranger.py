import sys
import torch.nn as nn
import torch
import sys;
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
             
def Ranger_bounds(model:nn.Module, train_loader, device="cuda", bound_type='layer',bitflip = 'float'):
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
    
    if bound_type =="layer":
        for key, val in results.items():
            results[key] = torch.max(val)  
            tresh[key] = torch.min(tresh[key]) 
            
       
    return results,tresh,None


   

