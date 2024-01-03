from re import X
from turtle import forward
import torch
import torch.nn as nn
from torch.autograd import Function
from relu_bound.bound_relu import Relu_bound
class bounded_relu_fitact(nn.Module,Relu_bound):
    def __init__(self,bounds,tresh=None,alpha = None,k=-20):
        super().__init__()
        bounds_param={}
        param_name= "bounds_param"
        self.tresh = None
        bounds_param[param_name] = nn.Parameter(data=bounds.cuda(), requires_grad=True)  
        self.k = k 
        for name, param in bounds_param.items():
            self.register_parameter(name, param) 
        self.bounds =  self.__getattr__("bounds_param")   
    def forward(self,input):
        # input = torch.nan_to_num(input)
        output = input - input * torch.sigmoid(-self.k * (input-self.__getattr__("bounds_param")))
        # print(output)
        return torch.maximum(torch.tensor(0.0),output)   

if __name__=="__main__":
    bounds= torch.tensor(3.0,device="cuda",requires_grad=True)
    a = bounded_relu_fitact(bounds, k=-20 )
    for param in a.parameters():
        print(param)
    input = torch.tensor(3.1,device='cuda',requires_grad=True)
    b = a(input) 
    b.backward()
    print(a.__getattr__("bounds_param").grad)
    # loss_fn = torch.nn.MSELoss()
    # loss = loss_fn(input,b)
    # loss.backward()
    # print(loss)
    # for param in a.parameters():
    #     print(param.grad)
        