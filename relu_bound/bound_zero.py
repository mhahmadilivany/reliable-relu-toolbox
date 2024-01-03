import torch
import torch.nn as nn
from torch.autograd import Function
from relu_bound.bound_relu import Relu_bound
class bounded_relu_zero(nn.Module,Relu_bound):
    '''
    Bound the relu activatoin and back the values to zero
    ------------------------------------
    bound : the bound for the activation
    --------------------------------------
    pytorch module with forward function
    '''
    def __init__(self, bounds,tresh = None,alpha=None):
        super().__init__()
        self.bounds = bounds
        self.tresh = None
    def forward(self, input):
        # input = torch.nan_to_num(input)
        output = torch.ones_like(input) * input
        output[torch.gt(input,self.bounds)] = torch.tensor(0.0)
        return torch.maximum(torch.tensor(0.0),output)  

if __name__=="__main__":
    a = bounded_relu_zero()
    input = torch.tensor([[1.0,2.0,0.5,3.5],[0.2,0.3,1.5,4.0]])
    print(a(input))     