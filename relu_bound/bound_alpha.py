
import torch
import torch.nn as nn
from torch.autograd import Function
from relu_bound.bound_relu import Relu_bound
import torch.nn.functional as F
class bounded_relu_alpha_learn(Function):
    '''
    Autograd function with forward and backward to train alpha parameters for activatoin functoin 
    --------------------------------------------------------------------------------------------
    ctx: for save tensor for backward
    input : input data to the module
    bound : bound for activatoin 
    alpha  : alpha parameters for each neurons
    '''
    @staticmethod
    def forward(ctx, inputs, bounds,tresh, alpha):
       ctx.save_for_backward(inputs)
       ctx.bounds = bounds
       ctx.alpha = alpha
       ctx.tresh = tresh
       output = torch.ones_like(inputs) * inputs 
       index_g = torch.where(torch.gt(output,bounds))
       index_i = torch.where(torch.logical_and(torch.lt(output,bounds),torch.gt(output,torch.tensor(0.0))))
       output[index_g] = tresh[index_g[1:]]
       output[index_i] = alpha[index_i[1:]] * output[index_i]
       return torch.maximum(torch.tensor(0.0),output)
    @staticmethod
    def backward(ctx, grad_output):
        # print(grad_output.max())
        inputs, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_bounds = None
        grad_tresh = None
        grad_alpha = grad_output.clone()
        index_g = torch.where(torch.gt(inputs,ctx.bounds))
        index_i = torch.where(torch.logical_and(torch.lt(inputs,ctx.bounds),torch.gt(inputs,torch.tensor(0.0))))
        index_l = torch.where(torch.lt(inputs,torch.tensor(0.0)))
        grad_input[index_l] = torch.tensor(0.0)
        grad_input[index_g] = torch.tensor(0.0)
        grad_alpha[index_l] = torch.tensor(0.0)
        grad_alpha[index_g] = torch.tensor(0.0)
        grad_input[index_i] =  ctx.alpha[index_i[1:]] * grad_input[index_i]
        grad_alpha[index_i] =  grad_alpha[index_i] * inputs[index_i]
        # print(grad_input.max())
        # print(grad_alpha.max())
        return grad_input, grad_bounds,grad_tresh,grad_alpha


# class bounded_relu_alpha(nn.Module,Relu_bound): 
#     def __init__(self,bounds,tresh=None,alpha_param=None):
#         super().__init__()
#         self.bounds = bounds
#         self.tresh= tresh
#         alpha={}
#         param_name= "alpha"
#         if alpha_param ==None:
#             alpha[param_name] = nn.Parameter(data=torch.ones_like(bounds).cuda(), requires_grad=True)  
#         else:
#             alpha[param_name] = nn.Parameter(data=alpha_param.cuda(), requires_grad=True)    
#         for name, param in alpha.items():
#             self.register_parameter(name, param) 
#     def forward(self,x):
#         return bounded_relu_alpha_learn.apply(x,self.bounds,self.tresh, self.__getattr__("alpha")) 

class bounded_relu_alpha(nn.Module,Relu_bound):
    def __init__(self,bounds=None,tresh=None,alpha_param = None,k=-20):
        super().__init__()
        bounds_param={}
        param_name1= "bounds_param"
        self.tresh = tresh
        if tresh ==None:
            bounds_param[param_name1] = nn.Parameter(data=torch.zeros_like(bounds).cuda(), requires_grad=True) 
        else:
            bounds_param[param_name1] = nn.Parameter(data=bounds.cuda(), requires_grad=True) 
          
        self.k = k 
        for name, param in bounds_param.items():
            self.register_parameter(name, param)   
        self.bounds =  self.__getattr__("bounds_param")  
    def forward(self,input):
        # input = torch.nan_to_num(input)
        output =   input - input * torch.sigmoid(-self.k* (input-self.__getattr__("bounds_param"))) # + self.__getattr__("bounds_param")
        # print(output)
        return torch.maximum(torch.tensor(0.0),output)   

# class bounded_relu_alpha(nn.Module,Relu_bound):
#     def __init__(self,bounds=None,tresh=None,alpha_param = None,k=-20):
#         super().__init__()
#         bounds_param={}
#         param_name1= "bounds_param"
#         self.tresh = tresh
#         if tresh ==None:
#             bounds_param[param_name1] = nn.Parameter(data=torch.rand_like(bounds) * torch.tensor(10.0).cuda(), requires_grad=True) 
#         else:
#             bounds_param[param_name1] = nn.Parameter(data=bounds.cuda(), requires_grad=True) 
          
#         self.k = k 
#         for name, param in bounds_param.items():
#             self.register_parameter(name, param)   
#         self.bounds =  self.__getattr__("bounds_param")   
#     def forward(self,input):
#         # input = torch.nan_to_num(input)
#         output =(input - input * torch.tanh(- self.k* (input-self.__getattr__("bounds_param"))))
#         # print(output)
#         return torch.maximum(torch.tensor(0.0),output)   


# class bounded_relu_alpha(nn.Module,Relu_bound):
#     def __init__(self,bounds,tresh=None,alpha_param = None,k=-0.5):
#         super().__init__()
#         self.bounds = bounds
#         alpha={}
#         param_name= "alpha"
#         self.tresh = tresh
#         if alpha_param ==None:
#             alpha[param_name] = nn.Parameter(data=torch.ones_like(bounds)*torch.tensor(0.5).cuda(), requires_grad=True)  
#         else:
#             alpha[param_name] = nn.Parameter(data=alpha_param.cuda(), requires_grad=True) 
#         self.k = k 
#         for name, param in alpha.items():
#             self.register_parameter(name, param) 
#         self.alpha =  self.__getattr__("alpha")  
#     def forward(self,input):
#         output =self.__getattr__("alpha") * (input - input * torch.tanh(-self.k * (input-self.bounds)))
#         return torch.maximum(torch.tensor(0.0),output)   


if __name__=="__main__":
    input = torch.tensor([[1.0,2.0,0.5,3.5],[0.2,0.3,1.5,4.0]],device='cuda')
    a = bounded_relu_alpha(bounds= torch.tensor([[1.0,1.0,2,4],[0.0,1.0,2,3.0]],device="cuda") )
    for param in a.parameters():
        print(param)
    input = torch.tensor([[1.0,2.0,0.5,3.5],[0.2,0.3,1.5,4.0]],device='cuda')
    b = a(input) 
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(input,b)
    loss.backward()
    print(loss)
    for param in a.parameters():
        print(param.grad)
        