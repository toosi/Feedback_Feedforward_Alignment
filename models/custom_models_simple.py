import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
# from modules import customized_modules_simple as customized_modules
# from modules import customized_modules_layerwise as customized_modules
from modules import BiHebb_modules as customized_modules  # used for NeurIPS2020

Linear = customized_modules.Linear
# ReLU = customized_modules.ReLU
Conv2d = customized_modules.Conv2d

class FullyConnectedF(nn.Module):
    def __init__(self, hidden_layers, nonlinearfunc, input_length, algorithm):
        super(FullyConnectedF, self).__init__()
        self.fc_0 = Linear(input_length,hidden_layers[0],False, algorithm=algorithm)
        self.hidden_layers = hidden_layers
        self.nonlinearfunc = nonlinearfunc
        for l in np.arange(1, len(hidden_layers)):
            setattr(self,'fc_%d'%l, Linear(hidden_layers[l-1], hidden_layers[l],False,algorithm=algorithm))

    def forward(self,x):
        x = x.view(x.shape[0],-1)
        for l in range(len(self.hidden_layers)-1):
            x = getattr(F,self.nonlinearfunc)(getattr(self,'fc_%d'%l)(x))

        x = getattr(self,'fc_%d'%(len(self.hidden_layers)-1))(x)
        return x, x # just to satisfy the requirements

          

class FullyConnectedB(nn.Module):
    def __init__(self, hidden_layers, nonlinearfunc, input_length, algorithm):
        super(FullyConnectedB, self).__init__()
        self.hidden_layers = hidden_layers
        self.nonlinearfunc = nonlinearfunc
        for l in range(len(hidden_layers)-1):
            setattr(self,'fc_%d'%l, Linear(hidden_layers[::-1][l], hidden_layers[::-1][l+1],False,algorithm=algorithm))
        setattr(self,'fc_%d'%(len(hidden_layers)-1), Linear(hidden_layers[0], input_length,False,algorithm=algorithm))

    def forward(self,x):
        for l in range(len(self.hidden_layers)-1):
            x = getattr(F,self.nonlinearfunc)(getattr(self,'fc_%d'%l)(x))

        x = getattr(self,'fc_%d'%(len(self.hidden_layers)-1))(x)
        input_size = int(np.sqrt(x.shape[-1]))
        x = x.view(x.shape[0], 1, input_size, input_size)
        return x, x # just to satisfy the requirements (the first one shoudl be V1 layer)
    
    
class LeNetNoMaxPool(nn.Module):
    def __init__(self, algorithm):
        super(LeNetNoMaxPool, self).__init__()
        
        self.conv1 = Conv2d(1, 6, 5, bias=False, algorithm=algorithm)      
        self.conv2 = Conv2d(6, 6, 2,2, bias=False, algorithm=algorithm) 
        self.conv3 = Conv2d(6, 16, 5, bias=False, algorithm=algorithm) 
        self.conv4 = Conv2d(16, 16, 2,2, bias=False, algorithm=algorithm) 
        self.fc1   = Linear(16*5*5, 120, bias=False, algorithm=algorithm) 
        self.fc2   = Linear(120, 84, bias=False, algorithm=algorithm) 
        self.fc3   = Linear(84, 10, bias=False, algorithm=algorithm) 



    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))# out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))# out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out