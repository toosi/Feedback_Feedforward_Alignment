import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
# from modules import customized_modules_simple as customized_modules
# from modules import customized_modules_layerwise as customized_modules
from modules import BiHebb_modules as customized_modules

Linear = customized_modules.LinearModule


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