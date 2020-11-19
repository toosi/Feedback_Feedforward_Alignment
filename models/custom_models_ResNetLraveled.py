import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from modules import customized_modules_simple as customized_modules
# from modules import customized_modules_layerwise as customized_modules
# from modules import BiHebb_modules as customized_modules

# print('******* in custom_models',torch.__version__,'*******')
# print(torch.__file__)

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

ReLU = nn.ReLU(inplace=True) #customized_modules.ReLUGrad
Conv2d = customized_modules.AsymmetricFeedbackConv2d
ConvTranspose2d = customized_modules.AsymmetricFeedbackConvTranspose2d
Linear = customized_modules.LinearModule

#---------------------------- No maxpool NoFC BN doesnot track ResNetL10 ------------------------

class AsymResLNet10F(nn.Module):
    def __init__(self, image_channels=3, n_classes = 10, kernel_size=7, stride=2 ,base_channels=64, algorithm='FA', normalization='BatchNorm2d', normalization_affine=False): 
        super(AsymResLNet10F, self).__init__()
        self.n_classes = n_classes 
        self.base_channels = base_channels
        self.conv1 = Conv2d(image_channels, self.base_channels, kernel_size=kernel_size, stride=stride, padding=3, bias=False, algorithm=algorithm)
        self.bn1 = getattr(nn, normalization)(self.base_channels, affine=normalization_affine, momentum=0.1, track_running_stats=False)
        
        self.relu = ReLU

        # layer 1
        self.conv11 = Conv2d(self.base_channels, self.base_channels, kernel_size=3, stride=1, groups=1, padding=1, algorithm=algorithm)
        self.bn11 = getattr(nn, normalization)(self.base_channels, affine=normalization_affine ,momentum=0.1, track_running_stats=False)
        self.relu = ReLU
        self.conv12 = Conv2d(self.base_channels, self.base_channels, kernel_size=3,stride=1, padding=1, algorithm=algorithm)
        self.bn12 = getattr(nn, normalization)(self.base_channels, affine=normalization_affine, momentum=0.1,track_running_stats=False)

        self.conv21 = Conv2d(self.base_channels, self.base_channels, kernel_size=3,stride=1, padding=1, algorithm=algorithm)
        self.bn21 = getattr(nn, normalization)(self.base_channels, affine=normalization_affine, momentum=0.1,track_running_stats=False)
        self.relu = ReLU
        self.conv22 = Conv2d(self.base_channels, self.base_channels*2, kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn22 = getattr(nn, normalization)(self.base_channels*2, affine=normalization_affine, momentum=0.1, track_running_stats=False)
        self.downsample1 =  Conv2d(self.base_channels, self.base_channels*2,kernel_size=1, stride=1, padding=0, algorithm=algorithm)
        self.bn23 = getattr(nn, normalization)(self.base_channels*2, affine=normalization_affine,momentum=0.1, track_running_stats=False)


        # layer 2
        self.conv31 = Conv2d(self.base_channels*2, self.base_channels*2, kernel_size=3, stride=2, groups=1, padding=1, algorithm=algorithm)
        self.bn31 = getattr(nn, normalization)(self.base_channels*2, affine=normalization_affine,momentum=0.1, track_running_stats=False)
        self.relu = ReLU
        self.conv32 = Conv2d(self.base_channels*2, self.base_channels*2, kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn32 = getattr(nn, normalization)(self.base_channels*2, affine=normalization_affine,momentum=0.1, track_running_stats=False)

        self.conv41 = Conv2d(self.base_channels*2, self.base_channels*2,  kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn41 = getattr(nn, normalization)(self.base_channels*2, affine=normalization_affine,momentum=0.1, track_running_stats=False)
        self.relu = ReLU
        self.conv42 = Conv2d(self.base_channels*2, self.n_classes, kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn42 = getattr(nn, normalization)(self.n_classes, affine=normalization_affine, momentum=0.1, track_running_stats=False)
        self.downsample2 =  Conv2d(self.base_channels*2, self.n_classes,kernel_size=1, stride=2, padding=0,  algorithm=algorithm, )
        # self.bn43 = nn.BatchNorm2d(self.base_channels*4,momentum=0.1, track_running_stats=False)



        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(self.base_channels*16, 1000)

    def forward(self, x):
        
        x = self.conv1(x)
        
        x = self.relu(self.bn1(x)) #self.relu(self.bn1(x))

        # layer 1
        identity = x
        
        x = self.conv11(x)
        x = self.relu(self.bn11(x))
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu(x)

        x = self.conv21(x)
        x = self.relu(self.bn21(x))
        x = self.conv22(x)
        x = self.bn22(x)
        x += self.bn23(self.downsample1(identity)) 
        x = self.relu(x)

        # layer 2
        identity = x
        
        x = self.conv31(x)
        x = self.relu(self.bn31(x))
        x = self.conv32(x)
        x = self.bn32(x)
        x = self.relu(x)

        x = self.conv41(x)
        x = self.relu(self.bn41(x))
        x = self.conv42(x)
        x = self.bn42(x)
        # x += self.bn43(self.downsample2(identity)) 
        x += self.downsample2(identity)
        latent = self.relu(x)
        

        x = self.avgpool(latent)
        pooled = torch.flatten(x, 1)
        # x = self.fc(x)
        
        dum = pooled

        return latent, pooled




class AsymResLNet10B(nn.Module):
    def __init__(self, image_channels=3, n_classes=10, algorithm='FA', kernel_size=7,stride=2 , base_channels=64, normalization='BatchNorm2d', normalization_affine=False):
        super(AsymResLNet10B, self).__init__()
        self.base_channels = base_channels
        self.n_classes = n_classes
        
        
         # layer 2
        # self.bn43 = nn.BatchNorm2d(self.base_channels*4,momentum=0.1, track_running_stats=False)
        self.upsample2 = ConvTranspose2d(self.n_classes, self.base_channels*2,kernel_size=1, stride=2, padding=0, output_padding=1, algorithm=algorithm,)
        self.bn42 = getattr(nn, normalization)(self.n_classes, affine=normalization_affine,momentum=0.1, track_running_stats=False)
        self.conv42 = ConvTranspose2d(self.n_classes, self.base_channels*2, kernel_size=3, stride=1, padding=1, output_padding=0, algorithm=algorithm,)
        self.relu = ReLU
        self.bn41 = getattr(nn, normalization)(self.base_channels*2, affine=normalization_affine,momentum=0.1, track_running_stats=False)
        self.conv41 = ConvTranspose2d(self.base_channels*2, self.base_channels*2, stride=1, kernel_size=3, padding=1, output_padding=0, algorithm=algorithm, )
        self.bn32 = getattr(nn, normalization)(self.base_channels*2, affine=normalization_affine, momentum=0.1, track_running_stats=False)
        self.conv32 = ConvTranspose2d(self.base_channels*2, self.base_channels*2, kernel_size=3,stride=1,  padding=1, output_padding=0, algorithm=algorithm, )
        
        
        self.bn31 = getattr(nn, normalization)(self.base_channels*2, affine=normalization_affine,momentum=0.1, track_running_stats=False)
        self.conv31 = ConvTranspose2d(self.base_channels*2, self.base_channels*2, kernel_size=3, stride=2, groups=1, padding=1, output_padding=1, algorithm=algorithm, ) #output_padding=1
 
        # layer 1
        self.bn23 = getattr(nn, normalization)(self.base_channels*2, affine=normalization_affine,momentum=0.1, track_running_stats=False)
        self.upsample1 =  ConvTranspose2d(self.base_channels*2, self.base_channels,kernel_size=1, stride=1, padding=0,output_padding=0, algorithm=algorithm,)
        self.bn22 = getattr(nn, normalization)(self.base_channels*2, affine=normalization_affine,momentum=0.1, track_running_stats=False)
        self.conv22 = ConvTranspose2d(self.base_channels*2, self.base_channels, kernel_size=3, stride=1, padding=1, output_padding=0, algorithm=algorithm, )
        self.relu = ReLU
        self.bn21 = getattr(nn, normalization)(self.base_channels, affine=normalization_affine, momentum=0.1,track_running_stats=False)
        self.conv21 = ConvTranspose2d(self.base_channels, self.base_channels, stride=1, kernel_size=3, padding=1, output_padding=0, algorithm=algorithm, )
        self.bn12 = getattr(nn, normalization)(self.base_channels, affine=normalization_affine, momentum=0.1,track_running_stats=False)

        self.conv12 = ConvTranspose2d(self.base_channels, self.base_channels, kernel_size=3,stride=1,  padding=1, output_padding=0, algorithm=algorithm,)
        self.relu = ReLU
        self.bn11 = getattr(nn, normalization)(self.base_channels, affine=normalization_affine, momentum=0.1,track_running_stats=False)
        self.conv11 = ConvTranspose2d(self.base_channels, self.base_channels, kernel_size=3, stride=1, groups=1, padding=1, output_padding=0, algorithm=algorithm, )


        self.relu = ReLU
        self.bn1 = getattr(nn, normalization)(self.base_channels, affine=normalization_affine, momentum=0.1,track_running_stats=False)
        self.conv1 = ConvTranspose2d(self.base_channels, image_channels , kernel_size=kernel_size, stride=stride, padding=2, bias=False, output_padding=1, algorithm=algorithm, ) #output_padding=1
        

    def forward(self, x, s=None):

        
        # layer 2 
        identity = x 
        x = self.bn42(x)
        
        x = self.conv42(x)
        x = self.relu(self.bn41(x))
        x = self.conv41(x)
        x = self.relu(x)
        x = self.bn32(x)
        x = self.conv32(x)
        x = self.relu(self.bn31(x))
        x = self.conv31(x)
        # x += self.upsample2(self.bn43(identity))
        x += self.upsample2(identity)
        x = self.relu(x)
        


        # layer 1
        identity = x 
        x = self.bn22(x)
        x = self.conv22(x)
        x = self.relu(self.bn21(x))
        x = self.conv21(x)

        x = self.relu(x)
        x = self.bn12(x)
        x = self.conv12(x)
        x = self.relu(self.bn11(x))
        x = self.conv11(x)
        x += self.upsample1(self.bn23(identity)) 
        x = self.relu(x)
        

        preconv1 = self.relu(self.bn1(x))
        x = self.conv1(preconv1)


        return preconv1, x




class AsymLNet5F(nn.Module):
    def __init__(self, image_channels=3, n_classes = 10, kernel_size=7, stride=2 ,base_channels=64, algorithm='FA', normalization='BatchNorm2d'): 
        super(AsymLNet5F, self).__init__()
        self.n_classes = n_classes 
        self.base_channels = base_channels
        self.conv1 = Conv2d(image_channels, self.base_channels, kernel_size=kernel_size, stride=stride, padding=3, bias=False, algorithm=algorithm)
        self.bn1 = getattr(nn, normalization)(self.base_channels, momentum=0.1, track_running_stats=False)
        self.relu = ReLU

        # # layer 1
        # self.conv11 = Conv2d(self.base_channels, self.base_channels, kernel_size=3, stride=1, groups=1, padding=1, algorithm=algorithm)
        # self.bn11 = getattr(nn, normalization)(self.base_channels,momentum=0.1, track_running_stats=False)
        # self.relu = ReLU
        # self.conv12 = Conv2d(self.base_channels, self.base_channels, kernel_size=3,stride=1, padding=1, algorithm=algorithm)
        # self.bn12 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False)

        # self.conv21 = Conv2d(self.base_channels, self.base_channels, kernel_size=3,stride=1, padding=1, algorithm=algorithm)
        # self.bn21 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False)
        # self.relu = ReLU
        # self.conv22 = Conv2d(self.base_channels, self.base_channels*2, kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        # self.bn22 = getattr(nn, normalization)(self.base_channels*2, momentum=0.1, track_running_stats=False)
        # self.downsample1 =  Conv2d(self.base_channels, self.base_channels*2,kernel_size=1, stride=1, padding=0, algorithm=algorithm)
        # self.bn23 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1, track_running_stats=False)


        # layer 2
        self.conv31 = Conv2d(self.base_channels, self.base_channels, kernel_size=3, stride=2, groups=1, padding=1, algorithm=algorithm)
        self.bn31 = getattr(nn, normalization)(self.base_channels,momentum=0.1, track_running_stats=False)
        self.relu = ReLU
        self.conv32 = Conv2d(self.base_channels, self.base_channels, kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn32 = getattr(nn, normalization)(self.base_channels,momentum=0.1, track_running_stats=False)

        self.conv41 = Conv2d(self.base_channels, self.base_channels,  kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn41 = getattr(nn, normalization)(self.base_channels,momentum=0.1, track_running_stats=False)
        self.relu = ReLU
        self.conv42 = Conv2d(self.base_channels, self.n_classes, kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn42 = getattr(nn, normalization)(self.n_classes, momentum=0.1, track_running_stats=False)
        # self.downsample2 =  Conv2d(self.base_channels, self.n_classes,kernel_size=1, stride=2, padding=0,  algorithm=algorithm)
        # self.bn43 = nn.BatchNorm2d(self.base_channels*4,momentum=0.1, track_running_stats=False)



        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(self.base_channels*16, 1000)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(self.bn1(x)) #self.relu(self.bn1(x))

        # # layer 1
        # identity = x
        
        # x = self.conv11(x)
        # x = self.relu(self.bn11(x))
        # x = self.conv12(x)
        # x = self.bn12(x)
        # x = self.relu(x)

        # x = self.conv21(x)
        # x = self.relu(self.bn21(x))
        # x = self.conv22(x)
        # x = self.bn22(x)
        # x += self.bn23(self.downsample1(identity)) 
        # x = self.relu(x)

        # layer 2
        # identity = x
        
        x = self.conv31(x)
        x = self.relu(self.bn31(x))
        x = self.conv32(x)
        x = self.bn32(x)
        x = self.relu(x)

        x = self.conv41(x)
        x = self.relu(self.bn41(x))
        x = self.conv42(x)
        x = self.bn42(x)
        # x += self.bn43(self.downsample2(identity)) 
        # x += self.downsample2(identity)
        latent = self.relu(x)
        

        x = self.avgpool(latent)
        pooled = torch.flatten(x, 1)
        # x = self.fc(x)
        
        dum = pooled

        return latent, pooled




class AsymLNet5B(nn.Module):
    def __init__(self, image_channels=3, n_classes=10, algorithm='FA', kernel_size=7,stride=2 , base_channels=64, normalization='BatchNorm2d'):
        super(AsymLNet5B, self).__init__()
        self.base_channels = base_channels
        self.n_classes = n_classes
        
        
         # layer 2
        # self.bn43 = nn.BatchNorm2d(self.base_channels*4,momentum=0.1, track_running_stats=False)
        self.upsample2 =  ConvTranspose2d(self.n_classes, self.base_channels,kernel_size=1, stride=2, padding=0, output_padding=1, algorithm=algorithm)
        self.bn42 = getattr(nn, normalization)(self.n_classes,momentum=0.1, track_running_stats=False)
        self.conv42 = ConvTranspose2d(self.n_classes, self.base_channels, kernel_size=3,stride=1,  padding=1, output_padding=0, algorithm=algorithm)
        self.relu = ReLU
        self.bn41 = getattr(nn, normalization)(self.base_channels,momentum=0.1, track_running_stats=False)
        self.conv41 = ConvTranspose2d(self.base_channels, self.base_channels, stride=1, kernel_size=3, padding=1, output_padding=0, algorithm=algorithm)
        self.bn32 = getattr(nn, normalization)(self.base_channels*2, momentum=0.1, track_running_stats=False)
        self.conv32 = ConvTranspose2d(self.base_channels, self.base_channels, kernel_size=3,stride=1,  padding=1, output_padding=0, algorithm=algorithm)
        self.relu = ReLU
        self.bn31 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1, track_running_stats=False)
        self.conv31 = ConvTranspose2d(self.base_channels, self.base_channels, kernel_size=3, stride=2, groups=1, padding=1, output_padding=1, algorithm=algorithm) #output_padding=1
 
        # # layer 1
        # self.bn23 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1, track_running_stats=False)
        # self.upsample1 =  ConvTranspose2d(self.base_channels*2, self.base_channels,kernel_size=1, stride=1, padding=0,output_padding=0, algorithm=algorithm)
        # self.bn22 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1, track_running_stats=False)
        # self.conv22 = ConvTranspose2d(self.base_channels*2, self.base_channels, kernel_size=3, stride=1, padding=1, output_padding=0, algorithm=algorithm)
        # self.relu = ReLU
        # self.bn21 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False)
        # self.conv21 = ConvTranspose2d(self.base_channels, self.base_channels, stride=1, kernel_size=3, padding=1, output_padding=0, algorithm=algorithm)
        # self.bn12 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False)

        # self.conv12 = ConvTranspose2d(self.base_channels, self.base_channels, kernel_size=3,stride=1,  padding=1, output_padding=0, algorithm=algorithm)
        # self.relu = ReLU
        # self.bn11 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False)
        # self.conv11 = ConvTranspose2d(self.base_channels, self.base_channels, kernel_size=3, stride=1, groups=1, padding=1, output_padding=0, algorithm=algorithm)


        self.relu = ReLU
        self.bn1 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False)
        self.conv1 = ConvTranspose2d(self.base_channels, image_channels , kernel_size=kernel_size, stride=stride, padding=2, bias=False, output_padding=1, algorithm=algorithm) #output_padding=1
        



    def forward(self, x, s=None):

        
        # layer 2 
        identity = x 
        x2 = x
        x = self.bn42(x)
        x = self.conv42(x)
        x = self.relu(self.bn41(x))
        x = self.conv41(x)
        x = self.relu(x)
        x = self.bn32(x)
        x = self.conv32(x)
        x = self.relu(self.bn31(x))
        x = self.conv31(x)
        # x += self.upsample2(self.bn43(identity))
        x += self.upsample2(identity)
        x = self.relu(x)
        


        # layer 1
        identity = x 
        x1 = x
        x = self.bn22(x)
        x = self.conv22(x)
        x = self.relu(self.bn21(x))
        x = self.conv21(x)

        x = self.relu(x)
        x = self.bn12(x)
        x = self.conv12(x)
        x = self.relu(self.bn11(x))
        x = self.conv11(x)
        x += self.upsample1(self.bn23(identity)) 
        x = self.relu(x)
        

        x = self.relu(self.bn1(x))
        x = self.conv1(x)


        return x



class AsymResLNet14F(nn.Module):
    def __init__(self, image_channels=3, n_classes = 10, kernel_size=7, stride=2 , base_channels=64, algorithm='FA', normalization='BatchNorm2d',normalization_affine=False): 
        super(AsymResLNet14F, self).__init__()
        self.n_classes = n_classes 
        self.base_channels = base_channels
        self.conv1 = Conv2d(image_channels, self.base_channels, kernel_size=kernel_size, stride=stride, padding=3, bias=False, algorithm=algorithm, )
        self.bn1 = getattr(nn, normalization)(self.base_channels, momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.relu = ReLU

        # layer 1
        self.conv11 = Conv2d(self.base_channels, self.base_channels, kernel_size=3, stride=1, groups=1, padding=1, algorithm=algorithm, )
        self.bn11 = getattr(nn, normalization)(self.base_channels,momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.relu = ReLU
        self.conv12 = Conv2d(self.base_channels, self.base_channels, kernel_size=3,stride=1, padding=1, algorithm=algorithm, )
        self.bn12 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False, affine=normalization_affine)

        self.conv21 = Conv2d(self.base_channels, self.base_channels, kernel_size=3,stride=1, padding=1, algorithm=algorithm, )
        self.bn21 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False, affine=normalization_affine)
        self.relu = ReLU
        self.conv22 = Conv2d(self.base_channels, self.base_channels*2, kernel_size=3, stride=1, padding=1, algorithm=algorithm,)
        self.bn22 = getattr(nn, normalization)(self.base_channels*2, momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.downsample1 =  Conv2d(self.base_channels, self.base_channels*2,kernel_size=1, stride=1, padding=0, algorithm=algorithm, )
        self.bn23 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1, track_running_stats=False, affine=normalization_affine)

        # layer 2
        self.conv31 = Conv2d(self.base_channels*2, self.base_channels*2, kernel_size=3, stride=1, groups=1, padding=1, algorithm=algorithm, )
        self.bn31 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.relu = ReLU
        self.conv32 = Conv2d(self.base_channels*2, self.base_channels*2, kernel_size=3,stride=1, padding=1, algorithm=algorithm,  )
        self.bn32 = getattr(nn, normalization)(self.base_channels*2, momentum=0.1,track_running_stats=False, affine=normalization_affine)

        self.conv41 = Conv2d(self.base_channels*2, self.base_channels*2, kernel_size=3,stride=1, padding=1, algorithm=algorithm,  )
        self.bn41 = getattr(nn, normalization)(self.base_channels*2, momentum=0.1,track_running_stats=False, affine=normalization_affine)
        self.relu = ReLU
        self.conv42 = Conv2d(self.base_channels*2, self.base_channels*4, kernel_size=3, stride=2, padding=1, algorithm=algorithm,  )
        self.bn42 = getattr(nn, normalization)(self.base_channels*4, momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.downsample2 =  Conv2d(self.base_channels*2, self.base_channels*4,kernel_size=1, stride=2, padding=0, algorithm=algorithm,  )
        self.bn43 = getattr(nn, normalization)(self.base_channels*4,momentum=0.1, track_running_stats=False, affine=normalization_affine)


        # layer 3
        self.conv51 = Conv2d(self.base_channels*4, self.base_channels*4, kernel_size=3, stride=1, groups=1, padding=1, algorithm=algorithm,  )
        self.bn51 = getattr(nn, normalization)(self.base_channels*4,momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.relu = ReLU
        self.conv52 = Conv2d(self.base_channels*4, self.base_channels*4, kernel_size=3, stride=1, padding=1, algorithm=algorithm,  )
        self.bn52 = getattr(nn, normalization)(self.base_channels*4,momentum=0.1, track_running_stats=False, affine=normalization_affine)

        self.conv61 = Conv2d(self.base_channels*4, self.base_channels*4,  kernel_size=3, stride=1, padding=1, algorithm=algorithm,  )
        self.bn61 = getattr(nn, normalization)(self.base_channels*4,momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.relu = ReLU
        self.conv62 = Conv2d(self.base_channels*4, self.n_classes, kernel_size=3, stride=2, padding=1, algorithm=algorithm,  )
        self.bn62 = getattr(nn, normalization)(self.n_classes, momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.downsample3 =  Conv2d(self.base_channels*4, self.n_classes, kernel_size=1, stride=2, padding=0,  algorithm=algorithm,  )
        # self.bn43 = nn.BatchNorm2d(self.base_channels*4,momentum=0.1, track_running_stats=False)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(self.base_channels*16, 1000)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(self.bn1(x)) #self.relu(self.bn1(x))

        # layer 1
        identity = x
        
        x = self.conv11(x)
        x = self.relu(self.bn11(x))
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu(x)

        x = self.conv21(x)
        x = self.relu(self.bn21(x))
        x = self.conv22(x)
        x = self.bn22(x)
        x += self.bn23(self.downsample1(identity)) 
        x = self.relu(x)

        # layer 2
        identity = x

        x = self.conv31(x)
        x = self.relu(self.bn31(x))
        x = self.conv32(x)
        x = self.bn32(x)
        x = self.relu(x)

        x = self.conv41(x)
        x = self.relu(self.bn41(x))
        x = self.conv42(x)
        x = self.bn42(x)
        # x += self.bn43(self.downsample2(identity)) 
        x += self.downsample2(identity)
        x = self.relu(x)

        # layer 3
        identity = x

        x = self.conv51(x)
        x = self.relu(self.bn51(x))
        x = self.conv52(x)
        x = self.bn52(x)
        x = self.relu(x)

        x = self.conv61(x)
        x = self.relu(self.bn61(x))
        x = self.conv62(x)
        x = self.bn62(x)
        # x += self.bn63(self.downsample3(identity)) 
        x += self.downsample3(identity)
        latent = self.relu(x)
        

        x = self.avgpool(latent)
        pooled = torch.flatten(x, 1)
        # x = self.fc(x)
                

        return latent, pooled




class AsymResLNet14B(nn.Module):
    def __init__(self, image_channels=3, n_classes=10, algorithm='FA', kernel_size=7, stride=2 ,base_channels=64, normalization='BatchNorm2d',normalization_affine=False ):
        super(AsymResLNet14B, self).__init__()
        self.base_channels = base_channels
        self.n_classes = n_classes
        
        
         # layer 3
        # self.bn43 = nn.BatchNorm2d(self.base_channels*4,momentum=0.1, track_running_stats=False)
        self.upsample3 =  ConvTranspose2d(self.n_classes, self.base_channels*4,kernel_size=1, stride=2, padding=0, output_padding=0, algorithm=algorithm,  )
        self.bn62 = getattr(nn, normalization)(self.n_classes,momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.conv62 = ConvTranspose2d(self.n_classes, self.base_channels*4, kernel_size=3,stride=2,  padding=1, output_padding=0, algorithm=algorithm,  )
        self.relu = ReLU
        self.bn61 = getattr(nn, normalization)(self.base_channels*4,momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.conv61 = ConvTranspose2d(self.base_channels*4, self.base_channels*4, stride=1, kernel_size=3, padding=1, output_padding=0, algorithm=algorithm,  )
        self.bn52 = getattr(nn, normalization)(self.base_channels*4, momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.conv52 = ConvTranspose2d(self.base_channels*4, self.base_channels*4, kernel_size=3,stride=1,  padding=1, output_padding=0, algorithm=algorithm,  )
        self.relu = ReLU
        self.bn51 = getattr(nn, normalization)(self.base_channels*4,momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.conv51 = ConvTranspose2d(self.base_channels*4, self.base_channels*4, kernel_size=3, stride=1, groups=1, padding=1, output_padding=0, algorithm=algorithm,  ) #output_padding=1


        # layer 2
        self.bn43 = getattr(nn, normalization)(self.base_channels*4,momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.upsample2 =  ConvTranspose2d(self.base_channels*4, self.base_channels*2,kernel_size=1, stride=2, padding=0,output_padding=0, algorithm=algorithm,  )
        self.bn42 = getattr(nn, normalization)(self.base_channels*4,momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.conv42 = ConvTranspose2d(self.base_channels*4, self.base_channels*2, kernel_size=3, stride=2, padding=1, output_padding=0, algorithm=algorithm,  )
        self.relu = ReLU
        self.bn41 = getattr(nn, normalization)(self.base_channels*2, momentum=0.1,track_running_stats=False, affine=normalization_affine)
        self.conv41 = ConvTranspose2d(self.base_channels*2, self.base_channels*2, stride=1, kernel_size=3, padding=1, output_padding=0, algorithm=algorithm,  )
        self.bn32 = getattr(nn, normalization)(self.base_channels*2, momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.conv32 = ConvTranspose2d(self.base_channels*2, self.base_channels*2, kernel_size=3,stride=1,  padding=1, output_padding=0, algorithm=algorithm,  )
        self.relu = ReLU
        self.bn31 = getattr(nn, normalization)(self.base_channels*2, momentum=0.1,track_running_stats=False, affine=normalization_affine)
        self.conv31 = ConvTranspose2d(self.base_channels*2, self.base_channels*2, kernel_size=3, stride=1, groups=1, padding=1, output_padding=0, algorithm=algorithm,  )


        # layer 1
        self.bn23 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.upsample1 =  ConvTranspose2d(self.base_channels*2, self.base_channels,kernel_size=1, stride=1, padding=0,output_padding=0, algorithm=algorithm,  )
        self.bn22 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1, track_running_stats=False, affine=normalization_affine)
        self.conv22 = ConvTranspose2d(self.base_channels*2, self.base_channels, kernel_size=3, stride=1, padding=1, output_padding=0, algorithm=algorithm,  )
        self.relu = ReLU
        self.bn21 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False, affine=normalization_affine)
        self.conv21 = ConvTranspose2d(self.base_channels, self.base_channels, stride=1, kernel_size=3, padding=1, output_padding=0, algorithm=algorithm,  )
        self.bn12 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False, affine=normalization_affine)

        self.conv12 = ConvTranspose2d(self.base_channels, self.base_channels, kernel_size=3,stride=1,  padding=1, output_padding=0, algorithm=algorithm,  )
        self.relu = ReLU
        self.bn11 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False, affine=normalization_affine)
        self.conv11 = ConvTranspose2d(self.base_channels, self.base_channels, kernel_size=3, stride=1, groups=1, padding=1, output_padding=0, algorithm=algorithm,  )


        self.relu = ReLU
        self.bn1 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False, affine=normalization_affine)
        self.conv1 = ConvTranspose2d(self.base_channels, image_channels , kernel_size=kernel_size, stride=stride, padding=2, bias=False, output_padding=0, algorithm=algorithm,  ) #output_padding=1
        



    def forward(self, x, s=None):

        
        # layer 3 
        identity = x 
        x = self.bn62(x)
        x = self.conv62(x)
        x = self.relu(self.bn61(x))
        x = self.conv61(x)
        x = self.relu(x)
        x = self.bn52(x)
        x = self.conv52(x)
        x = self.relu(self.bn51(x))
        x = self.conv51(x)
        # x += self.upsample2(self.bn43(identity))
        x += self.upsample3(identity)
        x = self.relu(x)
        

        # layer 2
        identity = x 
        x = self.bn42(x)
        x = self.conv42(x)
        x = self.relu(self.bn41(x))
        x = self.conv41(x)

        x = self.relu(x)
        x = self.bn32(x)
        x = self.conv32(x)
        x = self.relu(self.bn31(x))
        x = self.conv31(x)
        x += self.upsample2(self.bn43(identity)) 
        x = self.relu(x)



        # layer 1
        identity = x 
        x = self.bn22(x)
        x = self.conv22(x)
        x = self.relu(self.bn21(x))
        x = self.conv21(x)

        x = self.relu(x)
        x = self.bn12(x)
        x = self.conv12(x)
        x = self.relu(self.bn11(x))
        x = self.conv11(x)
        x += self.upsample1(self.bn23(identity)) 
        x = self.relu(x)
        

        preconv1 = self.relu(self.bn1(x))
        x = self.conv1(preconv1)


        return preconv1, x


#----------------- stochastic discrimination -------------------------
class AsymResLNetLimited10F(nn.Module):
    def __init__(self, image_channels=3, n_classes = 10, kernel_size=7, stride=2 ,base_channels=64, algorithm='FA', normalization='BatchNorm2d'): 
        super(AsymResLNetLimited10F, self).__init__()
        self.n_classes = n_classes 
        self.base_channels = base_channels
        self.conv1 = Conv2d(image_channels, self.base_channels, kernel_size=kernel_size, stride=stride, padding=3, bias=False, algorithm=algorithm)
        self.bn1 = getattr(nn, normalization)(self.base_channels, momentum=0.1, track_running_stats=False)
        self.relu = ReLU

        # layer 1
        self.conv11 = Conv2d(self.base_channels, self.base_channels, kernel_size=3, stride=1, groups=1, padding=1, algorithm=algorithm)
        self.bn11 = getattr(nn, normalization)(self.base_channels,momentum=0.1, track_running_stats=False)
        self.relu = ReLU
        self.conv12 = Conv2d(self.base_channels, self.base_channels, kernel_size=3,stride=1, padding=1, algorithm=algorithm)
        self.bn12 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False)

        self.conv21 = Conv2d(self.base_channels, self.base_channels, kernel_size=3,stride=1, padding=1, algorithm=algorithm)
        self.bn21 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False)
        self.relu = ReLU
        self.conv22 = Conv2d(self.base_channels, self.base_channels*2, kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn22 = getattr(nn, normalization)(self.base_channels*2, momentum=0.1, track_running_stats=False)
        self.downsample1 =  Conv2d(self.base_channels, self.base_channels*2,kernel_size=1, stride=1, padding=0, algorithm=algorithm)
        self.bn23 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1, track_running_stats=False)


        # layer 2
        self.conv31 = Conv2d(self.base_channels*2, self.base_channels*2, kernel_size=3, stride=2, groups=1, padding=1, algorithm=algorithm)
        self.bn31 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1, track_running_stats=False)
        self.relu = ReLU
        self.conv32 = Conv2d(self.base_channels*2, self.base_channels*2, kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn32 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1, track_running_stats=False)

        self.conv41 = Conv2d(self.base_channels*2, self.base_channels*2,  kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn41 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1, track_running_stats=False)
        self.relu = ReLU
        self.conv42 = Conv2d(self.base_channels*2, self.n_classes, kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn42 = getattr(nn, normalization)(self.n_classes, momentum=0.1, track_running_stats=False)
        self.downsample2 =  Conv2d(self.base_channels*2, self.n_classes,kernel_size=1, stride=2, padding=0,  algorithm=algorithm)
        # self.bn43 = nn.BatchNorm2d(self.base_channels*4,momentum=0.1, track_running_stats=False)



        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(self.base_channels*16, 1000)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(self.bn1(x)) #self.relu(self.bn1(x))

        # layer 1
        identity = x
        
        x = self.conv11(x)
        x = self.relu(self.bn11(x))
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu(x)

        x = self.conv21(x)
        x = self.relu(self.bn21(x))
        x = self.conv22(x)
        x = self.bn22(x)
        x += self.bn23(self.downsample1(identity)) 
        x = self.relu(x)

        # layer 2
        identity = x
        
        x = self.conv31(x)
        x = self.relu(self.bn31(x))
        x = self.conv32(x)
        x = self.bn32(x)
        x = self.relu(x)

        x = self.conv41(x)
        x = self.relu(self.bn41(x))
        x = self.conv42(x)
        x = self.bn42(x)
        # x += self.bn43(self.downsample2(identity)) 
        x += self.downsample2(identity)
        latent = self.relu(x)
        
        latent_size = latent.shape[2]

        x = self.avgpool(latent[:,:, :int(latent_size/2), :int(latent_size/2)])
        pooled = torch.flatten(x, 1)
        # x = self.fc(x)
        
        return latent, pooled



class AsymResLNetLimited14F(nn.Module):
    def __init__(self, image_channels=3, n_classes = 10, kernel_size=7, stride=2 , base_channels=64, algorithm='FA', normalization='BatchNorm2d'): 
        super(AsymResLNetLimited14F, self).__init__()
        self.n_classes = n_classes 
        self.base_channels = base_channels
        self.conv1 = Conv2d(image_channels, self.base_channels, kernel_size=kernel_size, stride=stride, padding=3, bias=False, algorithm=algorithm)
        self.bn1 = getattr(nn, normalization)(self.base_channels, momentum=0.1, track_running_stats=False)
        self.relu = ReLU

        # layer 1
        self.conv11 = Conv2d(self.base_channels, self.base_channels, kernel_size=3, stride=1, groups=1, padding=1, algorithm=algorithm)
        self.bn11 = getattr(nn, normalization)(self.base_channels,momentum=0.1, track_running_stats=False)
        self.relu = ReLU
        self.conv12 = Conv2d(self.base_channels, self.base_channels, kernel_size=3,stride=1, padding=1, algorithm=algorithm)
        self.bn12 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False)

        self.conv21 = Conv2d(self.base_channels, self.base_channels, kernel_size=3,stride=1, padding=1, algorithm=algorithm)
        self.bn21 = getattr(nn, normalization)(self.base_channels, momentum=0.1,track_running_stats=False)
        self.relu = ReLU
        self.conv22 = Conv2d(self.base_channels, self.base_channels*2, kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn22 = getattr(nn, normalization)(self.base_channels*2, momentum=0.1, track_running_stats=False)
        self.downsample1 =  Conv2d(self.base_channels, self.base_channels*2,kernel_size=1, stride=1, padding=0, algorithm=algorithm)
        self.bn23 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1, track_running_stats=False)

        # layer 2
        self.conv31 = Conv2d(self.base_channels*2, self.base_channels*2, kernel_size=3, stride=1, groups=1, padding=1, algorithm=algorithm)
        self.bn31 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1, track_running_stats=False)
        self.relu = ReLU
        self.conv32 = Conv2d(self.base_channels*2, self.base_channels*2, kernel_size=3,stride=1, padding=1, algorithm=algorithm)
        self.bn32 = getattr(nn, normalization)(self.base_channels*2, momentum=0.1,track_running_stats=False)

        self.conv41 = Conv2d(self.base_channels*2, self.base_channels*2, kernel_size=3,stride=1, padding=1, algorithm=algorithm)
        self.bn41 = getattr(nn, normalization)(self.base_channels*2, momentum=0.1,track_running_stats=False)
        self.relu = ReLU
        self.conv42 = Conv2d(self.base_channels*2, self.base_channels*4, kernel_size=3, stride=2, padding=1, algorithm=algorithm)
        self.bn42 = getattr(nn, normalization)(self.base_channels*4, momentum=0.1, track_running_stats=False)
        self.downsample2 =  Conv2d(self.base_channels*2, self.base_channels*4,kernel_size=1, stride=2, padding=0, algorithm=algorithm)
        self.bn43 = getattr(nn, normalization)(self.base_channels*4,momentum=0.1, track_running_stats=False)


        # layer 3
        self.conv51 = Conv2d(self.base_channels*4, self.base_channels*4, kernel_size=3, stride=1, groups=1, padding=1, algorithm=algorithm)
        self.bn51 = getattr(nn, normalization)(self.base_channels*4,momentum=0.1, track_running_stats=False)
        self.relu = ReLU
        self.conv52 = Conv2d(self.base_channels*4, self.base_channels*4, kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn52 = getattr(nn, normalization)(self.base_channels*4,momentum=0.1, track_running_stats=False)

        self.conv61 = Conv2d(self.base_channels*4, self.base_channels*4,  kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn61 = getattr(nn, normalization)(self.base_channels*4,momentum=0.1, track_running_stats=False)
        self.relu = ReLU
        self.conv62 = Conv2d(self.base_channels*4, self.n_classes, kernel_size=3, stride=2, padding=1, algorithm=algorithm)
        self.bn62 = getattr(nn, normalization)(self.n_classes, momentum=0.1, track_running_stats=False)
        self.downsample3 =  Conv2d(self.base_channels*4, self.n_classes, kernel_size=1, stride=2, padding=0,  algorithm=algorithm)
        # self.bn43 = nn.BatchNorm2d(self.base_channels*4,momentum=0.1, track_running_stats=False)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(self.base_channels*16, 1000)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(self.bn1(x)) #self.relu(self.bn1(x))

        # layer 1
        identity = x
        
        x = self.conv11(x)
        x = self.relu(self.bn11(x))
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu(x)

        x = self.conv21(x)
        x = self.relu(self.bn21(x))
        x = self.conv22(x)
        x = self.bn22(x)
        x += self.bn23(self.downsample1(identity)) 
        x = self.relu(x)

        # layer 2
        identity = x

        x = self.conv31(x)
        x = self.relu(self.bn31(x))
        x = self.conv32(x)
        x = self.bn32(x)
        x = self.relu(x)

        x = self.conv41(x)
        x = self.relu(self.bn41(x))
        x = self.conv42(x)
        x = self.bn42(x)
        # x += self.bn43(self.downsample2(identity)) 
        x += self.downsample2(identity)
        x = self.relu(x)

        # layer 3
        identity = x

        x = self.conv51(x)
        x = self.relu(self.bn51(x))
        x = self.conv52(x)
        x = self.bn52(x)
        x = self.relu(x)

        x = self.conv61(x)
        x = self.relu(self.bn61(x))
        x = self.conv62(x)
        x = self.bn62(x)
        # x += self.bn63(self.downsample3(identity)) 
        x += self.downsample3(identity)
        latent = self.relu(x)
        

        latent_size = latent.shape[2]

        x = self.avgpool(latent[:,:, :int(latent_size/2), :int(latent_size/2)])
        pooled = torch.flatten(x, 1)
        # x = self.fc(x)
                

        return latent, pooled