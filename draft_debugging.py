

"""
to debug the architectures before using them
make sure you use inputs on cuda otherwise modules using
THNN can't handle them and will rase NotImplemented

"""
import os
import torch
# import torchvision
# import torchvision.models as torchmodels
import torch.nn as nn
# from PIL import Image
# model = torchmodels.resnet18(pretrained=True)
# import numpy as np
# from torchvision import transforms
# import socket
from utils import helper_functions
Hook = helper_functions.Hook
from models import custom_models_ResNetLraveled as custom_models
# print(torch.__version__)
# from modules import customized_modules

# ConvTranspose2d = customized_modules.AsymmetricFeedbackConvTranspose2d
# # Conv2d = customized_modules.AsymmetricFeedbackConv2d

# arch = 'AsymLNet5F'# 'AsymResNet18BNoMaxP'#'asymresnet18'
# args_model = {'algorithm':'FA', 'kernel_size':7, 'stride':1}

arch = 'AsymResLNet10F'# 'AsymResNet18BNoMaxP'#'asymresnet18'
args_model = {'algorithm':'FA', 'kernel_size':7, 'stride':2}

model = getattr(custom_models, arch)(**args_model).cuda()
# model = nn.Sequential(ConvTranspose2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False, output_padding=0)).cuda()
inputs = torch.zeros([6, 3, 32, 32]).cuda()



[print(layer[0]) for layer in list(model._modules.items())]

hookF = {}
[hookF.update({layer[0]:Hook(layer[1])}) for layer in list(model._modules.items())]

model._modules['conv1']
latents, output = model(inputs)
print('latent shape', latents.shape)

print('conv1', len(hookF['conv1'].input), len(hookF['conv1'].output))
print('conv1', hookF['conv1'].input[0].shape,  hookF['conv1'].output.shape)

# patch_size = 16
# patches = inputs.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).squeeze()
# print(patches.shape)
# latents, output = model(patches[:,0,1].squeeze())

# print(latents.shape)

        
# # latents = torch.zeros_like(modelF(images[:,:,0:patch_size,0:patch_size]))
# latent_size = model(inputs[:,:,0:patch_size,0:patch_size])[0].shape[2]
# Latents_spaital = torch.zeros_like(model(inputs[:,:,0:patch_size,0:patch_size])[0]).repeat(4*4, 1, 1, 1, 1)
# print('Latents_spaital.shape',Latents_spaital.shape)
# s = 0
# for r in range(4):
#     for c in range(4):
#         # print('model(patches[:,r,c].squeeze())[0].shape',model(patches[:,r,c].squeeze())[0].shape)
#         Latents_spaital[s] =  model(patches[:,r,c].squeeze())[0]
#         s += 1
# Latents_spaital = Latents_spaital.permute(1, 2, 0, 3, 4)
# Latents_spaital = Latents_spaital.reshape(256, 10, 16, 4, 4)
# pooled = torch.nn.functional.avg_pool3d(Latents_spaital, (16,4,4))
# print('pooled',pooled.shape)

# ------ Backward -----------
arch = 'AsymResLNet10B'# 'AsymResNet18BNoMaxP'#'asymresnet18'
args_model = {'algorithm':'FA', 'kernel_size':7, 'stride':2}

model = getattr(custom_models, arch)(**args_model).cuda()
# model = nn.Sequential(ConvTranspose2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False, output_padding=0)).cuda()
output = model(latents.detach())

print(output.shape)




# #%%
# import torch
# import torchvision
# import torchvision.models as torchmodels
# import torch.nn as nn
# from PIL import Image
# model = torchmodels.resnet18(pretrained=True)

# from torchvision import transforms

# # %%
# list_children_names = [list(model.named_children())[i][0] for i in range(len(list(model.named_children())))]
# layer_index = list_children_names.index('layer4')

# model_extract = nn.Sequential(*list(model.children())[:layer_index+1])
# model_extract.eval()
# # %%
# datadir = '/hdd6gig/Documents/Research/Data/imagenet/train/n01440764/'
# f = 'n01440764_18.JPEG'
# im = np.asarray(Image.open(datadir+f).convert("L"))

# def image_loader(transformer, image_name):
#     image = Image.open(image_name)
#     image = transformer(image).float()
#     image = torch.tensor(image, requires_grad=True)
#     image = image.unsqueeze(0)
#     return image

# data_transforms = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor()
# ])
# #%%
# features = model_extract(image_loader(data_transforms,datadir+f ))
# # %%