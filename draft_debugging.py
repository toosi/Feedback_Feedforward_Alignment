

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

from models import custom_models as custom_models
# print(torch.__version__)
# from modules import customized_modules

# ConvTranspose2d = customized_modules.AsymmetricFeedbackConvTranspose2d
# # Conv2d = customized_modules.AsymmetricFeedbackConv2d

arch = 'AsymResNet18BNoMaxP'#'asymresnet18'
args_model = {'algorithm':'FA'}
# model = getattr(custom_models, arch)(**args_model)
model = nn.Sequential(ConvTranspose2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False, output_padding=0)).cuda()
inputs = torch.zeros([1, 1024, 4, 4]).cuda()

output = model(inputs)

# import torch.nn as nn
# import torch

# device = torch.device("cuda:0")

# # This model doesn't throw the error
# # t = nn.Sequential(nn.Conv3d(3,128, (9,4,4), stride=(1,2,2), padding=(0,1,1)), nn.ReLU(True), nn.Conv3d(128,256, (1,4,4), stride=(1,2,2), padding=(0,1,1)), nn.ReLU(),nn.Conv3d(256,256, (1,4,4), stride=(1,2,2), padding=(0,1,1))).to(device)

# t = nn.Sequential(nn.Conv3d(3,256, (9,4,4), stride=(1,2,2), padding=(0,1,1)), nn.ReLU(True), nn.Conv3d(256,512, (1,4,4), stride=(1,2,2), padding=(0,1,1)), nn.ReLU(),nn.Conv3d(512,512, (1,4,4), stride=(1,2,2), padding=(0,1,1))).to(device)
# i = torch.ones([10, 3, 9, 64, 96]).to(device)
# o = t(i)
# criterion = nn.L1Loss()
# loss = criterion(o, torch.ones_like(o))
# loss.backward()
# print('done!')


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