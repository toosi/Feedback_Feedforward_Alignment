
# To test architectures before using them in our framework
# currently it contains a version of wideresnets and performing SL
# 
import copy
import numpy as np
from modules import customized_modules_layerwise as customized_modules
import torch.nn as nn

Conv2d = customized_modules.AsymmetricFeedbackConv2d #nn.Conv2d #
ConvTranspose2d = customized_modules.AsymmetricFeedbackConvTranspose2d #nn.ConvTranspose2d  #
Linear = customized_modules.LinearModule #nn.Linear #

#***************************************************************

def toggle_state_dict_normalize(state_dict):
    # this code copies the forward parameters to the backward parameters and vice versa
    # additionally it normalizes both the forward and backward path 

    new_state_dict = {}
    already_added = []

    for ik, k  in enumerate(state_dict.keys()):

        item = state_dict[k]

        if ('bn' not in k) and ('weight' in k) and ('fc' not in k):

            if k not in already_added:
                

                if 'feedback' not in k:
                    item_dual = state_dict[k+'_feedback']
                    if 'upsample' in k:
                        k = k.replace('up','down')
                    elif 'downsample' in k:
                        k = k.replace('down', 'up')

                    new_state_dict.update({k+'_feedback':item})
                    new_state_dict.update({k:item_dual})

                else:
                    item_dual = state_dict[k.split('_')[0]]
                    if 'upsample' in k:
                        k = k.replace('up','down')
                    elif 'downsample' in k:
                        k = k.replace('down', 'up')
                        
                    denom_item = 1#torch.sqrt(torch.abs(item)**2+torch.abs(item_dual)**2)
                    denom_item_dual = 1#torch.sqrt(torch.abs(item)**2+torch.abs(item_dual)**2)
                    
                    new_state_dict.update({k.split('_')[0]:item/denom_item })
                    new_state_dict.update({k:item_dual/denom_item_dual})

        elif ('fc'in k) and ('weight' in k):
            if 'feedback' in k:
                new_state_dict.update({k.strip('_feedback'): torch.transpose(item, 0, 1)})
            else:
                new_state_dict.update({k+'_feedback': torch.transpose(item, 0, 1)})
                
        else:
            new_state_dict.update({k:item})

        already_added.append(k)

    return new_state_dict



#*****************************
# modifications affine=False, track_running_stats=False for bn2 in BasicBlock
#track_running_stats=False for bn2 improves test by 10% on nn modules
#affine=False degrades test for bn2 by 2% on nn modules


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#---------------------------- No maxpool NoFC BN doesnot track ResNetL10 ------------------------

class ReLUB(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, criteria):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.

        Here, criteria is the input to the module ReLU in forward pass
        """
        ctx.save_for_backward(criteria)
        output = input.clone()
        output[criteria<0] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        criteria, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[criteria < 0] = 0
        return grad_input, None


class AsymResLNet10F(nn.Module):
    def __init__(self, image_channels=3, n_classes = 10, kernel_size=7, stride=2 ,base_channels=64, algorithm='FA', normalization='BatchNorm2d'): 
        super(AsymResLNet10F, self).__init__()
        self.n_classes = n_classes 
        self.base_channels = base_channels
        self.conv1 = Conv2d(image_channels, self.base_channels, kernel_size=kernel_size, stride=stride, padding=3, bias=False, algorithm=algorithm)
        self.bn1 = getattr(nn, normalization)(self.base_channels, affine=False, momentum=0.1, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

        # layer 1
        self.conv11 = Conv2d(self.base_channels, self.base_channels, kernel_size=3, stride=1, groups=1, padding=1, algorithm=algorithm)
        self.bn11 = getattr(nn, normalization)(self.base_channels, affine=False ,momentum=0.1, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv12 = Conv2d(self.base_channels, self.base_channels, kernel_size=3,stride=1, padding=1, algorithm=algorithm)
        self.bn12 = getattr(nn, normalization)(self.base_channels, affine=False, momentum=0.1,track_running_stats=False)

        self.conv21 = Conv2d(self.base_channels, self.base_channels, kernel_size=3,stride=1, padding=1, algorithm=algorithm)
        self.bn21 = getattr(nn, normalization)(self.base_channels, affine=False, momentum=0.1,track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv22 = Conv2d(self.base_channels, self.base_channels*2, kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn22 = getattr(nn, normalization)(self.base_channels*2, affine=False, momentum=0.1, track_running_stats=False)
        self.downsample1 =  Conv2d(self.base_channels, self.base_channels*2,kernel_size=1, stride=1, padding=0, algorithm=algorithm)
        self.bn23 = getattr(nn, normalization)(self.base_channels*2, affine=False,momentum=0.1, track_running_stats=False)


        # layer 2
        self.conv31 = Conv2d(self.base_channels*2, self.base_channels*2, kernel_size=3, stride=2, groups=1, padding=1, algorithm=algorithm)
        self.bn31 = getattr(nn, normalization)(self.base_channels*2, affine=False,momentum=0.1, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv32 = Conv2d(self.base_channels*2, self.base_channels*2, kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn32 = getattr(nn, normalization)(self.base_channels*2, affine=False,momentum=0.1, track_running_stats=False)

        self.conv41 = Conv2d(self.base_channels*2, self.base_channels*2,  kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn41 = getattr(nn, normalization)(self.base_channels*2, affine=False,momentum=0.1, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv42 = Conv2d(self.base_channels*2, self.n_classes, kernel_size=3, stride=1, padding=1, algorithm=algorithm)
        self.bn42 = getattr(nn, normalization)(self.n_classes, affine=False, momentum=0.1, track_running_stats=False)
        self.downsample2 =  Conv2d(self.base_channels*2, self.n_classes,kernel_size=1, stride=2, padding=0,  algorithm=algorithm, )
        # self.bn43 = nn.BatchNorm2d(self.base_channels*4,momentum=0.1, track_running_stats=False)



        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(self.base_channels*16, 1000)

    def forward(self, x):
        
        x = self.conv1(x)
        
        x = self.bn1(x)
        xrelu0 = x
        x = self.relu(x) #self.relu(self.bn1(x))
        

        # layer 1
        identity = x
        
        x = self.conv11(x)
        x = self.bn11(x)
        xrelu11 = x
        x = self.relu(x)
        
        x = self.conv12(x)
        x = self.bn12(x)
        xrelu12 = x
        x = self.relu(x)
        

        x = self.conv21(x)
        x = self.bn21(x)
        xrelu13 = x
        x = self.relu(x)
        
        x = self.conv22(x)
        x = self.bn22(x)
        x += self.bn23(self.downsample1(identity)) 
        xrelu14 = x
        x = self.relu(x)
        

        # layer 2
        identity = x
        
        x = self.conv31(x)
        x = self.bn31(x)
        xrelu21 = x
        x = self.relu(x)
        
        x = self.conv32(x)
        x = self.bn32(x)
        xrelu22 = x
        x = self.relu(x)
        

        x = self.conv41(x)
        x = self.bn41(x)
        xrelu23 = x
        x = self.relu(x)
        
        x = self.conv42(x)
        x = self.bn42(x)
        # x += self.bn43(self.downsample2(identity)) 
        x += self.downsample2(identity)
        xrelu24 = x
        latent = self.relu(x)

        x = self.avgpool(latent)
        pooled = torch.flatten(x, 1)
        # x = self.fc(x)
        
        dum = pooled
        list_relus = [xrelu0, xrelu11, xrelu12, xrelu13, xrelu14, xrelu21, xrelu22, xrelu23, xrelu24]

        return list_relus, latent, pooled


class AsymResLNet10B(nn.Module):
    def __init__(self, image_channels=3, n_classes=10, algorithm='FA', kernel_size=7,stride=2 , base_channels=64, normalization='BatchNorm2d'):
        super(AsymResLNet10B, self).__init__()
        self.base_channels = base_channels
        self.n_classes = n_classes
        
        
         # layer 2
        # self.bn43 = nn.BatchNorm2d(self.base_channels*4,momentum=0.1, track_running_stats=False)
        self.upsample2 =  ConvTranspose2d(self.n_classes, self.base_channels*2,kernel_size=1, stride=2, padding=0, output_padding=1, algorithm=algorithm, )
        self.bn42 = getattr(nn, normalization)(self.n_classes,momentum=0.1,affine=False, track_running_stats=False)
        self.conv42 = ConvTranspose2d(self.n_classes, self.base_channels*2, kernel_size=3, stride=1, padding=1, output_padding=0, algorithm=algorithm,   )
        self.relu = nn.ReLU(inplace=True)
        self.bn41 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1,affine=False, track_running_stats=False)
        self.conv41 = ConvTranspose2d(self.base_channels*2, self.base_channels*2, stride=1, kernel_size=3, padding=1, output_padding=0, algorithm=algorithm,   )
        self.bn32 = getattr(nn, normalization)(self.base_channels*2, momentum=0.1,affine=False, track_running_stats=False)
        self.conv32 = ConvTranspose2d(self.base_channels*2, self.base_channels*2, kernel_size=3,stride=1,  padding=1, output_padding=0, algorithm=algorithm,   )
        self.relu = nn.ReLU(inplace=True)
        self.bn31 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1,affine=False, track_running_stats=False)
        self.conv31 = ConvTranspose2d(self.base_channels*2, self.base_channels*2, kernel_size=3, stride=2, groups=1, padding=1, output_padding=1, algorithm=algorithm,   ) #output_padding=1
 
        # layer 1
        self.bn23 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1,affine=False, track_running_stats=False)
        self.upsample1 =  ConvTranspose2d(self.base_channels*2, self.base_channels,kernel_size=1, stride=1, padding=0,output_padding=0, algorithm=algorithm,   )
        self.bn22 = getattr(nn, normalization)(self.base_channels*2,momentum=0.1,affine=False, track_running_stats=False)
        self.conv22 = ConvTranspose2d(self.base_channels*2, self.base_channels, kernel_size=3, stride=1, padding=1, output_padding=0, algorithm=algorithm,   )
        self.relu = nn.ReLU(inplace=True)
        self.bn21 = getattr(nn, normalization)(self.base_channels, momentum=0.1,affine=False,track_running_stats=False)
        self.conv21 = ConvTranspose2d(self.base_channels, self.base_channels, stride=1, kernel_size=3, padding=1, output_padding=0, algorithm=algorithm,   )
        self.bn12 = getattr(nn, normalization)(self.base_channels, momentum=0.1,affine=False,track_running_stats=False)

        self.conv12 = ConvTranspose2d(self.base_channels, self.base_channels, kernel_size=3,stride=1,  padding=1, output_padding=0, algorithm=algorithm,   )
        self.relu = nn.ReLU(inplace=True)
        self.bn11 = getattr(nn, normalization)(self.base_channels, momentum=0.1,affine=False,track_running_stats=False)
        self.conv11 = ConvTranspose2d(self.base_channels, self.base_channels, kernel_size=3, stride=1, groups=1, padding=1, output_padding=0, algorithm=algorithm,   )


        self.relu = nn.ReLU(inplace=True)
        self.bn1 = getattr(nn, normalization)(self.base_channels, momentum=0.1,affine=False,track_running_stats=False)
        self.conv1 = ConvTranspose2d(self.base_channels, image_channels , kernel_size=kernel_size, stride=stride, padding=2, bias=False, output_padding=1, algorithm=algorithm,   ) #output_padding=1
        



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



class AsymResLNet10BReLU(nn.Module):
    def __init__(self, image_channels=3, n_classes=10, algorithm='FA', kernel_size=7, stride=2 , base_channels=64, normalization='BatchNorm2d'):
        super(AsymResLNet10BReLU, self).__init__()
        self.base_channels = base_channels
        self.n_classes = n_classes
        
        
         # layer 2
        # self.bn43 = nn.BatchNorm2d(self.base_channels*4,momentum=0.1, track_running_stats=False)
        self.upsample2 =  ConvTranspose2d(self.n_classes, self.base_channels*2,kernel_size=1, stride=2, padding=0, output_padding=1, algorithm=algorithm,)
        self.bn42 = getattr(nn, normalization)(self.n_classes, affine=False,momentum=0.1, track_running_stats=False)
        self.conv42 = ConvTranspose2d(self.n_classes, self.base_channels*2, kernel_size=3, stride=1, padding=1, output_padding=0, algorithm=algorithm,)
        self.relu = ReLUB.apply
        self.bn41 = getattr(nn, normalization)(self.base_channels*2, affine=False,momentum=0.1, track_running_stats=False)
        self.conv41 = ConvTranspose2d(self.base_channels*2, self.base_channels*2, stride=1, kernel_size=3, padding=1, output_padding=0, algorithm=algorithm, )
        self.bn32 = getattr(nn, normalization)(self.base_channels*2, affine=False, momentum=0.1, track_running_stats=False)
        self.conv32 = ConvTranspose2d(self.base_channels*2, self.base_channels*2, kernel_size=3,stride=1,  padding=1, output_padding=0, algorithm=algorithm, )
        
        self.bn31 = getattr(nn, normalization)(self.base_channels*2, affine=False,momentum=0.1, track_running_stats=False)
        self.conv31 = ConvTranspose2d(self.base_channels*2, self.base_channels*2, kernel_size=3, stride=2, groups=1, padding=1, output_padding=1, algorithm=algorithm, ) #output_padding=1
 
        # layer 1
        self.bn23 = getattr(nn, normalization)(self.base_channels*2, affine=False,momentum=0.1, track_running_stats=False)
        self.upsample1 =  ConvTranspose2d(self.base_channels*2, self.base_channels,kernel_size=1, stride=1, padding=0,output_padding=0, algorithm=algorithm,)
        self.bn22 = getattr(nn, normalization)(self.base_channels*2, affine=False,momentum=0.1, track_running_stats=False)
        self.conv22 = ConvTranspose2d(self.base_channels*2, self.base_channels, kernel_size=3, stride=1, padding=1, output_padding=0, algorithm=algorithm, )
        
        self.bn21 = getattr(nn, normalization)(self.base_channels, affine=False, momentum=0.1,track_running_stats=False)
        self.conv21 = ConvTranspose2d(self.base_channels, self.base_channels, stride=1, kernel_size=3, padding=1, output_padding=0, algorithm=algorithm, )
        self.bn12 = getattr(nn, normalization)(self.base_channels, affine=False, momentum=0.1,track_running_stats=False)

        self.conv12 = ConvTranspose2d(self.base_channels, self.base_channels, kernel_size=3,stride=1,  padding=1, output_padding=0, algorithm=algorithm,)
        
        self.bn11 = getattr(nn, normalization)(self.base_channels, affine=False, momentum=0.1,track_running_stats=False)
        self.conv11 = ConvTranspose2d(self.base_channels, self.base_channels, kernel_size=3, stride=1, groups=1, padding=1, output_padding=0, algorithm=algorithm, )
        
        self.bn1 = getattr(nn, normalization)(self.base_channels, affine=False, momentum=0.1,track_running_stats=False)
        self.conv1 = ConvTranspose2d(self.base_channels, image_channels , kernel_size=kernel_size, stride=stride, padding=2, bias=False, output_padding=1, algorithm=algorithm, ) #output_padding=1
        

    def forward(self, x, relu_list):

        # layer 2 
        identity = x 
        x = self.bn42(x)
        
        x = self.relu(x, relu_list[8])
        x = self.conv42(x)
        x= self.bn41(x)
        x = self.relu(x, relu_list[7])
        x = self.conv41(x)
        x = self.relu(x, relu_list[6])
        x = self.bn32(x)
        x = self.conv32(x)
        x = self.bn31(x)
        x = self.relu(x, relu_list[5])
        x = self.conv31(x)
        # x += self.upsample2(self.bn43(identity))
        x += self.upsample2(identity)
        # x = self.relu(x, relu_list[5])

        # layer 1
        identity = x 
        x = self.bn22(x)
        x = self.relu(x, relu_list[4])
        x = self.conv22(x)
        x = self.bn21(x)
        x = self.relu(x, relu_list[3])
        x = self.conv21(x)

        x = self.relu(x, relu_list[2])
        x = self.bn12(x)
        x = self.conv12(x)
        x = self.bn11(x)
        x = self.relu(x, relu_list[1])
        x = self.conv11(x)
        x += self.upsample1(self.bn23(identity)) 
        
        x = self.bn1(x)
        preconv1 = self.relu(x, relu_list[0])
        x = self.conv1(preconv1)

        return preconv1, x



#****************************************************************

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

def correlation(output, images):
    """Computes the correlation between reconstruction and the original images"""
    x = output.contiguous().view(-1)
    y = images.contiguous().view(-1) 

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return pearson_corr.item()


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='/hdd6gig/Documents/Research/Data/CIFAR10/', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='/hdd6gig/Documents/Research/Data/CIFAR10/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
method = 'SL'
algorithm = 'FA'
print(method, algorithm)
net = AsymResLNet10F(algorithm=algorithm).to(device)
netB = AsymResLNet10B(algorithm=algorithm).to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    netB = torch.nn.DataParallel(netB)
    cudnn.benchmark = True


# inputs = torch.rand(1, 3,32,32).to(device)
# outputs, latents = net(inputs)
# recons = netB(outputs)
# netB.load_state_dict(toggle_state_dict(net.state_dict()))

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('/hdd6gig/Downloads/checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
list_params = [p for n,p in list(net.named_parameters()) if 'feedback' not in n]
optimizer = optim.RMSprop(list_params, lr=args.lr/100,
                    momentum=0.9, weight_decay=1e-5)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50, factor=0.1)

criteriond = nn.TripletMarginLoss() #nn.MSELoss() ## 
list_paramsd = [p for n,p in list(netB.named_parameters()) if 'feedback' not in n]
optimizerd = optim.RMSprop(list_paramsd, lr=args.lr/100,
                    momentum=0.9, weight_decay=1e-6)
# schedulerd = torch.optim.lr_scheduler.StepLR(optimizerd, step_size=10, gamma=0.1, last_epoch=-1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_lossd = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        net.train()
        optimizer.zero_grad()
        _, latents, out = net(inputs)
        
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()

        

        # for batch_idx, (inputs, targets) in enumerate(trainloader):

    #     inputs, targets = inputs.to(device), targets.to(device)

    # ****** SL operations ******** 
        if method == 'SL':
            netB.eval()
            net.eval()
            netB.load_state_dict(toggle_state_dict_normalize(net.state_dict()))
            
            netB.train()
            optimizerd.zero_grad()
            list_relus, latents, out = net(inputs)
            _, recons = netB(latents.detach(), list_relus)
            shuffled = inputs[torch.randperm(inputs.shape[0])]
            lossd = criteriond(F.interpolate(recons, size=inputs.shape[-1]), inputs, shuffled)
            # lossd =  criteriond(F.interpolate(recons, size=inputs.shape[-1]), inputs) #criteriond(rout1, out2)  +  #+ + criteriond(rout1, out1)
            lossd.backward()
            optimizerd.step()

            netB.eval()
            net.eval()

            net.load_state_dict(toggle_state_dict_normalize(netB.state_dict()))
            train_lossd += lossd.item()
         # ****************************

        train_loss += loss.item()
        _, predicted = out.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    
    test_loss = 0
    correct = 0
    total = 0
    net.eval()
    netB.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, _, outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('/hdd6gig/Downloads/checkpoint/'):
            os.mkdir('/hdd6gig/Downloads/checkpoint/')
        torch.save(state, '/hdd6gig/Downloads/checkpoint/ckpt.pth')
        best_acc = acc
    return acc


for epoch in range(start_epoch, start_epoch+200):

    train(epoch)
    acc = test(epoch)
    scheduler.step(acc)

