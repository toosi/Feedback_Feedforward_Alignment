
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

def toggle_state_dict_wresnets(state_dict):

    """
    Works for modular resnet codes
    """
    # How many layers in each block of Wresnet? we need it because we reverse the order of layers in a block
    blocks = [int(k.split('layer.')[1].split('.')[0]) for k in state_dict.keys() if 'block' in k]
    #'check if you wrapped the model in a module like nn.parallel does'
    
    num_blocks = max(blocks) + 1
    # print(state_dict.keys())
    new_state_dict = {}

    for ik, k  in enumerate(state_dict.keys()):

        item = state_dict[k]
        # whithin blocks
        if 'layer' in k :

            assert k.split('layer.')[1].split('.')[0].isdigit()
            b = int(k.split('layer.')[1].split('.')[0])
            k_new = k.split('layer.')[0]+'layer.%d'%(num_blocks-b-1) +k.split('layer.')[1][len(str(b)):]
            
            if 'feedback' in k:
                new_state_dict.update({k_new.strip('_feedback'):item})
            else:
                new_state_dict.update({k_new+'_feedback':item})
        
        # outside blocks
        else:
            # to exclude bn
            if 'conv' in k:
                if 'feedback' in k:
                    new_state_dict.update({k.strip('_feedback'):item})
                else:
                    new_state_dict.update({k+'_feedback':item})
            
            elif 'fc' in k:
                if 'feedback' in k:
                    new_state_dict.update({k.strip('_feedback'):item.t()})
                else:
                    new_state_dict.update({k+'_feedback':item.t()})

            else:

                new_state_dict.update({k:item})
    
 
    return new_state_dict

#*****************************
# modifications affine=False, track_running_stats=False for bn2 in BasicBlock
#track_running_stats=False for bn2 improves test by 10% on nn modules
#affine=False degrades test for bn2 by 2% on nn modules


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, algorithm='FA'):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, affine=False, track_running_stats=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, algorithm=algorithm)
        self.bn2 = nn.BatchNorm2d(out_planes, affine=False, track_running_stats=False) 
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False, algorithm=algorithm)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False, algorithm=algorithm) or None
        

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, algorithm='FA'):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, algorithm=algorithm)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, algorithm):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, algorithm))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, algorithm='FA'):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False, algorithm=algorithm)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, algorithm=algorithm)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, algorithm=algorithm)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, algorithm=algorithm)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], affine=False, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc = Linear(nChannels[3], num_classes, algorithm=algorithm)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        latents = self.relu(self.bn1(out))
        out = F.avg_pool2d(latents, 8)
        out = out.view(-1, self.nChannels)
        return latents, self.fc(out)



#***************************************************************
#***************************************************************
class BasicBlockT(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, algorithm='FA'):
        super(BasicBlockT, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, affine=False, track_running_stats=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = ConvTranspose2d(out_planes, in_planes,  kernel_size=3, stride=stride,
                               padding=1, bias=False, algorithm=algorithm)
        self.bn2 = nn.BatchNorm2d(out_planes, affine=False, track_running_stats=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = ConvTranspose2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False, algorithm=algorithm)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and ConvTranspose2d(out_planes, in_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False, algorithm=algorithm) or None
    def forward(self, x):

        # if not self.equalInOut:
        #     x = self.conv2(x)
        # else:
        out = self.conv2(x)  

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.relu1(self.bn1(out))

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlockT(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, algorithm='FA'):
        super(NetworkBlockT, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, algorithm)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, algorithm):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == int(nb_layers)-1 and in_planes or out_planes, out_planes, i == int(nb_layers)-1 and stride or 1, dropRate, algorithm))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNetT(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, algorithm='FA'):
        super(WideResNetT, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlockT
        # 1st conv before any network block
        self.conv1 = ConvTranspose2d(nChannels[0],3, kernel_size=3, stride=1,
                               padding=1, bias=False, algorithm=algorithm)
        # 1st block
        self.block1 = NetworkBlockT(n, nChannels[0], nChannels[1], block, 1, dropRate, algorithm=algorithm)
        # 2nd block
        self.block2 = NetworkBlockT(n, nChannels[1], nChannels[2], block, 2, dropRate, algorithm=algorithm)
        # 3rd block
        self.block3 = NetworkBlockT(n, nChannels[2], nChannels[3], block, 2, dropRate, algorithm=algorithm)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], affine=False, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc = Linear(num_classes, nChannels[3], algorithm=algorithm)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):

        # out = self.fc(x)
        # out = out.view(-1, self.nChannels, 1, 1)
        # out = F.interpolate(out, 8)
        # out = self.relu(self.bn1(x)) #removing this incread test performance
        out = self.block3(x)
        out = self.block2(out)
        out = self.block1(out)
        out = self.conv1(out)

        out = self.relu(self.bn1(out))
         
        return out

#***************************************************************






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
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='/hdd6gig/Documents/Research/Data/CIFAR10/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

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
net = WideResNet(depth=10, num_classes=10, widen_factor=4, dropRate=0.0,  algorithm=algorithm).to(device)
netB = WideResNetT(depth=10, num_classes=10, widen_factor=4, dropRate=0.0, algorithm=algorithm).to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    netB = torch.nn.DataParallel(netB)
    cudnn.benchmark = True


# inputs = torch.rand(1, 3,32,32).to(device)
# outputs = net(inputs)
# recons = netB(outputs)
# netB.load_state_dict(toggle_state_dict_wresnets(net.state_dict()))

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
optimizer = optim.SGD(list_params, lr=args.lr,
                    momentum=0.9, weight_decay=5e-4)

criteriond = nn.TripletMarginLoss()# 
list_paramsd = [p for n,p in list(netB.named_parameters()) if 'feedback' not in n]
optimizerd = optim.Adam(list_paramsd, lr=args.lr/100,
                      weight_decay=5e-4)

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


        # ****** SL operations ******** 
        if method == 'SL':
            netB.eval()
            net.eval()
            netB.load_state_dict(toggle_state_dict_wresnets(net.state_dict()))
            
            netB.train()
            optimizerd.zero_grad()
            latents, outputs = net(inputs)
            recons = netB(latents.detach())
            shuffled = inputs[torch.randperm(inputs.shape[0])]
            lossd = criteriond(F.interpolate(recons, size=inputs.shape[-1]), inputs, -shuffled)
            lossd.backward()
            optimizerd.step()

            netB.eval()
            net.eval()

            net.load_state_dict(toggle_state_dict_wresnets(netB.state_dict()))
            train_lossd += lossd.item()
         # ****************************

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        net.train()
        optimizer.zero_grad()
        latents, outputs = net(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        

        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)
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


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)

