print('CONFIGURATION')
import torch.nn as nn
import torch.optim as optim
import torchvision
import yaml 
import os
from datetime import datetime
import copy
import numpy as np
import scipy.stats as ss
import scipy
import h5py
import pprint 
pp = pprint.PrettyPrinter(indent=4)

import random
import argparse
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision import models as torchmodels
import matplotlib.pylab as plt

from utils import state_dict_utils

#%%
toggle_state_dict = state_dict_utils.toggle_state_dict
# toggle_state_dict = state_dict_utils.toggle_state_dict_resnets
toggle_state_dict_YYtoBP = state_dict_utils.toggle_state_dict_YYtoBP

from models import custom_models_ResNetLraveled as custom_models

# from models import custom_models as custom_models
# from models import custom_models_Fixup as custom_models


model_names = sorted(name for name in torchmodels.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='Pytorch Training')
parser.add_argument('--note', default=None, type=str)
parser.add_argument('-time', default='Now', type=str)
parser.add_argument('--hash', dest='hash', default=None,
                    help='None to get random hash or int')
parser.add_argument('-dataset', default='imagenet', type=str, metavar='N',
                    help='imagenet | CIFAR10')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-ae', '--arche', metavar='ARCHE', default='ConvDiscr',
                    choices=model_names+['autoencoder','SeeSaw','See','Saw', 'ResNet18F','AsymResNet10F',\
                         'AsymResNet18F','AsymResNet18FNoBN', 'FCDiscrNet',
                         'AsymResLNet10FNoMaxP', 'fixup_resnet20', 'fixup_resnet14','AsymResLNet10F','AsymResLNet14F','AsymResLNetLimited10F','AsymResLNetLimited14F'],
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-ad', '--archd', metavar='ARCHD', default='ConvGene',
                    choices=model_names+['autoencoder','SeeSaw','See','Saw', 'ResNet18B','AsymResNet18B','AsymResNet10B',\
                        'AsymResNet18BUpsamp','AsymResNet18BNoBN', 'FCGeneNet','AsymResLNet10BNoMaxP',
                        'AsymResNet18BNoMaxPLessReLU', 'fixup_resnetT20','fixup_resnetT14','AsymResLNet10B','AsymResLNet14B'],
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument('-AdvTraining', default=False, type=bool, metavar='N',
                    help='if True it does FGSM advresarial training for each batch')
rundatetime = datetime.now().strftime('%b%d-%H-%M')
import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
commit = repo.head.commit.hexsha
print('commit message:',commit )
# ---------- path to save data and models
import socket
if socket.gethostname()[0:4] in  ['node','holm','wats']:
    path_prefix = '/rigel/issa/users/Tahereh/Research'
elif socket.gethostname() == 'SYNPAI':
    path_prefix = '/hdd6gig/Documents/Research'
elif socket.gethostname()[0:2] == 'ax':
    path_prefix = '/home/tt2684/Research'


parser.add_argument('--base_channels', default=64, type=int, metavar='N',
                    help='base_channels')

parser.add_argument('--input_size', default=None, type=int, metavar='N',
                    help='input image size')

parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lrF', '--learning-ratee', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lrF')

parser.add_argument('--lrB', '--learning-rated', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lrB')

parser.add_argument('--step', '--step-scheduler', default=100, type=int,
                    metavar='Step', help='how many epochs in a step', dest='step')

parser.add_argument('--patiencee', '--patiencee-scheduler', default=50, type=float,
                    metavar='DR', help='patience in scheduler encoder', dest='patiencee')
parser.add_argument('--patienced', '--patienced-scheduler', default=40, type=int,
                    metavar='FCT', help='patience in scheduler decoder', dest='patienced')

parser.add_argument('--factore', '--factore-scheduler', default=0.1, type=float,
                    metavar='FacE', help='factor in scheduler encoder', dest='factore')
parser.add_argument('--factord', '--factord-scheduler', default=0.1, type=float,
                    metavar='FacD', help='factor in scheduler decoder', dest='factord')


parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wdF', '--weight-decayF', default=1e-7, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='wdF')
parser.add_argument('--wdB', '--weight-decayB', default=1e-7, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='wdB')
parser.add_argument('--optimizerF', default='RMSprop', type=str,
                    help='optimizer for encoder')
parser.add_argument('--optimizerB', default='RMSprop', type=str,
                    help='optimizer for decoder')
parser.add_argument('--lossfuncB', default='MSELoss', type=str,
                    help='MSELoss|SSIM')
parser.add_argument('-p', '--print-freq', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')



args = parser.parse_args()

if args.dataset == 'imagenet':
        args.n_classes = 1000
        input_size = 224
        image_channels = 3
        
elif args.dataset == 'CIFAR10':
    args.n_classes = 10
    input_size = 32
    image_channels = 3
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

elif args.dataset == 'CIFAR100':
    args.n_classes = 100
    input_size = 32
    image_channels = 3
    
elif 'MNIST' in args.dataset:
    args.n_classes = 10
    input_size = 32
    image_channels = 1
    
if args.hash is None:
    hash = random.getrandbits(10)
else:
    hash = args.hash

project = 'Symbio' #'SYY_MINST'
arch = 'E%sD%s'%(args.arche, args.archd)

args.imagesetdir = path_prefix+'/Data/'
args.runname = rundatetime+'_%s_'%commit[0:min(len(commit), 10)]+str(hash)
args.resultsdir = path_prefix+'/Results/Symbio/Symbio/%s/'%args.runname
args.tensorboarddir = path_prefix + '/Results/Tensorboard_runs/runs'+'/%s/'%project +args.runname
args.path_prefix = path_prefix
args.path_save_model = path_prefix+'/Models/%s_trained/%s/%s/%s/'%(args.dataset,project,arch,args.runname)
#print(args.path_save_model)
# args.databasedir = path_prefix+'/Results/database/%s/%s/%s/'%(project,arch,args.dataset)
args.imagesetdir = path_prefix+'/Data/%s/'%args.dataset
args.customdatasetdir_train = path_prefix+'/Data/Custom_datasets/%s/'%args.dataset
args.databasedir = path_prefix+'/Results/database/%s/%s/%s/'%(project,arch,args.dataset)

path_list = [args.resultsdir, args.path_save_model, args.databasedir, args.imagesetdir,args.customdatasetdir_train, args.tensorboarddir]


for path in path_list:
    if not(os.path.exists(path)):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass



# args.drop = 0.8
# args.step = 40
args.offset = 10
# args.lrF = 1e-3
# args.lrB = 1e-4
# args.wdF = 1e-5
# args.wdB = 1e-6

# algorithm = 'FA'
# args.algorithm = algorithm
print('runname=\'%s\''%args.runname)
print('directory:',args.resultsdir)

pp.pprint(vars(args))

with open(args.resultsdir+'configs.yml', 'w') as outfile:
    
    yaml.dump(vars(args), outfile, default_flow_style=False)


if 'MNIST' in args.dataset:
    image_channels = 1
else:
    image_channels = 3
    
modelF = nn.parallel.DataParallel(getattr(custom_models, args.arche)(algorithm='FA', base_channels=args.base_channels, image_channels=image_channels, n_classes=args.n_classes)).cuda() #Forward().cuda() # main model
modelB = nn.parallel.DataParallel(getattr(custom_models, args.archd)(algorithm='FA', base_channels=args.base_channels, image_channels=image_channels, n_classes=args.n_classes)).cuda() # backward network to compute gradients for modelF

modelC = nn.parallel.DataParallel(getattr(custom_models, args.arche)(algorithm='BP', base_channels=args.base_channels, image_channels=image_channels, n_classes=args.n_classes)).cuda() # Forward Control model to compare to BP



# # modelC = nn.DataParallel(modelC)
# print(modelF.state_dict().keys())
# print("***** ***********")
# print(modelC.state_dict().keys())

modelC.load_state_dict(toggle_state_dict_YYtoBP(modelF.state_dict(), modelC.state_dict()))
# print(modelC.state_dict().keys())


# print(modelC.state_dict()['conv0.weight'][0,0])
# print(modelF.state_dict()['conv0.weight'][0,0])
# start symmetric
# modelB.load_state_dict(transpose_weights(modelF.state_dict()) )
#%%
if 'fixup' in args.arche:
    parameters_bias = [p[1] for p in modelF.named_parameters() if 'bias' in p[0]]
    parameters_scale = [p[1] for p in modelF.named_parameters() if 'scale' in p[0]]
    parameters_others = [p[1] for p in modelF.named_parameters() if not ('bias' in p[0] or 'scale' in p[0])]
    optimizerF = getattr(torch.optim,args.optimizerF)(
            [{'params': parameters_bias, 'lr': args.lrF/10.}, 
            {'params': parameters_scale, 'lr': args.lrF/10.}, 
            {'params': parameters_others}], 
            lr=args.lrF, 
            momentum=args.momentum, 
            weight_decay=args.wdF)
                                

    parameters_bias = [p[1] for p in modelB.named_parameters() if 'bias' in p[0]]
    parameters_scale = [p[1] for p in modelB.named_parameters() if 'scale' in p[0]]
    parameters_others = [p[1] for p in modelB.named_parameters() if not ('bias' in p[0] or 'scale' in p[0])]
    optimizerB = getattr(torch.optim,args.optimizerB)(
            [{'params': parameters_bias, 'lr': args.lrB/10.}, 
            {'params': parameters_scale, 'lr': args.lrB/10.}, 
            {'params': parameters_others}], 
            lr=args.lrB, 
            momentum=args.momentum, 
            weight_decay=args.wdB) 
    
    parameters_bias = [p[1] for p in modelC.named_parameters() if 'bias' in p[0]]
    parameters_scale = [p[1] for p in modelC.named_parameters() if 'scale' in p[0]]
    parameters_others = [p[1] for p in modelC.named_parameters() if not ('bias' in p[0] or 'scale' in p[0])]
    optimizerC = getattr(torch.optim,args.optimizerF)(
            [{'params': parameters_bias, 'lr': args.lrF/10.}, 
            {'params': parameters_scale, 'lr': args.lrF/10.}, 
            {'params': parameters_others}], 
            lr=args.lrF, 
            momentum=args.momentum, 
            weight_decay=args.wdF)

else:
    if 'Adam' in args.optimizerF:
        optimizerF = getattr(optim, args.optimizerF)(modelF.parameters(),  lr=args.lrF, weight_decay=args.wdF)
        optimizerC = getattr(optim, args.optimizerF)(modelC.parameters(),  lr=args.lrF, weight_decay=args.wdF)
    else:
        optimizerF = getattr(optim, args.optimizerF)(modelF.parameters(),  lr=args.lrF, weight_decay=args.wdF, momentum=args.momentum)
        optimizerC = getattr(optim, args.optimizerF)(modelC.parameters(),  lr=args.lrF, weight_decay=args.wdF, momentum=args.momentum)
    if 'Adam' in args.optimizerB:
        optimizerB = getattr(optim, args.optimizerB)(modelB.parameters(),  lr=args.lrB, weight_decay=args.wdB)
    else:
        optimizerB = getattr(optim, args.optimizerB)(modelB.parameters(),  lr=args.lrB, weight_decay=args.wdB, momentum=args.momentum)

# schedulerF = optim.lr_scheduler.MultiStepLR(optimizerF, np.arange(0, args.epochs, args.step),args.drop)
# schedulerB = optim.lr_scheduler.MultiStepLR(optimizerB, np.arange(args.offset, args.epochs, args.step),args.drop)
# schedulerC = optim.lr_scheduler.MultiStepLR(optimizerC, np.arange(0,args.epochs, args.step),args.drop)

schedulerF = optim.lr_scheduler.ReduceLROnPlateau(optimizerF, 'max', patience=args.patiencee)
schedulerB = optim.lr_scheduler.ReduceLROnPlateau(optimizerB, 'max', patience=args.patienced)
schedulerC = optim.lr_scheduler.ReduceLROnPlateau(optimizerC, 'max', patience=args.patiencee)

criterionF = nn.CrossEntropyLoss() #
criterionB = nn.MSELoss() #


print('***** Creating initial state dicts *********')

modelF_nottrained = copy.deepcopy(modelF.state_dict())
modelB_nottrained = copy.deepcopy(modelB.state_dict())
modelC_nottrained = copy.deepcopy(modelC.state_dict())

optimizerF_original = copy.deepcopy(optimizerF.state_dict())
optimizerB_original = copy.deepcopy(optimizerB.state_dict())
optimizerC_original = copy.deepcopy(optimizerC.state_dict())

schedulerF_original = copy.deepcopy(schedulerF.state_dict())
schedulerB_original = copy.deepcopy(schedulerB.state_dict())
schedulerC_original = copy.deepcopy(schedulerC.state_dict())

torch.save(modelF_nottrained, args.resultsdir+'modelF_untrained.pt')
torch.save(modelB_nottrained, args.resultsdir+'modelB_untrained.pt')
torch.save(modelC_nottrained, args.resultsdir+'modelC_untrained.pt')

torch.save(optimizerF_original, args.resultsdir+'optimizerF_original.pt')
torch.save(optimizerB_original, args.resultsdir+'optimizerB_original.pt')
torch.save(optimizerC_original, args.resultsdir+'optimizerC_original.pt')

torch.save(schedulerF_original, args.resultsdir+'schedulerF_original.pt')
torch.save(schedulerB_original, args.resultsdir+'schedulerB_original.pt')
torch.save(schedulerC_original, args.resultsdir+'schedulerC_original.pt')


with open(args.resultsdir+"architecture_modelF.txt", "w") as text_file:
    print("modelF: {}".format(modelF), file=text_file)

with open(args.resultsdir+"architecture_modelB.txt", "w") as text_file:
    print("modelB: {}".format(modelB), file=text_file)