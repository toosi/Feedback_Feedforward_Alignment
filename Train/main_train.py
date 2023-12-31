
import argparse
import os
import random
import shutil
import time
import warnings
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tensorboardX import SummaryWriter

import torch.nn as nn
import torch.optim as optim
import torchvision
import yaml 
import os
import copy
import json
import numpy as np
import scipy.stats as ss
import scipy
import h5py
import random
import argparse
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib
import matplotlib.pylab as plt
matplotlib.use('agg')
import pprint 
pp = pprint.PrettyPrinter(indent=4)

from utils import state_dict_utils

import pytorch_ssim
from optimizers import LARC


# # toggle_state_dict = state_dict_utils.toggle_state_dict # for ResNetLraveled
# toggle_state_dict = state_dict_utils.toggle_state_dict_resnets # for custom_resnets
# toggle_state_dict_YYtoBP = state_dict_utils.toggle_state_dict_YYtoBP

# # from models import custom_models_ResNetLraveled as custom_models
# from models import custom_resnets as custom_models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


path_prefix = '/path/tosave/results/directory'

parser = argparse.ArgumentParser(description='Training with FFA and controls')
parser.add_argument(
        '--config-file',
        dest='config_file',
        type=argparse.FileType(mode='r'))


parser.add_argument('--method', type=str, default='BP', metavar='M',
                    help='method:BP|FFA|FA')

parser.add_argument('--resume_training_epochs', type=int, default=0,
                    help='if greater than 0 reads the checkpoint from resultsdir and append results to jsons and csvs')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
args = parser.parse_args()
assert args.config_file, 'Please specify a config file path'
if args.config_file:
    data = yaml.load(args.config_file)
    delattr(args, 'config_file')
    delattr(args, 'epochs')
    arg_dict = args.__dict__
    for key, value in data.items():
        setattr(args, key, value)
    if args.resume_training_epochs:
        args.epochs = args.resume_training_epochs

pp.pprint(arg_dict)
print(args.method)
with open(args.resultsdir+'args.yml', 'w') as outfile:
    
    yaml.dump(vars(args), outfile, default_flow_style=False)

writer = SummaryWriter(log_dir=args.tensorboarddir)


if 'AsymResLNet' in args.arche:
    toggle_state_dict = state_dict_utils.toggle_state_dict_normalize
    from models import custom_models_ResNetLraveled as custom_models

elif 'asymresnet' in args.arche:
    toggle_state_dict = state_dict_utils.toggle_state_dict_resnets
    # from models import custom_resnets as custom_models
    from models import custom_resnets_cifar_tmp as custom_models


elif args.arche.startswith('resnet'):
    from models import resnets as custom_models
    #just for compatibality
    toggle_state_dict = state_dict_utils.toggle_state_dict_resnets
elif  'FullyConnected' in args.arche:
    toggle_state_dict = state_dict_utils.toggle_state_dict

    from models import custom_models

toggle_state_dict_YYtoBP = state_dict_utils.toggle_state_dict_YYtoBP


best_acce = 0
best_lossd = 10


def main():

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)



def main_worker(gpu, ngpus_per_node, args):
    global best_acce, best_lossd
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.dataset == 'imagenet':
        args.n_classes = 1000
        if args.input_size is None:
            input_size = 224
        else:
            input_size = args.input_size
        image_channels = 3
        
    elif args.dataset == 'CIFAR10':
        args.n_classes = 10
        if args.input_size is None:
            input_size = 32
        else:
            input_size = args.input_size
        image_channels = 3
        train_mean = (0.4914, 0.4822, 0.4465)
        train_std = (0.2023, 0.1994, 0.2010)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    elif args.dataset == 'CIFAR100':
        args.n_classes = 100
        if args.input_size is None:
            input_size = 32
        else:
            input_size = args.input_size
        image_channels = 3

        train_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        train_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    

        
    elif 'MNIST' in args.dataset:
        args.n_classes = 10
        if args.input_size is None:
            input_size = 32
        else:
            input_size = args.input_size
        image_channels = 1
    
        
    
    # create encoder and decoder model
    def get_model(arch, agpu=args.gpu, args_model={}):
        if arch in model_names:
            if args.pretrained:
                print("=> using pre-trained model '{}'".format(arch))
                model = models.__dict__[arch](pretrained=True)
            else:
                print("=> creating model '{}'".format(arch))
                model = models.__dict__[arch]()
        else:
            model = getattr(custom_models,arch)(**args_model) 

        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if agpu is not None:
                print("Use GPU: {} for training".format(agpu))
                torch.cuda.set_device(agpu)
                model.cuda(agpu)
                    
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[agpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)

        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if arch.startswith('alexnet') or arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
        if agpu == 0:
            print(model)

        return model
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
                #print('in loop',args.gpu)
 
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    
    if args.method == 'BP':
        args.algorithm = 'BP'
        modelidentifier = 'C' #'Control
    else:
        args.algorithm = args.method
        modelidentifier = 'F'

    if 'FullyConnected' in args.arche:
        kwargs_asym = {'algorithm':args.algorithm, 'hidden_layers':[256, 256, 10], 'nonlinearfunc':'relu', 'input_length':1024}
    else:
        kwargs_asym = {'algorithm':args.algorithm, 'base_channels':args.base_channels, 'image_channels':image_channels, 'n_classes':args.n_classes, 'normalization_affine': not(args.normalization_noaffine)}

    print(kwargs_asym)
    modelF = nn.parallel.DataParallel(getattr(custom_models, args.arche)(**kwargs_asym)).cuda() #Forward().cuda() # main model
    modelB = nn.parallel.DataParallel(getattr(custom_models, args.archd)(**kwargs_asym)).cuda() # backward network to compute gradients for modelF

    
    # define loss function (criterion) and optimizer
    criterione = nn.CrossEntropyLoss().cuda(args.gpu)                                                                                       
    if args.lossfuncB == 'MSELoss':                                                                                       
        criteriond = nn.MSELoss().cuda(args.gpu)
    elif args.lossfuncB == 'SSIM':
        criteriond = pytorch_ssim.SSIM(window_size = int(input_size/10)).cuda(args.gpu)
    elif args.lossfuncB == 'TripletMarginLoss':
        criteriond = nn.TripletMarginLoss().cuda(args.gpu)

    list_paramsF = [p for n,p in list(modelF.named_parameters()) if 'feedback' not in n]
    list_paramsFB_local = [p for n,p in list(modelF.named_parameters()) if 'feedback' in n]
    list_paramsB = [p for n,p in list(modelB.named_parameters()) if 'feedback' not in n]

    optimizerFB_local = torch.optim.Adam(list_paramsFB_local, lr=0.0097)

    if 'Adam' in args.optimizerF:

        optimizerF = getattr(torch.optim,args.optimizerF)(list_paramsF, args.lrF,
                    weight_decay=args.wdF)
        optimizerF3 = getattr(torch.optim,args.optimizerF)(list_paramsF, args.lrF,
                    weight_decay=args.wdF)
    else:
        
        optimizerF = getattr(torch.optim,args.optimizerF)(list_paramsF, args.lrF,
                                momentum=args.momentumF,
                                weight_decay=args.wdF,)
        optimizerF3 = getattr(torch.optim,args.optimizerF)(list_paramsF, args.lrF,
                    momentum=args.momentumF,
                    weight_decay=args.wdF)

    if 'Adam' in args.optimizerB:                       

        optimizerB = getattr(torch.optim,args.optimizerB)(list_paramsB, args.lrB,
                                    
                                    weight_decay=args.wdB) 
    else:
        optimizerB = getattr(torch.optim,args.optimizerB)(list_paramsB, args.lrB,
                            momentum=args.momentumB,
                            weight_decay=args.wdB)
    
    schedulerF = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerF, 'max', patience=args.patiencee, factor=args.factore)
    schedulerB = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerB, 'max', patience=args.patienced, factor=args.factord)

    schedulerF3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerF3, 'max', patience=args.patiencee, factor=args.factore)

    
    modelF_nottrained = torch.load(args.resultsdir+'model%s_untrained.pt'%modelidentifier)
    modelB_nottrained = torch.load(args.resultsdir+'modelB_untrained.pt')

    optimizerF_original = torch.load(args.resultsdir+'optimizer%s_original.pt'%modelidentifier)
    optimizerB_original = torch.load(args.resultsdir+'optimizerB_original.pt')

    schedulerF_original = torch.load(args.resultsdir+'scheduler%s_original.pt'%modelidentifier)
    schedulerB_original = torch.load(args.resultsdir+'schedulerB_original.pt')

    if args.resume_training_epochs == 0:
        modelF.load_state_dict(modelF_nottrained)
        optimizerF.load_state_dict(optimizerF_original)        
        optimizerF3.load_state_dict(optimizerF_original)
        schedulerF.load_state_dict(schedulerF_original)
        schedulerF3.load_state_dict(schedulerF_original)
        
    else:
        checkpointe = torch.load(args.resultsdir+'checkpointe_%s.pth.tar'%args.method)
        modelF_trained = checkpointe['state_dict']
        args.start_epoch = checkpointe['epoch']

        best_acce = checkpointe['best_loss']
        if args.gpu is not None:
            # best_loss may be from a checkpoint from a different GPU
            best_acce = best_acce.to(args.gpu)
        modelF.load_state_dict(checkpointe['state_dict'])
        optimizerF.load_state_dict(checkpointe['optimizer'])

        if 'scheduler' in checkpointe.keys():
            schedulerF.load_state_dict(checkpointe['scheduler'])
        else:
            schedulerF.load_state_dict(schedulerF_original)

    if args.resume_training_epochs == 0 or args.method in ['FA','BP']:  
        modelB.load_state_dict(modelB_nottrained)   
        optimizerB.load_state_dict(optimizerB_original) 
        schedulerB.load_state_dict(schedulerB_original)

    else:
        checkpointd = torch.load(args.resultsdir+'checkpointd_%s.pth.tar'%args.method)
        modelB_trained = checkpointd['state_dict']
        optimizerB.load_state_dict(checkpointd['optimizer'])

        if 'scheduler' in checkpointd.keys():
            schedulerB.load_state_dict(checkpointd['scheduler'])
        else:
            schedulerB.load_state_dict(schedulerB_original)



    
    # Data loading code
    
    if 'CIFAR' in args.dataset:

        transform_train = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size, padding=4),
        # transforms.RandomAffine(degrees=30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

        train_dataset = getattr(datasets, args.dataset)(root=args.imagesetdir, train=True, download=True, transform=transform_train)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,sampler=train_sampler, shuffle=(train_sampler is None), num_workers=args.workers, drop_last=True)

        test_dataset = getattr(datasets, args.dataset)(root=args.imagesetdir, train=False, download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

    elif 'STL' in args.dataset:

        transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
        ])

        transform_test = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

        train_dataset = getattr(datasets, args.dataset)(root=args.imagesetdir, split='train', download=True, transform=transform_train)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,sampler=train_sampler, shuffle=(train_sampler is None), num_workers=args.workers, drop_last=True)

        test_dataset = getattr(datasets, args.dataset)(root=args.imagesetdir, split='test', download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

        
    
    elif 'MNIST' in args.dataset:
        transform_train = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])

        train_dataset = getattr(datasets, args.dataset)(root=args.imagesetdir, train=True, download=True, transform=transform_train)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,sampler=train_sampler, shuffle=(train_sampler is None), num_workers=args.workers, drop_last=True)

        test_dataset = getattr(datasets, args.dataset)(root=args.imagesetdir, train=False, download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

        # n_classes = 10
        

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume+'checkpointe.pth.tar'):
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpointe = torch.load(args.resume+'checkpointe.pth.tar')
            args.start_epoch = checkpointe['epoch']
            best_acce = checkpointe['best_loss']
            if args.gpu is not None:
                # best_loss may be from a checkpoint from a different GPU
                best_acce = best_acce.to(args.gpu)
            modelF.load_state_dict(checkpointe['state_dict'])
            optimizerF.load_state_dict(checkpointe['optimizer'])

            checkpointd = torch.load(args.resume+'checkpointd.pth.tar')
            args.start_epoch = checkpointd['epoch']
            best_lossd = checkpointd['best_loss']
            if args.gpu is not None:
                # best_loss may be from a checkpoint from a different GPU
               best_lossd = best_lossd.to(args.gpu)
            modelB.load_state_dict(checkpointd['state_dict'])
            optimizerB.load_state_dict(checkpointd['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpointe['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        if os.path.isfile(args.resume+'checkpointeFA.pth.tar'):
            print("=> loading checkpoint FA '{}'".format(args.resume))

            checkpointeFA = torch.load(args.resume+'checkpointeFA.pth.tar')
            best_acce = checkpointeFA['best_loss']
            if args.gpu is not None:
                # best_loss may be from a checkpoint from a different GPU
                best_acce = best_acce.to(args.gpu)
            modelFFA.load_state_dict(checkpointeFA['state_dict'])
            optimizerFFA.load_state_dict(checkpointeFA['optimizer'])


            print("=> loaded checkpoint FA'{}' (epoch {})"
                  .format(args.resume, checkpointeFA['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        if os.path.isfile(args.resume+'checkpointeBP.pth.tar'):
            print("=> loading checkpoint BP '{}'".format(args.resume))

            checkpointeBP = torch.load(args.resume+'checkpointeBP.pth.tar')
            best_acce = checkpointeBP['best_loss']
            if args.gpu is not None:
                # best_loss may be from a checkpoint from a different GPU
                best_acce = best_acce.to(args.gpu)
            modelFBP.load_state_dict(checkpointeBP['state_dict'])
            optimizerFBP.load_state_dict(checkpointeBP['optimizer'])


            print("=> loaded checkpoint BP'{}' (epoch {})"
                  .format(args.resume, checkpointeBP['epoch']))
        else:
            print("=> no checkpoint BP found at '{}'".format(args.resume))



    if args.evaluate:

        validate(val_loader, modelF, modelB, criterione, criteriond, args, args.start_epoch)

        return

    # a json to keep the records
    run_json_dict = {}
    Train_acce_list = []
    Train_corrd_list = []
    Train_lossd_list= []
    Train_lossl_list= []

    Test_acce_list  = []
    Test_corrd_list = []
    Test_lossd_list = []
    Test_lossl_list= []

    lrF_list = []

    Alignments_corrs_first_layer_list =  [] 
    Alignments_corrs_last_layer_list =  []

    Forward_norm_first_layer_list =  []
    Forward_norm_last_layer_list =  []

    Alignments_ratios_first_layer_list =  []
    Alignments_ratios_last_layer_list =  []

    results = {'train_acc': [],  'test_acc': [], 'train_lossd': [],  'test_lossd': [], 'train_corrd': [],  'test_corrd': []}
    for epoch in np.arange(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)
    
        for param_group in optimizerF.param_groups:
            lrF = param_group['lr'] 
        lrF_list.extend([lrF])
        run_json_dict.update({'lrF':lrF_list})

        print('*****  lrF=%1e'%(lrF))
        
        # train for one epoch
        modelF, modelB, train_results = train(train_loader, modelF, modelB, criterione, criteriond, optimizerF, optimizerB,optimizerF3, schedulerF,schedulerB,schedulerF3, optimizerFB_local, epoch, args)
        Train_acce_list.extend( [round(train_results[0],3)])
        Train_corrd_list.extend([round(train_results[1],3)])
        Train_lossd_list.extend([train_results[2]])
        Train_lossl_list.extend([train_results[3]])

        results['train_acc'].append(round(train_results[0],3))
        results['train_corrd'].append(train_results[1])
        results['train_lossd'].append(train_results[2])

        run_json_dict.update({'Train_acce':Train_acce_list})
        run_json_dict.update({'Train_corrd':Train_corrd_list})
        run_json_dict.update({'Train_lossd':Train_lossd_list})
        run_json_dict.update({'Train_lossl':Train_lossl_list})

        # evaluate on validation set
        _, _, test_results = validate(val_loader, modelF,modelB, criterione, criteriond, args, epoch)
        
        results['test_acc'].append(round(test_results[0],3))
        results['test_lossd'].append(test_results[2])
        results['test_corrd'].append(test_results[1])

        acce = test_results[0]
        corrd = test_results[1]
        Test_acce_list.extend( [round(test_results[0],3)])
        Test_corrd_list.extend([round(test_results[1],3)])
        Test_lossd_list.extend([test_results[2]])
        Test_lossl_list.extend([test_results[3]])

        run_json_dict.update({'Test_acce':Test_acce_list})
        run_json_dict.update({'Test_corrd':Test_corrd_list})
        run_json_dict.update({'Test_lossd':Test_lossd_list})
        run_json_dict.update({'Test_lossl':Test_lossl_list})

        # save statistics
        if args.resume_training_epochs:
            df = pd.read_csv(args.resultsdir + 'training_results_%s.csv'%args.algorithm)

            for k in results.keys():
                if k != 'Unnamed: 0':
                    
                    new_item =  list(df[k]) + results[k]
                    results.update({k: new_item})

        data_frame = pd.DataFrame(data=results)
        data_frame.to_csv(args.resultsdir + 'training_results_%s.csv'%args.algorithm)

        # evaluate alignments
        list_WF = [k for k in modelF.state_dict().keys() if 'feedback' in k]
        first_layer_key = list_WF[0]
        last_layer_key = list_WF[-1]

        corrs_first_layer = correlation(modelF.state_dict()[first_layer_key.strip('_feedback')], modelF.state_dict()[first_layer_key])
        ratios_first_layer = torch.norm(modelF.state_dict()[first_layer_key.strip('_feedback')]).item()/torch.norm(modelF.state_dict()[first_layer_key]).item() 

        corrs_last_layer = correlation(modelF.state_dict()[last_layer_key.strip('_feedback')], modelF.state_dict()[last_layer_key])
        ratios_last_layer = torch.norm(modelF.state_dict()[last_layer_key.strip('_feedback')]).item()/torch.norm(modelF.state_dict()[last_layer_key]).item() 

        norm_first_layer = torch.norm(modelF.state_dict()[first_layer_key.strip('_feedback')]).item()
        norm_last_layer = torch.norm(modelF.state_dict()[first_layer_key.strip('_feedback')]).item()

        norm_first_layer_back = torch.norm(modelF.state_dict()[first_layer_key]).item()
        norm_last_layer_back = torch.norm(modelF.state_dict()[first_layer_key]).item()


        Alignments_corrs_first_layer_list.extend([round(corrs_first_layer, 3)])
        Alignments_corrs_last_layer_list.extend([round(corrs_last_layer, 3)])
        
        Alignments_ratios_first_layer_list.extend([round(ratios_first_layer, 3)])
        Alignments_ratios_last_layer_list.extend([round(ratios_last_layer, 3)])
        
        Forward_norm_first_layer_list.extend([round(norm_first_layer, 3)])
        Forward_norm_last_layer_list.extend([round(norm_last_layer, 3)])
        
        
        run_json_dict.update({'Alignments_corrs_first_layer':Alignments_corrs_first_layer_list})
        run_json_dict.update({'Alignments_corrs_last_layer':Alignments_corrs_last_layer_list})
        
        run_json_dict.update({'Alignments_ratios_first_layer':Alignments_ratios_first_layer_list})
        run_json_dict.update({'Alignments_ratios_last_layer':Alignments_ratios_last_layer_list})

        
        run_json_dict.update({'Forward_norm_first_layer':Forward_norm_first_layer_list})
        run_json_dict.update({'Forward_norm_last_layer':Forward_norm_last_layer_list})


        # ---- adjust learning rates ----------

        adjust_learning_rate(schedulerF, acce)
        adjust_learning_rate(schedulerB, acce)# corrd


        # remember best acc@1 and save checkpoint
        is_beste = acce > best_acce
        best_acce = max(acce, best_acce)
    
        if args.method == 'BSL':
            break

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arche,
                'state_dict': modelF.state_dict(),
                'best_loss': best_acce,
                'optimizer' : optimizerF.state_dict(),
                'scheduler' : schedulerF.state_dict(),
            }, is_beste, filename='checkpointe_%s.pth.tar'%args.method)

            if args.method.startswith('SL') or args.method == 'BSL':
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.archd,
                    'state_dict': modelB.state_dict(),
                    'best_loss': best_acce,
                    'optimizer' : optimizerB.state_dict(),
                    'scheduler' : schedulerB.state_dict(),
                }, is_beste,  filename='checkpointd_%s.pth.tar'%args.method)
            
            if args.resume_training_epochs:
                with open('%srun_json_dict_%s.json'%(args.resultsdir, args.method), 'r') as fp:
                    chkp_run_json_dict = json.load(fp)
                for k in chkp_run_json_dict.keys():
                    new_item = chkp_run_json_dict[k] + run_json_dict[k]
                    run_json_dict.update({k:new_item})

            with open('%srun_json_dict_%s.json'%(args.resultsdir, args.method), 'w') as fp:
                
                json.dump(run_json_dict, fp, indent=4, sort_keys=True)        
                fp.write("\n")

        run_json_dict = {}
        Train_acce_list = []
        Train_corrd_list = []
        Train_lossd_list= []

        Test_acce_list  = []
        Test_corrd_list = []
        Test_lossd_list = []

        lrF_list = []


        if 'AsymResLNet' in args.arche:
            modelB.load_state_dict(toggle_state_dict(modelF.state_dict()))
        elif 'asymresnet' in args.arche:
            modelB.load_state_dict(modelF.state_dict(), toggle_state_dict(modelB.state_dict()))
        

        for epoch in range(args.start_epoch, args.epochs):

            if args.distributed:
            
                train_sampler.set_epoch(epoch)

        
            for param_group in optimizerF.param_groups:
                lrF = param_group['lr'] 
            lrF_list.extend([lrF])
            run_json_dict.update({'lrF':lrF_list})


            print('*****  lrF=%1e'%(lrF))
            
            # train for one epoch
            modelF, modelB, train_results = train(train_loader, modelF, modelB, criterione, criteriond, optimizerF, optimizerB,optimizerF3, schedulerF,schedulerB,schedulerF3, epoch, args)
            Train_acce_list.extend( [round(train_results[0],3)])
            Train_corrd_list.extend([round(train_results[1],3)])
            Train_lossd_list.extend([train_results[2]])
            run_json_dict.update({'Train_acce':Train_acce_list})
            run_json_dict.update({'Train_corrd':Train_corrd_list})
            run_json_dict.update({'Train_lossd':Train_lossd_list})
            # evaluate on validation set
            _, _, test_results = validate(val_loader, modelF,modelB, criterione, criteriond, args, epoch)
            
            acce = test_results[0]
            corrd = test_results[1]
            Test_acce_list.extend( [round(test_results[0],3)])
            Test_corrd_list.extend([round(test_results[1],3)])
            Test_lossd_list.extend([test_results[2]])
            run_json_dict.update({'Test_acce':Test_acce_list})
            run_json_dict.update({'Test_corrd':Test_corrd_list})
            run_json_dict.update({'Test_lossd':Test_lossd_list})

            # ---- adjust learning rates ----------

            adjust_learning_rate(schedulerF, acce)
            adjust_learning_rate(schedulerB, corrd)


            # remember best acc@1 and save checkpoint
            is_beste = acce > best_acce
            best_acce = max(acce, best_acce)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):

                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arche,
                    'state_dict': modelF.state_dict(),
                    'best_loss': best_acce,
                    'optimizer' : optimizerF.state_dict(),
                    'scheduler' : schedulerF.state_dict(),
                }, is_beste, filename='checkpointe_%s.pth.tar'%args.method)

                if args.method.startswith('SL') or args.method == 'BSL':
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.archd,
                        'state_dict': modelB.state_dict(),
                        'best_loss': best_acce,
                        'optimizer' : optimizerB.state_dict(),
                        'scheduler' : schedulerB.state_dict(),
                    }, is_beste,  filename='checkpointd_%s.pth.tar'%args.method)
                
                
                if args.resume_training_epochs:
                    with open('%srun_json_dict_%s.json'%(args.resultsdir, args.method), 'r') as fp:
                        chkp_run_json_dict = json.load(fp)
                    for k in chkp_run_json_dict.keys():
                        new_item = chkp_run_json_dict[k] + run_json_dict[k]
                        run_json_dict.update({k:new_item})

                with open('%srun_json_dict_%s.json'%(args.resultsdir, args.method), 'w') as fp:
                    
                    json.dump(run_json_dict, fp, indent=4, sort_keys=True)        
                    fp.write("\n")
        
def train(train_loader, modelF, modelB,  criterione, criteriond, optimizerF, optimizerB,optimizerF3,schedulerF,schedulerB,schedulerF3, optimizerFB_local, epoch, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losslatent = AverageMeter('LossL', ':.4e')
    corr = AverageMeter('corr', ':6.2f')
    
    top1 = AverageMeter('Acc@1', ':6.2f')
    m1, m2 = top1, corr
    
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, m1, m2],
        prefix=args.method + "Epoch: [{}]".format(epoch))

    if args.gpu is not None:
        
        onehot = torch.FloatTensor(args.batch_size, args.n_classes).cuda(args.gpu, non_blocking=True)
    else:
        onehot = torch.FloatTensor(args.batch_size, args.n_classes).cuda()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data ssading time
        data_time.update(time.time() - end)

        
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        else:
            images = images.cuda()
            target = target.cuda()   

        onehot.zero_()
        onehot.scatter_(1, target.view(target.shape[0], 1), 1)
        
        if ('MNIST' in args.dataset) and args.arche[0:2]!='FC':
            images= images.expand(-1, 1, -1, -1) #images= images.expand(-1, 3, -1, -1)

        

        # # ----- encoder ---------------------
        # switch to train mode
        modelF.train()
        modelB.train()

        optimizerF.zero_grad()
        # compute output
        sigma2 = 0.1
        noisy_images = images + torch.empty_like(images).normal_(mean=0, std=np.sqrt(sigma2))
        latents, output = modelF(noisy_images)

        losse = criterione(output, target)

        # compute gradient and do SGD step
        losse.backward()
        optimizerF.step()



        if args.method == 'BP':
            modelF.load_state_dict(toggle_state_dict_YYtoBP(modelF.state_dict(), modelF.state_dict()))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0].item(), images.size(0))

            
        if 'AsymResLNet' in args.arche:
            modelB.load_state_dict(toggle_state_dict(modelF.state_dict()))
        elif 'asymresnet' in args.arche:
            modelB.load_state_dict(toggle_state_dict(modelF.state_dict(), modelB.state_dict()))


        if any(m in args.method for m in ['FA','BP','BSL']):
            if args.arche.startswith('resnet18c'):
                recons = images # Diabled modelB
            else:
                _, recons = modelB(latents.detach())
            
            gener = recons
            reference = images
            reference = F.interpolate(reference, size=gener.shape[-1])

     
            lossd = criteriond(gener, reference) #+ args.gamma * nn.MSELoss()(gener,torch.zeros_like(gener)) #+ criterione(modelF(pooled), target)

            # measure correlation and record loss
            pcorr = correlation(gener, reference)
            losses.update(lossd.item(), images.size(0))
            corr.update(pcorr, images.size(0))
                

        elif args.method.startswith('FFA'):
            # ----- decoder ------------------  
            noisy_images = images + torch.empty_like(images).normal_(mean=0, std=np.sqrt(sigma2))
  
            latents,  output = modelF(noisy_images)
            

            # switch to train mode
            
            _,recons = modelB(latents.detach()) 

            if 'FFA' in args.method:
                gener = recons
                reference = images
    

            reference = F.interpolate(reference, size=gener.shape[-1])

            lossd = criteriond(gener, reference) #

            # measure correlation and record loss
            pcorr = correlation(gener, reference)
            losses.update(lossd.item(), images.size(0))
            corr.update(pcorr, images.size(0))
                
            optimizerB.zero_grad()
            lossd.backward()
            optimizerB.step()
            #schedulerB.step()

            if 'AsymResLNet' in args.arche:
                modelF.load_state_dict(toggle_state_dict(modelB.state_dict()))
            elif 'asymresnet' in args.arche:
                modelF.load_state_dict(toggle_state_dict(modelB.state_dict(),modelF.state_dict()))


        
        latents, _ = modelF(images)
        _, recons = modelB(latents.detach())

      
        lossL = torch.tensor([0]) 
        losslatent.update(lossL.item(), images.size(0))

        # # compute the accuracy after all training magics    
        # _, output = modelF(images)
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # top1.update(acc1[0].item(), images.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    print(args.method + ': Train avg  * lossd {losses.avg:.3f}'
    .format(losses=losses), flush=True)

    print(args.method + ': Train avg   * Acc@1 {top1.avg:.3f}'
        .format(top1=top1), flush=True)
    
    writer.add_scalar('Train%s/acc1'%args.method, top1.avg , epoch)
    writer.add_scalar('Train%s/corr'%args.method, corr.avg, epoch)
    writer.add_scalar('Train%s/loss'%args.method, losses.avg, epoch)
   
    return modelF, modelB,[top1.avg, corr.avg, losses.avg, losslatent.avg]



def validate(val_loader, modelF, modelB, criterione, criteriond, args, epoch):

    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losslatent = AverageMeter('LossL', ':.4e')
    corr = AverageMeter('corr', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    m1, m2 = top1, corr

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, m1, m2],
        prefix='Test %s: '%args.method)
    
    if args.gpu is not None:
        
        onehot = torch.FloatTensor(args.batch_size, args.n_classes).cuda(args.gpu, non_blocking=True)
    else:
        onehot = torch.FloatTensor(args.batch_size, args.n_classes).cuda()


    # switch to evaluate mode
    modelF.eval()
    modelB.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            else:
                images = images.cuda()
                target = target.cuda()
            
            onehot.zero_()
            onehot.scatter_(1, target.view(target.shape[0], 1), 1)
            
            if ('MNIST' in args.dataset) and args.arche[0:2]!='FC':
                images= images.expand(-1, 1, -1, -1) #images.expand(-1, 3, -1, -1)
            
            # ----- encoder ---------------------
                       
            # compute output
            latents, output = modelF(images)

            losse = criterione(output, target) #+ criteriond(modelB(latents.detach(), switches), images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0].item(), images.size(0))

            # ----- decoder ------------------ 

            latents,  _ = modelF(images)
            if args.arche.startswith('resnet18c'):
                recons = images # Diabled modelB
            else:
                _, recons = modelB(latents.detach())

            if args.method == 'SLTemplateGenerator':
                repb = onehot.detach()#modelB(onehot.detach())
                
                
                repb = repb.view(args.batch_size, args.n_classes, 1, 1)
                        
                        
                _,targetproj = modelB(repb) #, switches

                inputs_avgcat = torch.zeros_like(images)
                for t in torch.unique(target):
                    inputs_avgcat[target==t] = images[target==t].mean(0) #-inputs[target!=t].mean(0)
            
                gener = targetproj
                reference = inputs_avgcat


            
            elif args.method == 'SLError':
                #TODO: check the norm of subtracts
                prob = nn.Softmax(dim=1)(output.detach())
                repb = onehot - prob
                repb = repb.view(args.batch_size, args.n_classes, 1, 1)
                _, gener = modelB(repb.detach())
                reference = images - F.interpolate(recons, size=images.shape[-1])

            elif args.method == 'SLRobust':
                
                prob = nn.Softmax(dim=1)(output.detach())
                repb = onehot - prob
                repb = repb.view(args.batch_size, args.n_classes, 1, 1)
                _, gener = modelB(repb.detach())
                reference = images 

            elif args.method == 'SLErrorTemplateGenerator':
                prob = nn.Softmax(dim=1)(output.detach())
                repb = onehot - prob#modelB(onehot.detach())
                
                
                repb = repb.view(args.batch_size, args.n_classes, 1, 1)
                        
                        
                _,targetproj = modelB(repb) #, switches

                inputs_avgcat = torch.zeros_like(images)
                for t in torch.unique(target):
                    inputs_avgcat[target==t] = images[target==t].mean(0) #-inputs[target!=t].mean(0)
            
                gener = targetproj
                reference = inputs_avgcat
            
            else: #args.method in ['FFA','BP','FA']:
                gener = recons
                reference = images
            
            reference = F.interpolate(reference, size=gener.shape[-1])

            if args.lossfuncB == 'TripletMarginLoss':
                
                shuffled_batch = reference[torch.randperm(reference.shape[0])]
                lossd = criteriond(gener, reference, shuffled_batch)
            else:
                lossd = criteriond(gener, reference) #+ criterione(modelF(pooled), target)       
            

            latents_gener, output_gener = modelF(F.interpolate(recons, size=images.shape[-1]).detach())
            if args.lossfuncB == 'TripletMarginLoss':
                shuffled_images = images[torch.randperm(images.shape[0])]
                latents_shuffled, output = modelF(shuffled_images)

                lossL = criteriond(latents_gener, latents.detach(), latents_shuffled.detach())
            else:
                lossL = criteriond(latents_gener, latents.detach())

            pcorr = correlation(gener, reference)
            losses.update(lossd.item(), images.size(0))
            corr.update(pcorr, images.size(0))
            losslatent.update(lossL.item(), images.size(0))

                

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)


        print('Test avg {method} * lossd {losses.avg:.3f}'
            .format(method=args.method,losses=losses), flush=True)

        # TODO: this should also be done with the ProgressMeter
        print('Test avg  {method} * Acc@1 {top1.avg:.3f}'
            .format(method=args.method, top1=top1), flush=True)
    
        
    writer.add_scalar('Test%s/acc1'%args.method, top1.avg , epoch)
    writer.add_scalar('Test%s/corr'%args.method, corr.avg, epoch)
    writer.add_scalar('Test%s/loss'%args.method,losses.avg,epoch)
            

    return modelF, modelB, [top1.avg, corr.avg, losses.avg, losslatent.avg]


def save_checkpoint(state, is_best, filepath=args.resultsdir ,filename='checkpoint.pth.tar'):
    torch.save(state, filepath+filename)
    if is_best:
        shutil.copyfile(filepath+filename, filepath+'best'+filename)
  


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'




def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# FGSM attack code from pytorch tutorial
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def correlation(output, images):
    """Computes the correlation between reconstruction and the original images"""
    x = output.contiguous().view(-1)
    y = images.contiguous().view(-1) 

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return pearson_corr.item()


# A simple hook class that returns the input and output of a layer during forward/backward pass
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

if __name__ == '__main__':
    main()
 
    