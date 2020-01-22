 
import argparse
import os
import random
import shutil
import time
import warnings

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
matplotlib.use('agg')
import matplotlib.pylab as plt
from PIL import Image
import matplotlib.image as mpimg

import pprint 
pp = pprint.PrettyPrinter(indent=4)


import pytorch_ssim

from utils import state_dict_utils
from utils import helper_functions

toggle_state_dict = state_dict_utils.toggle_state_dict
# toggle_state_dict = state_dict_utils.toggle_state_dict_resnets
toggle_state_dict_YYtoBP = state_dict_utils.toggle_state_dict_YYtoBP
toggle_state_dict_BPtoYY = state_dict_utils.toggle_state_dict_BPtoYY
from models import custom_models_ResNetLraveled as custom_models

# from models import custom_models as custom_models
# from models import custom_models_Fixup as custom_models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

import socket
if socket.gethostname()[0:4] in  ['node','holm','wats']:
    path_prefix = '/rigel/issa/users/Tahereh/Research'
elif socket.gethostname() == 'SYNPAI':
    path_prefix = '/hdd6gig/Documents/Research'
elif socket.gethostname()[0:2] == 'ax':
    path_prefix = '/home/tt2684/Research'

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument(
        '--config-file',
        dest='config_file',
        type=argparse.FileType(mode='r'))


parser.add_argument('--method', type=str, default='BP', metavar='M',
                    help='method:BP|SYYVanilla|SYYBP|FA|SYYTemplateGenerator')
parser.add_argument('--eval_maxitr', type=int, default=0, 
                    help='maximun of iteration number for loopy inference')
parser.add_argument('--eval_time', type=str, default='Now', 
                    help='evaluation time')
parser.add_argument('--eval_epsilon', type=float, default=0, 
                    help='epsilon in FGSM attack')
parser.add_argument('--eval_sigma2', type=float, default=0, 
                    help='added guassian noise sigma2')
parser.add_argument('--eval_save_sample_images', type=bool, default=False, 
                    help='save sample images')

args = parser.parse_args()
assert args.config_file, 'Please specify a config file path'
if args.config_file:
    data = yaml.load(args.config_file)
    delattr(args, 'config_file')
    arg_dict = args.__dict__
    for key, value in data.items():
        setattr(args, key, value)


# ------- to cope with older configurations -------------------
if 'ConvMNIST_playground' in args.resultsdir:
    arch = 'E%sD%s'%(args.arche, args.archd)
    args.resultsdir = args.resultsdir.replace('ConvMNIST_playground', 'SYY2020')
    args.path_save_model = path_prefix+'/Models/%s_trained/%s/%s/%s/'%(args.dataset,'SYY_MNIST',arch,args.runname)

if not hasattr(args,'tensorboarddir'):
    project = 'SYY2020' 
    
    args.tensorboarddir = path_prefix + '/Results/Tensorboard_runs/runs'+'/%s/'%project +args.runname
    
    args.base_channels = 64
    args.databasedir = path_prefix+'/Results/database/%s/%s/%s/'%(project,arch,args.dataset)
    args.arche = args.arche.replace('NoMaxP','')
    args.archd = args.archd.replace('NoMaxP','')

with open(args.resultsdir+'args.yml', 'w') as outfile:
    
    yaml.dump(vars(args), outfile, default_flow_style=False)

pp.pprint(vars(args))
print(args.method)

writer = SummaryWriter(log_dir=args.tensorboarddir)

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
        args.algorithm = 'FA'
        modelidentifier = 'F'

    modelF = get_model(args.arche, args.gpu, {'algorithm': args.algorithm, 'base_channels':args.base_channels, 'image_channels':image_channels, 'n_classes':args.n_classes}) #, 'woFullyConnected':True
    modelB = get_model(args.archd, args.gpu, {'algorithm': 'FA','base_channels':args.base_channels, 'image_channels':image_channels, 'n_classes':args.n_classes})

    
    # define loss function (criterion) and optimizer
    criterione = nn.CrossEntropyLoss().cuda(args.gpu)                                                                                       
    if args.lossfuncB == 'MSELoss':                                                                                       
        criteriond = nn.MSELoss().cuda(args.gpu)
    elif args.lossfuncB == 'SSIM':
        criteriond = pytorch_ssim.SSIM(window_size = int(input_size/10))

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

    else:

        optimizerF = getattr(torch.optim,args.optimizerF)(list(modelF.parameters()), args.lrF,
                                momentum=args.momentum,
                                weight_decay=args.wdF)
                                

        optimizerB = getattr(torch.optim,args.optimizerB)(modelB.parameters(), args.lrB,
                                    momentum=args.momentum,
                                    weight_decay=args.wdB) 
        
    schedulerF = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerF, 'max', patience=args.patiencee, factor=args.factore)
    schedulerB = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerB, 'max', patience=args.patienced, factor=args.factord)

    # ------load Trained models ---------
    modelF_trained = torch.load(args.path_save_model+'checkpointe_%s.pth.tar'%args.method)['state_dict']
    if args.method.startswith('SL') or args.method == 'BSL':
        modelB_trained = torch.load(args.path_save_model+'checkpointd_%s.pth.tar'%args.method)['state_dict']
    else:
        modelB_trained= toggle_state_dict_BPtoYY(modelF_trained, modelB.state_dict())
    
        
    modelF.load_state_dict(modelF_trained)
    modelB.load_state_dict(modelB_trained)

    
    # Data loading code
    if args.dataset == 'imagenet':
        valdir = os.path.join(args.imagesetdir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(args.imagecrop),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=True)
        # n_classes = 1000

    elif 'CIFAR' in args.dataset:


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_dataset = getattr(datasets, args.dataset)(root=args.imagesetdir, train=False, download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

    elif 'MNIST' in args.dataset:
        
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

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

    cudnn.benchmark = True



    # a json to keep the records
    run_json_dict = {}
    run_json_dict.update(vars(args))

    Test_acce_list  = []
    Test_corrd_list = []
    Test_lossd_list = []
    
    assert (args.eval_sigma2==0) or (args.eval_epsilon == 0), 'Gaussian noise OR adversarial attack, choose one'
    
    if (args.eval_epsilon == 0) :
        # evaluate on validation set
        for itr in range(args.eval_maxitr):

            _, _, test_results = validate(val_loader, modelF, modelB, criterione, criteriond, args, itr, args.eval_sigma2)
            
            acce = test_results[0]
            Test_acce_list.extend( [round(test_results[0],3)])
            Test_corrd_list.extend([round(test_results[1],3)])
            Test_lossd_list.extend([test_results[2]])
            run_json_dict.update({'Test_acce':Test_acce_list})
            run_json_dict.update({'Test_corrd':Test_corrd_list})
            run_json_dict.update({'Test_lossd':Test_lossd_list})

            writer.add_scalar('Test%s/acc1'%args.method, test_results[0], itr)
            writer.add_scalar('Test%s/corr'%args.method, test_results[1], itr)
            writer.add_scalar('Test%s/loss'%args.method, test_results[2], itr)


    elif (args.eval_epsilon>0):

        for itr in range(args.eval_maxitr):

            _, _, test_results = validate_robustness(val_loader, modelF, modelB, criterione, criteriond, args, itr)

            acce = test_results[0]
            Test_acce_list.extend( [round(test_results[0],3)])
            Test_corrd_list.extend([round(test_results[1],3)])
            Test_lossd_list.extend([test_results[2]])
            run_json_dict.update({'Test_acce':Test_acce_list})
            run_json_dict.update({'Test_corrd':Test_corrd_list})
            run_json_dict.update({'Test_lossd':Test_lossd_list})

            writer.add_scalar('Test%s/acc1'%args.method, test_results[0], itr)
            writer.add_scalar('Test%s/corr'%args.method, test_results[1], itr)
            writer.add_scalar('Test%s/loss'%args.method, test_results[2], itr)
    
    
    

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        json_name =  '%s%s_%s_eval%s_maxitr%d_epsilon%0.1e_noisesigma2%s.json'%(args.databasedir,args.runname, args.method, args.eval_time, args.eval_maxitr, args.eval_epsilon, args.eval_sigma2)
        print('json saved at: ',json_name)
        with open(json_name, 'w') as fp:
            run_json_dict.update(arg_dict)
            json.dump(run_json_dict, fp, indent=4, sort_keys=True)        
            fp.write("\n")
    print('eval_time argument to generate_figure:',args.eval_time)



def validate(val_loader, modelF, modelB, criterione, criteriond, args, itr, sigma2):

    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

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

    not_saved = True
    not_saved_itr = True

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

            if 'MNIST' in args.dataset  and args.arche[0:2]!='FC':
                images= images.expand(-1, 1, -1, -1) #images.expand(-1, 3, -1, -1)
            
            # ----- encoder ---------------------
            images_noisy = images + torch.empty_like(images).normal_(mean=0, std=np.sqrt(sigma2)).cuda()
            
            if (args.eval_save_sample_images) and not_saved:
                not_saved = False
                helper_functions.generate_sample_images(images, target, title='original', param_dict={}, args=args)
                helper_functions.generate_sample_images(images_noisy, target, title='noisy inputs', param_dict={'sigma2':sigma2}, args=args)

            # compute output
            latents, output = modelF(images_noisy)
            # ----- decoder ------------------ 
            recons_before_interpolation = modelB(latents.detach()) 
            recons = F.interpolate(recons_before_interpolation, size=images.shape[-1])
            
            if args.method == 'SLTemplateGenerator':
                repb = onehot.detach()#modelB(onehot.detach())
                repb = repb.view(args.batch_size, args.n_classes, 1, 1)
                        
                        
                targetproj = modelB(repb) #, switches

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
                gener = modelB(repb.detach())
                reference = images - F.interpolate(recons, size=images.shape[-1])

            elif args.method == 'SLRobust':
                
                prob = nn.Softmax(dim=1)(output.detach())
                repb = onehot - prob
                repb = repb.view(args.batch_size, args.n_classes, 1, 1)
                gener = modelB(repb.detach())
                reference = images 

            elif args.method == 'SLErrorTemplateGenerator':
                prob = nn.Softmax(dim=1)(output.detach())
                repb = onehot - prob#modelB(onehot.detach())
                
                
                repb = repb.view(args.batch_size, args.n_classes, 1, 1)
                        
                        
                targetproj = modelB(repb) #, switches

                inputs_avgcat = torch.zeros_like(images)
                for t in torch.unique(target):
                    inputs_avgcat[target==t] = images[target==t].mean(0) #-inputs[target!=t].mean(0)
            
                gener = targetproj
                reference = inputs_avgcat

            else:# args.method in ['SLVanilla','BP','FA']:
                gener = recons
                reference = images

        

            for _ in range(itr):
                # compute output
                latents, output = modelF(gener.detach())

                # ----- decoder ------------------ 
                recons_before_interpolation = modelB(latents.detach()) 
                recons = F.interpolate(recons_before_interpolation, size=images.shape[-1])

                if args.method == 'SLTemplateGenerator':
                    repb = onehot.detach()#modelB(onehot.detach())
                    repb = repb.view(args.batch_size, args.n_classes, 1, 1)
                            
                            
                    targetproj = modelB(repb) #, switches

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
                    gener = modelB(repb.detach())
                    reference = images - F.interpolate(recons, size=images.shape[-1])

                elif args.method == 'SLRobust':
                    
                    prob = nn.Softmax(dim=1)(output.detach())
                    repb = onehot - prob
                    repb = repb.view(args.batch_size, args.n_classes, 1, 1)
                    gener = modelB(repb.detach())
                    reference = images 

                elif args.method == 'SLErrorTemplateGenerator':
                    prob = nn.Softmax(dim=1)(output.detach())
                    repb = onehot - prob#modelB(onehot.detach())
                    
                    
                    repb = repb.view(args.batch_size, args.n_classes, 1, 1)
                    targetproj = modelB(repb) #, switches

                    inputs_avgcat = torch.zeros_like(images)
                    for t in torch.unique(target):
                        inputs_avgcat[target==t] = images[target==t].mean(0) #-inputs[target!=t].mean(0)
            
                    gener = targetproj
                    reference = inputs_avgcat
                
                else: # args.method in ['SLVanilla','BP','FA']:
                    gener = recons
                    reference = images
            
            
            if (args.eval_save_sample_images) and not_saved_itr:
                not_saved_itr = False
                helper_functions.generate_sample_images(gener, target, title='gener by '+args.method, param_dict={'sigma2':sigma2, 'itr':itr}, args=args)

            
            # measure accuracy and record loss
            losse = criterione(output, target) 
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0].item(), images.size(0))

            # measure correlation and record loss
            reference = F.interpolate(reference, size=gener.shape[-1])
            lossd = criteriond(gener, reference) #+ criterione(modelF(pooled), target)

            pcorr = correlation(gener, reference)
            losses.update(lossd.item(), images.size(0))
            corr.update(pcorr, images.size(0))
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)


        print('Test avg {method} sigma2 {sigma2} itr: {itr} * lossd {losses.avg:.3f}'
            .format(method=args.method, sigma2=sigma2, itr=itr,losses=losses), flush=True)

        # TODO: this should also be done with the ProgressMeter
        print('Test avg  {method} sigma2 {sigma2} itr: {itr} * Acc@1 {top1.avg:.3f}'
            .format(method=args.method, sigma2=sigma2, itr=itr, top1=top1), flush=True)
              

    return modelF, modelB, [top1.avg, corr.avg, losses.avg]




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


def validate_robustness(val_loader, modelF, modelB, criterione, criteriond, args, itr):

    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

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

    not_saved = True
    not_saved_itr = True

    
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

        
        if 'MNIST' in args.dataset and args.arche[0:2]!='FC':
            images= images.expand(-1, 1, -1, -1) #images.expand(-1, 3, -1, -1)
        
        images.requires_grad = True

        # ----- encoder ---------------------
                    
        # compute output
        _, output = modelF(images)

        # # If the initial prediction is wrong, dont bother attacking, just move on
        # init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability     
        # if init_pred.item() != target.item():
        #     continue

        losse = criterione(output, target) #+ criteriond(modelB(latents.detach(), switches), images)

        # Zero all existing gradients
        modelF.zero_grad()

        # Calculate gradients of model in backward pass
        losse.backward()

        # Collect datagrad
        images_grad = images.grad.data

        # Call FGSM Attack
        perturbed_images = fgsm_attack(images, args.eval_epsilon, images_grad)
        latents, output = modelF(perturbed_images)

        if (args.eval_save_sample_images) and not_saved:
                not_saved = False
                helper_functions.generate_sample_images(images.detach(), target, title='original', param_dict={}, args=args)
                helper_functions.generate_sample_images(perturbed_images.detach(), target, title='perturbed inputs by %s'%args.method, param_dict={'epsilon':args.eval_epsilon}, args=args)

        losse = criterione(output, target) #+ criteriond(modelB(latents.detach(), switches), images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0].item(), images.size(0))

        # ----- decoder ------------------ 
        recons = modelB(latents.detach())

        
        if args.method == 'SLTemplateGenerator':
            repb = onehot.detach()#modelB(onehot.detach())           
            repb = repb.view(args.batch_size, args.n_classes, 1, 1)                   
                    
            targetproj = modelB(repb) #, switches

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
            gener = modelB(repb.detach())
            reference = images - F.interpolate(gener, size=images.shape[-1])

        elif args.method == 'SLRobust':
            
            prob = nn.Softmax(dim=1)(output.detach())
            repb = onehot - prob
            repb = repb.view(args.batch_size, args.n_classes, 1, 1)
            gener = modelB(repb.detach())
            reference = images 

        elif args.method == 'SLErrorTemplateGenerator':
            prob = nn.Softmax(dim=1)(output.detach())
            repb = onehot - prob#modelB(onehot.detach())
            
            
            repb = repb.view(args.batch_size, args.n_classes, 1, 1)
                    
                    
            targetproj = modelB(repb) #, switches

            inputs_avgcat = torch.zeros_like(images)
            for t in torch.unique(target):
                inputs_avgcat[target==t] = images[target==t].mean(0) #-inputs[target!=t].mean(0)
        
            gener = targetproj
            reference = inputs_avgcat
        
        else: #args.method in ['SLVanilla','BP','FA']:
            gener = recons
            reference = images
        
        

        for _ in range(itr):

            # compute output
            latents, output = modelF(gener.detach())
            # ----- decoder ------------------ 
            recons = modelB(latents.detach())

            if args.method == 'SLTemplateGenerator':

                repb = onehot.detach()#modelB(onehot.detach())    
                repb = repb.view(args.batch_size, args.n_classes, 1, 1)
                                  
                targetproj = modelB(repb) #, switches

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
                gener = modelB(repb.detach())
                reference = images - F.interpolate(recons, size=images.shape[-1])

            elif args.method == 'SLRobust':
                
                prob = nn.Softmax(dim=1)(output.detach())
                repb = onehot - prob
                repb = repb.view(args.batch_size, args.n_classes, 1, 1)
                gener = modelB(repb.detach())
                reference = images 

            elif args.method == 'SLErrorTemplateGenerator':

                prob = nn.Softmax(dim=1)(output.detach())
                repb = onehot - prob#modelB(onehot.detach())
                repb = repb.view(args.batch_size, args.n_classes, 1, 1)
                        
                        
                targetproj = modelB(repb) #, switches

                inputs_avgcat = torch.zeros_like(images)
                for t in torch.unique(target):
                    inputs_avgcat[target==t] = images[target==t].mean(0) #-inputs[target!=t].mean(0)
            
                gener = targetproj
                reference = inputs_avgcat
            
            else: # args.method in ['SLVanilla','BP','FA']:
                gener = recons
                reference = images
            
                    
        if (args.eval_save_sample_images) and not_saved_itr:
            not_saved_itr = False
            helper_functions.generate_sample_images(gener.detach(), target, title='gener by '+args.method, param_dict={'epsilon':args.eval_epsilon, 'itr':itr}, args=args)

        
        # measure accuracy and record loss
        losse = criterione(output, target) #+ criteriond(modelB(latents.detach(), switches), images)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0].item(), images.size(0))
        
        reference = F.interpolate(reference, size=gener.shape[-1])
        lossd = criteriond(gener, reference) #+ criterione(modelF(pooled), target)
        # measure correlation and record loss
        pcorr = correlation(gener, reference)
        losses.update(lossd.item(), images.size(0))
        corr.update(pcorr, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)


    print('Test avg {method} itr{itr} epsilon{epsilon} * lossd {losses.avg:.3f}'
        .format(method=args.method, itr=itr, epsilon=args.eval_epsilon,losses=losses), flush=True)

    # TODO: this should also be done with the ProgressMeter
    print('Test avg  {method} itr{itr} epsilon{epsilon} * Acc@1 {top1.avg:.3f}'
        .format(method=args.method, itr=itr, epsilon=args.eval_epsilon, top1=top1), flush=True)
            

    return modelF, modelB, [top1.avg, corr.avg, losses.avg]


def save_checkpoint(state, is_best, filepath=args.path_save_model ,filename='checkpoint.pth.tar'):
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


def adjust_learning_rate(scheduler, value):
    scheduler.step(value)
    # for param_group in optimizer.param_groups:
    #     lr = param_group['lr']
    # """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""


    # lr = lr * (drop ** (epoch // scale))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    

    #sd = scheduler.state_dict()
    #sd['base_lr'] = lr
    #sd['max_lr'] = lr*factor
    #scheduler.load_state_dict(sd)   


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def correlation(output, images):
    """Computes the correlation between reconstruction and the original images"""
    x = output.contiguous().view(-1)
    y = images.contiguous().view(-1) 

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return pearson_corr.item()
    


if __name__ == '__main__':
    main()