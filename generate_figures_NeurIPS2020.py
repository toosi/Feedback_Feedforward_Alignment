import torch
import torchsummary
from torchsummary import summary
from torchvision import models

import copy
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
#%matplotlib inline
import os
import yaml

import socket
if socket.gethostname()[0:4] in  ['node','holm','wats']:
    path_prefix = '/rigel/issa/users/Tahereh/Research'
elif socket.gethostname() == 'SYNPAI':
    path_prefix = '/hdd6gig/Documents/Research'
elif socket.gethostname()[0:2] == 'ax':
    path_prefix = '/home/tt2684/Research'

resultsdir = path_prefix + '/Results/Symbio/Symbio/'


def get_measure_dicts(hashname):
    #'RMSpropRMSpropMNISTAsymResLNet10' #'RMSpropRMSpropMNISTFullyConn' ##'RMSpropRMSpropMNISTFullyConnE150'
    # 'RMSpRMSpMNISTAsymResLNet10BNaffine'
    methods = ['SLVanilla','BP', 'FA']
    colors = {'FA':'k', 'BP':'k', 'SLVanilla':'red'}
    linestyles = {'FA':'--', 'BP':'-', 'SLVanilla':'-'}
    if 'AsymResLNet' in hashname:
        markers = {'FA':'o', 'BP':'o', 'SLVanilla':'o'}
    elif 'FullyConn' in hashname:
        markers = {'FA':'s', 'BP':'s', 'SLVanilla':'s'}

    facecolors = {'FA':'none', 'BP':'k', 'SLVanilla':'red'}

    with open(path_prefix + '/Results/Symbio/runswithhash/%s.txt'%hashname) as f:
        Lines = f.readlines() 

    valid_runnames = []
#     fig, ax = plt.subplots(1,1, figsize=(12,8))
    for l in Lines:
        runname = l.strip('\n')

        configs = yaml.safe_load(open(path_prefix + '/Results/Symbio/Symbio/%s/configs.yml'%runname, 'r'))
        list_csv_paths = []
        for method in methods:
            p = path_prefix + '/Results/Symbio/Symbio/%s/training_results_%s.csv'%(runname, method)
            if os.path.exists(p):
                df = pd.read_csv(p)
                if len(list(df['test_acc'])) == configs['epochs']:
                    list_csv_paths.append(p)
        if len(list_csv_paths) == len(methods):
            valid_runnames.append(runname)
            configs = yaml.safe_load(open(path_prefix + '/Results/Symbio/Symbio/%s/configs.yml'%runname, 'r'))

    print('number of valid runs',len(valid_runnames))

    n_epochs = configs['epochs']
    arch =  configs['arche'][:-1]

    test_init = np.zeros((len(valid_runnames),n_epochs))
    test_acc_dict = {}
    test_corrd_dict = {}
    test_lossd_dict = {}
    for method in methods:
        test_acc_dict[method] = copy.deepcopy(test_init)
        test_corrd_dict[method] = copy.deepcopy(test_init)
        test_lossd_dict[method] = copy.deepcopy(test_init)

    for r, runname in enumerate(valid_runnames):
        configs = yaml.safe_load(open(path_prefix + '/Results/Symbio/Symbio/%s/configs.yml'%runname, 'r'))

        for method in methods:
            p = path_prefix + '/Results/Symbio/Symbio/%s/training_results_%s.csv'%(runname, method)
            df = pd.read_csv(p)
            label = method 
#             pl = ax.plot(df['test_corrd'], label=label, color=colors[method], ls=linestyles[method])
            test_acc_dict[method][r] = list(df['test_acc'])
            test_corrd_dict[method][r] = list(df['test_corrd'])
            test_lossd_dict[method][r] = list(df['test_lossd'])
    return test_acc_dict, test_corrd_dict, test_lossd_dict, configs




def get_measure_dicts_autoencoders(hashname):
    #'RMSpropRMSpropMNISTAsymResLNet10' #'RMSpropRMSpropMNISTFullyConn' ##'RMSpropRMSpropMNISTFullyConnE150'
    # 'RMSpRMSpMNISTAsymResLNet10BNaffine'
    methods = ['BP', 'FA']

    with open(path_prefix + '/Results/Symbio/runswithhash/%s.txt'%hashname) as f:
        Lines = f.readlines() 

    valid_runnames = []
#     fig, ax = plt.subplots(1,1, figsize=(12,8))
    for l in Lines:
        runname = l.strip('\n')

        configs = yaml.safe_load(open(path_prefix + '/Results/Symbio/Symbio/%s/configs.yml'%runname, 'r'))
        list_csv_paths = []
        for method in methods:
            p = path_prefix + '/Results/Symbio/Symbio/%s/training_results_autoencoders_%s.csv'%(runname, method)
            if os.path.exists(p):
                df = pd.read_csv(p)
                if len(list(df['test_acc'])) == configs['epochs']:
                    list_csv_paths.append(p)
        if len(list_csv_paths) == len(methods):
            valid_runnames.append(runname)
            configs = yaml.safe_load(open(path_prefix + '/Results/Symbio/Symbio/%s/configs.yml'%runname, 'r'))

    print('autoencoders number of valid runs',len(valid_runnames))

    n_epochs = configs['epochs']
    arch =  configs['arche'][:-1]

    test_init = np.zeros((len(valid_runnames),n_epochs))
    test_acc_dict = {}
    test_corrd_dict = {}
    test_lossd_dict = {}
    for method in methods:
        test_acc_dict[method] = copy.deepcopy(test_init)
        test_corrd_dict[method] = copy.deepcopy(test_init)
        test_lossd_dict[method] = copy.deepcopy(test_init)

    for r, runname in enumerate(valid_runnames):
        configs = yaml.safe_load(open(path_prefix + '/Results/Symbio/Symbio/%s/configs.yml'%runname, 'r'))

        for method in methods:
            p = path_prefix + '/Results/Symbio/Symbio/%s/training_results_autoencoders_%s.csv'%(runname, method)
            df = pd.read_csv(p)
            label = method 
#             pl = ax.plot(df['test_corrd'], label=label, color=colors[method], ls=linestyles[method])
            test_acc_dict[method][r] = list(df['test_acc'])
            test_corrd_dict[method][r] = list(df['test_corrd'])
            test_lossd_dict[method][r] = list(df['test_lossd'])
    return test_acc_dict, test_corrd_dict, test_lossd_dict, configs

#'RMSpropRMSpropMNISTAsymResLNet10' #'RMSpropRMSpropMNISTFullyConn' ##'RMSpropRMSpropMNISTFullyConnE150'
    # 'RMSpRMSpMNISTAsymResLNet10BNaffine'
hashname= 'RMSpRMSpFaMNISTAsymResLNet10BNaff' #'RMSpRMSpMNISTAsymResLNet10BNaffine'
test_acc_dict, test_corrd_dict, test_lossd_dict, configs = get_measure_dicts(hashname)
test_acc_dict_ae, test_corrd_dict_ae, test_lossd_dict_ae, configs = get_measure_dicts_autoencoders(hashname)
n_epochs = configs['epochs']

methods = ['SLVanilla', 'BP','FA']
colors = {'FA':'k', 'BP':'k', 'SLVanilla':'red'}
linestyles = {'FA':'--', 'BP':'-', 'SLVanilla':'-'}

# Test Acc
fig, ax = plt.subplots(1,1, figsize=(5,3.5))
for method in methods:
    # plot discriminative
    measure = test_acc_dict
    ax.plot(range(n_epochs ), np.median(measure[method], 0), colors[method], label=method, ls=linestyles[method])
#     ax.fill_between(range(n_epochs ), np.median(measure[method], 0)-measure[method].std(0),
#                     np.median(measure[method], 0)+measure[method].std(0),
#                     alpha=0.1, color=colors[method], ls=linestyles[method])
    print(method, measure[method].mean(0)[-1])
    
    if method in ['FA','BP']:
        # plot autoencoders
        measure = test_acc_dict_ae
        ax.plot(range(n_epochs ), np.median(measure[method], 0), 'lightgray', label=method + ' AE', ls=linestyles[method])
#         ax.fill_between(range(n_epochs ), np.median(measure[method], 0)-measure[method].std(0),
#                         np.median(measure[method], 0)+measure[method].std(0),
#                         alpha=0.1, color='gray', ls=linestyles[method])
        print('autoencoder test_acc',method, measure[method].mean(0)[-1])
        
    
ax = plt.gca()
ax.patch.set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(axis='y',direction='out', right=False)
plt.tick_params(axis='x',direction='out', top=False)
ax.set_xlabel('epochs')
ax.set_ylabel('Test accuracy')
ax.legend(loc='center right')

ax.set_title('%s %s (%s)'%(configs['dataset'], arch, hashname))
plt.tight_layout()

savedir = path_prefix + '/Results/Symbio/runswithhash/%s/'%hashname
if not os.path.exists(savedir):
    os.makedirs(savedir)

fig.savefig(savedir + '5lineplots_Testacc_%s.png'%(hashname), dpi=200)
fig.savefig(savedir + '5lineplots_Testacc_%s.pdf'%(hashname), dpi=200)


ax.set_title('Convolutional architecture')


savedir = path_prefix + '/Results/Symbio/runswithhash/Final/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
    
fig.savefig(savedir + '5lineplots_Testcorrd_%s.png'%(hashname), dpi=200)
fig.savefig(savedir + '5lineplots_Testcorrd_%s.pdf'%(hashname), dpi=200)
#-------------------------------------------------------------------------------

# Test correlation
fig, ax = plt.subplots(1,1, figsize=(5,3.5))
for method in methods:
    # plot discriminative
    measure = test_corrd_dict
    ax.plot(range(n_epochs ), np.median(measure[method], 0), colors[method], label=method, ls=linestyles[method])
#     ax.fill_between(range(n_epochs ), np.median(measure[method], 0)-measure[method].std(0),
#                     np.median(measure[method], 0)+measure[method].std(0),
#                     alpha=0.1, color=colors[method], ls=linestyles[method])
    print(method, measure[method].mean(0)[-1])
    
    if method in ['FA','BP']:
        # plot autoencoders
        measure = test_corrd_dict_ae
        ax.plot(range(n_epochs ), np.median(measure[method], 0), 'lightgray', label=method + ' AE', ls=linestyles[method])
#         ax.fill_between(range(n_epochs ), np.median(measure[method], 0)-measure[method].std(0),
#                         np.median(measure[method], 0)+measure[method].std(0),
#                         alpha=0.1, color='gray', ls=linestyles[method])
        print('autoencoder test_corrd',method, measure[method].mean(0)[-1])
        
    
ax = plt.gca()
ax.patch.set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(axis='y',direction='out', right=False)
plt.tick_params(axis='x',direction='out', top=False)
ax.set_xlabel('epochs')
ax.set_ylabel('Test reconstruction correlation')
ax.legend(loc='center right')

ax.set_title('%s %s (%s)'%(configs['dataset'], arch, hashname))
plt.tight_layout()

savedir = path_prefix + '/Results/Symbio/runswithhash/%s/'%hashname
if not os.path.exists(savedir):
    os.makedirs(savedir)
    
fig.savefig(savedir + '5lineplots_Testcorrd_%s.png'%(hashname), dpi=200)
fig.savefig(savedir + '5lineplots_Testcorrd_%s.pdf'%(hashname), dpi=200)


ax.set_title('Convolutional architecture')


savedir = path_prefix + '/Results/Symbio/runswithhash/Final/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
    
fig.savefig(savedir + '5lineplots_Testcorrd_%s.png'%(hashname), dpi=200)
fig.savefig(savedir + '5lineplots_Testcorrd_%s.pdf'%(hashname), dpi=200)
