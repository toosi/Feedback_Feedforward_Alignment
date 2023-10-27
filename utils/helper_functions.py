import yaml 
import json
import copy

'''
    For the given path, get the List of all files in the directory tree 
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles



def get_measure_dicts_json(hashname_disc, n_epochs_disc, path_prefix, resultsdir):
#     fig, ax = plt.subplots(1,1, figsize=(5,3.5))
    #'RMSpropRMSpropMNISTAsymResLNet10' #'RMSpropRMSpropMNISTFullyConn' ##'RMSpropRMSpropMNISTFullyConnE150'
    # 'RMSpRMSpMNISTAsymResLNet10BNaffine'
    methods = ['SLVanilla','BP', 'FA']
    colors = {'FA':'k', 'BP':'k', 'SLVanilla':'red'}
    linestyles = {'FA':'--', 'BP':'-', 'SLVanilla':'-'}
    if 'AsymResLNet' in hashname_disc:
        markers = {'FA':'o', 'BP':'o', 'SLVanilla':'o'}
    elif 'FullyConn' in hashname_disc:
        markers = {'FA':'s', 'BP':'s', 'SLVanilla':'s'}

    facecolors = {'FA':'none', 'BP':'k', 'SLVanilla':'red'}

    with open(path_prefix + '/Results/Symbio/runswithhash/%s.txt'%hashname_disc) as f:
        Lines = f.readlines() 

    valid_runnames = []
#     fig, ax = plt.subplots(1,1, figsize=(12,8))
    for l in Lines:
        runname = l.strip('\n')

        configs = yaml.safe_load(open(resultsdir + '/%s/configs.yml'%runname, 'r'))
        list_json_paths = []
        for method in methods:
            p = resultsdir + '/%s/run_json_dict_%s.json'%(runname, method)
            if os.path.exists(p):
                
                with open(p,"r") as jfile:
                    dj = json.load(jfile)
                
                if len(dj['Test_acce']) >= n_epochs_disc: #configs['epochs']:
                    list_json_paths.append(p)
                else:
                    print(len(dj['Test_acce']),n_epochs_disc, configs['epochs'])
                    
        
        if len(list_json_paths) == len(methods):
            valid_runnames.append(runname)
            configs = yaml.safe_load(open(resultsdir + '/%s/configs.yml'%runname, 'r'))
        else:
            print(list_json_paths)

    print('number of valid runs discriminative',len(valid_runnames))

#     n_pochs = 370 #configs['epochs']
    arch =  configs['arche'][:-1]

    test_init = np.zeros((len(valid_runnames),n_epochs_disc))
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
            p = resultsdir + '%s/run_json_dict_%s.json'%(runname, method)
            with open(p,"r") as jfile:
                dj = json.load(jfile)
            label = method 
#             ax.plot(dj['lrF'], label=label, color=colors[method], ls=linestyles[method])
            test_acc_dict[method][r] = dj['Test_acce'][0:n_epochs_disc]
            test_corrd_dict[method][r] = dj['Test_corrd'][0:n_epochs_disc]
            test_lossd_dict[method][r] = dj['Test_lossd'][0:n_epochs_disc]
    return test_acc_dict, test_corrd_dict, test_lossd_dict, configs, valid_runnames




def get_measure_dicts_ae_json(hashname_ae, n_epochs_ae, path_prefix, resultsdir):
    #'RMSpropRMSpropMNISTAsymResLNet10' #'RMSpropRMSpropMNISTFullyConn' ##'RMSpropRMSpropMNISTFullyConnE150'
    # 'RMSpRMSpMNISTAsymResLNet10BNaffine'
#     fig, ax = plt.subplots(1,1, figsize=(5,3.5))
    methods = ['BP', 'FA']
    colors = {'FA':'k', 'BP':'k'}
    linestyles = {'FA':'--', 'BP':'-'}

    with open(path_prefix + '/Results/Symbio/runswithhash/%s.txt'%hashname_ae) as f:
        Lines = f.readlines() 

    valid_runnames = []
#     fig, ax = plt.subplots(1,1, figsize=(12,8))
    for l in Lines:
        runname = l.strip('\n')

        configs = yaml.safe_load(open(resultsdir + '/%s/configs.yml'%runname, 'r'))
        list_json_paths = []
        for method in methods:
            p = resultsdir + '/%s/run_json_dict_autoencoder_%s.json'%(runname, method)
            if os.path.exists(p):
                
                with open(p,"r") as jfile:
                    dj = json.load(jfile)
                
                if len(dj['Test_acce']) >= n_epochs_ae: #configs['epochs']:
                    list_json_paths.append(p)
                else:
                    print('auto',len(dj['Test_acce']), n_epochs_ae, configs['epochs'])
                    
        
        if len(list_json_paths) == len(methods):
            valid_runnames.append(runname)
            configs = yaml.safe_load(open(path_prefix + '/Results/Symbio/Symbio/%s/configs.yml'%runname, 'r'))

    print('number of valid runs autoencoder',len(valid_runnames))

#     n_pochs = 370 #configs['epochs']
    arch =  configs['arche'][:-1]

    test_init = np.zeros((len(valid_runnames),n_epochs_ae))
    test_acc_dict = {}
    test_corrd_dict = {}
    test_lossd_dict = {}
    for method in methods:
        test_acc_dict[method] = copy.deepcopy(test_init)
        test_corrd_dict[method] = copy.deepcopy(test_init)
        test_lossd_dict[method] = copy.deepcopy(test_init)

    for r, runname in enumerate(valid_runnames):
        configs = yaml.safe_load(open(resultsdir + '/%s/configs.yml'%runname, 'r'))

        for method in methods:
            p = resultsdir + '/%s/run_json_dict_autoencoder_%s.json'%(runname, method)
            with open(p,"r") as jfile:
                dj = json.load(jfile)
            label = method 
#             ax.plot(dj['lrF'], label=label, color=colors[method], ls=linestyles[method])
            test_acc_dict[method][r] = dj['Test_acce'][0:n_epochs_ae]
            test_corrd_dict[method][r] = dj['Test_corrd'][0:n_epochs_ae]
            test_lossd_dict[method][r] = dj['Test_lossd'][0:n_epochs_ae]
            if method == 'FA' and np.all(test_corrd_dict[method][r]<0.9):
                print('inja',runname, len(dj['Test_corrd']))
    return test_acc_dict, test_corrd_dict, test_lossd_dict, configs, valid_runnames









class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model, ind_output):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()
        self.ind_output = ind_output

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1]
        # if wrapped in DataParallel (not sure):
        # first_layer = list(list(self.model._modules.items())[0][1]._modules.items())[0][1]

        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, onehot):
        # Forward
        model_output = self.model(input_image)[self.ind_output]

        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        # one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        # one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=onehot)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients
        return gradients_as_arr


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
        
    
import os, fnmatch
def find(pattern, path):
    """ this code match a pattern to find a file in a folder  source: stacko"""
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                # result.append(os.path.join(root, name))
                result.append(name)
    return result

# find('*.txt', '/path/to/dir')

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def generate_sample_images(images, target, title, param_dict, args):

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=[6,2])
    for im in range(5):
        implot = images[im].cpu().numpy()
        # if the image has three channels like in CIFAR
        if images.shape[1] == 3:
        
            implot = np.swapaxes(implot,0,2)        
            implot = (implot - np.min(implot))/np.ptp(implot)                
        else:
            implot = implot.squeeze()
        if args.dataset == 'MNIST':
            axes[im].imshow(implot, cmap='gray')
        else:
            axes[im].imshow(implot)
        axes[im].axis('off')
        if args.dataset == 'CIFAR10':
            axes[im].set_title('C=%s'%classes[target[im].item()])
        else: 
            axes[im].set_title('C=%s'%target[im].item())
    
    fig.suptitle(args.runname + ', %s: %s'%(title, param_dict), fontsize=8)
    if not os.path.exists(args.resultsdir+'evaluate'):
        os.makedirs(args.resultsdir+'evaluate')
    
    fig.savefig(args.resultsdir+'evaluate/%s_samples_eval%s_%s_%s.png'%(title, args.eval_time, args.method, param_dict), dpi=200)
    fig.savefig(args.resultsdir+'evaluate/%s_samples_eval%s_%s_%s.pdf'%(title, args.eval_time, args.method, param_dict), dpi=200)
    print('%s_samples_eval%s_%s_%s.png saved at %sevaluate/'%(title, args.eval_time, args.method, param_dict, args.resultsdir))
    plt.clf()


import torchvision
import torch
def load_dataset(data_path):

    """
    load imagesets from pytorch-friendly organized image  folders
    """
    
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader

def correlation(output, images):
    """Computes the correlation between reconstruction and the original images"""
    x = output.contiguous().view(-1)
    y = images.contiguous().view(-1) 

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return pearson_corr.item()



def plot_RDMs(tensor, n_samples, title, args):

    implot = np.zeros((tensor.shape[0], tensor.shape[0]))
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[0]):
            implot[i,j]= correlation(tensor[i],tensor[j])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6,5])

    title_font = 13
    axis_font = 11

    im = ax.matshow(implot, vmin=0, vmax=1, origin='lower', cmap=plt.cm.get_cmap('Spectral_r'))
    # ax.axis('off')
    ax.set_xticks(np.linspace(args.n_samples_RDM/2,args.n_classes*args.n_samples_RDM-args.n_samples_RDM/2,args.n_classes))
    ax.set_xticklabels(range(args.n_classes), fontsize=axis_font)

    ax.set_yticks(np.linspace(args.n_samples_RDM/2,args.n_classes*args.n_samples_RDM-args.n_samples_RDM/2,args.n_classes))
    ax.set_yticklabels(range(args.n_classes), fontsize=axis_font)
    ax.set_ylim(ax.get_ylim()[::-1])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        
    fig.colorbar(im, cax=cbar_ax)
    # axes.set_title('C=%s'%target[im].item())
    if not os.path.exists(args.resultsdir+'evaluate'):
        os.makedirs(args.resultsdir+'evaluate')
        
    fig.suptitle(args.runname + ' method=%s, %s'%(args.method, title), fontsize=title_font)
    fig.savefig(args.resultsdir+'evaluate/RDMs%s_eval%s_%s.png'%(title, args.eval_time, args.method), dpi=200)
    fig.savefig(args.resultsdir+'evaluate/RDMs%s_eval%s_%s.pdf'%(title, args.eval_time, args.method), dpi=200)
    print('RDM %s_eval%s_%s.png saved at %sevaluate/'%(title, args.eval_time, args.method, args.resultsdir))
    plt.clf()

    return implot


# def zscore(tensor):

#     mean = tensor.mean(dim=0, keepdim=True)
#     print(mean.shape, mean)
#     std = tensor.mean(dim=0, keepdim=True)

#     mean = mean.repeat(100,1)
#     std = std.repeat(100,1)
    
#     tensor = (tensor - mean)/std
#     print(tensor)
#     return tensor

def generate_spectrum(tensor1,title1,tensor2, title2, args):

    array1 = tensor1.view(tensor1.shape[0], -1).cpu().numpy()
    array2 = tensor2.view(tensor2.shape[0], -1).cpu().numpy()
    from sklearn.decomposition import PCA

    n_components = 1000

    pca = PCA(n_components=n_components)

    def get_ev_ratios(array):
        pca.fit(array)
        ev_ratios = pca.explained_variance_ratio_
        return ev_ratios

    ev_ratios1 = get_ev_ratios(array1)
    ev_ratios2 = get_ev_ratios(array2)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6,5])
    ax.loglog(range(n_components), ev_ratios1,color='b', label=title1)
    ax.loglog(range(n_components), ev_ratios2,color='r', label=title2)
    ax.loglog(np.arange(1,n_components), [1/i for i in np.arange(1, n_components)], label='1/f')
    ax.set_xlabel('PC dimensions')
    ax.set_ylabel('explained variance ratio')
    ax.legend()

    from scipy.optimize import curve_fit
    def f_linear(x, a, b):
        return a*x+b

    a, b = curve_fit(f_linear, np.log(np.arange(1, n_components)), np.log(ev_ratios1[1:]))[0]
    print(a, 'a')
    ax.loglog(range(n_components), np.array([a*i+b for i in range(n_components)]), ls='--', color='b')
    ax.text(1,10e-9,'%s a=%0.2f'%(title1,a))

    a, b = curve_fit(f_linear, np.log(np.arange(1, n_components)), np.log(ev_ratios2[1:]))[0]
    print(a, 'a')
    ax.text(1,10e-10,'%s a=%0.2f'%(title2,a))
    ax.loglog(range(n_components), np.array([a*i+b for i in range(n_components)]), ls='--', color='r')
    ax.set_title(args.runname + ' method=%s, n_comp=%d'%(args.method, n_components), fontsize=13)
    fig.savefig(args.resultsdir+'eigen_spectrum%sand%s_eval%s_%s_n%d.png'%(title1,title2, args.eval_time, args.method, n_components), dpi=200)
