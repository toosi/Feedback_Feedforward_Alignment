


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
        axes[im].imshow(implot)
        axes[im].axis('off')
        if args.dataset == 'CIFAR10':
            axes[im].set_title('C=%s'%classes[target[im].item()])
        else: 
            axes[im].set_title('C=%s'%target[im].item())
    fig.suptitle(args.runname + ', %s: %s'%(title, param_dict), fontsize=8)
    fig.savefig(args.resultsdir+'%s_samples_eval%s_%s_%s.png'%(title, args.eval_time, args.method, param_dict), dpi=200)
    fig.savefig(args.resultsdir+'%s_samples_eval%s_%s_%s.pdf'%(title, args.eval_time, args.method, param_dict), dpi=200)
    print('%s_samples_eval%s_%s_%s.png saved at %s'%(title, args.eval_time, args.method, param_dict, args.resultsdir))
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
    fig.suptitle(args.runname + ' method=%s, %s'%(args.method, title), fontsize=title_font)
    fig.savefig(args.resultsdir+'RDMs%s_eval%s_%s.png'%(title, args.eval_time, args.method), dpi=200)
    fig.savefig(args.resultsdir+'RDMs%s_eval%s_%s.pdf'%(title, args.eval_time, args.method), dpi=200)
    print('RDM %s_eval%s_%s.png saved at %s'%(title, args.eval_time, args.method, args.resultsdir))
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
