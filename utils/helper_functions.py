


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
        axes[im].set_title('C=%s'%target[im].item())
    fig.suptitle(args.runname + ', %s: %s'%(title, param_dict), fontsize=8)
    fig.savefig(args.resultsdir+'%s_samples_eval%s_%s_%s.png'%(title, args.eval_time, args.method, param_dict), dpi=200)
    fig.savefig(args.resultsdir+'%s_samples_eval%s_%s_%s.pdf'%(title, args.eval_time, args.method, param_dict), dpi=200)
    print('%s_samples_eval%s_%s_%s.png saved at %s'%(title, args.eval_time, args.method, param_dict, args.resultsdir))
    plt.clf()


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