#%%
import socket
if socket.gethostname()[0:4] in  ['node','holm','wats']:
    path_prefix = '/rigel/issa/users/Tahereh/Research'
elif socket.gethostname() == 'SYNPAI':
    path_prefix = '/hdd6gig/Documents/Research'
elif socket.gethostname()[0:2] == 'ax':
    path_prefix = '/scratch/issa/users/tt2684/Research'
import torch
from torch.utils import data as torchdata
import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import scipy.io as io
import h5py
import pickle
import pandas as pd
import matplotlib.pylab as plt
import torch.nn as nn


robust = True
time_interval = [70, 170]
region = 'V4'
n_epochs = 20
print('********* time_interval: ',time_interval,'robust=%s'%robust, 'region=%s'%region)

if region == 'IT':
    n_neurons = 168
elif region == 'V4':
    n_neurons = 128

class ReadMeta:
    def __init__(self, neuralfeaturesdir):
        self.neuralfeaturesdir = neuralfeaturesdir
    def get_times(self):
        file = open(self.neuralfeaturesdir + 'Data_imagesAndtimes.pickle', 'rb')
        data_times = pickle.load(file)
        file.close()
        times = data_times[1]
        return times
    def get_DF_neu(self):
        DF_neu = pd.read_csv(self.neuralfeaturesdir + 'DataFrame_Neural.csv', sep=",", index_col=False)
        print(self.neuralfeaturesdir )
        return DF_neu
    def get_DF_img(self):
        DF_img = pd.read_csv(self.neuralfeaturesdir + 'DataFrame_Images.csv', sep=",", index_col=False)
        return DF_img
#%%
neuralfeaturesdir = path_prefix+'/features/neural_features/'
datadir = path_prefix+'/Data/DiCarlo/hvm_images_all/'
Meta = ReadMeta(neuralfeaturesdir)
DF_img = Meta.get_DF_img()
DF_neu = Meta.get_DF_neu()
times = Meta.get_times()
#%%
list_im_names = [i[99:] for i in DF_img['filename']]
images = np.zeros((5760,3, 256,256))
for find, f in enumerate(list_im_names):
      im = np.asarray(Image.open(datadir+f))#.convert("L")
      images[find] = np.swapaxes(im, 2,0)
#%%
class HvMImageDataset(torchdata.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, crossval, transform=None, variations=['V3','V6'] ,meta_type='category', seed=0):
        'Initialization'
        print('meta_type can be any of:', DF_img.columns)
        print('Variations is a list of variations e.g. ["V3","V6"]')
        print('crossval should be "Train" or "Test"')
        DF_Var = pd.DataFrame(DF_img[DF_img['var'].isin( variations)])
        inds_Var= np.where(DF_img['var'].isin( variations))[0].tolist()
        self.images_var = images[inds_Var]
        self.transform = transform
        if meta_type in  ['category','obj']:
              categories = np.unique(list(DF_Var[meta_type])).tolist()
              labels = {}
              [labels.update({u:categories.index(list(DF_Var[meta_type])[u])}) for u in range(len(DF_Var))]
        np.random.seed(seed)
        rand_perms = np.random.permutation(len(inds_Var))
        print('first five random gen inds:',rand_perms[:5])
        if crossval== 'Train':
            train_rand_perms  = rand_perms[0:int(0.8*len(inds_Var))]
            self.labels = {k:labels[k] for k in train_rand_perms}
            self.list_IDs = [range(len(inds_Var))[i] for i in train_rand_perms]
        elif crossval== 'Test':
            test_rand_perms  = rand_perms[int(0.8*len(inds_Var)):]
            self.labels = {k:labels[k] for k in test_rand_perms}
            self.list_IDs = [range(len(inds_Var))[i] for i in test_rand_perms]
      #   print(labels)
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        f = list_im_names[ID]
        im = Image.open(datadir+f)
        # Load data and get label
        X = im #self.images_var[ID]
        y = self.labels[ID]
        sample = {'image': X, 'category': y}
        if self.transform:
            im= self.transform(im)
        return im, y, ID
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 6}
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            
        ])
test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            
        ])
train_loader = torchdata.DataLoader(HvMImageDataset(crossval='Train', transform=train_transforms, seed=0), **params)
sample_train = next(iter(train_loader))
test_loader = torchdata.DataLoader(HvMImageDataset(crossval='Test', transform=test_transforms, seed=0), **params)
sample_test = next(iter(test_loader))


mean = 0.
std = 0.
nb_samples = 0.
for data, _, _ in train_loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

for data, _, _ in test_loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(mean, std)
normalize = transforms.Normalize(mean=mean,
                                     std=std)
train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
train_loader = torchdata.DataLoader(HvMImageDataset(crossval='Train', transform=train_transforms, seed=0), **params)
sample_train = next(iter(train_loader))
test_loader = torchdata.DataLoader(HvMImageDataset(crossval='Test', transform=test_transforms, seed=0), **params)
sample_test = next(iter(test_loader))


# %%
import os
import sys
import torch
import torch as ch
import numpy as np
import seaborn as sns
from scipy import stats
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
from robustness import model_utils, datasets
from robustness.tools.vis_tools import show_image_column
from user_constants import DATA_PATH_DICT
# %matplotlib inline

# %%
# Constants
DATA = 'RestrictedImageNet' # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
BATCH_SIZE = 4
NUM_WORKERS = 8
NOISE_SCALE = 20

DATA_SHAPE = 32 if DATA == 'CIFAR' else 224 # Image size (fixed for dataset)
REPRESENTATION_SIZE = 2048 # Size of representation vector (fixed for model)



# %%
dataset_function = getattr(datasets, DATA)
dataset = dataset_function(DATA_PATH_DICT[DATA])

# %%

# Load model
model_kwargs = {
    'arch': 'resnet50',
    'dataset': dataset,
    'resume_path': './models/%s.pt'%DATA
}
model = getattr(torchvision.models,model_kwargs['arch'])(pretrained=True)

model.eval()
# %%

list_children_names = [list(model.named_children())[i][0] for i in range(len(list(model.named_children())))]
layer_index = list_children_names.index('layer4') #'layer4'
model_extract = nn.Sequential(*list(model.children())[:layer_index+1]).cuda()
model_extract.eval()

# %%
sample_test = next(iter(test_loader))
features = model_extract(sample_test[0].cuda())
print(features.shape)

# %%
# get all the images in batches and compute the features


# Neural HvM dataloader
# given list of images, and region and time and trials give neural features

import scipy.io
import numpy as np

class ReadData:

    def __init__(self, datadir, DF_neu, DF_img):
        self.datadir = datadir
        self.DF_neu = DF_neu
    def get_data_allimages(self,region, time_range):
        n_images = 640 + 2560 + 2560
        n_neurons = 168 + 128
        n_trials = 29 + 51 + 47
        # 2017.08.16_hvmdata_mats
        
        times = np.arange(-100, 290, 10).tolist()
        t_ind0 = times.index(time_range[0])
        t_ind1 = times.index(time_range[1])

        time_inds = np.arange(t_ind0,t_ind1)
        Data_image = np.zeros((n_neurons, len(time_inds), n_images))
        Data_trial_V0 = np.zeros((n_neurons, len(time_inds), 640, 29))
        Data_trial_V3 = np.zeros((n_neurons, len(time_inds), 2560, 51))
        Data_trial_V6 = np.zeros((n_neurons, len(time_inds), 2560, 47))

        for ib,b in enumerate(time_inds):
            fname = 'hvm_allrep_t=%02d.mat' % b
            mat = scipy.io.loadmat(self.datadir + 'DiCarlo/2017.08.16_hvmdata_mats/' + fname)
            v0 = np.mean(mat['repdata'][0][0][0], 0)
            v1 = np.mean(mat['repdata'][0][0][1], 0)
            v2 = np.mean(mat['repdata'][0][0][2], 0)
            Data_image[:, ib, :] = np.concatenate((v0, v1, v2)).T
            v0 = mat['repdata'][0][0][0]
            v1 = mat['repdata'][0][0][1]
            v2 = mat['repdata'][0][0][2]
            Data_trial_V0[:, ib, :, :] = np.swapaxes(v0, 0, 2)
            Data_trial_V3[:, ib, :, :] = np.swapaxes(v1, 0, 2)
            Data_trial_V6[:, ib, :, :] = np.swapaxes(v2, 0, 2)
        
        # averaging over times and trials #TODO: trial selection
        Neu_V0 = Data_trial_V0[np.where(self.DF_neu['region'] == region)[0], :, :, :].mean(1).mean(2)
        Neu_V3 = Data_trial_V3[np.where(self.DF_neu['region'] == region)[0], :, :, :].mean(1).mean(2)
        Neu_V6 = Data_trial_V6[np.where(self.DF_neu['region'] == region)[0], :, :, :].mean(1).mean(2)

        Neu_all = np.concatenate((Neu_V0, Neu_V3, Neu_V6), axis=1).T
        return Neu_all

    def get_data(self, Neu_all, image_inds_in_all):
        Neu_images = Neu_all[image_inds_in_all]
        return Neu_images

#%%
def correlation(output, images):
    """Computes the correlation between reconstruction and the original images"""
    x = output.contiguous().view(-1)
    y = images.contiguous().view(-1) 

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return pearson_corr.item()

class Flatten(nn.Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.
    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).

    """
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)

model_map = nn.Sequential(nn.Conv2d(2048, 2048, 3), 
                          nn.Conv2d(2048, 2048, 3), 
                          nn.Conv2d(2048, 2048, 3),
                          Flatten(),
                          nn.Linear(2048, n_neurons)).cuda()
model_map(features).shape
# %%
# train a small convolutional net to map each neuron from featues
# and test to calculate the consistency for both not robust and robust nets
ReadDataClass = ReadData('/home/tt2684/Research/Data/', DF_neu, DF_img)

Neu_all = ReadDataClass.get_data_allimages(region, time_interval)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model_map.parameters(), lr=0.01)

for epoch in range(n_epochs):

    running_loss = 0
    running_corr = 0
    i = 0
    for images, _, inds in train_loader:

        model_features = model_extract(images.cuda())
        neural_features = ReadDataClass.get_data(Neu_all, inds)
        
        output = model_map(model_features)
        target = torch.tensor(neural_features).float().cuda()
        corr = correlation(output, target)

        optimizer.zero_grad()
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_corr += corr
        i +=1
    
    print('%sTrain Epoch %d:'%(robust,epoch),'   ', running_loss/(i+1),'   ',running_corr/(i+1))


#%% 
# Test
i = 0
running_loss = 0
running_corr = 0
for images, _, inds in test_loader:

    model_features = model_extract(images.cuda())
    neural_features = ReadDataClass.get_data(Neu_all, inds)
    
    output = model_map(model_features)
    target = torch.tensor(neural_features).float().cuda()

    loss = criterion(output, target)
    corr = correlation(output, target)
    
    running_loss += loss.item()
    running_corr += corr
    i +=1

print('%s Test'%robust, running_loss/(i+1), running_corr/(i+1))