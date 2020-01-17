import argparse
import numpy as np
import os
import json
from utils import helper_functions
import yaml
find = helper_functions.find
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt
import pprint 
pp = pprint.PrettyPrinter(indent=4)
import socket
if socket.gethostname()[0:4] in  ['node','holm','wats']:
    path_prefix = '/rigel/issa/users/Tahereh/Research'
elif socket.gethostname() == 'SYNPAI':
    path_prefix = '/hdd6gig/Documents/Research'
elif socket.gethostname()[0:2] == 'ax':
    path_prefix = '/home/tt2684/Research'

parser = argparse.ArgumentParser(description='Generate figures')
parser.add_argument(
        '--config-file',
        dest='config_file',
        type=argparse.FileType(mode='r'))
parser.add_argument(
        '--eval_robust',
        dest='eval_robust',default=False)
parser.add_argument(
        '--eval_time',
        dest='eval_time',type=str)

args = parser.parse_args()
if args.config_file:
    data = yaml.load(args.config_file)
    delattr(args, 'config_file')
    arg_dict = args.__dict__
    for key, value in data.items():
        setattr(args, key, value)
print(args.resultsdir.split('/Research')[1])

if not(hasattr(args, 'databasedir')):
    project = 'Symbio'#'SYY_MNIST'
    arch = 'E%sD%s'%(args.arche, args.archd)
    args.databasedir  = path_prefix+'/Results/database/%s/%s/%s/'%(project,arch,args.dataset)


methods = ['BP', 'FA', 'SLVanilla','SLRobust',  'SLError']#,'BSL',  'SLErrorTemplateGenerator' 'SLError', 
colors =  {'BP':'k', 'FA':'grey', 'SLVanilla':'r','SLRobust':'salmon',
            'SLError':'orange', 'SLErrorTemplateGenerator':'yellow', 'BSL':'b'}

if args.eval_robust:
    sigma2 = 0.0
    maxitr = 4
    Test_acce = {}
    epsilons = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for method in  methods:#,'SLError','SLTemplateGenerator'
        Test_acce_arr = np.zeros((4, 6))
        for ie , epsilon in enumerate(epsilons):
            # json_name = '%s_%s_eval%s_maxitr4_epsilon%0.1e.json'%(args.runname, method, args.eval_time, epsilon)
            json_name = '%s%s_%s_eval%s_maxitr%d_epsilon%0.1e_noisesigma2%s.json'%(args.databasedir,args.runname, method, args.eval_time, maxitr, epsilon, sigma2)

            print(json_name)
            with open(json_name, 'r') as fp:
                            
                results = json.load(fp)   
                Test_acce_arr[:, ie] = results['Test_acce']     
                Test_acce.update({method:Test_acce_arr})
else:
    Test_acce = {}
    Train_acce = {}
    Test_lossd = {}
    Train_lossd = {}
    Test_corrd = {}
    Train_corrd = {}

    for method in  methods:#,'SLError','SLTemplateGenerator'
        with open('%s%s_%s.json'%(args.databasedir,args.runname, method), 'r') as fp:
                        
            results = json.load(fp)        
            Test_acce.update({method:results['Test_acce']})
            Train_acce.update({method:results['Train_acce']})

            Test_lossd.update({method:results['Test_lossd']})
            Train_lossd.update({method:results['Train_lossd']})

            Test_corrd.update({method:results['Test_corrd']})
            Train_corrd.update({method:results['Train_corrd']})



# args.resultsdir = args.resultsdir.replace('ConvMNIST_playground', 'SYY2020')
resultsdir = path_prefix + args.resultsdir.split('/Research')[1]
pp.pprint(vars(args))
# I used to save results into numpy files

# list_npy = find( '*.npy', resultsdir )
# print(list_npy)
# print(os.listdir(resultsdir))

# results = {}
# for f in list_npy:
#     print(f[:-4])
#     k = f[:-4].split('_')[0] + '_%s'%f[:-4].split('_')[1]
#     r = np.load(resultsdir + f)
#     results.update({k:r})
# print(results.keys())



if args.eval_robust:

    #------ accuracy ------------
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=[20,4])

    for itr in range(4):
        for method in methods:

            axes[itr].plot(epsilons, Test_acce[method][itr], color=colors[method], label=method)

        axes[itr].set_xlabel('epsilon')
        axes[itr].set_ylabel('Discrimination accuracy: percent correct')
        axes[itr].set_title('itr = %d'%itr)
        axes[itr].set_ylim([0,100])
        

        fig.suptitle('%s dataset %s eval%s sigma2=%s'%(args.dataset, args.runname, args.eval_time, sigma2))

        axes[3].legend()

        # axes[1].set_xticks([])
        # axes[1].set_yticks([])
        # axes[1].axis('off')
        # txt= axes[1].text(-0.1,-0.1,str(args),wrap=True, fontsize=8 )
        # txt._get_wrap_line_width = lambda : 200


    fig.savefig(resultsdir+'robust_results_eval%s_sigma2_%s.pdf'%(args.eval_time, sigma2), dpi=200)
    fig.savefig(resultsdir+'robust_results_eval%s_sigma2_%s.png'%(args.eval_time, sigma2), dpi=200)

    plt.clf()


else:

    #------ accuracy ------------
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10,6])

    for method in methods:
            axes[0].plot(Train_acce[method], color=colors[method], ls='--')

            axes[0].plot(Test_acce[method], color=colors[method], label=method)



    # # axes[0].plot(results['SLBP_Train'][:,1],  color='lightgrey', label='SLBP %.2f'%results['SLBP_Test'][0,1])
    # axes[0].plot(results['SLDecoderRobustOutput_Train'][:,1],  color='lightgrey', label='SLDecoderRobustOutput %.2f'%results['SLDecoderRobustOutput_Test'][0,1])

    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('Discrimination accuracy: percent correct')
    axes[0].set_title('%s dataset %s '%(args.dataset, args.runname))
    axes[0].legend()

    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].axis('off')
    txt= axes[1].text(-0.1,-0.1,str(args),wrap=True, fontsize=8 )
    txt._get_wrap_line_width = lambda : 200


    fig.savefig(resultsdir+'acc_results_%depochs.pdf'%args.epochs, dpi=200)
    fig.savefig(resultsdir+'acc_results_%depochs.png'%args.epochs, dpi=200)

    plt.clf()

    print('************')
    for method in methods:

        print('Train ACC','%s:'%method, Train_acce[method][-1], end =" ")
    print('************')
    for method in methods:    
        print('Test ACC','%s:'%method, Test_acce[method][-1], end =" ")
    print('************')
    #------ reconstruction loss ------------
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10,6])

    for method in methods:
            axes[0].plot(Train_lossd[method], color=colors[method], ls='--')

            axes[0].plot(Test_lossd[method], color=colors[method], label=method)

    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('Reconstruction loss: perpixel')
    axes[0].set_title('%s dataset %s '%(args.dataset, args.runname))
    axes[0].legend()

    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].axis('off')
    txt= axes[1].text(-0.1,-0.1,str(args),wrap=True, fontsize=8 )
    txt._get_wrap_line_width = lambda : 200


    fig.savefig(resultsdir+'lossd_results_%depochs.pdf'%args.epochs, dpi=200)
    fig.savefig(resultsdir+'lossd_results_%depochs.png'%args.epochs, dpi=200)

    plt.clf()

    print('************')
    for method in methods:

        print('Train LossD','%s:'%method, Train_lossd[method][-1], end =" ")
    print('************')
    for method in methods:    
        print('Test LossD','%s:'%method, Test_lossd[method][-1], end =" ")
    print('************')


    #------ reconstruction corr ------------
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10,6])

    for method in methods:
            axes[0].plot(Train_corrd[method], color=colors[method], ls='--')

            axes[0].plot(Test_corrd[method], color=colors[method], label=method)

    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('Reconstruction correlation')
    axes[0].set_title('%s dataset %s '%(args.dataset, args.runname))
    axes[0].legend()

    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].axis('off')
    txt= axes[1].text(-0.1,-0.1,str(args),wrap=True, fontsize=8 )
    txt._get_wrap_line_width = lambda : 200


    fig.savefig(resultsdir+'corrd_results_%depochs.pdf'%args.epochs, dpi=200)
    fig.savefig(resultsdir+'corrd_results_%depochs.png'%args.epochs, dpi=200)

    plt.clf()

    print('************')
    for method in methods:

        print('Train corrD','%s:'%method, Train_corrd[method][-1], end =" ")
    print('************')
    for method in methods:    
        print('Test corrD','%s:'%method, Test_corrd[method][-1], end =" ")
    print('************')