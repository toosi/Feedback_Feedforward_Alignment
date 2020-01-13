#!/bin/sh
#SBATCH --job-name=Toy # The job name.
#SBATCH -o /scratch/issa/users/tt2684/Research/Report/output_ToyModels.%j.out # STDOUT
#SBATCH -c 16
#SBATCH --gres=gpu:4
#SBATCH --mem=24gb
#SBATCH 

printf "********************************************************** \n"
printf " replicate AsymResNet track_running False \n"
# printf " Finally test is not that bad, even when using track_mean=True \n"
# printf " 1) I excluded bn form prameters to be exchanged in toggle_state_dict_FF_to_FB \n"
# printf " 2) using the latent before forward weight update \n"
# printf " the new problem is overfitting  \n"
# printf " Examining BN, testing new toggle_FFtoFB , Resnet10 MNIST optimizers\n"
printf "********************************************************** \n"


now=$(date +'%Y-%m-%d_%H-%M')
module load anaconda3-2019.03

# source activate /axsys/home/tt2684/conda-envs/pytorch_tensorflow_cuttingedge
# source activate pytorch_tensorflow_SYNPAI
# source activate /axsys/home/tt2684/conda-envs/pytorch_tensorflow_SYNPAI_CUDA10
source activate /axsys/home/tt2684/conda-envs/pytorch_tensorflow_SYNPAI_CUDA100cudnn75

####Command to execute Python program
config=0
# python -u create_config.py -dataset CIFAR10 -j 16 --base_channels 64 --batch-size 256 --epoch 400 -ae AsymResLNet10F -ad AsymResLNet10B --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --patiencee 50 --patienced 40 -p 100; config=1  #

if [ $config == 0 ]
    then
    runname='Jan08-14-54_b479c8f095_460'  #'Dec30-16-38_04f1282767_1'   #'Dec29-10-13_fd73bdce0b_2'  #Dec06-23-07_81c6b0578b_2' #'Dec06-12-23_81c6b0578b_2'   #'Dec06-09-41_81c6b0578b_2' #'Dec06-12-09_81c6b0578b_2'
    # configpath="/axsys/scratch/issa/users/tt2684/Research/Results/Toy_models/ConvMNIST_playground/$runname/configs.yml"
    configpath="/axsys/scratch/issa/users/tt2684/Research/Results/Toy_models/SYY2020/$runname/configs.yml"

    # python -u main_train.py --method SYYVanilla  --config-file $configpath
    # python -u main_train.py --method FA  --config-file $configpath
    # python -u main_train.py --method BP  --config-file $configpath
    # python -u main_train.py --method SYYDecoderRobustOutput  --config-file $configpath
    # python -u main_train.py --method SYYError  --config-file $configpath
    # python -u main_train.py --method SYYDecoderRobustOnehot  --config-file $configpath
    # python -u main_train.py --method SYYTemplateGenerator  --config-file $configpath

    # python -u main_train_SYYtarget.py --method SYYVanilla  --config-file $configpath


    python -u generate_figures.py --config-file $configpath



    # python -u main_evaluate_withimages.py --eval_save_sample_images True  --method $method --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time $now

    # ## robustness evaluation
    # methods='BP FA SYYVanilla'
    # for method in $methods
    # do 
    # for eval_epsilon in `seq 0.0 0.2 1.0`
    # do 
    # for eval_sigma2 in `seq 0.0 0.5 2`
    # do
    # echo $method
    # echo $eval_sigma2
    # echo $eval_epsilon
    # python -u main_evaluate.py --method $method --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time $now 
    # done
    # done
    # done
    
    # python -u generate_figures.py --eval_robust True --eval_time 2020-01-09_17-07 --config-file $configpath
    
    fi
#------------ history
# python -u main_compare.py  -j 48 -dataset MNIST -b 512 -p 50 --lre 1e-4 --lrd 1e-4   -ae FCDiscrNet -ad FCGeneNet -time $now  --dist-url 'tcp://127.0.0.1:6008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 
# python -u create_config.py -ae AsymResLNet10FNoMaxP -ad AsymResLNet10BNoMaxP --optimizere 'SGD' --hash 2 --batch-size 256 --epoch 20 --lrF 1e-2 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 # --step 50 --drop 0.5 
# python -u create_config.py -dataset MNIST -ae fixup_resnet14 -ad fixup_resnetT14 --optimizerF 'SGD' --hash 2 --batch-size 256 --epoch 20 --lrF 1e-2 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 -p 10 # --step 50 --drop 0.5 

## YY: python -u main_dual.py  -j 48 --train_scheduler alternating  -ae AsymResNet18F -ad AsymResNet18B  -b 512 --evaluate --resume /scratch/issa/users/tt2684/Research/Models/imagenet_trained/ForwardResidualProp/alternating/EAsymResNet18FDAsymResNet18B/Aug21_11-11_RMSpropforYY_ax01/  
## SeeSaw: python -u main.py  -j 16 --train_scheduler alternating  -a SeeSawRes  -b 512 --dist-url 'tcp://127.0.0.1:6009' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0

###python main.py -j 20 -a ResNet18S --dist-url 'tcp://127.0.0.1:6009' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0##--resume /scratch/issa/users/tt2684/Research/Models/imagenet_trained/ForwardResidualProp/SeeSaw/checkpoint.pth.tar 
## srun -c 10  --gres=gpu:2 python evaluate.py -j 18 -a SeeSaw --imageresize 64 --imagecrop 64 --losstype reconstruction --evaluate --resume /scratch/issa/users/tt2684/Research/Models/imagenet_trained/ForwardResidualProp/SeeSaw/checkpoint.pth.tar --dist-url 'tcp://127.0.0.1:6004' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 
## stdbuf -o0 -e0 srun -c 48 --gres=gpu:gtx2080:8 python main_dual.py  -j 48 --train_scheduler alternating  -ae AssyResNet18F -ad AssyResNet18B  -b 512 --dist-url 'tcp://127.0.0.1:6009' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 

##now=$(date +'%Y-%m-%d_%H-%M'); srun -c 48 --gres=gpu:8 python -u main_dual_greedy.py -j 0 -dataset imagenet -b 16 -p 50  -ae AsymResNet18F -ad AsymResNet18B -time $now  --dist-url 'tcp://127.0.0.1:6008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
#End of script


