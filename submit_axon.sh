#!/bin/sh
#SBATCH --job-name=Symbio # The job name.
#SBATCH -o /scratch/issa/users/tt2684/Research/Report/output_Symbio.%j.out # STDOUT
#SBATCH -c 20
#SBATCH --gres=gpu:gtx2080:1
#SBATCH --mem=60gb
#SBATCH --exclude ax07
#SBATCH 


module load anaconda3-2019.03

source activate /home/tt2684/conda-envs/pytorch_tensorflow_SYNPAI_CUDA100cudnn75

now=$(date +'%Y-%m-%d_%H-%M')
note='**ErrorNorm**'
# learning_scheduler_of_decoder_depends_on_its_own_cost
printf "********************************************************** \n"

printf " $note \n"

printf "********************************************************** \n"

####Command to execute Python program
config=0
# python -u create_config.py -dataset imagenet  -j 24 --input_size 224 --base_channels 128 --batch-size 128 --epoch 500 -ae AsymResLNetLimited14F -ad AsymResLNet14B --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --patiencee 50 --patienced 40 -p 100 --note  $note ; config=1  #
# python -u create_config.py -dataset CIFAR10  -j 24 --input_size 32 --base_channels 64 --batch-size 256 --epoch 500 -ae AsymResLNet10F -ad AsymResLNet10B --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --patiencee 50 --patienced 40 -p 100 --note  $note ; config=1  #

if [ $config == 0 ]
    then
    runname='Jan19-13-12_70f10333a4_810' #'Jan28-12-36_a4e68fe8a6_916'  #'Jan19-09-23_70f10333a4_657'  #'Jan17-15-32_40a474bced_872'  # 'Jan15-11-58_fc540af17a_346'  #'Jan15-11-38_fc540af17a_617'  #'Jan15-11-58_fc540af17a_346'  #   #'Dec30-16-38_04f1282767_1'   #'Dec29-10-13_fd73bdce0b_2'  #Dec06-23-07_81c6b0578b_2' #'Dec06-12-23_81c6b0578b_2'   #'Dec06-09-41_81c6b0578b_2' #'Dec06-12-09_81c6b0578b_2'
    configpath="/home/tt2684/Research/Results/Symbio/Symbio/$runname/configs.yml"

    # python -u main_train.py --method BSL  --config-file $configpath
    # python -u main_train.py --method SLVanilla  --config-file $configpath
    # python -u main_train.py --method FA --config-file $configpath
    # python -u main_train.py --method BP  --config-file $configpath
    # python -u main_train.py --method SLDecoderRobustOutput  --config-file $configpath
    # python -u main_train.py --method SLError  --config-file $configpath
    # python -u main_train.py --method SLDecoderRobustOnehot  --config-file $configpath
    # python -u main_train.py --method SLTemplateGenerator  --config-file $configpath
    # python -u main_train.py --method SLRobust  --config-file $configpath

    # python -u main_train.py --method SLErrorTemplateGenerator  --config-file $configpath

    # python -u main_train_SLtarget.py --method SLVanilla  --config-file $configpath


    # python -u generate_figures.py --config-file $configpath
    # python -u generate_figures.py --config-file $configpath --eval_RDMs True
    # # remember to run on 1 gpu to make sure hooks work as expected
    # methods='SLVanilla BP FA' # SLRobust SLError
    # for method in $methods
    # do
    # python -u main_evaluate.py --eval_time $now --config-file $configpath --eval_generate_RDMs True --method $method
    # done

    # python -u main_evaluate.py --eval_save_sample_images False  --method $method --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time $now

    # ## robustness evaluation
    # methods='SLVanilla BP FA SLRobust SLError' # SLRobust SLError 
    # for method in $methods
    # do 
    # for eval_epsilon in `seq 0.0 0.2 1.0`
    # do 
    # # for eval_sigma2 in `seq 0.0 0.0 0.0`
    # # do
    # eval_sigma2=0.0
    # echo $method
    # echo $eval_sigma2
    # echo $eval_epsilon
    # python -u main_evaluate.py  --eval_save_sample_images False --method $method --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time $now 
    # # done
    # done
    # done
    
    # python -u generate_figures.py --eval_robust True --eval_time 2020-01-20_09-13 --config-file $configpath
    
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


