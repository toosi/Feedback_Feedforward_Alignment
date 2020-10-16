#!/bin/sh
#SBATCH --job-name=Symbio # The job name.
#SBATCH -o /scratch/issa/users/tt2684/Research/Report/output_Symbio.%j.out # STDOUT
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH --mem=80gb
#  $*#SBATCH --array=0-2
#SBATCH --time=5-00:00:00

if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]
then
  echo " =========================================="
  echo " SLURM_JOB_ID = $SLURM_JOB_ID"
  echo " SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
  echo " =========================================="
fi

module load anaconda3-2019.03

source activate /home/tt2684/conda-envs/pytorch_tensorflow_latest

now=$(date +'%Y-%m-%d_%H-%M')
note='**FashionMNIST**'
# Revisit_Asymresnet_AffineFalse
# imagenet_with_modified_resnets_wobn1_trackFalse_wolastAcc
# Cycle_Consistency AdvTrainingFGSM_epsilon0.2_withOUTSperateOptimizerscheduler
# AdvTrainingFGSM_epsilon0.2_withscheduler
printf "********************************************************** \n"

printf " $note \n"

printf "********************************************************** \n"

####Command to execute Python program
config=0
# init='May25-14-32_CIFAR10_9fb773a12e_987' # use: -loadinitialization $init
# Fully Connected
# python -u create_config.py  --hash RMSpRMSpFaMNISTFullyConnE150  --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-4 --lrB 1e-4 --wdF 1e-5 --wdB 1e-5 --patiencee 20 --patienced 15 -dataset FashionMNIST -j16 --input_size 32 --batch-size 256 --epoch 150 -ae FullyConnectedF -ad FullyConnectedB -p 100 --note  $note ; config=1 
# Convolutional
# python -u create_config.py --hash TwoCostAEcontrolMNIST -dataset MNIST  -j 24 --input_size 32 --base_channels 64 --batch-size 256 --epoch 300 -ae AsymResLNet10F -ad AsymResLNet10B --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --patiencee 50 --patienced 45 -p 100 --note  $note ; config=1  #
# python -u create_config.py --hash FaMNISTAsymResLNet10BNaff -dataset FashionMNIST  -j 24 --input_size 32 --base_channels 64 --batch-size 256 --epoch 600 -ae AsymResLNet10F -ad AsymResLNet10B --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --patiencee 50 --patienced 40 -p 100 --note  $note ; config=1  #



# python -u create_config.py --hash CIFAR10AsymResLNet10BNaffPatience30 -dataset CIFAR10  -j 24 --input_size 32 --base_channels 64 --batch-size 256 --epoch 400 -ae AsymResLNet10F -ad AsymResLNet10B --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --patiencee 30 --patienced 20 -p 100 --note  $note ; config=1  #
# python -u create_config.py  -dataset CIFAR10  -j 24 --input_size 32 --base_channels 64 --batch-size 256 --epoch 400 -ae AsymResLNet10F -ad AsymResLNet10B --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --patiencee 50 --patienced 40 -p 100 --note  $note ; config=1  #



# python -u create_config.py -dataset imagenet  -j 4 --input_size 224  --batch-size 256 --epoch 500 -ae 'asymresnet18' -ad 'asymresnetT18' --optimizerF 'SGD' --optimizerB 'RMSprop' --lrF 1e-1 --lrB 1e-3 --wdF 1e-4 --wdB 1e-6 --patiencee 10 --patienced 8 -p 100 --note  $note ; config=1  #
# python -u create_config.py -dataset CIFAR10  --normalization_affine True -j 4  --gamma 0  --input_size 32 --base_channels 64 --batch-size 256 --epoch 1000 -ae AsymResLNet10F -ad AsymResLNet10B --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-1 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --patiencee 50 --patienced 40 -p 100 --note  $note ; config=1  #
# python -u create_config.py -dataset MNIST  -j 24 --input_size 32 --base_channels 64 --batch-size 256 --epoch 150 -ae AsymResLNet10F -ad AsymResLNet10B --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --patiencee 50 --patienced 40 -p 100 --note  $note ; config=1  #

# gammas=(1e-10 1e-5 1e-0 1e1) --primitive_weights $alpha $beta $gamma

# optimizerB='RMSprop'
# optimizerF='RMSprop'
# lrF=0.001
# for gamma in "${gammas[@]}"
# do
# python -u create_config.py  -dataset CIFAR10   -j 24  --input_size 32  --batch-size 256 --epoch 1000 -ae 'asymresnet18' -ad 'asymresnetT18' --optimizerF $optimizerF --optimizerB $optimizerB --lrF $lrF --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --momentumF 0.9 --momentumB 0.9 --patiencee 50 --patienced 40 -p 100 --note  $note ; config=1  #
# done

# python -u create_config.py -dataset imagenet  -j 4 --input_size 224  --batch-size 64 --epoch 100 -ae 'asymresnet18' -ad 'asymresnetT18' --optimizerF 'Adam' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --patiencee 10 --patienced 8 -p 100 --note  $note ; config=1  #
#--primitive_weights 1019.1 7.9 3e-6

if [ $config == 0 ]
  then
  # runname='May09-13-40_CIFAR10_adf3aac7c7_866' #'May06-11-39_CIFAR10_adf3aac7c7_918' # 'Feb14-09-08_CIFAR10_a3e0466f41_592'  #
  
  # runnames=('May09-08-33_CIFAR10_adf3aac7c7_878' 'May09-08-33_CIFAR10_adf3aac7c7_835' 'May09-08-33_CIFAR10_adf3aac7c7_85' 'May09-08-34_CIFAR10_adf3aac7c7_372')
  
  # runnames=('May09-13-39_CIFAR10_adf3aac7c7_94' 'May09-13-40_CIFAR10_adf3aac7c7_866' 'May09-13-40_CIFAR10_adf3aac7c7_1020')
  # runnames=('May10-09-29_CIFAR10_adf3aac7c7_907' 'May10-09-29_CIFAR10_adf3aac7c7_844')
  # runnames=('May10-16-05_CIFAR10_adf3aac7c7_500' 'May10-16-05_CIFAR10_adf3aac7c7_318')
  # runnames=('May11-11-50_CIFAR10_adf3aac7c7_138' 'May11-11-50_CIFAR10_adf3aac7c7_125' 'May11-11-50_CIFAR10_adf3aac7c7_222' 'May11-11-50_CIFAR10_adf3aac7c7_804')
  # runnames=('May11-19-15_CIFAR10_adf3aac7c7_412' 'May11-19-15_CIFAR10_adf3aac7c7_149')
  # runnames=('May11-22-05_CIFAR10_adf3aac7c7_524' 'May11-22-05_CIFAR10_adf3aac7c7_456' 'May11-22-06_CIFAR10_adf3aac7c7_685' 'May11-22-06_CIFAR10_adf3aac7c7_731')
  # runnames=('May12-08-19_CIFAR10_adf3aac7c7_89' 'May12-08-19_CIFAR10_adf3aac7c7_676' 'May12-08-19_CIFAR10_adf3aac7c7_16' 'May12-08-19_CIFAR10_adf3aac7c7_160')
  # runnames=('May13-08-23_CIFAR10_367d0e639c_837' 'May13-08-24_CIFAR10_367d0e639c_157' 'May13-08-24_CIFAR10_367d0e639c_438' 'May13-08-24_CIFAR10_367d0e639c_898' )
  # runnames=('May15-11-23_CIFAR10_367d0e639c_623')
  # runnames=('May15-18-17_CIFAR10_367d0e639c_588')
  # runnames=('May16-09-00_CIFAR10_367d0e639c_19')
  # runnames=('May16-12-59_CIFAR10_367d0e639c_526')
  # runnames=('May17-07-26_CIFAR10_367d0e639c_180')
  # runnames=('May17-14-39_CIFAR10_367d0e639c_577')
  # runnames=('May18-13-25_CIFAR10_367d0e639c_715')
  # runnames=('May18-14-10_imagenet_367d0e639c_279')
  # runname="${runnames[$SLURM_ARRAY_TASK_ID]}"



  runname=Oct16-14-41_CIFAR10_7f7153644c_872 #Oct14-09-25_CIFAR10_84b9f34c93_166  #Oct14-09-21_CIFAR10_84b9f34c93_391 # Sep30-11-43_MNIST_4c4b77d125_516   #May25-09-00_CIFAR10_b784081472_181
  # configpath="/home/tahereh/Documents/Research/Results/Symbio/Symbio/$runname/configs.yml"
  configpath="/home/tt2684/Research/Results/Symbio/Symbio/$runname/configs.yml"
  methods=('SLVanilla' 'BP' 'FA') # ('SLError' 'SLAdvImg' 'SLLatentRobust')'IA'  'BP' 'FA' 'SLError' 'SLAdvImg' 'SLAdvCost' 'SLLatentRobust' 'SLConv1')


  # methods=('SLAdvImgCC0' 'SLAdvCostCC0' 'BPCC0' 'FACC0' 'SLVanillaCC0' 'SLErrorCC0' )
  # methods=('BPCC1' 'FACC1' 'SLVanillaCC1' 'SLErrorCC1' 'SLAdvImgCC1' 'SLAdvCostCC1')
  # methods=('BP' 'FA' 'SLVanilla' 'SLLatentRobust' 'SLAdvImg' 'SLError')
  # python -u main_train.py   --config-file $configpath --method "${methods[$SLURM_ARRAY_TASK_ID]}"
  
  python -u main_train_PCGrad.py   --config-file $configpath --method PCGRrad
  
  # python -u main_train.py --method 'SLError'  --config-file $configpath

  # printf " $methods \n"
  # SLConv1 needs to be run in one gpu because of hooks

  # python -u generate_figures.py --config-file $configpath
  # python -u generate_figures.py --config-file $configpath --eval_RDMs True

  # # to compute aligments between forward and backward weights
  # for method in "${methods[@]}"
  # do
  # python -u main_evaluate.py --eval_alignments True --config-file $configpath --eval_time debug --method $method
  # done


  # # remember to run on 1 gpu to make sure hooks work as expected
  # methods='SLVanilla BP FA' # SLRobust SLError
  # for method in $methods
  # do
  # # python -u main_evaluate.py --eval_time $now --config-file $configpath --eval_generate_RDMs True --method $method
  # # done
  # for eval_sigma2 in `seq 0.0 0.1 1.0`
  # do
  # eval_epsilon=0.0
  # python -u main_evaluate.py --eval_save_sample_images False  --method $method --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time $now
  # done
  # done
  # robustness evaluation
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
  # python -u main_evaluate.py  --eval_save_sample_images False --method "${methods[$SLURM_ARRAY_TASK_ID]}" --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time 2020-02-20_09-14 #$now 
  # done
  # done
  # done
  
  # python -u generate_figures.py --eval_swept_var sigma2 --eval_time 2020-03-12_11-55 --config-file $configpath
  
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


