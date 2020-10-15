#!/bin/sh
module load anaconda3-2019.03
source activate pytorch_tensorflow_latest

now=$(date +'%Y-%m-%d_%H-%M')
note='**Autoencoder_TwoCosts**'
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
# python -u create_config.py --hash MNISTAsymResLNet10BNaffPatience30 -dataset MNIST  -j 24 --input_size 32 --base_channels 64 --batch-size 256 --epoch 200 -ae AsymResLNet10F -ad AsymResLNet10B --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --patiencee 30 --patienced 25 -p 100 --note  $note ; config=1  #
# python -u create_config.py --hash FaMNISTAsymResLNet10BNaff -dataset FashionMNIST  -j 24 --input_size 32 --base_channels 64 --batch-size 256 --epoch 600 -ae AsymResLNet10F -ad AsymResLNet10B --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --patiencee 50 --patienced 40 -p 100 --note  $note ; config=1  #

# python -u create_config.py -dataset MNIST  -j 24 --input_size 32 --base_channels 64 --batch-size 256 --epoch 300 -ae AsymResLNet10F -ad AsymResLNet10B --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --patiencee 50 --patienced 40 -p 100 --note  $note ; config=1  #


# python -u create_config.py  -dataset STL10    -j 24 --input_size 96 --base_channels 64 --batch-size 64 --epoch 400 -ae AsymResLNet10F -ad AsymResLNet10B --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-6 --patiencee 30 --patienced 20 -p 100 --note  $note ; config=1  #



if [ $config == 0 ]
  then
    runname=Oct15-10-51_MNIST_2925cdefbe_654
    #Oct10-11-44_MNIST_3a134f421d_RMSpRMSpMNISTAsymResLNet10BNaff3_251 # Sep30-11-43_MNIST_4c4b77d125_516   #May25-09-00_CIFAR10_b784081472_181
    configpath="/home/tahereh/Documents/Research/Results/Symbio/Symbio/$runname/configs.yml"

    method=BP

    printf " Here $configpath \n"
    # python -u main_train.py --method $method  --config-file $configpath # --resume_training_epochs 400
    # python -u main_train_autoencoders.py --method $method  --config-file $configpath
    # python -u main_train_autoencoders_twocosts.py --method $method  --config-file $configpath


    robustness to noise evaluation
    for eval_sigma2 in `seq 1e-3 0.1 1.0` # why start at 1e-3 instead of zero? becuase numpy. doesn't accept zero as sigma2 
    do
    eval_epsilon=0.0
    # python -u main_evaluate.py --eval_save_sample_images False  --method $method --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time now
    # python -u main_evaluate_autoencoders.py --eval_save_sample_images False  --method $method --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time now
    python -u main_evaluate_autoencoders_twocosts.py --eval_save_sample_images False  --method $method --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time now

    done


    # # robustness to adversarial attacks evaluation
    # for eval_epsilon in `seq 0.0 0.2 1.0`
    # do 
    # # for eval_sigma2 in `seq 0.0 0.0 0.0`
    # # do
    # eval_sigma2=0.0
    # echo $method
    # echo $eval_sigma2
    # echo $eval_epsilon
    # python -u main_evaluate.py  --eval_save_sample_images False --method $method --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time now 
    # done
    
    fi