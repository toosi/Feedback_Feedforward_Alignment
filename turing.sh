#!/bin/sh
module load anaconda3-2019.03

source activate pytorch_tensorflow_latest
runname=Oct07-11-36_MNIST_8bef0c74ea_RMSpRMSpMNISTAsymResLNet10BNaffine2_252
#Oct10-11-44_MNIST_3a134f421d_RMSpRMSpMNISTAsymResLNet10BNaff3_251 # Sep30-11-43_MNIST_4c4b77d125_516   #May25-09-00_CIFAR10_b784081472_181
configpath="/home/tahereh/Documents/Research/Results/Symbio/Symbio/$runname/configs.yml"

method=SLVanilla

printf " Here $configpath \n"
# python -u main_train.py --method $method  --config-file $configpath # --resume_training_epochs 400
# python -u main_train_autoencoders.py --method $method  --config-file $configpath



# robustness to noise evaluation
for eval_sigma2 in `seq 1e-3 0.1 1.0` # why start at 1e-3 instead of zero? becuase numpy. doesn't accept zero as sigma2 
do
eval_epsilon=0.0
# python -u main_evaluate.py --eval_save_sample_images False  --method $method --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time now
python -u main_evaluate_autoencoders.py --eval_save_sample_images False  --method $method --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time now
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
    