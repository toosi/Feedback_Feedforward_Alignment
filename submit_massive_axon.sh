#!/bin/sh
#SBATCH --job-name=Symbio # The job name.
#SBATCH -o /scratch/issa/users/tt2684/Research/Report/output_Symbio.%j.out # STDOUT
#SBATCH -c 20
#SBATCH --gres=gpu:2
#SBATCH --mem=20gb
#SBATCH --array=0-10
#SBATCH --time=5-00:00:00
#SBATCH -A issa

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
note='**MNIST_Conv**'

printf "********************************************************** \n"

printf " $note \n"

printf "********************************************************** \n"

####Command to execute Python program
config=0
# init='May25-14-32_CIFAR10_9fb773a12e_987' # use: -loadinitialization $init



#Fully Connected
# python -u create_config.py  --optimizerF 'SGD' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-5 --patiencee 20 --patienced 15 -dataset MNIST -j16 --input_size 32 --batch-size 256 --epoch 100 -ae FullyConnectedF -ad FullyConnectedB -p 100 --note  $note ; config=1 
# python -u create_config.py  --optimizerF 'SGD' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-3 --wdB 1e-5 --patiencee 20 --patienced 15 -dataset MNIST -j16 --input_size 32 --batch-size 256 --epoch 100 -ae FullyConnectedF -ad FullyConnectedB -p 100 --note  $note ; config=1 
# python -u create_config.py  --optimizerF 'SGD' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-3 --patiencee 20 --patienced 15 -dataset MNIST -j16 --input_size 32 --batch-size 256 --epoch 100 -ae FullyConnectedF -ad FullyConnectedB -p 100 --note  $note ; config=1 
# python -u create_config.py  --optimizerF 'SGD' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-3 --wdB 1e-3 --patiencee 20 --patienced 15 -dataset MNIST -j16 --input_size 32 --batch-size 256 --epoch 100 -ae FullyConnectedF -ad FullyConnectedB -p 100 --note  $note ; config=1 

# python -u create_config.py  --optimizerF 'SGD' --optimizerB 'RMSprop' --lrF 1e-2 --lrB 1e-3 --wdF 1e-5 --wdB 1e-5 --patiencee 20 --patienced 15 -dataset MNIST -j16 --input_size 32 --batch-size 256 --epoch 100 -ae FullyConnectedF -ad FullyConnectedB -p 100 --note  $note ; config=1 
# python -u create_config.py  --optimizerF 'SGD' --optimizerB 'RMSprop' --lrF 1e-2 --lrB 1e-3 --wdF 1e-3 --wdB 1e-5 --patiencee 20 --patienced 15 -dataset MNIST -j16 --input_size 32 --batch-size 256 --epoch 100 -ae FullyConnectedF -ad FullyConnectedB -p 100 --note  $note ; config=1 
# python -u create_config.py  --optimizerF 'SGD' --optimizerB 'RMSprop' --lrF 1e-2 --lrB 1e-3 --wdF 1e-5 --wdB 1e-3 --patiencee 20 --patienced 15 -dataset MNIST -j16 --input_size 32 --batch-size 256 --epoch 100 -ae FullyConnectedF -ad FullyConnectedB -p 100 --note  $note ; config=1 
# python -u create_config.py  --optimizerF 'SGD' --optimizerB 'RMSprop' --lrF 1e-2 --lrB 1e-3 --wdF 1e-3 --wdB 1e-3 --patiencee 20 --patienced 15 -dataset MNIST -j16 --input_size 32 --batch-size 256 --epoch 100 -ae FullyConnectedF -ad FullyConnectedB -p 100 --note  $note ; config=1 


# python -u create_config.py  --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-3 --wdF 1e-5 --wdB 1e-5 --patiencee 20 --patienced 15 -dataset MNIST -j16 --input_size 32 --batch-size 256 --epoch 100 -ae FullyConnectedF -ad FullyConnectedB -p 100 --note  $note ; config=1 
# python -u create_config.py  --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-3 --lrB 1e-4 --wdF 1e-5 --wdB 1e-5 --patiencee 20 --patienced 15 -dataset MNIST -j16 --input_size 32 --batch-size 256 --epoch 100 -ae FullyConnectedF -ad FullyConnectedB -p 100 --note  $note ; config=1 
# python -u create_config.py  --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-4 --lrB 1e-4 --wdF 1e-5 --wdB 1e-5 --patiencee 20 --patienced 15 -dataset MNIST -j16 --input_size 32 --batch-size 256 --epoch 100 -ae FullyConnectedF -ad FullyConnectedB -p 100 --note  $note ; config=1 
# python -u create_config.py  --optimizerF 'RMSprop' --optimizerB 'RMSprop' --lrF 1e-4 --lrB 1e-3 --wdF 1e-5 --wdB 1e-5 --patiencee 20 --patienced 15 -dataset MNIST -j16 --input_size 32 --batch-size 256 --epoch 100 -ae FullyConnectedF -ad FullyConnectedB -p 100 --note  $note ; config=1 

# Sep30-16-51_MNIST_4c4b77d125_618
# Sep30-16-51_MNIST_4c4b77d125_715
# Sep30-16-51_MNIST_4c4b77d125_901
# Sep30-16-51_MNIST_4c4b77d125_298

if [ $config == 0 ]
    then
    # filename='/home/tt2684/Research/Results/Symbio/runswithhash/RMSpropRMSpropMNISTAsymResLNet10.txt'
    # filename='/home/tt2684/Research/Results/Symbio/runswithhash/RMSpropRMSpropMNISTFullyConn.txt'
    # filename='/home/tt2684/Research/Results/Symbio/runswithhash/RMSpropRMSpropMNISTFullyConnE150.txt'
    filename='/home/tt2684/Research/Results/Symbio/runswithhash/RMSpRMSpMNISTAsymResLNet10BNaffine.txt'
    # filename='/home/tt2684/Research/Results/Symbio/runswithhash/RMSpRMSpMNISTAsymResLNet10BNaffine2.txt'
    # filename='/home/tt2684/Research/Results/Symbio/runswithhash/RMSpRMSpMNISTAsymResLNet10BNaff3.txt'

    # filename='/home/tt2684/Research/Results/Symbio/runswithhash/RMSpRMSpFaMNISTFullyConnE150.txt'
    # filename='/home/tt2684/Research/Results/Symbio/runswithhash/RMSpRMSpFaMNISTAsymResLNet10BNaff.txt'
    # filename='/home/tt2684/Research/Results/Symbio/runswithhash/RMSpRMSpFaMNISTAsymResLNet10BNaffPtnc30.txt'
    # filename='/home/tt2684/Research/Results/Symbio/runswithhash/FaMNISTAsymResLNet10BNaff.txt'
    
    # filename='/home/tt2684/Research/Results/Symbio/runswithhash/RMSpCIFAR10AsymResLNet10BNaff.txt'
    # filename='/home/tt2684/Research/Results/Symbio/runswithhash/MNISTAsymResLNet10BNaffPatience30.txt'
    # filename='/home/tt2684/Research/Results/Symbio/runswithhash/MNISTAsymResLNet10BNaffPatience30.txt'

    n=1
    runnames=()
    while read line; do
    # reading each line
    echo "Line No. $n : $line"
    n=$((n+1))
    
    runname=$line #Sep30-14-41_MNIST_4c4b77d125_237 # Sep30-11-43_MNIST_4c4b77d125_516   #May25-09-00_CIFAR10_b784081472_181
    runnames+=( $runname )
    # configpath="/home/tahereh/Documents/Research/Results/Symbio/Symbio/$runname/configs.yml"
    # configpath="/home/tt2684/Research/Results/Symbio/Symbio/$runname/configs.yml"
    # methods=('SLVanilla' 'BP' 'FA') # ('SLError' 'SLAdvImg' 'SLLatentRobust')'IA'  'BP' 'FA' 'SLError' 'SLAdvImg' 'SLAdvCost' 'SLLatentRobust' 'SLConv1')

    # python -u main_train.py --method "${methods[$SLURM_ARRAY_TASK_ID]}"  --config-file $configpath
    done < $filename


    method=BP

    configpath="/home/tt2684/Research/Results/Symbio/Symbio/${runnames[$SLURM_ARRAY_TASK_ID]}/configs.yml"
    printf " Here $configpath \n"
    # python -u main_train.py --method $method  --config-file $configpath # --resume_training_epochs 400
    # python -u main_train_autoencoders.py --method $method  --config-file $configpath

    

    # robustness to noise evaluation
    for eval_sigma2 in `seq 1e-3 0.1 1.0` # why start at 1e-3 instead of zero? becuase numpy. doesn't accept zero as sigma2 
    do
    eval_epsilon=0.0
    # python -u main_evaluate.py --eval_save_sample_images False  --method $method --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time now
    # python -u main_evaluate_autoencoders.py --eval_save_sample_images False  --method $method --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time now
    python -u main_evaluate_dynamic_decoder.py --eval_save_sample_images False  --method $method --eval_epsilon $eval_epsilon --eval_sigma2 $eval_sigma2  --eval_maxitr 4 --config-file $configpath --eval_time now

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
    # # done
    

    # python -u generate_figures.py --eval_swept_var sigma2 --eval_time now --config-file $configpath
  
    fi



