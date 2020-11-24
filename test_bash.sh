#!/bin/sh
#SBATCH --job-name=Symbio # The job name.
#SBATCH -o /scratch/issa/users/tt2684/Research/Report/output_Symbio.%j.out # STDOUT
#SBATCH -c 20
#SBATCH --gres=gpu:1
#SBATCH --mem=20gb

if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]
then
  echo " =========================================="
  echo " SLURM_JOB_ID = $SLURM_JOB_ID"
  echo " SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
  echo " =========================================="
fi

module load anaconda3-2019.03

conda activate /home/tt2684/conda-envs/pytorch_tensorflow_latest

python test.py
