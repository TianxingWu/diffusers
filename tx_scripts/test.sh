#!/bin/sh
#PBS -N SVD
#PBS -l select=1:ngpus=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -P 12003828
#PBS -q ai

module load cuda/11.6.2
module load miniforge3
conda activate diffusers

cd /home/users/ntu/tianxin3/project/diffusers/tx_scripts

python test_svd.py