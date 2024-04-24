#!/bin/bash
#
#SBATCH -A astro           # Set Account name
#SBATCH --job-name=torchtest   # The job name
#SBATCH --gres=gpu:1            # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH --constraint=rtx8000    # You may specify rtx8000 or v100s or omit this line for either
#SBATCH -c 1                    # Number of cores
#SBATCH -t 0-02:00              # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=5gb       # Memory per cpu core

module load anaconda
 
#cd "/burg/home/tjk2147/src/GitHub/torchinterp1d" 
#pip install -e .

#cd "/burg/home/tjk2147/src/GitHub/spender"
#pip install -e . 

#cd "/burg/home/tjk2147/src" 

conda update -n base -c defaults conda

#Command to execute Python program
#python python_script.py

#python /burg/home/tjk2147/src/GitHub/spender/train/train_sdss.py --dir='/burg/home/tjk2147/src/GitHub/spender/train' --outfile='/burg/home/tjk2147/src/GitHub/spender/train/checkpoint.pt'
 
python /burg/home/tjk2147/src/GitHub/spender/train/train_sdss.py '/burg/home/tjk2147/src/GitHub/spender/train' '/burg/home/tjk2147/src/GitHub/spender/train/checkpoint.pt'

#End of script