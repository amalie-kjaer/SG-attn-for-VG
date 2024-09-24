#!/bin/sh
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --mem-per-cpu=100G
#SBATCH --time=24:00:00
#SBATCH -n 4

#SBATCH --output=exp_xattn_maskfixed.txt
#SBATCH --open-mode=append
#SBATCH --job-name=xattn

source ~/.bashrc 
source activate
conda deactivate 
conda activate ms

python main.py --train --config_path config.yaml

exit 0