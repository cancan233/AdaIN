#!/bin/bash
#SBATCH --account=brubenst-condo

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH -J NST_preprocess_data
#SBATCH --mem=24G

# run command
module load python/3.7.4
source ~/pythonenv/tf_cpu/bin/activate
python preprocess.py ./images/style_kaggle
