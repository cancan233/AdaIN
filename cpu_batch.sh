#!/bin/bash
#SBATCH --account=brubenst-condo

#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH -J clean_content_data
#SBATCH --mem=24G

# run command
module load python/3.7.4
source ~/pythonenv/tf_cpu/bin/activate
python preprocess.py ./images/content
# python run.py --content-dir ./images/content --style-dir ./images/style_kaggle --pretrained-vgg19 ./images/vgg19_normalised.npz
