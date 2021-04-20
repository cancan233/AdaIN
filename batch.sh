#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -J AdaIN_NST_training
#SBATCH --mem=48G

module load python/3.7.4
module load cuda/11.1.1
module load gcc/10.2
module load cudnn/8.1.0
source ~/pythonenv/tf_gpu/bin/activate

# clean style images. remove corrupted images or too large images
# python preprocess.py ./images/style_kaggle

python run.py --content-dir ./images/content --style-dir ./images/style_kaggle --pretrained-vgg19 ./images/vgg19_normalised.npz

# deal the data file

