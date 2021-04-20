# Repo for implementation of AdaIN in tensorflow-2.0

We have implememted the AdaIN model described in [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) using Tensorflow 2.0 version.

Tested based upon environments below
* python 3.7.9
* tensorflow 2.4
* [kaggle-api](https://github.com/Kaggle/kaggle-api)

## Download data 
The training content images are [MS-COCO](https://cocodataset.org/#home) and style images are [WikiArt](https://www.kaggle.com/c/painter-by-numbers). You can run `setup.sh` directly to download both images dataset. The total dataset size will take about 40 GB.

## Some defects
* We take the "same" padding in our encoder and decoder model, while in the paper, they use "reflect" padding to avoid border artifacts.

## Reference
1. Repository from Xun Huang, [AdaIN-style](https://github.com/xunhuang1995/AdaIN-style), who is the first author of AdaIN model paper.
2. [AdaIN-style-transfer---Tensorflow-2.0](https://github.com/rasmushr97/AdaIN-style-transfer---Tensorflow-2.0): The repo resize all images to (256, 256) without preserving the ratio, which is different from the data preparation described in the paper. 
3. [Tensorflow-Style-Transfer-with-Adain](https://github.com/JunbinWang/Tensorflow-Style-Transfer-with-Adain)