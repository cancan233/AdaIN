#!/bin/sh

## set up environment ##
# pip install tensorflow_datasets as tfds

if [ ! -d "./images" ]
then
    echo "Directory ./images DOES NOT exists. Creating"
    mkdir ./images
fi

cd images
echo "Download MS-COCO train2014 dataset..."
# axel is recommended if download speed is too slow
wget -O ./train2014.zip http://images.cocodataset.org/zips/train2014.zip
unzip ./train2014.zip
mv train2014 content
rm -rf ./train2014.zip

# All of the images from the Kaggle dataset (corrupted images fixed), resized so that their smaller side is 256 pixels long to allow enough wiggle-room for data augmentation while not becoming a bottleneck to read and process.
echo "Download WikiArt preprocessed data..."
wget -O ./WikiArt.tgz https://github.com/somewacko/painter-by-numbers/releases/download/data-v1.0/train.tgz
tar -xzvf ./WikiArt.tgz
mv train style
rm -rf ./WikiArt.tgz

echo "Download WikiArt normal-size data..."
kaggle competitions download -f train.zip painter-by-numbers
unzip train.zip
mv train style_normal
kaggle competitions download -f replacements_for_corrupted_files.zip painter-by-numbers
rm -rf train.zip

unzip replacements_for_corrupted_files.zip
mv train/* style_normal
rm -rf train/ test/ __MACOSX/ replacements_for_corrupted_files.zip

echo "Download VGG19 pretrained weights..."
wget https://s3-us-west-2.amazonaws.com/wengaoye/vgg19_normalised.npz

<<<<<<< HEAD
cd ..
## TODO: add pretrained model download link ##

=======
## Download pretrained normalized vgg19 model ##
wget https://s3-us-west-2.amazonaws.com/wengaoye/vgg19_normalised.npz
>>>>>>> 4a01fe49c53aa0e51db68626d6dfdca0d4a020a9
