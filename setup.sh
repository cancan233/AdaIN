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


<<'COMMENT'
# All of the images from the Kaggle dataset resized so that their smaller side is 256 pixels long to allow enough wiggle-room for data augmentation while not becoming a bottleneck to read and process.

echo "Download WikiArt resized data..."
wget -O ./WikiArt.tgz https://github.com/somewacko/painter-by-numbers/releases/download/data-v1.0/train.tgz
tar -xzvf ./WikiArt.tgz
mv train style
rm -rf ./WikiArt.tgz
COMMENT

<<'COMMENT'
echo "Download WikiArt data directly from Kaggle ..."
kaggle competitions download -f train.zip painter-by-numbers
unzip train.zip
mv train style_kaggle
rm -rf train.zip

kaggle competitions download -f replacements_for_corrupted_files.zip painter-by-numbers
unzip replacements_for_corrupted_files.zip
mv -f train/* style_kaggle
rm -rf train/ test/ __MACOSX/ replacements_for_corrupted_files.zip
COMMENT

<< 'COMMENT'
# Below code does not work for such a huge file. 
# Here is the link for the file https://drive.google.com/file/d/1Y7O3J_htkaFc1FSPIoc8oEWvzD3843ug/view?usp=sharing

echo "Download WikiArt cleaned data..."
FILEID="1n1s23CdPUy--D_DgRH1i-BAnt75joLqg"
FILENAME="train_cleaned.tar.gz"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${FILEID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${FILENAME} && rm -rf /tmp/cookies.txt
COMMENT


echo "Download VGG19 pretrained weights..."
wget https://s3-us-west-2.amazonaws.com/wengaoye/vgg19_normalised.npz

cd ..
## TODO: add pretrained model download link ##

