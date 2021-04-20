#!/bin/sh

if [ ! -d "./images" ]
then
    echo "Directory ./images DOES NOT exists. Creating"
    mkdir ./images
fi

cd images

<<'COMMENT' Uncomment this block to download MS-COCO train2014 dataset
echo "Download MS-COCO train2014 dataset..."
# axel is recommended if download speed is too slow
wget -O ./train2014.zip http://images.cocodataset.org/zips/train2014.zip
unzip ./train2014.zip
mv train2014 content
rm -rf ./train2014.zip
COMMENT 

<< 'COMMENT' Uncomment this block to download the preprocessed MS-COCO dataset
# Below code does not work for such a huge file. 
# Here is the link for the file https://drive.google.com/file/d/1835Ce_fDPRBHLblOtqsmjiKSyf0TIwSp/view?usp=sharing

FILEID='1835Ce_fDPRBHLblOtqsmjiKSyf0TIwSp'

echo "Download WikiArt cleaned data..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1835Ce_fDPRBHLblOtqsmjiKSyf0TIwSp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1835Ce_fDPRBHLblOtqsmjiKSyf0TIwSp" -O content_cleaned.tar.gz && rm -rf /tmp/cookies.txt
COMMENT


<<'COMMENT' Uncomment this block to download WikiArt dataset from Kaggle
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

<< 'COMMENT' Uncomment this block to download preprocessed WikiArt datase
# Below code does not work for such a huge file. 
# Here is the link for the file https://drive.google.com/file/d/1Y7O3J_htkaFc1FSPIoc8oEWvzD3843ug/view?usp=sharing

FILEID='1Y7O3J_htkaFc1FSPIoc8oEWvzD3843ug'

echo "Download WikiArt cleaned data..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Y7O3J_htkaFc1FSPIoc8oEWvzD3843ug' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Y7O3J_htkaFc1FSPIoc8oEWvzD3843ug" -O style_cleaned.tar.gz && rm -rf /tmp/cookies.txt
COMMENT


echo "Download VGG19 pretrained weights..."
wget https://s3-us-west-2.amazonaws.com/wengaoye/vgg19_normalised.npz

cd ..
## TODO: add pretrained model download link ##

