import os
import random
import numpy as np
import sys
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

import tensorflow as tf
import hyperparameters as hp


class ImageDataset:
    def __init__(self, content_dir, style_dir):
        self.content_dir = content_dir
        self.style_dir = style_dir
        # Note: in case of class_mode None, the data still needs to reside in a subdirectory of directory for it to work correctly. Thus, here I manually define the class as the directory name of the content_dir and style_dir respectively.
        self.content_dir_head, self.content_dir_tail = os.path.split(self.content_dir)
        self.style_dir_head, self.style_dir_tail = os.path.split(self.style_dir)
        self.content_data = self.get_data(
            self.content_dir_head, classes=[self.content_dir_tail]
        )
        self.style_data = self.get_data(
            self.style_dir_head, classes=[self.style_dir_tail]
        )

    def preprocess_fn(self, img):
        """
        height, width, _ = img.shape
        if height < width:
            new_height = 512
            new_width = int(width * new_height / height)
        else:
            new_width = 512
            new_height = int(height * new_width / width)
        img = resize(img, [new_height, new_width], anti_aliasing=True)
        start_h = np.random.choice(new_height - hp.img_size + 1)
        start_w = np.random.choice(new_width - hp.img_size + 1)
        img = img[
            start_h : (start_h + hp.img_size), start_w : (start_w + hp.img_size), :
        ]
        """
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    def get_data(self, directory, classes):
        images = []
        paths = os.listdir(directory)
        for path in paths:
            path = os.path.join(os.path.abspath(directory), path)
            image = tf.keras.preprocessing.image.load_img(path)
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            # image_array = tf.expand_dims(image_array, 0)
            resized_image = tf.image.resize(
                image_array, [512, 512], method="nearest", preserve_aspect_ratio=False
            )
            cropped_image = tf.image.random_crop(
                resized_image, size=[hp.img_size, hp.img_size, 3]
            )
            images.append(cropped_image)
        print(np.array(images).shape)
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocess_fn
        )
        """
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=None,
            color_mode="rgb",
            classes=classes,
            class_mode=None,
            batch_size=None,
            shuffle=True,
        )
        """
        data_gen = data_gen.flow(images, batch_size=hp.batch_size, shuffle=True)
        exit()
        return data_gen


def get_image(path):
    img = tf.keras.preprocessing.image.load_img(
        path,
        color_mode="rgb",
        target_size=None,
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
    return img_array


def clean_images():
    """
    clean all corrupted images files.
    """
    directory = sys.argv[1]
    paths = os.listdir(directory)
    num_delete = 0
    for i in tqdm(range(len(paths))):
        path = paths[i]
        img_path = os.path.join(os.path.abspath(directory), path)
        try:
            image = imread(img_path)
            if len(image.shape) != 3 or image.shape[2] != 3:
                num_delete += 1
                os.remove(img_path)
                print("\nimage.shape:", image.shape, " Remove image <%s>\n" % img_path)
            else:
                height, width, _ = image.shape

                if height < width:
                    new_height = 512
                    new_width = int(width * new_height / height)
                else:
                    new_width = 512
                    new_height = int(height * new_width / width)

                try:
                    resize(image, [new_height, new_width], anti_aliasing=True)
                except:
                    print("Cant resize this file, will delete it")
                    num_delete += 1
                    os.remove(img_path)
        except:
            num_delete += 1
            print("Cant read this file, will delete it")
            os.remove(img_path)

    print(
        "\n\ndelete %d files! Current number of files: %d\n\n"
        % (num_delete, len(paths) - num_delete)
    )


if __name__ == "__main__":
    clean_images()