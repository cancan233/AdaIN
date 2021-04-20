import os
import random
import numpy as np
import sys
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_ubyte
from tqdm import tqdm

import tensorflow as tf
import hyperparameters as hp


class ImageDataset:
    def __init__(self, content_paths, style_paths):
        self.content_paths = content_paths
        self.style_paths = style_paths

        # Note: (only if you flow_from_directory) in case of class_mode None, the data still needs to reside in a subdirectory of directory for it to work correctly. Thus, here I manually define the class as the directory name of the content_dir and style_dir respectively.
        """
        self.content_dir_head, self.content_dir_tail = os.path.split(self.content_dir)
        self.style_dir_head, self.style_dir_tail = os.path.split(self.style_dir)
        self.content_data = self.get_data(
            self.content_dir_head, classes=[self.content_dir_tail]
        )
        self.style_data = self.get_data(
            self.style_dir_head, classes=[self.style_dir_tail]
        )
        """
        self.content_data = self.get_data(self.content_paths)
        self.style_data = self.get_data(self.style_paths)

    def preprocess_fn(self, img):
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    # def get_data(self, paths, classes):
    def get_data(self, paths):
        images = []

        for path in paths:
            image = tf.keras.preprocessing.image.load_img(path)
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            # Cannot use tf.image.resize() because it will yield images whose largest dimension is 512 while the smallest one can be smaller than 256, when preserve_aspect_ratio is True. Use smart_resize() here, which is a little different than what described in the paper.

            resized_image = tf.keras.preprocessing.image.smart_resize(
                image_array, [512, 512]
            )

            # resized_image = image_array
            cropped_image = tf.image.random_crop(
                resized_image, size=[hp.img_size, hp.img_size, 3]
            )

            image = tf.keras.applications.vgg19.preprocess_input(cropped_image)
            images.append(image)

        """
        Don't know how to implement the custom resize into flow_from_directory.

        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocess_fn
        )

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

        return np.array(images)


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

    You have to make sure that the directory + "_cleaned" exists.

    TODO: This operation can be parallelized using joblib.Parallel. But it might require too large memory.
    """
    directory = sys.argv[1]
    if directory[-1] == "/":
        directory = directory[:-1]
    paths = os.listdir(directory)
    num_delete = 0
    for i in tqdm(range(len(paths))):
        path = paths[i]
        img_path = os.path.join(os.path.abspath(directory), path)
        try:
            image = imread(img_path)
            if len(image.shape) != 3 or image.shape[2] != 3 or image.size > 89478485:
                num_delete += 1
                # os.remove(img_path)
                print(
                    "\nimage.shape:",
                    image.shape,
                    "image.size:",
                    image.size,
                    " Remove image <%s>\n" % img_path,
                )
            else:
                image = imread(img_path)
                height, width, _ = image.shape

                if height < width:
                    new_height = 512
                    new_width = int(width * new_height / height)
                else:
                    new_width = 512
                    new_height = int(height * new_width / width)
                try:
                    resized_image = resize(image, [new_height, new_width])
                    # if not os.path.isdir(directory + "_cleaned"):
                    #     os.mkdir(directory + "_cleaned")
                    save_name = (
                        directory + "_cleaned" + os.sep + os.path.split(img_path)[1]
                    )
                    imsave(save_name, img_as_ubyte(resized_image))

                except:
                    print("Cant resize this file, will delete it")
                    num_delete += 1
                    # os.remove(img_path)
        except:
            num_delete += 1
            print("Cant read this file, will delete it")
            # os.remove(img_path)

    print(
        "\n\ndelete %d files! Current number of files: %d\n\n"
        % (num_delete, len(paths) - num_delete)
    )


def test_cleaned_images():
    directory = sys.argv[1]
    if directory[-1] == "/":
        directory = directory[:-1]
    paths = os.listdir(directory)
    for i in tqdm(range(len(paths))):
        path = paths[i]
        img_path = os.path.join(os.path.abspath(directory), path)
        try:
            image = imread(img_path)
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            resized_image = image_array
            resized_image = tf.image.random_crop(
                resized_image, size=[hp.img_size, hp.img_size, 3]
            )
        except:
            print(img_path)


if __name__ == "__main__":
    clean_images()
    # test_cleaned_images()
