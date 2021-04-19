import os
import random
import numpy as np

# from PIL import Image
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
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    def get_data(self, path, classes):
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocess_fn
        )

        target_size = (hp.img_size, hp.img_size)

        data_gen = data_gen.flow_from_directory(
            path,
            target_size=target_size,
            classes=classes,
            class_mode=None,
            batch_size=hp.batch_size,
            shuffle=True,
        )
        return data_gen


def get_image(path):
    img = tf.keras.preprocessing.image.load_img(
        path,
        color_mode="rgb",
        target_size=None,
        #target_size=(hp.img_size, hp.img_size),
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
    return img_array
