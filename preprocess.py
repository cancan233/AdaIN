import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import hyperparameters as hp


class ImageDataset:
    def __init__(self, data_path):
        self.data_path = data_path
        print(data_path)
        self.content_data = self.get_data(self.data_path, classes=["content"])
        self.style_data = self.get_data(self.data_path, classes=["style"])

    def preprocess_fn(self, img):
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    def get_data(self, path, classes):
        # not sure if we need to augment the images
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocess_fn
        )

        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(hp.img_size, hp.img_size),
            classes=classes,
            class_mode=None,
            batch_size=hp.batch_size,
            shuffle=True,
        )
        return data_gen
