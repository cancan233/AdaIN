# Modified based on https://github.com/JunbinWang/Tensorflow-Style-Transfer-with-Adain/blob/master/adain_norm.py

import tensorflow as tf
import numpy as np
import hyperparameters as hp
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.models import Model


class AdaIn(Layer):
    def __init__(self, epsilon=hp.epsilon, alpha=hp.alpha):
        self.epsilon = epsilon
        self.alpha = alpha
        super(AdaIn, self).__init__()

    def call(self, content_features, style_features):
        content_mean, content_variance = tf.nn.moments(
            content_features, axes=[1, 2], keep_dims=True
        )
        style_mean, style_variance = tf.nn.moments(
            style_features, axes=[1, 2], keep_dims=True
        )
        normalized_content_features = tf.nn.batch_normalization(
            content_features,
            content_mean,
            content_variance,
            style_mean,
            tf.math.sqrt(style_variance),
            self.epsilon,
            name="AdaIN",
        )
        normalized_content_features = (
            self.alpha * normalized_content_features +
            (1 - self.alpha) * content_features
        )
        return normalized_content_features


# This can be simplified by directly loading VGG19 pretrained model
class encoder(tf.keras.Model):
    def __init__(self):
        super(encoder, self).__init__()

        self.vgg19 = tf.keras.applications.VGG19(
            include_top=False, weights="imagenet")

        self.vgg19_encoder = tf.keras.Model(
            inputs=self.vgg19, outputs=[
                self.vgg19.get_layer("block4_conv2").output]
        )

        self.vgg19_encoder.trainable = False

    def call(self, x):
        return self.vgg19_encoder(x)


class decoder(tf.keras.Model):
    def __init__(self):
        super(decoder, self).__init__()

        self.vgg19_decoder = tf.keras.Sequential(
            [
                Conv2D(512, 3, 1, padding="same", activation="relu"),
                UpSampling2D((2, 2)),
                Conv2D(256, 3, 1, padding="same", activation="relu"),
                Conv2D(256, 3, 1, padding="same", activation="relu"),
                Conv2D(256, 3, 1, padding="same", activation="relu"),
                Conv2D(256, 3, 1, padding="same", activation="relu"),
                UpSampling2D((2, 2)),
                Conv2D(128, 3, 1, padding="same", activation="relu"),
                Conv2D(128, 3, 1, padding="same", activation="relu"),
                UpSampling2D((2, 2)),
                Conv2D(64, 3, 1, padding="same", activation="relu"),
                Conv2D(64, 3, 1, padding="same", activation="relu"),
            ],
            name="vgg19_decoder",
        )

    def call(self, x):
        return self.vgg19_decoder(x)

    def loss_fn(self, content_loss, style_loss):
        return content_loss + hp.c_lambda * style_loss


class AdaIN_NST(tf.keras.Model):
    def __init__(self):
        super(AdaIn_NST, self).__init__()

    def call(self, content, style):
        enc_content = encoder().call(content)
        enc_style = encoder().call(style)
        adain = AdaIN().call()
        return x

    def train(self, content, style, loss, n_epoch=hp.n_epoch, save_path=None):
        save_interval = 100
        optimizer =
