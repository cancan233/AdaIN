# Based on https://github.com/JunbinWang/Tensorflow-Style-Transfer-with-Adain/blob/master/adain_norm.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Conv2D, UpSampling2D

import hyperparameters as hp

def deprocess_img(img):
    tmp = img + np.array([103.939, 116.779, 123.68])
    tmp = tmp[:,:,:,::-1]
    tmp = tf.clip_by_value(tmp, 0.0, 255.0)
    return tmp

class AdaIN(Layer):
    def __init__(self, epsilon=hp.epsilon, alpha=hp.alpha):
        self.epsilon = epsilon
        self.alpha = alpha
        super(AdaIN, self).__init__()

    def call(self, content_features, style_features):
        content_mean, content_variance = tf.nn.moments(
            content_features, axes=[1, 2], keepdims=True
        )
        style_mean, style_variance = tf.nn.moments(
            style_features, axes=[1, 2], keepdims=True
        )
        content_std = tf.math.sqrt(content_variance+self.epsilon)
        style_std = tf.math.sqrt(style_variance+self.epsilon)
        adain_content_features = (content_features-content_mean)/content_std*style_std+style_mean

        adain_content_features = (
            self.alpha * adain_content_features +
            (1 - self.alpha) * content_features
        )
        return adain_content_features


# This can be simplified by directly loading VGG19 pretrained model
class encoder(tf.keras.Model):
    def __init__(self, layer_names):
        super(encoder, self).__init__()

        self.vgg19 = tf.keras.applications.VGG19(
            include_top=False, weights="imagenet")

        self.vgg19.trainable = False
        outputs = [self.vgg19.get_layer(name).output for name in layer_names]

        self.vgg19_encoder = tf.keras.Model(
            inputs=self.vgg19.input, outputs=outputs
        )

    def call(self, x):
        return self.vgg19_encoder(x)


class decoder(tf.keras.Model):
    def __init__(self):
        super(decoder, self).__init__()

        self.vgg19_decoder = tf.keras.Sequential(
            [
                Conv2D(256, 3, 1, padding="same", activation="relu"),
                UpSampling2D((2, 2)),
                Conv2D(256, 3, 1, padding="same", activation="relu"),
                Conv2D(256, 3, 1, padding="same", activation="relu"),
                Conv2D(256, 3, 1, padding="same", activation="relu"),
                Conv2D(128, 3, 1, padding="same", activation="relu"),
                UpSampling2D((2, 2)),
                Conv2D(128, 3, 1, padding="same", activation="relu"),
                Conv2D(64, 3, 1, padding="same", activation="relu"),
                UpSampling2D((2, 2)),
                Conv2D(64, 3, 1, padding="same", activation="relu"),
                Conv2D(3, 3, 1, padding="same"),
            ],
            name="vgg19_decoder",
        )

    def call(self, x):
        return self.vgg19_decoder(x)

class AdaIN_NST(tf.keras.Model):
    def __init__(self, epsilon=hp.epsilon, alpha=hp.alpha, s_lambda=hp.s_lambda, lr=hp.learning_rate):
        super(AdaIN_NST, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.s_lambda = s_lambda
        self.enc = encoder(['block1_conv1','block2_conv1','block3_conv1','block4_conv1'])
        self.adain = AdaIN(epsilon, alpha)
        self.dec = decoder()
        self.optimizer = tf.keras.optimizers.Adam(lr)

    def call(self, content, style):
        enc_content = self.enc(content)
        enc_style = self.enc(style)
        adain_content = self.adain(enc_content[-1],enc_style[-1])
        output = self.dec(adain_content)
        return [enc_style, adain_content, output]

    def loss_fn(self, enc_style, adain_content, output):
        output = deprocess_img(output)
        output = tf.keras.applications.vgg19.preprocess_input(output)
        enc_adain = self.enc(output)
        content_loss = tf.reduce_sum(tf.square(enc_adain[-1]-adain_content))
        style_loss_list = []
        for i in range(len(enc_adain)):
            style_mean, style_variance = tf.nn.moments(
                enc_style[i], axes=[1, 2]
            )
            adain_mean, adain_variance = tf.nn.moments(
                enc_adain[i], axes=[1, 2]
            )
            style_std = tf.math.sqrt(style_variance+self.epsilon)
            adain_std = tf.math.sqrt(adain_variance+self.epsilon)
            layer_loss = tf.reduce_sum(tf.square(style_mean-adain_mean)+tf.square(style_std-adain_std))
            style_loss_list.append(layer_loss)
        style_loss = tf.reduce_sum(style_loss_list)
        return content_loss+self.s_lambda*style_loss
