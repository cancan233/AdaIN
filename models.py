# Based on https://github.com/JunbinWang/Tensorflow-Style-Transfer-with-Adain/blob/master/adain_norm.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Layer,
    Conv2D,
    UpSampling2D,
    MaxPooling2D,
    Lambda,
    Input,
)

import hyperparameters as hp


def deprocess_img(img):
    tmp = img + np.array([103.939, 116.779, 123.68])
    tmp = tmp[:, :, :, ::-1]
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
        content_std = tf.math.sqrt(content_variance + self.epsilon)
        style_std = tf.math.sqrt(style_variance + self.epsilon)
        adain_content_features = (
            content_features - content_mean
        ) / content_std * style_std + style_mean

        adain_content_features = (
            self.alpha * adain_content_features + (1 - self.alpha) * content_features
        )
        return adain_content_features


# This can be simplified by directly loading VGG19 pretrained model
class encoder(tf.keras.Model):
    def __init__(self, layer_names, weight_path):
        super(encoder, self).__init__()

        self.vgg19 = tf.keras.applications.VGG19(
            include_top=False, weights=None, input_shape=hp.input_shape
        )

        weights = np.load(weight_path)
        i = 0
        for layer in self.vgg19.layers:
            if layer.name[-5:-1] == "conv":
                kernel = weights["arr_%d" % i].transpose([2, 3, 1, 0])
                kernel = kernel.astype(np.float32)
                bias = weights["arr_%d" % (i + 1)]
                bias = bias.astype(np.float32)
                layer.set_weights([kernel, bias])
                i += 2

        layers = [l for l in self.vgg19.layers]
        x = layers[0].output
        for i in range(1, len(layers)):
            if layers[i].name[-5:-1] == "conv":
                x = Lambda(
                    lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
                )(x)
                setattr(layers[i], "padding", "valid")
            x = layers[i](x)
        self.vgg19 = tf.keras.Model(inputs=layers[0].input, outputs=x)

        self.vgg19.trainable = False
        outputs = [self.vgg19.get_layer(name).output for name in layer_names]

        self.vgg19_encoder = tf.keras.Model(inputs=self.vgg19.input, outputs=outputs)

    def call(self, x):
        return self.vgg19_encoder(x)


class decoder(tf.keras.Model):
    def __init__(self):
        super(decoder, self).__init__()

        self.vgg19_decoder = tf.keras.Sequential(
            [
                Conv2D(
                    256,
                    3,
                    1,
                    padding="valid",
                    activation="relu",
                    name="dec_block4_conv1",
                ),
                UpSampling2D((2, 2), name="dec_block3_pool"),
                Conv2D(
                    256,
                    3,
                    1,
                    padding="valid",
                    activation="relu",
                    name="dec_block3_conv4",
                ),
                Conv2D(
                    256,
                    3,
                    1,
                    padding="valid",
                    activation="relu",
                    name="dec_block3_conv3",
                ),
                Conv2D(
                    256,
                    3,
                    1,
                    padding="valid",
                    activation="relu",
                    name="dec_block3_conv2",
                ),
                Conv2D(
                    128,
                    3,
                    1,
                    padding="valid",
                    activation="relu",
                    name="dec_block3_conv1",
                ),
                UpSampling2D((2, 2), name="dec_block2_pool"),
                Conv2D(
                    128,
                    3,
                    1,
                    padding="valid",
                    activation="relu",
                    name="dec_block2_conv2",
                ),
                Conv2D(
                    64,
                    3,
                    1,
                    padding="valid",
                    activation="relu",
                    name="dec_block2_conv1",
                ),
                UpSampling2D((2, 2), name="dec_block1_pool"),
                Conv2D(
                    64,
                    3,
                    1,
                    padding="valid",
                    activation="relu",
                    name="dec_block1_conv2",
                ),
                Conv2D(3, 3, 1, padding="valid", name="dec_block1_conv1"),
            ],
            name="vgg19_decoder",
        )

        layers = [l for l in self.vgg19_decoder.layers]
        inputs = Input(shape=(None, None, 512))
        x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT"))(
            inputs
        )
        x = layers[0](x)
        for i in range(1, len(layers)):
            if layers[i].name[-5:-1] == "conv":
                x = Lambda(
                    lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
                )(x)
            x = layers[i](x)
        self.vgg19_decoder = tf.keras.Model(inputs=inputs, outputs=x)

    def call(self, x):
        return self.vgg19_decoder(x)


class AdaIN_NST(tf.keras.Model):
    def __init__(self, weight_path, alpha=hp.alpha):
        super(AdaIN_NST, self).__init__()
        self.epsilon = hp.epsilon
        self.alpha = alpha
        self.style_lambda = hp.style_lambda
        self.enc = encoder(
            ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"],
            weight_path,
        )
        self.adain = AdaIN(hp.epsilon, alpha)
        self.dec = decoder()
        self.optimizer = tf.keras.optimizers.Adam(
            (
                tf.keras.optimizers.schedules.InverseTimeDecay(
                    hp.learning_rate, decay_steps=1, decay_rate=5e-5
                )
            )
        )

    def call(self, content, style):
        enc_content = self.enc(content)
        enc_style = self.enc(style)
        adain_content = self.adain(enc_content[-1], enc_style[-1])
        output = self.dec(adain_content)
        output = deprocess_img(output)
        return [enc_style, adain_content, output]

    def loss_fn(self, enc_style, adain_content, output):
        # output = deprocess_img(output)
        output = tf.keras.applications.vgg19.preprocess_input(output)
        enc_adain = self.enc(output)
        content_loss = tf.reduce_sum(
            tf.reduce_mean(tf.square(enc_adain[-1] - adain_content), axis=(1, 2))
        )
        style_loss_list = []
        for i in range(len(enc_adain)):
            style_mean, style_variance = tf.nn.moments(enc_style[i], axes=[1, 2])
            adain_mean, adain_variance = tf.nn.moments(enc_adain[i], axes=[1, 2])
            style_std = tf.math.sqrt(style_variance + self.epsilon)
            adain_std = tf.math.sqrt(adain_variance + self.epsilon)
            layer_loss = tf.reduce_sum(
                tf.square(style_mean - adain_mean) + tf.square(style_std - adain_std)
            )
            style_loss_list.append(layer_loss)
        style_loss = tf.reduce_sum(style_loss_list)
        return content_loss + self.style_lambda * style_loss
