"""
Modified from Project 4, CS1430
"""

import os
import argparse
import datetime
import tensorflow as tf
import numpy as np
from skimage.io import imshow, imread, imsave
import hyperparameters as hp

from models import decoder, encoder, AdaIN, AdaIN_NST

from preprocess import ImageDataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def train(model, content_data, style_data):
    np.random.shuffle(content_data)
    np.random.shuffle(style_data)
    nbatch = len(content_data) // hp.batch_size
    for i in range(nbatch):
        batch_content = content_data[i * hp.batch_size : (i + 1) * hp.batch_size]
        batch_style = style_data[i * hp.batch_size : (i + 1) * hp.batch_size]
        with tf.GradientTape() as tape:
            enc_style, adain_content, output = model(batch_content, batch_style)
            loss = model.loss_fn(enc_style, adain_content, output)
        print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_data):
    pass


def main():
    """
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    if ARGS.load_checkpoint:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)

    content_data = ImageDataset(content_path, batch_size=hp.batch_size)
    style_data = ImageDataset(style_path, batch_size=hp.batch_size)

    os.chdir(sys.path[0])
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # content_data = np.expand_dims(
    # np.array(imread('c/brownspring.jpg'), dtype=np.float32)/255., 0)
    # print(content_data.shape)
    # style_data = np.expand_dims(
    # np.array(imread('s/starry_night.jpg'), dtype=np.float32)/255., 0)

    dataset = ImageDataset("./images/")

    model = AdaIN_NST()
    for epoch in range(hp.num_epochs):
        train(model, dataset.content_data, dataset.style_data)
        time = datetime.datetime.now()
        model.save(
            "./checkpoints/models/epoch{}_{}_{}_{}.h5".format(
                epoch, time.date(), time.hour, time.minute
            )
        )
    # imsave("test.png", model(content_data, style_data)[-1][0].numpy())


if __name__ == "__main__":
    main()
