"""
Modified from Project 4, CS1430
"""

import os
import argparse
import datetime
import tensorflow as tf
import numpy as np
from skimage.io import imshow, imsave
import hyperparameters as hp

from models import decoder, encoder, AdaIN, AdaIN_NST, deprocess_img

from preprocess import ImageDataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def train(model, content_data, style_data):
    with tf.GradientTape() as tape:
        enc_style, adain_content, output = model(content_data, style_data)
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

    """
    datasets = ImageDataset()

#    os.chdir(sys.path[0])
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    model = AdaIN_NST()
    epo_interval = 50
    for epoch in range(hp.num_epochs):
        for i in range(min(len(datasets.content_data),len(datasets.style_data))):
            content_data = next(datasets.content_data)
            style_data = next(datasets.style_data)
            train(model, content_data, style_data)
        time = datetime.datetime.now()
        if not (epoch+1) % epo_interval:
            fname = "./checkpoints/models/epoch{}_{}_{}_{}".format(
                epoch, time.date(), time.hour, time.minute
            )
            model.save_weights(
                fname, save_format="tf"
            )


    imsave("test.png", deprocess_img(model(next(datasets.content_data), next(datasets.style_data))[-1].numpy())[0])


if __name__ == "__main__":
    main()
