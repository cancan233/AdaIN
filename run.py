"""
Modified from Project 4, CS1430
"""

import os
import glob
import sys
import argparse
import re
import datetime
import tensorflow as tf
import numpy as np
from skimage.io import imshow, imread, imsave
import hyperparameters as hp
from models import decoder, encoder, AdaIN, AdaIN_NST, deprocess_img

from preprocess import ImageDataset, get_image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args():
    """Usage: Train: python run.py --content-dir [content-dir] --style-dir [style-dir] \n
    Test: python run.py --load-checkpoint [checkpoint-file] --evaluate --content-evaulate [content-img] --style-evaluate [style-img]"""

    parser = argparse.ArgumentParser(
        description="arguments parser for AdaIN-NST model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--content-dir",
        default="." + os.sep + "images" + os.sep + "content",
        help="directory where the content is stored",
    )
    parser.add_argument(
        "--style-dir",
        default="." + os.sep + "images" + os.sep + "style",
        help="directory where the style is stored",
    )
    parser.add_argument(
        "--pretrained-vgg19",
        default="./images/vgg19_normalised.npz",
        help="fila path for vgg19 pretrained weights, end with .npz",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="not save the checkpoints, logs and model. Used only for develop purpose",
    )
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help="path to model checkpoint file, should be similar to ./output/checkpoints/041721-201121/epoch19",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="skips training and evalutes the style transfer on examples once.",
    )
    parser.add_argument("--content-evaluate", help="path the image to transform")
    parser.add_argument("--style-evaluate", help="path to style image")
    return parser.parse_args()


class CustomModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        super(CustomModelSaver, self).__init__()
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, logs=None):
        save_name = "epoch{}".format(epoch)
        tf.keras.models.save_model(
            self.model, self.checkpoint_path + os.sep + save_name, save_format="tf"
        )


def train(model, content_data, style_data, logs_path, checkpoint_path):
    input = [content_data, style_data]

    # callback_list = [
    #     tf.keras.callbacks.TensorBoard(
    #         log_dir=logs_path, update_freq="batch", profile_batch=0
    #     ),
    #     CustomModelSaver(checkpoint_path),
    # ]
    # model.fit(
    #     x=input,
    #     epochs=hp.num_epochs,
    #     batch_size=None,
    #     callbacks=callback_list,
    #     # initial_epoch=init_epoch,
    # )

    with tf.GradientTape() as tape:
        enc_style, adain_content, output = model(input)
        loss = model.loss_fn(enc_style, adain_content, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def test(model, content_image, style_image, output_name):
    if os.path.isdir("./examples/results"):
        os.mkdir("./examples/results")
    imsave(
        "./examples/results/{}".format(output_name),
        deprocess_img(model([content_image, style_image])[-1].numpy())[0],
    )


def main():
    time_now = datetime.datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    content_images = glob.glob(ARGS.content_dir)
    style_images = glob.glob(ARGS.style_dir)
    num_images = min(len(content_images), len(style_images))
    content_images = content_images[:num_images]
    style_images = style_images[:num_images]

    model = AdaIN_NST(ARGS.pretrained_vgg19)
    checkpoint_path = "./output/checkpoints" + os.sep + timestamp + os.sep
    logs_path = "./output/logs" + os.sep + timestamp + os.sep
    logs_path = os.path.abspath(logs_path)
    model.compile(optimizer=model.optimizer, loss=model.loss_fn)

    if ARGS.load_checkpoint is not None:
        # model = tf.keras.models.load_model(ARGS.load_checkpoint, compile=False)
        model.load_weights(ARGS.load_checkpoint)

    if not ARGS.evaluate and not os.path.exists(checkpoint_path) and not ARGS.no_save:
        os.makedirs(checkpoint_path)
        os.makedirs(logs_path)

    if ARGS.evaluate:
        content_image = get_image(ARGS.content_evaluate)
        style_image = get_image(ARGS.style_evaluate)
        output_name = (
            os.path.split(ARGS.content_evaluate)[1][:-4]
            + "_"
            + os.path.split(ARGS.style_evaluate)[1]
        )
        test(model, content_image, style_image, output_name)

    else:
        num_batches = int(num_images // hp.batch_size)
        for epoch in range(hp.num_epochs):
            np.random.shuffle(content_images)
            np.random.shuffle(style_images)

            for batch in range(num_batches):
                content_batch_path = content_images[
                    batch * hp.batch_size : (batch * hp.batch_size + hp.batch_size)
                ]
                style_batch_path = style_images[
                    batch * hp.batch_size : (batch * hp.batch_size + hp.batch_size)
                ]

                content_batch = 
                style_batch = 
                train(model, content_batch, style_batch, logs_path, checkpoint_path)
        # datasets = ImageDataset(ARGS.content_dir, ARGS.style_dir)
        # train(
        #     model,
        #     datasets.content_data,
        #     datasets.style_data,
        #     logs_path,
        #     checkpoint_path,
        # )

        """
        num_batches = min(len(datasets.content_data), len(datasets.style_data))
        for epoch in range(hp.num_epochs):
            for i in range(num_batches):
                content_data = next(datasets.content_data)
                style_data = next(datasets.style_data)
                loss = train(
                    model, content_data, style_data, logs_path, checkpoint_path
                )
                if i % 10 == 0:
                    tf.print(
                        "Epoch {}\t Batch {}\t: Loss {}\t".format(epoch, i, loss),
                        output_stream=sys.stdout
                        # output_stream="file://{}/loss.log".format(logs_path),
                    )
            save_name = "epoch{}".format(epoch)
            if not ARGS.no_save:
                model.save_weights(filepath=checkpoint_path + os.sep + save_name)
        """


if __name__ == "__main__":
    ARGS = parse_args()
    main()
