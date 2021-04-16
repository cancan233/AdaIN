"""
Modified from Project 4, CS1430
"""

import os
import argparse
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import decoder, encoder, AdaIN, AdaIN_NST


def loss_fn():
    pass


def train():
    pass


def test(model, test_data):
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    if ARGS.load_checkpoint in not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)

    content_data = ImageDataset(content_path, batch_size=hp.batch_size)
    style_data = ImageDataset(style_path, batch_size=hp.batch_size)

    os.chdir(sys.path[0])

    model = AdaIN_NST()
    model.compile(
        optimizer=model.optimizer,
        loss=loss_fn,
    )

    train(model)


if __name__ == "__main__":
    main()
