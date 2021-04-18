from skimage.io import imsave
from preprocess import ImageDataset
import tensorflow as tf

datasets = ImageDataset("./examples/content", "./examples/style")
from models import AdaIN_NST, deprocess_img

model = tf.keras.models.load_model("./output/checkpoints/041721-201958/epoch399")
model.get_weights()
imsave(
    "./examples/output/test.png",
    deprocess_img(
        model([next(datasets.content_data), next(datasets.style_data)])[-1].numpy()
    )[0],
)
