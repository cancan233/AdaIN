from skimage.io import imsave
from preprocess import ImageDataset, get_image
import tensorflow as tf
from models import AdaIN_NST, deprocess_img

content_image = get_image("./examples/content/brownspring.jpg")
style_image = get_image("./examples/style/starry_night.jpg")

output_name = "brownspring_starry_night.jpg"
model = tf.keras.models.load_model(
    "./output/checkpoints/041721-230915/epoch0", compile=False
)
imsave(
    "./examples/output/test.png",
    deprocess_img(model([content_image, style_image])[-1].numpy())[0],
)
