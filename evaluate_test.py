from skimage.io import imsave
from preprocess import ImageDataset, get_image
import tensorflow as tf
from models import AdaIN_NST, deprocess_img

content_image = get_image("./examples/content/brownspring.jpg")
style_image = get_image("./examples/style/starry_night.jpg")

output_name = "brownspring_starry_night.jpg"
model=AdaIN_NST("./vgg19_normalised.npz")
model.load_weights("./output/checkpoints/training")
'''
model = tf.keras.models.load_model(
    "./output/checkpoints/041821-124334/epoch0/", compile=False
)
'''
imsave(
    "./examples/output/test.png",
    deprocess_img(model([content_image, style_image])[-1].numpy())[0],
)
