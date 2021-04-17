from skimage.io import imsave
from preprocess import ImageDataset
datasets = ImageDataset()
from models import AdaIN_NST, deprocess_img
model = AdaIN_NST()
model.load_weights('checkpoints/models/epoch499_2021-04-17_10_33')
imsave("test.png", deprocess_img(model(next(datasets.content_data), next(datasets.style_data))[-1].numpy())[0])
