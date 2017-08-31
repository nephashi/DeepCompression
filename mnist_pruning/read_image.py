import numpy as np
from PIL import Image

def read_image(path):
    tmp =  np.array(Image.open(path).resize((28, 28), resample=2))
    img = np.zeros((28, 28, 1))
    img[:, :, 0] = tmp[:, :, 0]
    return img