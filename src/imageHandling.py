import numpy as np
from PIL import Image as pilImage

def img_to_array(imgPath):
    img = pilImage.open(imgPath)
    return np.array(img)

def array_to_img(array, filename):
    img = pilImage.fromarray(array.reshape(28, 28), 'L')
    img.save(filename)