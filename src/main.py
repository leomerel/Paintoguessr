import numpy as np
from PIL import Image as pilImage

imgPath = "../img/airplane.npy"

def img_to_array(imgPath):
    img = pilImage.open(imgPath)
    return np.array(img)

def array_to_img(array, title):
    img = pilImage.fromarray(array.reshape(28, 28), 'L')
    img.save(title)


if __name__ == '__main__':
    imgs = np.load(imgPath)
    img_title = "../output/airplane" + str(0) + ".png"
    array_to_img(imgs[0], img_title)

