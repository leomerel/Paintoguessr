from os import listdir
from os.path import isfile, join
import numpy as np

from sklearn.model_selection import train_test_split

path = "./numpy_bitmap/img/"

def createSets():
    npyFiles = [f for f in listdir(path) if isfile(join(path, f))]

    classNames = []
    images = []
    labels = []

    for i in range(len(npyFiles)):
        print(npyFiles[i])
        imgs = np.load(path + npyFiles[i])
        className = npyFiles[i].split('.n')[0]
        classNames.append(className)
        for j in range(len(imgs)):
            if j >= 10000:  # We limit the number of images per class
                break
            images.append(imgs[j].reshape(28, 28))
            labels.append(i)

    return classNames, np.array(images), np.array(labels)


def splitData(data, target, test_size=0.1, shuffle=True):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=42,
                                                        shuffle=shuffle)
    return X_train, X_test, y_train, y_test


def formatImages(images):
    return images / 255.0

