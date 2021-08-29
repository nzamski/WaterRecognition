import os
import cv2
import skimage.segmentation
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from PIL import Image
from pathlib import Path
from random import shuffle
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu


def get_train_test_paths(test_ratio: float = 0.2):
    # extract the data from the dataset folder
    files = [file_name for file_name in Path(os.getcwd()+os.sep+'Water Bodies Dataset'+os.sep+'Images').rglob("*.jpg")]
    # randomize the order of the data
    shuffle(files)
    # separate test and train files
    first_train = int(test_ratio * len(files))
    test_path = files[:first_train]
    train_path = files[first_train:]
    return train_path, test_path


def get_mask_path(file_path):
    # gets source image path, returns mask path
    file_path = str(file_path).replace('Images', 'Masks')
    return file_path


def chan_vese(path):
    img = cv2.imread(path)
    blur = cv2.GaussianBlur(img, (11, 11), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    cv_gray = skimage.segmentation.chan_vese(gray)
    mask = cv2.imread(get_mask_path(path))

    plt.subplot(1, 2, 1)
    plt.imshow(cv_gray, cmap='gray')
    plt.title("Chan Vese")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")

    plt.show()


def threshold_otsu(path):
    img = cv2.imread(path, 0)
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = img > thresh

    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].hist(img.ravel(), bins=256)
    ax[1].set_title('Histogram')
    ax[1].axvline(float(ret), color='r')

    ax[2].imshow(binary, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded')
    ax[2].axis('off')

    plt.show()


if __name__ == '__main__':
    train, test = get_train_test_paths()
    for path in train[:5]:
        print(path)
        threshold_otsu(str(path))
