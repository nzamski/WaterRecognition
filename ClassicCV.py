import cv2
import os
from pathlib import Path
from random import shuffle
import skimage.segmentation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import pyplot as plt


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
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    cv_gray = skimage.segmentation.chan_vese(gray)
    return cv_gray


if __name__ == '__main__':
    train, test = get_train_test_paths()
    for path in train:
        print(str(path))
        mask = cv2.imread(get_mask_path(path))
        plt.subplot(2, 1, 1)
        plt.imshow(chan_vese(str(path)), cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(mask, cmap='gray')
        plt.show()
        break
