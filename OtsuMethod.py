import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt


def get_source_paths():
    # extract the data from the dataset folder
    files = [file_name for file_name in Path(os.getcwd()+os.sep+'Water Bodies Dataset'+os.sep+'Images').rglob("*.jpg")]
    return files


def get_mask(file_path):
    # gets source image path, returns mask path
    file_path = str(file_path).replace('Images', 'Masks')
    mask = cv2.imread(file_path, cv2.imread_grayscale)
    return mask


def get_img_index(path):
    index = str(path).split('_')[-1].split('.')[0]
    return index


def otsu_predict(source_path):
    source_img = cv2.imread(str(source_path), cv2.imread_grayscale)
    mask_img = get_mask(source_path)
    ret, thresh = cv2.threshold(source_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = source_img > thresh

    correct = np.sum(binary == mask_img)
    size = source_img.shape[0] * source_img.shape[1]
    gray = binary * 255

    return gray, correct, size


def main():
    correct_pixels, image_size, indices = list(), list(), list()
    source_paths = get_source_paths()
    for source_path in tqdm(source_paths):
        gray, correct, size = otsu_predict(source_path)
        correct_pixels.append(correct)
        image_size.append(size)
        index = get_img_index(source_path)
        indices.append(index)
        cv2.imwrite(f'Otsu Images{os.sep}{index}.png', gray)
    df = pd.DataFrame(data={'correct_pixels': correct_pixels, 'image_size': image_size, 'index': indices})
    df['accuracy'] = df['correct_pixels'] / df['image_size']
    df.to_csv('Otsu_Results.csv', index=False)

# def plots():
#     fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
#     ax = axes.ravel()
#     ax[0] = plt.subplot(1, 3, 1)
#     ax[1] = plt.subplot(1, 3, 2)
#     ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])
#
#     ax[0].imshow(img, cmap=plt.cm.gray)
#     ax[0].set_title('Original')
#     ax[0].axis('off')
#
#     ax[1].hist(img.ravel(), bins=256)
#     ax[1].set_title('Histogram')
#     ax[1].axvline(float(ret), color='r')
#
#     ax[2].imshow(binary, cmap=plt.cm.gray)
#     ax[2].set_title('Thresholded')
#     ax[2].axis('off')
#
#     plt.show()


if __name__ == '__main__':
    main()
