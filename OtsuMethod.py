import os
import PIL
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt


def get_source_paths():
    # extract the data from the dataset folder
    files = [file_name for file_name in
             Path(os.getcwd() + os.sep + 'Water Bodies Dataset' + os.sep + 'Images').rglob("*.jpg")]
    return files


def get_mask(file_path):
    # gets source image path, returns mask path
    file_path = str(file_path).replace('Images', 'Masks')
    mask = cv2.imread(file_path, 0)
    mask = (mask < 128).astype(int) * 255
    return mask


def get_img_index(path):
    index = str(path).split('_')[-1].split('.')[0]
    return index


def otsu_predict(source_path, coeff=1):
    source_img = cv2.imread(str(source_path), 0)
    mask_img = get_mask(source_path)
    ret, thresh = cv2.threshold(source_img, 0, 255, int((cv2.THRESH_BINARY + cv2.THRESH_OTSU)*coeff))
    binary = (source_img > thresh)

    # switch thresholding if classes are inverted
    colored_img = cv2.imread(str(source_path))
    blue = colored_img[:, :, 2]
    binary = binary if np.mean(blue[binary]) > np.mean(~blue[binary]) else ~binary
    binary = binary.astype(int)*255

    correct = np.sum(binary == mask_img)
    size = source_img.shape[0] * source_img.shape[1]

    return binary, correct, size, coeff


def main():
    correct_pixels, image_size, indices, coeffs = list(), list(), list(), list()
    source_paths = get_source_paths()
    for i in [9, 10, 12]:
        for source_path in tqdm(source_paths):
            segmented, correct, size, coeff = otsu_predict(source_path, i/10)
            correct_pixels.append(correct)
            image_size.append(size)
            index = get_img_index(source_path)
            indices.append(index)
            coeffs.append(coeff)
            cv2.imwrite(f'Otsu Images{os.sep}-{coeff}-{index}.png', segmented)
    df = pd.DataFrame(data={'correct_pixels': correct_pixels, 'image_size': image_size,
                            'index': indices, 'coeff': coeffs})
    df['accuracy'] = df['correct_pixels'] / df['image_size']
    df.to_csv('Otsu_Results.csv', index=False)


if __name__ == '__main__':
    main()
    # source = 'D:/Noam/Desktop/img.jpg'
    # segmented, correct, size, coeff = otsu_predict(source)
    # print(correct / size)
    # image = PIL.Image.fromarray(segmented)
    # image.show()
