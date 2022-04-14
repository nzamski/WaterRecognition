import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from datetime import datetime


def get_source_paths():
    # extract the data from the dataset folder
    files = [str(file_name) for file_name in
             Path(os.getcwd() + os.sep + 'Water Bodies Dataset' + os.sep + 'Images').rglob("*.jpg")]
    return files


def get_mask(file_path):
    # gets source image path, returns mask path
    file_path = file_path.replace('Images', 'Masks')
    mask = cv2.imread(file_path, 0)
    mask = (mask < 128).astype(int) * 255
    return mask


def get_img_index(path):
    index = path.split('_')[-1].split('.')[0]
    return index


def otsu_predict(source_path):
    img = cv2.imread(source_path)
    height, width = img.shape[0], img.shape[1]
    mask_img = get_mask(source_path)
    gray_img = cv2.imread(source_path, 0)

    start = datetime.now()
    ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    end = datetime.now()
    duration = (end - start).total_seconds()

    # rule for thresh inversion by the red channel
    red = img[:, :, 0]
    thresh = thresh == 255
    thresh = thresh if np.mean(red[thresh]) > np.mean(red[~thresh]) else ~thresh
    thresh = thresh.astype(int) * 255

    true_positive = ((thresh == 255) & (mask_img == 255)).sum().item()
    false_positive = ((thresh == 255) & (mask_img == 0)).sum().item()
    false_negative = ((thresh == 0) & (mask_img == 255)).sum().item()

    denominator = (true_positive + 0.5 * (false_positive + false_negative))
    if denominator == 0:
        return thresh, height * width, 0, ret

    f1 = true_positive / denominator

    return thresh, height * width, f1, ret, duration


def main():
    scores, thresholds, image_size, indices, durations = list(), list(), list(), list(), list()
    source_paths = get_source_paths()
    for source_path in tqdm(source_paths):
        prediction, size, f1, ret, duration = otsu_predict(source_path)
        scores.append(f1)
        thresholds.append(ret)
        image_size.append(size)
        index = get_img_index(source_path)
        indices.append(index)
        durations.append(duration)
        path = f'{os.getcwd()}{os.sep}Otsu Images{os.sep}otsu_{index} ({round(f1, 3)}).jpg'
        plt.imsave(path, prediction, cmap='Greys')
    df = pd.DataFrame(data={'F1': scores,
                            'Threshold': thresholds,
                            'Image Size': image_size,
                            'Index': indices,
                            'Duration': durations})
    df.to_csv('Otsu_Results.csv', index=False)


if __name__ == '__main__':
    main()
