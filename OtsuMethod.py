import os
import cv2
import PIL.Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from datetime import datetime


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


def hist_plot(source_path, segmented, ret):
    source_img = cv2.imread(str(source_path))
    gray_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)

    hist_r = cv2.calcHist([source_img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([source_img], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([source_img], [2], None, [256], [0, 256])
    hist_k = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    plt.subplot(221), plt.imshow(source_img)
    plt.subplot(222), plt.imshow(segmented, cmap='gray')
    plt.subplot(223), plt.plot(hist_r, 'r'), plt.plot(hist_g, 'g'), plt.plot(hist_b, 'b')
    plt.axvline(ret, color='y')
    plt.subplot(224), plt.plot(hist_k, 'k')
    plt.xlim([0, 256])
    plt.axvline(ret, color='y')
    plt.xlim([0, 256])

    plt.show()


def otsu_predict(source_path, pad, k, dif):
    gray_img = cv2.imread(str(source_path), 0)[20:-20, 20:-20]
    mask_img = get_mask(source_path)[20:-20, 20:-20]
    height, width = gray_img.shape[0], gray_img.shape[1]

    start = datetime.now()
    ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    end = datetime.now()
    seconds = (end - start).total_seconds()

    if height * width == 0:
        return cv2.imread('empty.jpg'), 1, 0, ret, seconds

    # rule for thresh inversion
    frame = thresh.copy()
    # replace all the inside with neutral values
    frame[pad:-pad, pad:-pad] = -1
    frame[frame == 0] = 1  # to have a valid blacks count
    blacks = np.count_nonzero(frame == 1)
    whites = np.count_nonzero(frame == 255)
    if abs(blacks - whites) > dif:
        thresh = thresh if blacks > whites else 255 - thresh
    else:
        # get the central slice
        center = thresh[int(height / 2) - 1:int(height / 2) - 1 + k, int(width / 2) - 1:int(width / 2) - 1 + k]
        center_avg = np.mean(center)
        thresh = thresh if center_avg > 128 else 255 - thresh

    # old rule by red channel
    # colored_img = cv2.imread(str(source_path))[20:-20, 20:-20, :]
    # red = colored_img[:, :, 0]
    # thresh = thresh == 255
    # thresh = thresh if np.mean(red[thresh]) < np.mean(red[~thresh]) else ~thresh
    # thresh = thresh.astype(int) * 255

    predicted_positive = (thresh == 255).sum().item()
    true_positive = ((thresh == 255) & (mask_img == 255)).sum().item()
    false_negative = ((thresh == 0) & (mask_img == 255)).sum().item()

    if true_positive == 0 or predicted_positive == 0:
        return thresh, width * height, 1, ret, seconds

    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / predicted_positive

    f1 = (2 * precision * recall) / (precision + recall)

    return thresh, height * width, f1, ret, seconds


def main():
    scores, durations, thresholds, image_size, indices = list(), list(), list(), list(), list()
    source_paths = get_source_paths()
    for source_path in tqdm(source_paths):
        segmented, size, f1, ret, seconds = otsu_predict(source_path)
        scores.append(f1)
        durations.append(seconds)
        thresholds.append(ret)
        image_size.append(size)
        index = get_img_index(source_path)
        indices.append(index)
        cv2.imwrite(f'Otsu Images{os.sep}{index}.jpg', segmented)
    df = pd.DataFrame(data={'F1': scores,
                            'Seconds': durations,
                            'Threshold': thresholds,
                            'Image Size': image_size,
                            'Index': indices})
    # major = pd.DataFrame(data={'Padding': [pad],
    #                            'Kernel Size': [k],
    #                            'Dif': [dif],
    #                            'F1 Median': [df['F1'].median()],
    #                            'F1 Mean': [df['F1'].mean()]}, index=[0])
    # major.to_csv('Otsu_Conditions.csv', index=False, mode='a', header=False)
    df.to_csv('Otsu_Results.csv', index=False)


if __name__ == '__main__':
    main()
