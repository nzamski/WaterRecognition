import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path


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


def otsu_predict(source_path):
    source_img = cv2.imread(str(source_path), 0)
    mask_img = get_mask(source_path)
    ret, thresh = cv2.threshold(source_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    size = source_img.shape[0] * source_img.shape[1]
    # switch thresholding if classes are inverted
    colored_img = cv2.imread(str(source_path))
    red = colored_img[:, :, 0]
    thresh = thresh == 255
    thresh = thresh if np.mean(red[thresh]) < np.mean(red[~thresh]) else ~thresh
    thresh = thresh.astype(int) * 255

    if ((thresh == 255) & (mask_img == 255)).sum().item() == 0:
        return thresh, size, 1, ret

    predicted_positive = (thresh == 255).sum().item()
    true_positive = ((thresh == 255) & (mask_img == 255)).sum().item()
    false_negative = ((thresh == 0) & (mask_img == 255)).sum().item()

    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / predicted_positive

    f1 = (2 * precision * recall) / (precision + recall)

    return thresh, size, f1, ret


def main():
    scores, image_size, indices, thresholds = list(), list(), list(), list()
    source_paths = get_source_paths()
    for source_path in tqdm(source_paths):
        segmented, size, f1, ret = otsu_predict(source_path)
        scores.append(f1)
        thresholds.append(ret)
        image_size.append(size)
        index = get_img_index(source_path)
        indices.append(index)
        cv2.imwrite(f'Otsu Images{os.sep}{index}.jpg', segmented)
    df = pd.DataFrame(data={'F1': scores, 'Threshold': thresholds, 'Image Size': image_size, 'Index': indices})
    df.to_csv('Otsu_Results.csv', index=False)


if __name__ == '__main__':
    main()
    # source = 'D:/Noam/Desktop/img.jpg'
    # segmented, size, f1 = otsu_predict(source)
    # print(f1)
    # image = PIL.Image.fromarray(segmented)
    # image.show()
