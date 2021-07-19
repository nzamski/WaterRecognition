import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from PIL import Image
from cv2 import imread
from pathlib import Path
from random import shuffle
from TrainMaskAround import train_and_test_files_paths, load_image, get_mask_path
from collections import defaultdict
from tqdm import tqdm


def rgb_shaper(path):
    img = load_image(path)
    target = imread(get_mask_path(path), 0)
    pixel_as_row = img.reshape(-1, 3)
    target = target.reshape(-1)
    return pixel_as_row, target


def count_all_rgbs_mask_combs(path_list):
    example_count = defaultdict(int)
    for path in tqdm(path_list):
        pixel_as_row, target = rgb_shaper(path)
        data_frame = pd.DataFrame(data=pixel_as_row, columns=['r','g','b'])
        data_frame['value'] = target
        rows = data_frame.groupby(['r', 'g', 'b', 'value']).size().reset_index(name='counts').to_dict('split')['data']
        for row in rows:
            key = tuple(row[:-1])
            count = row[-1]
            example_count[key] += count
    r = list()
    g = list()
    b = list()
    values = list()
    counts = list()
    for key, value in tqdm(example_count.items()):
        r.append(key[0])
        g.append(key[1])
        b.append(key[2])
        values.append(key[3])
        counts.append(value)
    counts_df = pd.DataFrame(data={'r': r, 'g': g, 'b': b, 'value': values, 'count': counts})
    return counts_df


if __name__ == '__main__':
    train, test = train_and_test_files_paths()
    rgb_value_count = count_all_rgbs_mask_combs(train+test)
    rgb_value_count.to_csv('rgb_value_count.csv')
