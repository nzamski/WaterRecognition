import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from PIL import Image
from cv2 import imread
from pathlib import Path
from random import shuffle, seed

# set a fixed seed to shuffle paths by
seed(42)


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


def load_image(file_name):
    # get image path and return as array
    img = Image.open(file_name)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def get_mask_path(file_path):
    # gets source image path, returns mask path
    file_path = str(file_path).replace('Images', 'Masks')
    return file_path


def rgb_shaper(path):
    # load data and reshape
    img = load_image(path)
    target = imread(get_mask_path(path), 0)
    pixel_as_row = img.reshape(-1, 3)
    target = target.reshape(-1)
    return pixel_as_row, target


def count_all_rgb_mask_combs(path_list):
    # create default dictionary to store the combinations
    example_count = defaultdict(int)
    for path in tqdm(path_list):
        pixel_as_row, target = rgb_shaper(path)
        # create the data frame
        data_frame = pd.DataFrame(data=pixel_as_row, columns=['r', 'g', 'b'])
        # insert the mask values
        data_frame['value'] = target
        # order and arrange the data frame and convert to dictionary
        rows = data_frame.groupby(['r', 'g', 'b', 'value']).size().reset_index(name='counts').to_dict('split')['data']
        # iterate through the rows in the dictionary
        for row in rows:
            # store every RGB combination with its prevalence
            rgb_index = tuple(row[:-1])
            count = row[-1]
            example_count[rgb_index] += count
    # create lists for all columns
    r = list()
    g = list()
    b = list()
    values = list()
    counts = list()
    # store in every list its corresponding data
    for key, value in tqdm(example_count.items()):
        r.append(key[0])
        g.append(key[1])
        b.append(key[2])
        values.append(key[3])
        counts.append(value)
    # return a data frame of all pixel values and mask values combinations
    counts_df = pd.DataFrame(data={'r': r, 'g': g, 'b': b, 'value': values, 'count': counts})
    return counts_df


def main():
    # export the modified data to CSV
    train, test = get_train_test_paths()
    rgb_train = count_all_rgb_mask_combs(train)
    rgb_train.to_csv(index=False)
    rgb_test = count_all_rgb_mask_combs(test)
    rgb_test.to_csv(index=False)


if __name__ == '__main__':
    main()