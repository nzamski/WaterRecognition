import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from cv2 import imread
from pathlib import Path
from sklearn.metrics import f1_score


def arr_to_df(arr: np.ndarray):
    """ turns an image to a dataframe, each row a pixle """
    if len(arr.shape) == 2:
        result = pd.DataFrame(data=arr.flatten())
    else:
        data = np.stack([arr[:, :, i].flatten() for i in range(arr.shape[2])]).T
        result = pd.DataFrame(data=data)
    return result


def df_to_arr(df: pd.DataFrame, rows, cols):
    """ turns a dataframe created by arr_to_df to an image """
    if len(df.columns) == 1:
        arr = df.values.flatten().reshape(rows, cols)
    else:
        arr = np.dstack([df[i].values.flatten().reshape((rows, cols)) for i in df.columns])
    return arr


def get_mask_path(file_path):
    # gets source image path, returns mask path
    file_path = str(file_path).replace('Images', 'Masks')
    return file_path


def logit_predict(model, image):
    img_num = int(str(image).rsplit('_', 1)[1].split('.')[0])

    source = Image.open(image)
    source.load()
    source_arr = np.asarray(source, dtype="int32")

    mask_arr = imread(get_mask_path(image), 0)
    mask_arr = (mask_arr < 128).astype('int64')

    rows, cols, _ = source_arr.shape
    df = arr_to_df(source_arr)
    df.columns = ['r', 'g', 'b']
    df = df / 255
    df['prediction'] = model.predict(df.values)
    prediction = df_to_arr(df[['prediction']], rows, cols)
    f1 = f1_score(mask_arr.reshape(-1), prediction.reshape(-1), average='macro', zero_division=0)
    return prediction, f1, img_num


def main():
    # get image paths
    files = [file_name for file_name in
             Path(os.getcwd() + os.sep + 'Water Bodies Dataset' + os.sep + 'Images').rglob("*.jpg")]
    # load model
    model_path = f'{os.getcwd()}{os.sep}Models{os.sep}Logistic Regression.pkl'
    model = pickle.load(open(model_path, 'rb'))
    # predict and save each image
    for image in tqdm(files):
        prediction, f1, img_num = logit_predict(model, image)
        path = f'{os.getcwd()}{os.sep}Logit Images{os.sep}logit_{img_num} ({round(f1, 3)}).jpg'
        plt.imsave(path, prediction, cmap='Greys')


if __name__ == '__main__':
    main()
