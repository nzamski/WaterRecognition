import os
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn.functional as f

from tqdm import tqdm
from pathlib import Path
from DeepModels import Conv1
from MachineExport import logit_predict
from DeepExport import get_train_test_loaders, deep_predict
from OtsuMethod import get_source_paths, get_img_index, otsu_predict


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


def get_otsu():
    scores, indices = list(), list()
    source_paths = get_source_paths()
    for source_path in tqdm(source_paths):
        _, _, f1, _, _ = otsu_predict(source_path)
        scores.append(f1)
        index = get_img_index(source_path)
        indices.append(index)
    df = pd.DataFrame({'Index': indices, 'Otsu F1': scores})
    df['Index'] = df['Index'].astype(int)
    df.sort_values(by='Index', inplace=True)
    return df


def get_logit():
    # get image paths
    files = [file_name for file_name in
             Path(os.getcwd() + os.sep + 'Water Bodies Dataset' + os.sep + 'Images').rglob("*.jpg")]
    # load model
    model = f'{os.getcwd()}{os.sep}Models{os.sep}Logistic Regression.pkl'
    loaded_model = pickle.load(open(model, 'rb'))
    # predict and save each image
    scores, indices = list(), list()
    for image in tqdm(files):
        _, f1, img_num = logit_predict(loaded_model, image)
        scores.append(f1)
        indices.append(img_num)
    df = pd.DataFrame({'Index': indices, 'Logit F1': scores})
    df['Index'] = df['Index'].astype(int)
    df.sort_values(by='Index', inplace=True)
    return df


def get_deep():
    # if GPU is available, prepare it for heavy calculations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Conv1(100, 2000, f.relu).to(device)
    model_path = f'{os.getcwd()}{os.sep}Models{os.sep}Conv1-relu-2000.pt'
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        _, test_loader, _ = get_train_test_loaders(1, 100)
        scores, indices = list(), list()
        for image, mask, img_num in tqdm(test_loader):
            _, f1 = deep_predict(model, image, mask)
            scores.append(f1)
            indices.append(img_num.item())
    df = pd.DataFrame({'Index': indices, 'Deep F1': scores})
    df['Index'] = df['Index'].astype(int)
    df.sort_values(by='Index', inplace=True)
    return df


def main():
    otsu_f1 = get_otsu()
    logit_f1 = get_logit()
    deep_f1 = get_deep()

    # logit_f1 = logit_f1[logit_f1['Index'].isin(deep_f1['Index'])].copy()
    # otsu_f1 = otsu_f1[otsu_f1['Index'].isin(deep_f1['Index'])].copy()
    # results = pd.DataFrame({'Index': deep_f1['Index'],
    #                         'Otsu F1': otsu_f1['Otsu F1'],
    #                         'Logit F1': logit_f1['Logit F1'],
    #                         'Deep F1': deep_f1['Deep F1']})

    results = deep_f1.merge(right=logit_f1, on='Index', how='inner').merge(right=otsu_f1, on='Index', how='inner')

    results['Dif'] = results['Deep F1'] - results['Logit F1'] - results['Otsu F1']
    results.sort_values(by='Dif', inplace=True)
    results.to_csv('predictions.csv', index=False)


if __name__ == '__main__':
    main()
