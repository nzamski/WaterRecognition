import os
import numpy as np
from pathlib import Path
from random import shuffle

import torch
from PIL import Image
from cv2 import imread
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train_and_test_files_paths(path: str = os.getcwd(), test_ratio: float = 0.2) -> list:
    files = [file_name for file_name in Path(os.getcwd()+os.sep+'Water Bodies Dataset'+os.sep+'Images').rglob("*.jpg")]
    shuffle(files)
    first_train = int(test_ratio * len(files))
    test = files[:first_train]
    train = files[first_train:]
    return train, test


def get_mask_path(file_path):
    prefix = os.getcwd()+os.sep+'Water Bodies Dataset'+os.sep+'Masks'
    file_path = str(file_path).split(os.sep)[-1]
    full_path = prefix + os.sep + file_path
    return full_path


def load_image(file_name):
    img = Image.open(file_name)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def split_to_squares(image_path, length):
    rgb_array = load_image(image_path)
    maxX, maxY, _ = rgb_array.shape
    slices = []
    for x in range(maxX - length + 1):
        for y in range(maxY - length + 1):
            sub = rgb_array[x:x+length,y:y+length,:]
            slices.append(sub)
    return slices


def get_Y(image_path, length):
    binary_array = imread(image_path, 0)
    maxX, maxY = binary_array.shape
    binary_array = (binary_array < 128).astype(int)
    tags = []
    for x in range(int((length - 1) / 2), maxX - int((length - 1) / 2)):
        for y in range(int((length - 1) / 2), maxY - int((length - 1) / 2)):
            tag = binary_array[x, y]
            tags.append(tag)
    return tags


class Net(nn.Module):
    def __init__(self, length, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(length*length*3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    input_image_length = 5
    hidden_size = 12
    train, test = train_and_test_files_paths()
    losses = list()
    for path in train[:10]:
        path = train[0]
        tag_path = get_mask_path(path)
        X = split_to_squares(path, input_image_length)
        y = get_Y(tag_path, input_image_length)
        model = Net(input_image_length, hidden_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        running_loss = 0
        for x, target in zip(X, y):
            x = torch.tensor(x).flatten().float()
            tag = torch.tensor([target], dtype=torch.long)
            # forward + backward + optimize
            optimizer.zero_grad()
            prediction = model(x).reshape((1,2))
            loss = criterion(prediction, tag)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses.append(running_loss)
    print(losses)


#criterion(torch.tensor([[3.2, 1.3,0.2, 0.8]],dtype=torch.float), torch.tensor([0], dtype=torch.long))