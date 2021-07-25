import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from PIL import Image
from cv2 import imread
from pathlib import Path
from random import shuffle


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


def get_mask_path(file_path):
    # disassemble and assemble data path to return mask path
    wdr = os.getcwd()+os.sep+'Water Bodies Dataset'+os.sep+'Masks'
    file_path = str(file_path).split(os.sep)[-1]
    mask_path = wdr + os.sep + file_path
    return mask_path


def load_image(file_name):
    # get image path and return as array
    img = Image.open(file_name)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def split_to_squares(image_path, length):
    # get image array from the path
    rgb_array = load_image(image_path)
    # store image height and width (in pixels)
    max_x, max_y, _ = rgb_array.shape
    # initiate list for image slices
    slices = []
    # move along the image and save every square to the list
    for corner_x in range(max_x - length + 1):  # why not +2 ?
        for corner_y in range(max_y - length + 1):
            # append the squared matrix to the list
            sub = rgb_array[corner_x:corner_x+length, corner_y:corner_y+length, :]
            slices.append(sub)
    return slices


def get_y(image_path, length):  # expected odd length
    # get mask from path
    binary_array = imread(image_path, 0)
    # store mask height and width (in pixels)
    max_x, max_y = binary_array.shape
    # convert pixel colors to absolute black & white
    binary_array = (binary_array < 128).astype(int)
    # initiate list for mask slices
    tags = []
    # move along the mask and save every square to the list
    for x in range(int((length - 1) / 2), max_x - int((length - 1) / 2)):
        for y in range(int((length - 1) / 2), max_y - int((length - 1) / 2)):
            # append the pixel to the list
            tag = binary_array[x, y]
            tags.append(tag)
    return tags


class Net(nn.Module):
    # define the model
    def __init__(self, length, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(length * length * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

    # set activation functions for the layers
    def forward(self, x):
        x = f.leaky_relu(self.fc1(x))
        x = f.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # length of the squared matrix
    input_image_length = 5
    # number of neurons in the second and third layer
    hidden_size = 12
    # retrieve train and test files
    train, test = get_train_test_paths()
    # initiate a list for loss accumulation
    losses = list()
    # iterate through all data pairs
    for path in train:
        # retrieve corresponding mask files
        tag_path = get_mask_path(path)
        X = split_to_squares(path, input_image_length)
        Y = get_y(tag_path, input_image_length)
        # assign the model
        model = Net(input_image_length, hidden_size)
        # set a loss function
        criterion = nn.CrossEntropyLoss()
        # set an optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # initiate loss variable for current epoch
        running_loss = 0
        for x, target in zip(X, Y):
            # convert input pixel to tensor and flatten
            x = torch.flatten(torch.tensor(x)).float()
            # convert target to tensor
            tag = torch.tensor([target], dtype=torch.long)
            # set all gradients to to zero
            optimizer.zero_grad()
            prediction = model(x).reshape((1, 2))
            # activate cross entropy, calculate loss
            loss = criterion(prediction, tag)
            # back propagation
            loss.backward()
            optimizer.step()
            # update into current loss
            running_loss += loss.item()
        # add current loss to the list
        losses.append(running_loss)
    print(losses)
