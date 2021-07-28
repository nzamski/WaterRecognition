import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from cv2 import imread
from pathlib import Path
from random import shuffle
from Models import *


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
    for corner_x in range(max_x - length + 1):
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


def get_x_y(file_path, length):
    # returns a tuple of image and its mask
    X = split_to_squares(file_path, length)
    mask_path = get_mask_path(file_path)
    y = get_y(mask_path, length)
    return X, y


def fit_model(model, model_parameters, loss_function, optimizer, preprocessor, input_image_length):
    # retrieve train and test files
    train, test = get_train_test_paths()
    # initiate a list for loss accumulation
    losses = list()
    # assign the model
    model = model(*model_parameters)
    # set a loss function
    criterion = loss_function()
    # set an optimizer
    optimizer = optimizer(model.parameters(), lr=0.001)
    # iterate through all data pairs
    for path in train:
        X, y = preprocessor(path, input_image_length)
        # initiate loss variable for current epoch
        running_loss = 0
        for i, (x, target) in tqdm(enumerate(zip(X, y))):
            # convert input pixel to tensor and flatten
            x = torch.flatten(torch.tensor(x)).float()
            # convert target to tensor
            tag = torch.tensor([target], dtype=torch.long)
            # reset all gradients
            optimizer.zero_grad()
            # save current prediction
            prediction = model(x).reshape((1, 2))
            # activate cross entropy, calculate loss
            loss = criterion(prediction, tag)
            # back propagation
            loss.backward()
            optimizer.step()
            # update into current loss
            running_loss += loss.item()
            # print current loss value every 5000 iterations
            if i % 5_000 == 0:
                print(loss.item())
        # add current loss to the list
        losses.append(running_loss / len(y))
    print(losses)


if __name__ == '__main__':
    # define all activation functions, optimizers and loss functions
    # ReLU = f.relu
    # LeakyReLU = f.leaky_relu
    # Sigmoid = f.sigmoid
    # Adam = optim.Adam
    # SGD = optim.SGD
    # CrossEntropy = nn.CrossEntropyLoss()
    # MSE = nn.MSELoss()
    # L1 = nn.L1Loss()
    # fit_model(Hidden1)

    model = Hidden1
    input_image_length = 5
    hidden_layer_size = 10
    activation = f.relu
    model_parameters = (input_image_length, hidden_layer_size, activation)
    optimizer = optim.Adam
    loss_function = nn.CrossEntropyLoss
    preprocessor = get_x_y
    fit_model(model, model_parameters, loss_function, optimizer, preprocessor, input_image_length)
