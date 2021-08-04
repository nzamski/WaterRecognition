import os
import random
import numpy as np
from tqdm import tqdm
from DataLoader import DataLoader
from PIL import Image
from cv2 import imread
from pathlib import Path
from Models import *
random.seed(42)


def get_train_test_paths(test_ratio: float = 0.2):
    # extract the data from the dataset folder
    files = [file_name for file_name in Path(os.getcwd()+os.sep+'Water Bodies Dataset'+os.sep+'Images').rglob("*.jpg")]
    # randomize the order of the data
    random.shuffle(files)
    # separate test and train files
    first_train = int(test_ratio * len(files))
    test_path = files[:first_train]
    train_path = files[first_train:]
    return train_path, test_path


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
    # set a train loader
    train_loader = DataLoader(train, input_image_length)
    # iterate through all data pairs
    for x, y in train_loader:
        # initiate loss variable for current epoch
        running_loss = 0
        for i, (x, target) in tqdm(enumerate(zip(x, y))):
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
