import os
import cv2
import pandas as pd

from DeepModels import *
from tqdm import tqdm
from torch import optim
from datetime import datetime
from torchvision.utils import save_image
from torchvision.transforms import Resize
from DataLoader import get_train_test_loaders


def fit_model(model, model_parameters, loss_function, optimizer, batch_size, image_normalized_length, num_of_epochs):
    train_loader, test_loader = get_train_test_loaders(batch_size, image_normalized_length)
    # if GPU is available, prepare it for heavy calculations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model(*model_parameters).to(device)
    criterion = loss_function()
    optimizer = optimizer(model.parameters(), lr=3e-4)
    for epoch in range(1, num_of_epochs + 1):
        model.train()
        epoch_start = datetime.now()
        epoch_loss = 0
        for image, mask in tqdm(train_loader):
            x = image.float().to(device)
            tag = mask.flatten().long().to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, tag)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_end = datetime.now()
        epoch_seconds = (epoch_end - epoch_start).total_seconds()
        model.eval()
        total_predicted_positive, total_true_positive, total_false_negative, \
        total_true_prediction, total_false_prediction = 0, 0, 0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(test_loader):
                x = x.to(device)
                y = y.flatten().to(device)
                probabilities = model(x)
                prediction = torch.argmax(probabilities, dim=1)

                predicted_positive = (prediction == 1).sum().item()
                true_positive = ((prediction == 1) & (y == 1)).sum().item()
                false_negative = ((prediction == 0) & (y == 1)).sum().item()

                total_predicted_positive += predicted_positive
                total_true_positive += true_positive
                total_false_negative += false_negative
                total_true_prediction += (prediction == y).sum().item()
                total_false_prediction += (prediction != y).sum().item()
        recall = total_true_positive / (total_true_positive + total_false_negative)
        precision = total_true_positive / total_predicted_positive
        f1 = (2 * precision * recall) / (precision + recall)
        accuracy = total_true_prediction / (total_true_prediction + total_false_prediction)
        # append results to csv file
        df = pd.DataFrame({'Model Name': [model.__class__.__name__],
                           'Iteration': [epoch],
                           'Input Image Length': [image_normalized_length],
                           'Hidden Layer Size': [hidden_layer_size],
                           'Batch Size': [batch_size],
                           'Activation Function': [str(model_parameters[2].__name__)],
                           'Optimizer': [str(type(optimizer))],
                           'Loss Function': [str(loss_function)],
                           'Loss': [epoch_loss],
                           'Recall': [recall],
                           'Precision': [precision],
                           'F1': [f1],
                           'Accuracy': [accuracy],
                           'Iteration Training Seconds': [epoch_seconds]})
        df.to_csv('Deep_Results.csv', index=False, mode='a', header=False)
        print(df)
    # clean memory of current model on exit
    torch.cuda.empty_cache()


if __name__ == '__main__':
    models = (Hidden1, Hidden2, Conv1, Conv2, Conv3)
    activation_funcs = (f.relu, f.leaky_relu)
    hidden_layer_sizes = (4000, 3000, 2000)

    optimizer = optim.Adam
    loss_func = nn.CrossEntropyLoss
    image_normalized_length = 100
    batch_size = 16
    num_of_epochs = 10
    kernel_size = 3

    # train models with varying hyperparameters
    params = dict()
    for model in models:
        for activation_func in activation_funcs:
            for hidden_layer_size in hidden_layer_sizes:
                if 'Conv' in model.__name__:
                    model_parameters = (image_normalized_length, hidden_layer_size, activation_func, kernel_size)
                model_parameters = (image_normalized_length, hidden_layer_size, activation_func)
                fit_model(model, model_parameters, loss_func, optimizer,
                          batch_size, image_normalized_length, num_of_epochs)
