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


def get_img_index(path):
    index = str(path).split('_')[-1].split('.')[0]
    return index


def fit_model(model, model_parameters, loss_function, optimizer, batch_size, image_normalized_length, num_of_epochs):
    # retrieve train and test files
    train_loader, test_loader = get_train_test_loaders(batch_size, image_normalized_length)
    # if GPU is available, prepare it for heavy calculations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # assign the model
    model = model(*model_parameters).to(device)
    # assign the loss function
    criterion = loss_function()
    # set an optimizer
    optimizer = optimizer(model.parameters(), lr=3e-4)
    for epoch in range(1, num_of_epochs + 1):
        # MODEL TRAINING
        model.train()
        # start counting epoch duration
        epoch_start = datetime.now()
        # initiate epoch loss
        epoch_loss = 0
        # iterate through all data pairs
        for image, mask in tqdm(train_loader):
            # convert input pixel to tensor
            x = image.float().to(device)
            # convert target to tensor
            tag = mask.flatten().long().to(device)
            # reset all gradients
            optimizer.zero_grad()
            # save current prediction
            prediction = model(x)
            # activate loss function, calculate loss
            loss = criterion(prediction, tag)
            # back propagation
            loss.backward()
            optimizer.step()
            # update epoch loss
            epoch_loss += loss.item()
            break
        # stop counting epoch duration
        epoch_end = datetime.now()
        epoch_seconds = (epoch_end - epoch_start).total_seconds()
        # MODEL EVALUATION
        model.eval()
        # collect predicted results and real results
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
        # calculate accuracy and f1 score
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
        df.to_csv('Water_Bodies_Results.csv', index=False, mode='a', header=False)
        print(df)
    # save a prediction
    # with torch.no_grad():
    #     for i, image, mask in enumerate(test_loader):
    #         name = ''
    #         x = image.float().to(device)
    #         tag = mask.flatten().long().to(device)
    #         prediction = model(x)
    #         file_name, _ = test_loader.dataset.samples[i]
    #         file_index = get_img_index(file_name)
    #         save_prediction(prediction, file_index, name)
    torch.cuda.empty_cache()


def save_prediction(prediction, index, name):
    # convert two-valued pixels to single-max-value
    prediction = torch.argmax(prediction, dim=1)
    # reshape the flattened prediction to a matrix
    prediction = prediction.reshape(100, 100)
    # get original size
    source_path = f'{os.getcwd()}{os.sep}Water Bodies Dataset{os.sep}Images{os.sep}water_body_{index}.jpg'
    width, height = cv2.imread(source_path).shape
    # resize prediction to original source size
    prediction = Resize(prediction, size=(width, height))
    prediction_path = f'{os.getcwd()}{os.sep}Deep Images{os.sep}{name}_{index}.jpg'
    # save the prediction
    save_image(prediction, prediction_path)


def get_n_params(model):
    """https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6"""
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


if __name__ == '__main__':
    models = (Conv2,)
    activation_funcs = (f.relu, f.leaky_relu)
    hidden_layer_sizes = (4000, 3000, 2000)
    batch_sizes = (32,)
    optimizers = (optim.Adam,)

    image_normalized_length = 100
    num_of_epochs = 10
    loss_func = nn.CrossEntropyLoss
    kernel_size = 3

    # train models with varying hyperparameters
    for model in models:
        for activation_func in activation_funcs:
            for hidden_layer_size in hidden_layer_sizes:
                for batch_size in batch_sizes:
                    for optimizer in optimizers:
                        model_parameters = (image_normalized_length, hidden_layer_size, activation_func, kernel_size)
                        fit_model(model, model_parameters, loss_func, optimizer,
                                  batch_size, image_normalized_length, num_of_epochs)
