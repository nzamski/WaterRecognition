from Models import *
from tqdm import tqdm
from DataLoader import DataLoader, get_train_test_paths


def fit_model(model, model_parameters, loss_function, optimizer, matrix_length):
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
    train_loader = DataLoader(train, matrix_length)
    # iterate through all data pairs
    for slice, tag in train_loader:
        # initiate loss variable for current epoch
        running_loss = 0
        for i, (x, target) in tqdm(enumerate(zip(slice, tag))):
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
        losses.append(running_loss / len(tag))  # ok?
    print(losses)


if __name__ == '__main__':
    # define all activation functions, optimizers and loss functions
    ReLU = f.relu
    LeakyReLU = f.leaky_relu
    Sigmoid = f.sigmoid
    Adam = optim.Adam
    SGD = optim.SGD
    CrossEntropy = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()
    L1 = nn.L1Loss()

    # define model parameters
    matrix_length = 5
    hidden_layer_size = 10
    model_parameters = (matrix_length, hidden_layer_size, ReLU)

    # train the model
    fit_model(Hidden1, model_parameters, CrossEntropy, Adam, matrix_length)
