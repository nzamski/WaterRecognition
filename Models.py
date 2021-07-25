import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f


# define all activation functions, optimizers and loss functions
ReLU = f.relu()
LeakyReLU = f.leaky_relu()
Sigmoid = f.sigmoid()
Adam = optim.Adam()
SGD = optim.SGD()
CrossEntropy = nn.CrossEntropyLoss()
MSE = nn.MSELoss()
L1 = nn.L1Loss()


class Hidden1(nn.Module):
    # define the model
    def __init__(self, length, hidden_size, activation):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(length * length * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    # set activation functions for the layers
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class Hidden2(nn.Module):
    # define the model
    def __init__(self, length, hidden_size, activation):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(length * length * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

    # set activation functions for the layers
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class Hidden3(nn.Module):
    # define the model
    def __init__(self, length, hidden_size, activation):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(length * length * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2)

    # set activation functions for the layers
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


class Conv(nn.Module):
    # define the model
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(1, 3, 3, 3)
        self.conv2 = nn.Conv2d(3, 3, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

