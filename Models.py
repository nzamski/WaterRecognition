import torch
import torch.nn as nn
import torch.nn.functional as f


class Hidden1(nn.Module):
    # define the model
    def __init__(self, length, hidden_size, activation):
        super().__init__()
        self.activation = activation
        self.length = length

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(length * length * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * length * length)  # 2 for each class

    # set activation functions for the layers
    def forward(self, x):
        x = self.flat(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = x.reshape(x.size(0) * self.length**2, 2)
        return x


class Hidden2(nn.Module):
    # define the model
    def __init__(self, length, hidden_size, activation):
        super().__init__()
        self.activation = activation
        self.length = length

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(length * length * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, (2 * length * length))

    # set activation functions for the layers
    def forward(self, x):
        x = self.flat(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = x.view(x.size(0), 2, self.length, self.length)
        return x


class Conv1(nn.Module):
    # define the model
    def __init__(self, length, hidden_size, activation, kernel_size = 3):
        super().__init__()
        self.activation = activation
        self.length = length
        self.kernel_size = kernel_size  # expected odd kernel_size
        self.hidden_size = hidden_size  # hidden_size = 3 times a perfect square

        self.conv1 = nn.Conv2d(3, 3, self.kernel_size, padding=self.kernel_size - 1)
        self.fc1 = nn.Linear(3 * length * length, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 2 * length * length)

    def forward(self, x):
        x = f.max_pool2d(self.activation(self.conv1(x)), kernel_size=self.kernel_size)
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = x.reshape(x.size(0) * self.length ** 2, 2)
        return x


class Conv2(nn.Module):
    # define the model
    def __init__(self, length, hidden_size, kernel_size, activation):
        super().__init__()
        self.activation = activation
        self.length = length
        self.kernel_size = kernel_size  # expected odd kernel_size
        self.hidden_size = hidden_size  # hidden_size = 3 times a perfect square

        self.conv1 = nn.Conv2d(3, 3, (self.kernel_size, self.kernel_size), padding=(self.kernel_size - 1) / 2)
        self.conv2 = nn.Conv2d(3, 3, (self.kernel_size, self.kernel_size), padding=(self.kernel_size - 1) / 2)
        self.fc1 = nn.Linear(3 * length * length, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 2 * length * length)

    def forward(self, x):
        x = f.max_pool2d(self.activation(self.conv1(x)), kernel_size=self.kernel_size, stride=1)
        x = f.max_pool2d(self.activation(self.conv2(x)), kernel_size=self.kernel_size, stride=1)
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class Conv3(nn.Module):
    # define the model
    def __init__(self, length, hidden_size, kernel_size, activation):
        super().__init__()
        self.activation = activation
        self.length = length
        self.kernel_size = kernel_size  # expected odd kernel_size
        self.hidden_size = hidden_size  # hidden_size = 3 times a perfect square

        self.conv1 = nn.Conv2d(3, 3, (self.kernel_size, self.kernel_size), padding=(self.kernel_size - 1) / 2)
        self.conv2 = nn.Conv2d(3, 3, (self.kernel_size, self.kernel_size), padding=(self.kernel_size - 1) / 2)
        self.conv3 = nn.Conv2d(3, 3, (self.kernel_size, self.kernel_size), padding=(self.kernel_size - 1) / 2)
        self.fc1 = nn.Linear(3 * length * length, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 2 * length * length)

    def forward(self, x):
        x = f.max_pool2d(self.activation(self.conv1(x)), kernel_size=self.kernel_size, stride=1)
        x = f.max_pool2d(self.activation(self.conv2(x)), kernel_size=self.kernel_size, stride=1)
        x = f.max_pool2d(self.activation(self.conv3(x)), kernel_size=self.kernel_size, stride=1)
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
