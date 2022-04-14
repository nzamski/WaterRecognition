from DeepModels import Conv1
from torchviz import make_dot
import torch.nn.functional as f
import matplotlib.pyplot as plt
from torchvision import models
# from torchsummary import summary
from DataLoader import get_train_test_loaders
import os


os.environ["PATH"] += os.pathsep + r'D:\Graphviz\bin'

model = Conv1(100, 3000, f.leaky_relu)
train_loader, test_loader = get_train_test_loaders(1, 100)
# batch = next(iter(train_loader))
# yhat = model(batch.text)


def main():
    for i, _ in train_loader:
        yhat = model(i)
        make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
        break


if __name__ == '__main__':
    main()
