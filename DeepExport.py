import os
import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt

from tqdm import tqdm
from DeepModels import Conv1
from sklearn.metrics import f1_score
from DataLoader import get_train_test_loaders


def deep_predict(model, image, mask):
    x = image.float()
    tag = mask.long()[0, 0, :, :]
    prediction = model(x).float()
    prediction = torch.argmax(prediction, dim=1)
    prediction = prediction.reshape(100, 100)
    f1 = f1_score(tag.flatten(), prediction.flatten(), average='macro', zero_division=0)
    return prediction, f1


def main():
    model = Conv1(100, 2000, f.relu)
    model_path = f'{os.getcwd()}{os.sep}Models{os.sep}Conv1-relu-2000.pt'
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        train_loader, test_loader, _ = get_train_test_loaders(1, 100)
        for loader in (train_loader, test_loader):
            for image, mask, img_num in tqdm(loader):
                prediction, f1 = deep_predict(model, image, mask)
                img_num = str(img_num.item())
                path = f'{os.getcwd()}{os.sep}Deep Images{os.sep}deep_{img_num} ({round(f1, 3)}).jpg'
                plt.imsave(path, prediction, cmap='Greys')


if __name__ == '__main__':
    main()
