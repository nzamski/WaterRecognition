import os
import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt

from tqdm import tqdm
from DeepModels import Conv1
from sklearn.metrics import f1_score
from DataLoader import get_train_test_loaders


def save_triplets(loader, out_dir):
    for image, mask, img_num in tqdm(loader):
        img_num = str(img_num.item())
        x = image.float().to(device)
        tag = mask.long()[0, 0, :, :].to(device)
        prediction = model(x).float().to(device)
        prediction = torch.argmax(prediction, dim=1)
        prediction = prediction.reshape(100, 100)

        f1 = f1_score(tag.flatten(), prediction.flatten(), average='macro', zero_division=0)
        title = f'{img_num} ({round(f1, 3)})'
        plt.suptitle(title, y=0.8)
        plt.subplot(1, 3, 1)
        plt.imshow(x[0, :, :, :].permute(1, 2, 0), cmap='Greys')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(prediction, cmap='Greys')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(tag, cmap='Greys')
        plt.axis('off')
        plt.savefig(f'{out_dir}{os.sep}{title}.png', dpi=300)


if __name__ == '__main__':
    # if GPU is available, prepare it for heavy calculations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Conv1(100, 2000, f.relu).to(device)
    model_path = f'{os.getcwd()}{os.sep}Models{os.sep}Conv1-relu-2000.pt'
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        _, test_loader, _ = get_train_test_loaders(1, 100)

        path1 = f'D:{os.sep}Noam{os.sep}Desktop{os.sep}evidence1'
        save_triplets(test_loader, path1)

        path2 = f'D:{os.sep}Noam{os.sep}Desktop{os.sep}evidence2'
        save_triplets(test_loader, path2)
