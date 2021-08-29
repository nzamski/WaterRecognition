import os
import PIL
import random
import torchvision

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as f


def get_train_test_paths(test_ratio: float = 0.2):
    # extract the data from the dataset folder
    files = [file_name for file_name in
             Path(os.getcwd() + os.sep + 'Water Bodies Dataset' + os.sep + 'Images').rglob("*.jpg")]
    # randomize the order of the data
    random.shuffle(files)
    # separate test and train files
    first_train = int(test_ratio * len(files))
    test_path = files[:first_train]
    train_path = files[first_train:]
    return train_path, test_path


def get_mask_path(file_path):
    # gets source image path, returns mask path
    file_path = str(file_path).replace('Images', 'Masks')
    return file_path


class WaterDataset(Dataset):
    def __init__(self, path_list, transform_source=None, transform_both=None):
        self.sources = path_list
        self.transform_source = transform_source
        self.transform_both = transform_both

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        img_path = self.sources[index]
        source = f.to_tensor(PIL.Image.open(img_path))
        label = f.to_tensor(PIL.Image.open(get_mask_path(img_path)).convert('L'))

        if self.transform_source:
            source = self.transform_source(source)
        if self.transform_both:
            source = self.transform_both(source)
            label = self.transform_both(label)
            label = (label < 0.5).float()

            assert len(label.unique()) <= 2, "threshold didn't work"

        return source, label


def get_train_test_loaders(batch_size, length):
    train_path, test_path = get_train_test_paths()
    train_loader = DataLoader(dataset=WaterDataset(train_path,
                                                   transform_both=torchvision.transforms.Resize((length, length))),
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=WaterDataset(test_path,
                                                  transform_both=torchvision.transforms.Resize((length, length))),
                             batch_size=batch_size)

    return train_loader, test_loader


if __name__ == '__main__':
    pass
