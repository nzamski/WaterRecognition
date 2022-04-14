import os
import numpy as np

from PIL import Image
from glob import glob
from cv2 import imread
from pathlib import Path
from random import shuffle, seed

# set a fixed seed to shuffle paths by
seed(42)


def get_train_test_paths(test_ratio: float = 0.2):
    # extract the data from the dataset folder
    files = [file_name for file_name in Path(os.getcwd()+os.sep+'Water Bodies Dataset'+os.sep+'Images').rglob("*.jpg")]
    # randomize the order of the data
    shuffle(files)
    # separate test and train files
    first_train = int(test_ratio * len(files))
    test_path = files[:first_train]
    train_path = files[first_train:]
    return train_path, test_path


def load_image(file_name):
    # get image path and return as array
    img = Image.open(file_name)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def get_mask_path(file_path):
    # gets source image path, returns mask path
    file_path = str(file_path).replace('Images', 'Masks')
    return file_path


def load_y(file_path):
    mask_path = get_mask_path(file_path)
    raw_image = imread(mask_path, 0)
    # convert pixel colors to absolute black & white
    binary_array = (raw_image < 128).astype(int)
    return binary_array


class FileLoader:
    def __init__(self, path, length: int, load_both: bool = True):
        self.length = length
        self.load_both = load_both
        # source image variables initialization
        self.source_image = load_image(path)
        self.source_current_x, self.source_current_y = -1, 0  # -1 is a temp starting point
        self.source_max_x = self.source_image.shape[0] - self.length
        self.source_max_y = self.source_image.shape[1] - self.length
        # mask image variables initialization
        self.mask_image = load_y(path) if load_both else None
        self.mask_current_x = int((length - 1) / 2) - 1 if load_both else None  # -1 is a temp starting point
        self.mask_current_y = int((length - 1) / 2) if load_both else None
        self.mask_max_x = self.mask_image.shape[0] - int((length - 1) / 2) if load_both else None
        self.mask_max_y = self.mask_image.shape[1] - int((length - 1) / 2) if load_both else None

    def advance_source_index(self):
        if self.source_current_x < self.source_max_x - self.length:
            # the slice doesn't exceed the columns
            self.source_current_x += 1
        else:
            # go to the next row and reset the x
            self.source_current_x = 0
            if self.source_current_y < self.source_max_y - self.length:
                self.source_current_y += 1
            else:
                # finished going through image
                return None
        return self.source_current_x, self.source_current_y

    def advance_mask_index(self):
        if self.mask_current_x < self.mask_max_x:  # no need in self.length, right?
            # the tag doesn't exceed the columns
            self.mask_current_x += 1
        else:
            # go to the next row and reset the x
            self.mask_current_x = 0
            if self.mask_current_y < self.mask_max_y:  # same
                self.mask_current_y += 1
            else:
                # finished going through image
                return None
        return self.mask_current_x, self.mask_current_y

    def get_next(self):
        # returns the next slice (+ tag)
        source_indices = self.advance_source_index()
        if source_indices is None:
            return None
        else:
            source_x, source_y = source_indices
            source_slice = self.source_image[
                           source_x:source_x + self.length,
                           source_y:source_y + self.length,
                           :]
        if self.load_both:
            mask_x, mask_y = self.advance_mask_index()
            mask_tag = self.mask_image[mask_x, mask_y]
            return source_slice, mask_tag
        return source_slice


class DataLoader:
    def __init__(self, path_list, length, batch_size: int = 4, load_both: bool = True, return_file_name: bool = False):
        self.path_list = path_list
        shuffle(self.path_list)
        self.length = length
        self.current_file_index = 0
        # expected batch size smaller than number of paths
        self.batch_size = batch_size
        self.load_both = load_both
        self.return_file_name = return_file_name
        self.active_file_loaders, self.file_names = self.setup_active_file_loaders()
        self.file_index = batch_size
        self.max_file_index = len(self.path_list)

    def __iter__(self):
        return self

    def setup_active_file_loaders(self):
        active_file_loaders, file_names = list(), list()
        for _ in range(self.batch_size):
            new_loader, file_name = self.get_next_loader()
            active_file_loaders.append(new_loader)
            file_names.append(file_name)
        return active_file_loaders, file_names

    def get_next_loader(self):
        # return the next file loader (by order) and its path
        if self.current_file_index < len(self.path_list):
            path = self.path_list[self.current_file_index]
            new_loader = FileLoader(path, self.length, self.load_both)
            self.current_file_index += 1
            return new_loader, path
        return None, None

    def __next__(self):
        # initiate lists to store a batch of data
        x_batch, y_batch = list(), list()
        for loader_index, data_loader in enumerate(self.active_file_loaders):
            # get next potential data loader
            loaded_result = data_loader.get_next()
            # end of current loader
            if loaded_result is None:
                new_loader, file_name = self.get_next_loader()
                # finished going through path list
                if new_loader is None:
                    del(self.active_file_loaders[loader_index])
                    del(self.file_names[loader_index])
                    # end of all loaders in active file loaders
                    if len(self.active_file_loaders) == 0:
                        raise StopIteration()
                # didn't finished going through all the data yet
                else:
                    data_loader = new_loader
                    self.active_file_loaders[loader_index] = data_loader
                    self.file_names[loader_index] = file_name
                    loaded_result = data_loader.get_next()
            # finished a batch
            if loaded_result is not None:
                x_batch.append(loaded_result[0])
                if self.load_both:
                    y_batch.append(loaded_result[1])
        # return all x, y batches
        results = [x_batch]
        if self.load_both:
            results.append(y_batch)
        if self.return_file_name:
            results.append(self.file_names)
        return results


def main():
    main_path = r'C:\Users\nzams\Desktop\WaterRecognition\Water Bodies Dataset\Images\*'
    path_list, length = [path for path in glob(main_path)[:5]], 3
    loader = DataLoader(path_list, length)
    for x, y in loader:
        pass


if __name__ == '__main__':
    main()
