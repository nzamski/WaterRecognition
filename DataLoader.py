import numpy as np
from PIL import Image
from random import shuffle
from cv2 import imread


def load_image(file_name):
    # get image path and return as array
    img = Image.open(file_name)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def get_mask_path(file_path):
    # gets source image path, returns mask path
    mask_path = file_path.replace('Images', 'Masks')
    return mask_path


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
        self.source_current_x, self.source_current_y = 0, 0
        self.source_max_x = self.source_image.shape[0] - self.length
        self.source_max_y = self.source_image.shape[1] - self.length
        # mask image variables initialization
        self.mask_image = load_y(path) if load_both else None
        self.mask_current_x = int((length - 1) / 2) if load_both else None
        self.mask_current_y = int((length - 1) / 2) if load_both else None
        self.mask_max_x = self.mask_image.shape[0] - int((length - 1) / 2) if load_both else None
        self.mask_max_y = self.mask_image.shape[1] - int((length - 1) / 2) if load_both else None

    def advance_source_index(self):
        if self.source_current_x < self.source_max_x - self.length:
            self.source_current_x += 1
        else:
            self.source_current_x = 0
            if self.source_current_y < self.source_max_y - self.length:
                self.source_current_y += 1
            else:
                return None
        return self.source_current_x, self.source_current_y

    def advance_mask_index(self):
        if self.mask_current_x < self.mask_max_x - self.length:
            self.mask_current_x += 1
        else:
            self.mask_current_x = 0
            if self.mask_current_y < self.mask_max_y - self.length:
                self.mask_current_y += 1
            else:
                return None
        return self.mask_current_x, self.mask_current_y

    def get_next(self):
        if self.advance_source_index() is None:
            source_slice = None
        else:
            source_x, source_y = self.advance_source_index()
            source_slice = self.source_image[
                           source_x:source_x + self.length,
                           source_y:source_y + self.length,
                           :]
        if self.load_both:
            if self.advance_mask_index() is None:
                mask_tag = None
            else:
                mask_x, mask_y = self.advance_mask_index()
                mask_tag = self.mask_image[mask_x, mask_y]
            return source_slice, mask_tag
        return source_slice


class DataLoader:
    def __init__(self, path_list, batch_size: int = 4):
        self.path_list = path_list
        shuffle(self.path_list)
        self.batch_size = batch_size
        self.active_file_loaders = [FileLoader(path) for path in self.path_list[:batch_size]]  # length?
        self.file_index = batch_size
        self.max_file_index = len(self.path_list)

    def __next__(self):
        x_batch, y_batch = list(), list()
        for loader_index, data_loader in enumerate(self.active_file_loaders):
            loaded_result = data_loader.get_next()
            if loaded_result is None:  # end of current loader
                new_loader = self.get_next_loader()
                if new_loader is None:  # finished going through path list
                    del(self.active_file_loaders[loader_index])
                    if len(self.active_file_loaders) == 0:  # end of all loaders in active file loaders
                        return None
                else:
                    data_loader = new_loader
                    self.active_file_loaders[loader_index] = data_loader
                    loaded_result = data_loader.get_next()
            x_batch.append(loaded_result[0])
            y_batch.append(loaded_result[1])
        return x_batch, y_batch
