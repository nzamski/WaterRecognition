import numpy as np
from PIL import Image
from random import shuffle
from cv2 import imread


def split_to_squares(image_path, length):
    # get image array from the path
    rgb_array = load_image(image_path)
    # store image height and width (in pixels)
    max_x, max_y, _ = rgb_array.shape
    # initiate list for image slices
    slices = []
    # move along the image and save every square to the list
    for corner_x in range(max_x - length + 1):
        for corner_y in range(max_y - length + 1):
            # append the squared matrix to the list
            sub = rgb_array[corner_x:corner_x+length, corner_y:corner_y+length, :]
            slices.append(sub)
    return slices


def get_y(image_path, length):  # expected odd length
    # get mask from path
    binary_array = imread(image_path, 0)
    # store mask height and width (in pixels)
    max_x, max_y = binary_array.shape
    # convert pixel colors to absolute black & white
    binary_array = (binary_array < 128).astype(int)
    # initiate list for mask slices
    tags = []
    # move along the mask and save every square to the list
    for x in range(int((length - 1) / 2), max_x - int((length - 1) / 2)):
        for y in range(int((length - 1) / 2), max_y - int((length - 1) / 2)):
            # append the pixel to the list
            tag = binary_array[x, y]
            tags.append(tag)
    return tags


class FileLoader:
    def __init__(self, path, length: int, load_both: bool = True):
        self.length = length
        self.load_both = load_both
        self.source_image = self.load_image(path)
        self.mask_image = self.load_y(path) if load_both else None
        self.source_current_x, self.source_current_y = 0, 0
        self.mask_current_x = int((length - 1) / 2)
        self.mask_current_y = int((length - 1) / 2) if load_both else None, None
        self.source_max_x = self.source_image.shape[0] - self.length
        self.source_max_y = self.source_image.shape[1] - self.length
        self.mask_max_x = self.mask_image.shape[0] - int((length - 1) / 2)
        self.mask_max_y = self.mask_image.shape[1] - int((length - 1) / 2)

    def load_image(self, file_name):
        # get image path and return as array
        img = Image.open(file_name)
        img.load()
        data = np.asarray(img, dtype="int32")
        return data

    def get_mask_path(self, file_path):
        mask_path = file_path.replace('Images', 'Masks')
        return mask_path

    def load_y(self, file_path):
        mask_path = self.get_mask_path(file_path)
        raw_image = imread(mask_path, 0)
        # convert pixel colors to absolute black & white
        binary_array = (raw_image < 128).astype(int)
        return binary_array

    def advance_source_index(self):
        if self.source_current_x < self.source_max_x - self.length:
            self.source_current_x += 1
        else:
            self.source_current_x = 0
            if self.source_current_y < self.source_max_y - self.length:
                self.source_current_y += 1
            else:
                return None

    def get_next(self):
        next_source_slice = self.source_image[
                            self.source_current_x:self.source_current_x + self.length,
                            self.source_current_y:self.source_current_y + self.length,
                            :]
        next_mask_slice = self.mask_image[self.mask_current_x, self.mask_current_y]


class OurDataLoader:
    def __init__(self, path_list, batch_size: int = 4):
        self.path_list = path_list
        shuffle(self.path_list)
        self.batch_size = batch_size
        self.active_file_loaders = [FileLoader(path) for path in self.path_list[:batch_size]]
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