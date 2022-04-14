import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from PIL import Image


def get_size(image_path):
    width, height = Image.open(image_path).size
    return width * height


def main():
    evidence_images = [file_name for file_name in glob(f'{os.getcwd()}{os.sep}evidence1{os.sep}*')]
    img_num = [int(path.split(os.sep)[-1].split(' ')[0]) for path in evidence_images]
    F1 = [float(path.split(os.sep)[-1].split('(')[1].split(')')[0]) for path in evidence_images]

    source_images = [f'{os.getcwd()}{os.sep}Water Bodies Dataset{os.sep}Images{os.sep}water_body_{num}.jpg'
                     for num in img_num]
    size = [get_size(image) for image in source_images]

    results = pd.DataFrame({'num': img_num, 'size': size, 'F1': F1})

    p = sns.scatterplot(data=results, x='size', y='F1')
    plt.xlim(0, 12)
    plt.show()


if __name__ == '__main__':
    main()
