import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def hist_plot(source_path, segmented, ret):
    source_img = cv2.imread(source_path)
    gray_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)

    hist_r = cv2.calcHist([source_img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([source_img], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([source_img], [2], None, [256], [0, 256])
    hist_k = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    plt.subplot(221), plt.imshow(source_img)
    plt.subplot(222), plt.imshow(segmented, cmap='Greys')
    plt.subplot(223), plt.plot(hist_r, 'r'), plt.plot(hist_g, 'g'), plt.plot(hist_b, 'b')
    plt.axvline(ret, color='y')
    plt.subplot(224), plt.plot(hist_k, 'k')
    plt.xlim([0, 256])
    plt.axvline(ret, color='y')
    plt.xlim([0, 256])

    plt.show()


def channels_std():
    red_std, green_std, blue_std = list(), list(), list()
    non_red_std, non_green_std, non_blue_std = list(), list(), list()
    indices = list()
    source_paths = get_source_paths()
    for source_path in tqdm(source_paths):
        img = cv2.imread(str(source_path))
        mask_img = get_mask(str(source_path))
        red = np.ma.masked_where(mask_img == 0, img[:, :, 0])
        non_red = np.ma.masked_where(mask_img == 255, img[:, :, 0])
        green = np.ma.masked_where(mask_img == 0, img[:, :, 1])
        non_green = np.ma.masked_where(mask_img == 255, img[:, :, 1])
        blue = np.ma.masked_where(mask_img == 0, img[:, :, 2])
        non_blue = np.ma.masked_where(mask_img == 255, img[:, :, 2])
        indices.append(get_img_index(str(source_path)))
        red_std.append(np.std(red))
        non_red_std.append(np.std(non_red))
        green_std.append(np.std(green))
        non_green_std.append(np.std(non_green))
        blue_std.append(np.std(blue))
        non_blue_std.append(np.std(non_blue))
    df = pd.DataFrame(data={'Index': indices,
                            'Red Water STD': red_std,
                            'Red Non-Water STD': non_red_std,
                            'Green Water STD': green_std,
                            'Green Non-Water STD': non_green_std,
                            'Blue Water STD': blue_std,
                            'Blue Non-Water STD': non_blue_std})
    df.to_csv('STDs.csv', index=False)


def inversion_rule_2(thresh, pad, k, height, width):
    # rule for thresh inversion
    frame = thresh.copy()
    # replace all the inside with neutral values
    frame[pad:-pad, pad:-pad] = 2
    uniques, counts = np.unique(frame, return_counts=True)
    blacks = counts[uniques == 0].item() if 0 in uniques else 0
    whites = counts[uniques == 255].item() if 255 in uniques else 0
    if whites == 0 or blacks == 0 or max(blacks, whites) / (blacks + whites) > 0.7:
        thresh = thresh if blacks > whites else 255 - thresh
    else:
        # get the central slice
        center = thresh[int(height / 2) - 1:int(height / 2) - 1 + k, int(width / 2) - 1:int(width / 2) - 1 + k]
        center_avg = np.mean(center)
        thresh = thresh if center_avg > 128 else 255 - thresh


def inversion_rule_2(img, thresh, pad, height, width):
    # rule for thresh inversion
    frame = thresh.copy()
    # replace all the inside with neutral values
    frame[pad:-pad, pad:-pad] = 2
    uniques, counts = np.unique(frame, return_counts=True)
    blacks = counts[uniques == 0].item() if 0 in uniques else 0
    whites = counts[uniques == 255].item() if 255 in uniques else 0
    if whites == 0 or blacks == 0 or max(blacks, whites) / (blacks + whites) > 0.7:
        thresh = thresh if blacks > whites else 255 - thresh
    else:
        k = 30
        found = False
        while k > 15 and not found:
            blacks, whites = 0, 0
            for row in range(pad, height - k - pad + 1):
                for col in range(pad, width - k - pad + 1):
                    mat = thresh[row:row + k, col:col + k]
                    if (mat == np.full((k, k), 0)).all():
                        blacks += 1
                    elif (mat == np.full((k, k), 255)).all():
                        whites += 1
            if blacks * whites != 0 and max(blacks, whites) / (blacks + whites) > 0.7:
                thresh = thresh if whites > blacks else 255 - thresh
                found = True
            k -= 1
        if not found:
            std_blue = np.std(np.ma.masked_where(thresh == 0, img[:, :, 2])).item()
            std_non_blue = np.std(np.ma.masked_where(thresh == 255, img[:, :, 2])).item()
            thresh = thresh if std_blue < std_non_blue else 255 - thresh
