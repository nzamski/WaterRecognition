import cv2
import skimage.segmentation

from matplotlib import pyplot as plt

img = cv2.imread(r'Water Bodies Dataset\Images\water_body_22.jpg')
blur = cv2.GaussianBlur(img, (11, 11), 0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
cv_gray = skimage.segmentation.chan_vese(gray)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

plt.imshow(cv_gray, cmap='gray')
plt.show()
