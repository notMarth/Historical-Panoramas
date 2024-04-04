import Panorama as pan
import FileOps as op
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def mask():
    im = op.convertGrey(cv2.imread('CameraPhotos/STEPHEN (10).jpg'))
    mask = np.zeros(im.shape)
    cv2.rectangle(im, (3321, 517), (4268, 2486), 0, -1)
    cv2.rectangle(im, (2203, 730), (4268, 2200), 0, -1)
    cv2.rectangle(im, (0, 0), (1390, 2394), 0, -1)

    return im

if __name__ == '__main__':
    im = op.convertGrey(cv2.imread('CameraPhotos/STEPHEN (10).jpg'))
    mask = mask()

    plt.figure(figsize=(15, 10))
    plt.subplot(121)
    plt.imshow(im, 'gray')

    plt.subplot(122)
    plt.imshow(mask, 'gray')
    plt.show()

    cv2.imwrite('masks/STEPHEN (10).jpg', mask())