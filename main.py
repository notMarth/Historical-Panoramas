import Panorama as pan
import FileOps as op
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

im1 = cv2.imread("CameraPhotos/ATH PATCH (HIST 1).jpg")
im2 = cv2.imread("CameraPhotos/ATH (12).jpg")

#BRUTE FORCE
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

#im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]), interpolation=cv2.INTER_CUBIC)

result,imMatches = pan.stitch(im2,im1, match_type = 1, match_lim=10)
# show the images
plt.figure(figsize=(15, 10))
plt.subplot(1,2,1)
plt.imshow(im1,'gray')
plt.xticks([]), plt.yticks([])
plt.title("image 1")
plt.subplot(1,2,2)
plt.imshow(im2,'gray')
plt.xticks([]), plt.yticks([])
plt.title("image 2")
plt.show()


plt.figure(figsize=(15, 10))
plt.imshow(imMatches)
plt.xticks([]), plt.yticks([])
plt.title("Matches - Brute Force")
plt.show()

plt.figure(figsize=(15, 10))
plt.imshow(result.astype(np.uint8),'gray')
plt.xticks([]), plt.yticks([])
plt.title("Panorama")
plt.show()

