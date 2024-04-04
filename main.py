import Panorama as pan
import FileOps as op
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

imurl = "MED PATCH"
imNum = 10

im1 = cv2.imread(f"CameraPhotos/{imurl} (HIST).jpg")
mask1 = cv2.imread(f"masks/{imurl} (HIST).jpg")
if mask1 is None: mask1 = []

im2 = cv2.imread(f"CameraPhotos/{imurl} ({imNum}).jpg")
mask2 = cv2.imread(f"masks/{imurl} ({imNum}).jpg")
if mask2 is None: mask2 = []

#BRUTE FORCE
im1 = op.convertGrey(im1)
im2 = op.convertGrey(im2)
mask1 = op.convertGrey(mask1)
mask2 = op.convertGrey(mask2)
# print(im1.shape, im2.shape, mask1.shape, mask2.shape)
im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]), interpolation=cv2.INTER_CUBIC)
if len(mask2) > 0: mask2 = cv2.resize(mask2, (im1.shape[1], im1.shape[0]), interpolation=cv2.INTER_CUBIC)

result,imMatches = pan.stitch(im2,im1, mask2, mask1, match_type = 1, match_lim=10)
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

