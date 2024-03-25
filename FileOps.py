# Image and File loading Utilities
# Used to import photos and perform file operations

import numpy as np
import cv2

# return two given images but with appropriate padding
# imA will be the "bigger" image that imB is fit into
def getImage(imA, imB):

    heightA, widthA = imA.shape
    heightB, widthB = imB.shape

    bigA = np.pad(imA, ((heightA, ), (widthA, )), mode='constant', constant_values=0.)
    bigB = np.pad(imB, ((2*heightA - heightB, ), (2*widthA - widthB, )), mode='constant', constant_values=0.)

    return bigA, bigB

# attempt to convert to greyscale. If already grey, return image
def convertGrey(im):
    
    try:
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    except:
        return im
    

# feather the two images together where they share common points
# Idea: grab the distance between the furthest points and use that to determine
# the amount of feathering at each pixel
def feather(panA, panB):
    pass
    # common = np.nonzero(panA != 0 * panB != 0)
    # largest = np.max(np.lingalg.norm(common, axis=1))
    # smallest = np.min(np.linalg.norm(common, axis=1))

    # rate = largest/smallest

    # panorama = panA + panB
    # for i in range(len(common[0])//2):
    #     panorama(common[:, i]) = panorama(common[:, i])*rate
