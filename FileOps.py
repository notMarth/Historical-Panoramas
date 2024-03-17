# Image and File loading Utilities
# Used to import photos and perform file operations

import numpy as np

# return two given images but with appropriate padding
# imA will be the "bigger" image that imB is fit into
def getImage(imA, imB):

    heightA, widthA = imA.shape
    heightB, widthB = imB.shape

    bigA = np.pad(imA, ((heightA, ), (widthA, )), mode='constant', constant_values=0.)
    bigB = np.pad(imB, ((2*heightA - heightB, ), (2*widthA - widthB, )), mode='constant', constant_values=0.)

    return bigA, bigB