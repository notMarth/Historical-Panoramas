import cv2
import numpy as np

def detectAndDescribe(image, feat):
    
    orb = cv2.ORB_create()
    sift = cv2.SIFT_create()
    kp = orb.detect(image, None)
    brisk = cv2.BRISK_create()

    return brisk.detectAndCompute(image, None)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB, match_type, ratio=0.7):

    bf = cv2.BFMatcher()
    
    if match_type == 0:
        matches = bf.match(featuresA, featuresB)
        good_matches = sorted(matches, key = lambda x:x.distance)[:int(len(matches)/15)]
        
    else:
        matches = bf.knnMatch(featuresA, featuresB, k=2)
        
        good_matches = []
        for m,n in matches:
            if m.distance < ratio*n.distance:
                good_matches.append(m)

    #Adapted from https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    src_pts = np.float32([ kpsA[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([ kpsB[m.trainIdx].pt for m in good_matches]) 
    print(src_pts)
    print(dst_pts)
    H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    
    return good_matches, H[0]

def stitch(imageA, imageB, maskA, maskB, patchA, patchB, match_type = 1, match_lim=-1, mode=""):
    
    
    if mode == "mask" and len(maskA) > 0: kps1, feat1 = detectAndDescribe(maskA, None)
    elif mode == "patch" and len(patchA) > 0: kps1, feat1 = detectAndDescribe(patchA, None)
    else: kps1, feat1 = detectAndDescribe(imageA, None)

    if mode == "mask" and len(maskB) > 0: kps2, feat2 = detectAndDescribe(maskB, None)
    elif mode == "patch" and len(maskB) > 0: kps2, feat2 = detectAndDescribe(patchB, None)
    else: kps2, feat2 = detectAndDescribe(imageB, None)

    matches, H = matchKeypoints(kps1, kps2, feat1, feat2, match_type=match_type)
    result = np.zeros((2*imageB.shape[0], 2*imageB.shape[1]))
    result += cv2.warpPerspective(imageA, H, (2*imageB.shape[1], 2*imageB.shape[0]))
    warped = result
    padded = np.pad(imageB, ((0, result.shape[0] - imageB.shape[0]), (0, result.shape[1] - imageB.shape[1])), 'constant', constant_values=0)
    mask = np.ones(result.shape)
    mask[(result>0) * (padded>0)] = 2
    result = (padded + result) / mask
    
    imMatches = cv2.drawMatches(imageA, kps1, imageB, kps2, matches[:match_lim], None)
    #imMatches=None
    return (result, imMatches, warped)