import numpy as np 
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn import linear_model

GOOD_MATCH_PERCENT = 0.15


def main():

    # read image and make it greyscale
    img1 = cv.imread('Data/House/frame00000001.png')
    grey1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)

    img2 = cv.imread('Data/House/frame00000035.png')
    grey2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    # perform sift to get keypoints and descriptors 
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(grey1, None)
    kp2, des2 = sift.detectAndCompute(grey2, None)

    # match descriptors (using L1 norm now, because it's the only one that worked...)
    matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
    matches = matcher.match(des1, des2, None)

    # only keep good matcjes (why? idk)
    matches.sort(key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    
    # draw top matches
    imMatches = cv.drawMatches(img1, kp1, img2, kp2, matches, None)
    cv.imwrite("Data/SIFT/matches.jpg", imMatches)
        
    # draw keypoints on original image and save it
    img= cv.drawKeypoints(grey1, kp1, img1)
    cv.imwrite('Data/SIFT/other_sift_keypoints.jpg',img)

    # RANSAC 
    


if __name__ == "__main__":
    main()