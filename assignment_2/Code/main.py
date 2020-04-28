import numpy as np 
import cv2 
from matplotlib import pyplot as plt
from sklearn import linear_model



def main():

    # read image and make it greyscale
    img = cv2.imread('Data/House/frame00000001.png')
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


    # perform sift
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(grey, None)

    # not entirally sure what this does
    kp, des = sift.compute(grey, kp)
    
    # draw keypoints on original image and save it
    img= cv2.drawKeypoints(grey, kp, img)
    cv2.imwrite('Data/SIFT/sift_keypoints.jpg',img)




    # do RANSAC
    ransac = linear_model.RANSACRegressor()
    # ransac.fit(X, y)



if __name__ == "__main__":
    main()