import numpy as np 
import cv2 

def main():

    # read image and make it greyscale
    img = cv2.imread('Data/House/frame00000001.png')
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


    # perform sift
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(grey, None)

    # draw keypoints on original image and save it
    img= cv2.drawKeypoints(grey, kp, img)
    cv2.imwrite('Data/SIFT/sift_keypoints.jpg',img)



if __name__ == "__main__":
    main()