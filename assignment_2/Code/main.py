import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn import linear_model
import sys

def zeroes(size):
    return np.zeros(size)

def chaining():
    with open('PointViewMatrix.txt', 'r') as f:
        bigboi = np.zeros((205, 215))
        for i, line in enumerate(f):
            print(i)
            line = np.array(line.split(' '))
            print(line.shape)
            bigboi[i, :] = line.squeeze()
            # break

        plt.imshow(bigboi)
        plt.show()

    # this is very ugly and 6000 is a very random number but it works
    pvm = zeroes((100, 6000))

    addition = 0
    index = {}
    feature_count = 0

    # iterate over all images, compare 1-2, 2-3, 48-49, 49-1
    for img_number in range(1,49):
        print(f'Image {img_number} and {img_number + 1}')

        img1 = cv.imread(f'Data/House/frame000000{img_number:02}.png')
        img2 = cv.imread(f'Data/House/frame000000{img_number + 1:02}.png')

        grey1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
        grey2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

        # perform sift to get keypoints and descriptors
        sift = cv.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(grey1, None)
        kp2, des2 = sift.detectAndCompute(grey2, None)

        # match descriptors (using L1 norm now, because it's the only one that worked...)
        matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
        matches = matcher.match(des1, des2)

        # only keep good matcjes (why? idk)
        matches.sort(key=lambda x: x.distance, reverse=False)
        num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:num_good_matches]

        points1 = []
        points2 = []

        for i, match in enumerate(matches):
            points1.append(kp1[match.queryIdx].pt)
            points2.append(kp2[match.trainIdx].pt)

            # save index of match so that when found in next
            # imgage the feature can be added to the right column
            # (not 100% sure this is correct)
            index[match.trainIdx] = feature_count
            feature_count += 1

            # get correct index from dict
            try:
                ind = index[match.queryIdx]
            except KeyError:
                ind = index[match.trainIdx]

            # create new column
            z = zeroes((100))
            z[img_number + addition] = points1[-1][0] # x
            z[img_number + addition + 1] = points2[-1][1] # y
            z[img_number + addition + 2 ] = points2[-1][0]
            z[img_number + addition + 3] = points2[-1][1]

            # put column in proper place in array
            pvm[:, ind] = z

        addition += 1

    print('Image 49 and 1')
    # do this all again for last two images
    img1 = cv.imread('Data/House/frame00000049.png')
    img2 = cv.imread('Data/House/frame00000001.png')

    # perform sift to get keypoints and descriptors
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(grey1, None)
    kp2, des2 = sift.detectAndCompute(grey2, None)

    # match descriptors (using L1 norm now, because it's the only one that worked...)
    matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
    matches = matcher.match(des1, des2)

    # only keep good matcjes (why? idk)
    matches.sort(key=lambda x: x.distance, reverse=False)
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    points1 = []
    points2 = []

    for i, match in enumerate(matches):
        points1.append(kp1[match.queryIdx].pt)
        points2.append(kp2[match.trainIdx].pt)

        index[match.trainIdx] = feature_count
        feature_count += 1
        try:
            ind = index[match.queryIdx]
        except KeyError:
            ind = index[match.trainIdx]

        # create new column
        z = zeroes((100))
        z[img_number + addition] = points1[-1][0] # x
        z[img_number + addition + 1] = points2[-1][1] # y
        z[img_number + addition + 2 ] = points2[-1][0]
        z[img_number + addition + 3] = points2[-1][1]

        # put column in proper place in array
        pvm[:, ind] = z

    # remove excess zeroes
    pvm = pvm[:, :max(index.values())]

    norm = np.where(pvm > 0, 1, 0)
    plt.imshow(norm)
    plt.show()

if __name__ == "__main__":
    chaining()
