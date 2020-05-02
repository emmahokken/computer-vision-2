
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
import pandas as pd

def zeroes(size):
    return np.zeros(size)

def chaining():
    pvm = pd.DataFrame()

    # iterate over all images, compare 1-2, 2-3, 48-49, 49-1
    for img_number in range(1,50):

        if img_number + 1 < 50:
            img_number2 = img_number + 1
        else:
            img_number2 = 1
        # img_number2 = img_number + 1 if img_number + 1 < 50 else 1

        print(f'Image {img_number} and {img_number2}')

        matches, kp1, kp2 = get_matches(img_number, img_number2)

        p = []
        p_a = []

        # iterate over matches and create pvm
        for i, match in enumerate(matches):
            p.append(kp1[match.queryIdx].pt)
            p_a.append(kp2[match.trainIdx].pt)

            # very first iteration, create first column
            if pvm.shape[1] == 0:
                feature_point = f'{i + 1}'
                pvm = set_points(pvm, img_number, img_number2, feature_point, p, p_a)
            else:
                pvm = get_pvm(pvm, img_number, img_number2, p, p_a)

    print(pvm.shape)

    pvm = pvm.values
    plt.imshow(pvm, aspect='auto')
    plt.show()
    norm = np.where(pvm > 0, 1, 0)

    plt.imshow(norm, aspect='auto')
    plt.show()
    return pvm, norm

def get_pvm(pvm, img_number, img_number2, p, p_a):
    # very first iteration

    found = False
    for feature_point, point in enumerate(zip(pvm.loc[f'x{img_number}'], pvm.loc[f'y{img_number}'])):
        if nearby(p[-1], point):
            found = True
            pvm.ix[f'x{img_number2}', feature_point] = p_a[-1][0]
            pvm.ix[f'y{img_number2}', feature_point] = p_a[-1][1]
            break

    # when no new points are found, add a new column
    if not found:
        feature_point = f'{pvm.shape[1] + 1}'
        pvm = set_points(pvm, img_number, img_number2, feature_point, p, p_a)

    return pvm

def set_points(pvm, img_number, img_number2, feature_point, p, p_a):
    pvm.ix[f'x{img_number}', feature_point] = p[-1][0]
    pvm.ix[f'y{img_number}', feature_point] = p[-1][1]
    pvm.ix[f'x{img_number2}', feature_point] = p_a[-1][0]
    pvm.ix[f'y{img_number2}', feature_point] = p_a[-1][1]

    return pvm

def get_matches(img_number, img_number2, t=0.25):
    img1 = cv.imread(f'Data/House/frame000000{img_number:02}.png')
    img2 = cv.imread(f'Data/House/frame000000{img_number2:02}.png')

    grey1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    grey2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    # perform sift to get keypoints and descriptors
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(grey1, None)
    kp2, des2 = sift.detectAndCompute(grey2, None)

    # match descriptors (using L1 norm now, because it's the only one that worked...)
    matcher = cv.BFMatcher(cv.NORM_L1)
    matches = matcher.match(des1, des2)

    # only keep good matcjes (why? idk)
    matches.sort(key=lambda x: x.distance)
    num_good_matches = int(len(matches) * t)
    matches = matches[:num_good_matches]

    return matches, kp1, kp2

def nearby(points1, points2, t=10):
    x1, y1 = points1
    x2, y2 = points2
    close = x1 - t <= x2 <= x1 + t and y1 - t <= y2 <= y1 + t
    return close


def test_pvm():
    with open('PointViewMatrix.txt', 'r') as f:
        bigboi = np.zeros((205, 215))
        for i, line in enumerate(f):
            print(i)
            line = np.array(line.split(' '))
            print(line.shape)
            bigboi[i, :] = line.squeeze()
            # break

        plt.imshow(bigboi, aspect='auto')
        plt.show()


if __name__ == '__main__':
    chaining()
