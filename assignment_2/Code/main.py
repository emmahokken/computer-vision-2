import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn import linear_model

GOOD_MATCH_PERCENT = 0.15

def zeroes(size):
    return np.zeros(size)

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
    matches = matcher.match(des1, des2)

    # only keep good matcjes (why? idk)
    matches.sort(key=lambda x: x.distance, reverse=False)
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    # # draw top matches
    # img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches, None)
    # cv.imwrite("Data/SIFT/matches.jpg", img_matches)


    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    h, mask = cv.findHomography(points1, points2, cv.RANSAC)

    # find matrix A
    A = np.array([points1[:, 0]*points2[:, 0], points1[:, 0]*points2[:,1], points1[:,0], points1[:, 1]*points2[:,0],
                    points1[:,1]*points2[:,1], points1[:,1], points2[:,0], points2[:,1], np.ones((len(points1)))]).T

    F, mask = cv.findFundamentalMat(points1,points2)
    # We select only inlier points
    pts1 = points1[mask.ravel()==1]
    pts2 = points2[mask.ravel()==1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(grey1,grey2,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(grey2,grey1,lines2,pts2,pts1)
    # plt.subplot(121),plt.imshow(img5)
    # plt.subplot(122),plt.imshow(img3)
    # plt.show()

    F = np.random.normal(size=[3,3])
    p = np.random.normal(size=[3,10])
    p_a = np.random.normal(size=[3,10])
    A = construct_A(p, p_a)

    F_h = np.random.normal(size=[3,3])
    p_h = np.random.normal(size=[3,10])
    p_ah = np.random.normal(size=[3,10])
    A_h = construct_A(p_h, p_ah)

    lines = F.dot(p).T
    lines_h = F_h.dot(p_h).T
    p = p.T[:,:2]
    p = np.float32(p)
    p_a = p_a.T[:,:2]
    p_a = np.float32(p_a)

    img3,img6 = drawlines(grey1,grey2,lines,p,p_a)
    # plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()




def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def construct_A(p, p_a,):
    p = p.T
    p_a = p_a.T
    A = np.array([p[:,0]*p_a[:,0], p[:,0]*p_a[:,1], p[:,0], p[:, 1]*p_a[:,0],
                  p[:,1]*p_a[:,1], p[:,1], p_a[:,0], p_a[:,1], np.ones((len(p)))]).T
    return(A)

def compute_matches(img1, img2):
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

    return matches

def chaining():
    pvm = np.zeros((50, 6000, 2))

    index = {}
    feature_count = 0
    # iterate over all images, compare 1-2, 2-3, 48-49, 49-1
    for img_number in range(1,50):
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
            index[match.trainIdx] = feature_count
            feature_count += 1
            try:
                print('query index!')
                ind = index[match.queryIdx]
            except KeyError:
                print('train index :(')
                ind = index[match.trainIdx]

            # create new column
            z = zeroes((50, 2))
            z[img_number] = points1[-1]
            z[img_number + 1] = points2[-1]

            # put column in proper place in array
            pvm[:, ind, :] = z



        # temp = np.zeros((49, len(points1)), dtype=(float, 2))
        # temp[i - 1, :] = np.array(points1, dtype=(float, 2))
        # temp[i, :] = np.array(points2, dtype=(float, 2))
        #
        # pvm = np.hstack((pvm, temp))

    # TODO: do final iteration from 49 to 1
    img1 = cv.imread('Data/House/frame00000049.png')
    img2 = cv.imread('Data/House/frame00000001.png')

    # for each point found, add column to matrix
    points1, points2 = compute_points(img1, img2)
    temp = np.zeros((49, len(points1)), dtype=(float, 2))
    temp[i - 1, :] = np.array(points1, dtype=(float, 2))
    temp[i, :] = np.array(points2, dtype=(float, 2))

    pvm = np.hstack((pvm, temp))

if __name__ == "__main__":
    chaining()
