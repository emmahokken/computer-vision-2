import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
import pandas as pd
import argparse
from scipy.spatial import procrustes
from mpl_toolkits.mplot3d import Axes3D
import math
from sklearn.neighbors import NearestNeighbors

def sfm():
    pvm = pd.read_csv(f'results/chaining/pvm/pvm_{ARGS.match_method}_{ARGS.dist_filter}_{ARGS.nearby_filter}.csv', index_col=0).values
    row, col = pvm.shape[0], pvm.shape[1]

    S_list = []
    index_list = []
    for r in range(0, row, 2):
        for c in range(col):
            # get dense block
            dense_block, indices = get_dense_block(pvm[r:,c:], c)
            if isinstance(dense_block, bool):
                continue

            # normalize by translating to mean
            dense_norm = dense_block - np.mean(dense_block, axis=1).reshape(-1, 1)

            # apply SVD to dense submatrix
            S, M = get_motion_structure(dense_norm)
            S_list.append(S)
            index_list.append(indices)

    for i, s in enumerate(S_list):
        if i > 0 and i + 1 < len(S_list):
            look_at = []
            look_at = [l1 for l1 in index_list[i] if l1 in index_list[i + 1]]
            look_at = np.intersect1d(index_list[i], index_list[i + 1])
            from_1 = [int(np.where(index_list[i] == la)[0]) for la in look_at]
            from_2 = [int(np.where(index_list[i + 1] == la)[0]) for la in look_at]
            s1 = s[:, from_1]
            s2 = S_list[i + 1][:, from_2]

            mtx1, mtx2, disp = procrustes(s1, s2)

            

def get_motion_structure(dense_block):
    U, W, Vt = np.linalg.svd(dense_block)

    U_3 = U[:,:3]
    W_3 = np.diag(W[:3])
    Vt_3 = Vt[:3, :]

    M = U_3.dot(np.sqrt(W_3))
    S = np.sqrt(W_3).dot(Vt_3)

    return S, M

def get_dense_block(pvm, c):

    # check how many images have points
    dense_block = pvm[:, 0]
    dense_block = dense_block[np.invert(np.isnan(dense_block))]

    if dense_block.shape[0] < 2 * ARGS.consecutive_frames:
        return False, False

    dense_block = pvm[:2*ARGS.consecutive_frames]
    points = np.all(np.invert(np.isnan(dense_block)), axis=0)
    indices = np.where(points)

    dense_block = dense_block[:, points]

    if dense_block.shape[1] < ARGS.consecutive_frames:
        return False, False

    indices = indices[0] + c

    return dense_block, indices

if __name__ == '__main__':

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--viz', default=False, type=bool,
                        help='whether to visualize the result')
    PARSER.add_argument('--match_method', default='bf', type=str,
                        help='which method to use for matching feature points', choices=['bf', 'flann'])
    PARSER.add_argument('--dist_filter', default=0.5, type=float,
                        help='initial points filtering')
    PARSER.add_argument('--nearby_filter', default=10, type=int,
                        help='threshold for determining whether two points are similar')
    PARSER.add_argument('--consecutive_frames', default=3, type=int,
                        help='amount of consecutive frames')

    ARGS = PARSER.parse_args()

    sfm()
