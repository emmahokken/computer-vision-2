import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
import pandas as pd
import argparse
from scipy import sparse
from chaining import chaining
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

def get_motion_structure(dense_block):
    U, W, Vt = np.linalg.svd(dense_block)

    U_3 = U[:,:3]
    W_3 = np.diag(W[:3])
    Vt_3 = Vt[:3, :]

    M = U_3.dot(np.sqrt(W_3))
    S = np.sqrt(W_3).dot(Vt_3)
    S_list.append(S)

    return(S, M)




def sfm():
    pvm = pd.read_csv(f'../results/chaining/pvm/pvm_{ARGS.match_method}_{ARGS.dist_filter}_{ARGS.nearby_filter}.csv', index_col=0).values
    row, col = pvm.shape[0], pvm.shape[1]

    S_list = []
    for r in range(row):
        for c in range(col):

            # get dense block
            dense_block = get_dense_block(pvm[r:,c:])
            if isinstance(dense_block, bool):
                continue

            # normalize by translating to mean
            dense_norm = dense_block - np.mean(dense_block, axis=1).reshape(-1, 1)

            S, M = get_motion_structure(dense_norm)
            print(S)
            print(M)
            sys.exit()

    # print(np.array(S_list).shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(S_list[0][0], S_list[0][1], S_list[0][2])
    plt.show()

def get_dense_block(pvm):

    # check how many images have points
    d = pvm[:, 0]
    d = d[np.invert(np.isnan(d))]

    if d.shape[0] < 2 * ARGS.consecutive_frames:
        return False

    dense_block = pvm[:2*ARGS.consecutive_frames]
    points = np.all(np.invert(np.isnan(dense_block)), axis=0)
    dense_block = dense_block[:, points]

    if dense_block.shape[1] < ARGS.consecutive_frames:
        return False

    return dense_block

if __name__ == '__main__':

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--viz', default=False, type=bool,
                        help='whether to visualize the result')
    PARSER.add_argument('--match_method', default='bf', type=str,
                        help='which method to use for matching feature points', choices=['bf', 'flann'])
    PARSER.add_argument('--dist_filter', default=0.25, type=float,
                        help='initial points filtering')
    PARSER.add_argument('--nearby_filter', default=10, type=int,
                        help='threshold for determining whether two points are similar')
    PARSER.add_argument('--consecutive_frames', default=3, type=int,
                        help='amount of consecutive frames')

    ARGS = PARSER.parse_args()

    sfm()
