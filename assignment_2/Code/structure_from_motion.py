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
import open3d as o3d

def get_motion_structure(dense_block):
    U, W, Vt = np.linalg.svd(dense_block)

    U_3 = U[:,:3]
    W_3 = np.diag(W[:3])
    Vt_3 = Vt[:3, :]

    M = U_3.dot(np.sqrt(W_3))
    S = (np.sqrt(W_3)).dot(Vt_3)

    return(S, M)

def visualize(S):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(S)
    o3d.io.write_point_cloud("test.ply", pcd)

    pcd_load = o3d.io.read_point_cloud("test.ply")
    o3d.visualization.draw_geometries([pcd_load])

def get_dense_block(pvm):
    d = pvm[:, 0]
    d = d[np.invert(np.isnan(d))]

    if d.shape[0] < 2 * ARGS.consecutive_frames:
        return False

    dense_block = pvm[:2*ARGS.consecutive_frames]
    points = np.all(np.invert(np.isnan(dense_block)), axis=0)
    dense_block = dense_block[:, points]

    if dense_block.shape[1] < ARGS.consecutive_frames:
        return False

    return(dense_block)

def sfm():
    # Visualize benchmark point view matrix
    with open('../PointViewMatrix.txt', 'r') as f:
        pvm = np.zeros((202, 215))
        for i, line in enumerate(f):
            line = np.array(line.split(' '))
            pvm[i, :] = line.squeeze()

    pvm = pvm - np.mean(pvm, axis=1).reshape(-1, 1)
    S, M = get_motion_structure(pvm)
    #visualize(S.T)

    # Visualize constructed point view matrix
    pvm = pd.read_csv(f'../results/chaining/pvm/pvm_bf_0.25_10.csv', index_col=0).values

    row, col = pvm.shape[0], pvm.shape[1]
    for r in range(0,row,ARGS.consecutive_frames):
        S_all = []
        for c in range(col):
            # get dense block
            dense_block = get_dense_block(pvm[r:,c:])
            if isinstance(dense_block, bool):
                continue

            # normalize by translating to mean
            dense_block = dense_block - np.mean(dense_block, axis=1).reshape(-1, 1)
            S, M = get_motion_structure(dense_block)
            S_all.append(S)
            visualize(S.T)
            break
        break

if __name__ == '__main__':

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--consecutive_frames', default=3, type=int,
                        help='amount of consecutive frames')

    ARGS = PARSER.parse_args()

    sfm()
