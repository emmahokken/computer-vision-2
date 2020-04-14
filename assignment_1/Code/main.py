import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import open3d as o3d
import math
import matplotlib.pyplot as plt
import pickle as pkl
import sys
import argparse
import time
from tqdm import tqdm
plt.style.use('seaborn')

from data import read_pcd

def visualize_pcd(pcd_array):
    '''
    Converts an array to a point cloud and visualizes it.
    '''
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array)
    o3d.io.write_point_cloud("sync.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("sync.ply")
    o3d.visualization.draw_geometries([pcd_load])

    # # convert Open3D.o3d.geometry.PointCloud to numpy array
    # pcd_array = np.asarray(pcd_load.points)
    #
    # # save z_norm as an image (change [0,1] range to [0,255] range with uint8 type)
    # img = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
    # o3d.io.write_image("sync.png", img)
    # o3d.visualization.draw_geometries([img])


# EMMA. I have two methods here. Do you have a way to check which one is fastest?
def find_closest_points(base, target):
    # Nearest neighbors method
    nnb_model = NearestNeighbors(n_neighbors=1).fit(target)
    dis, ind = nnb_model.kneighbors(base, return_distance=True)

    dis = dis.squeeze()
    ind = list(ind.squeeze())

    # Tree method (I believe KDTree is a faster method, which we can use later for the sped-up version)
    # kdt_model = KDTree(target)
    # dis, ind = kdt_model.query(base)
    # dis = list(dis)
    # ind = list(ind)

    # Todo, check if the used distance metric (think its euclidean) equals the RMS from the assignment
    RMS = np.mean(dis)
    return ind, RMS


def compute_R_t(base, target):
    # Map points around 0
    base_centroid = np.mean(base, axis=0)
    target_centroid = np.mean(target, axis=0)

    base = base - base_centroid
    target = target - target_centroid

    # Compute R using singular value decomposition
    covariance_matrix = np.dot(base.T, target)
    U, S, V = np.linalg.svd(covariance_matrix)
    R = np.dot(V.T, U.T) # Not sure about the transpose here. See if the RMS goes down, then it should be fine.

    # Compute t
    t = np.dot(R, base_centroid) - target_centroid

    return R, t

def iterative_closest_point(base, target, stacked, i):
    all_RMS = []
    iters = []

    no_points = int(target.shape[0]*ARGS.sub_sampling_r)
    target  = sub_sampling(target, target[0:no_points])

    if ARGS.sub_sampling_method == 'uniform':
        base_sub = sub_sampling(base, target) # make base and target same shape

    # Optimize R and t using the EM-algorithm
    RMS = math.inf
    for iter in range(ARGS.max_icp_iters): #todo: maybe chance to a while loop.
        iters.append(iter)
        if ARGS.sub_sampling_method == 'random':
            base_sub = sub_sampling(base, target)

        ind, new_RMS = find_closest_points(base_sub, target) # E-step
        RMS = new_RMS
        #print(RMS)
        all_RMS.append(RMS)
        R, t = compute_R_t(base_sub, target[ind,:]) # M-step

        base = np.dot(R, base.T).T - t.T
        if ARGS.sub_sampling_method == 'uniform':
            base_sub = np.dot(R, base_sub.T).T - t.T

        stacked = np.dot(R, stacked.T).T - t.T


        if iter > ARGS.icp_treshold_w:
            std = np.std(all_RMS[-ARGS.icp_treshold_w:])
            if std < ARGS.icp_treshold:
                break


    # # Plot iteration
    # if i == 0:
    #     plt.plot(iters, all_RMS)
    #     plt.xlabel('iteration')
    #     plt.ylabel('RMS')
    #     plt.savefig('../results/rms_icp_{}_{}_{}.png'
    #                 .format(ARGS.sub_sampling_method, ARGS.sub_sampling_r,
    #                         ARGS.step_size))
    #     plt.clf()

    return stacked, RMS

def sub_sampling(in_array, out_array):
    ind = np.random.randint(0, in_array.shape[0], out_array.shape[0])
    return(in_array[ind])

def merge_pcds():
    base = read_pcd('../Data/data/0000000000.pcd') #load first base
    stacked = base
    RMSs = []
    iters = []

    start_time = time.time()

    for i in tqdm(range(ARGS.start, ARGS.end, ARGS.step_size)):
        target = read_pcd(f'../Data/data/00000000{i + 1:02}.pcd') #load target

        stacked_bu = np.copy(stacked) #back-up stacked
        stacked, RMS = iterative_closest_point(base, target, stacked, i) #transform stacked

        if RMS < 0.004:
            stacked = np.vstack((stacked, target))
            base = target
        else:
            print(i, RMS)
            stacked = stacked_bu

        RMSs.append(RMS)
        iters.append(i)

    pkl.dump(base, open('../results/{}_{}_{}.pkl'
                .format(ARGS.sub_sampling_method, ARGS.sub_sampling_r,
                        ARGS.step_size), "wb"))



    av_RMS = np.mean(RMSs)
    print(f'execution time: {time.time() - start_time} seconds')
    print('average RMS: {}'.format(av_RMS))

    plt.plot(iters, RMSs)
    plt.xlabel('point cloud pair')
    plt.ylabel('RMS')
    plt.savefig('../results/rms_main_{}_{}_{}.png'
                .format(ARGS.sub_sampling_method, ARGS.sub_sampling_r,
                        ARGS.step_size))
    plt.clf()

    visualize_pcd(stacked)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--start', default=0, type=int,
                        help='first pcd')
    PARSER.add_argument('--end', default=99, type=int,
                        help='final pcd')
    PARSER.add_argument('--step_size', default=1, type=int,
                        help='step size between the pcds')

    PARSER.add_argument('--sub_sampling_method', default='uniform', type=str,
                        help='method for sub sampling pcd rows, uniform or random')
    PARSER.add_argument('--sub_sampling_r', default=0.5, type=float,
                        help='ratio for sub sampling')

    PARSER.add_argument('--max_icp_iters', default=1000, type=int,
                        help='max number of iterations for icp algorithm')
    PARSER.add_argument('--icp_treshold', default=0.00001, type=float,
                        help='treshold for early stopping icp algorithm')
    PARSER.add_argument('--icp_treshold_w', default=10, type=int,
                        help='window for treshold for icp algorithm')

    ARGS = PARSER.parse_args()

    merge_pcds()
