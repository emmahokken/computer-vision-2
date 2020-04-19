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
from pathlib import Path
from scipy.io import loadmat
from collections import defaultdict

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

def find_closest_points(base, target):
    if ARGS.dist_measure == 'nn':
        # Nearest neighbors method
        nnb_model = NearestNeighbors(n_neighbors=1).fit(target)
        dis, ind = nnb_model.kneighbors(base, return_distance=True)

        dis = dis.squeeze()
        ind = list(ind.squeeze())


    elif ARGS.dist_measure == 'kd':
        # Tree method 
        kdt_model = KDTree(target)
        dis, ind = kdt_model.query(base)
        dis = list(dis)
        ind = list(ind)

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
    R = np.dot(V.T, U.T)

    # Compute t
    t = np.dot(R, base_centroid) - target_centroid

    return R, t


def sampling(in_array, no_points):
    ind = np.random.randint(0, in_array.shape[0], no_points)
    return(in_array[ind])


def normal_sampling(norm_pcd, base, no_points):
    buckets = create_buchets(norm_pcd)
    samps = []
    keys = buckets.keys()

    for buck in buckets.values():
        samps.extend(np.random.choice(buck, int(no_points / len(keys))))

    return base[samps]


def iterative_closest_point(base, target, stacked, base_norm, target_norm):
    all_RMS = []
    iters = []

    no_points = int(target.shape[0]*ARGS.sampling_r)

    if ARGS.sampling_method == 'normal':
        target = normal_sampling(target_norm, target, no_points)

    else:
        target  = sampling(target, no_points)

    if ARGS.sampling_method in ['none','uniform']:
        base_sub = sampling(base, no_points) # make base and target same shape


    if ARGS.sampling_method == 'normal':
        base_sub = normal_sampling(base_norm, base, no_points)

    # Optimize R and t using the EM-algorithm
    RMS = math.inf
    for iter in range(ARGS.max_icp_iters):
        iters.append(iter)
        if ARGS.sampling_method == 'random':
            base_sub = sampling(base, no_points)

        ind, new_RMS = find_closest_points(base_sub, target) # E-step
        RMS = new_RMS
        all_RMS.append(RMS)
        R, t = compute_R_t(base_sub, target[ind,:]) # M-step


        base = np.dot(R, base.T).T - t.T

        if ARGS.sampling_method in ['none','uniform', 'normal']:
            base_sub = np.dot(R, base_sub.T).T - t.T

        if stacked.shape[0] > 1:
            stacked = np.dot(R, stacked.T).T - t.T


        if iter > ARGS.icp_treshold_w:
            std = np.std(all_RMS[-ARGS.icp_treshold_w:])
            if (std < ARGS.icp_treshold):
                break

    # Plot iteration
    if stacked.shape[0] == 1:
        return(iters, all_RMS)

    return stacked, RMS



def add_noise(pcd, ratio):
    '''
    Add gaussian noise to a pcd based on the mean and variance of each axis.
    '''
    no_noise = int(ratio*len(pcd[:,0]))
    noise_x = np.random.normal(np.mean(pcd[:,0]), np.std(pcd[:,0]), (no_noise,1))
    noise_y = np.random.normal(np.mean(pcd[:,1]), np.std(pcd[:,1]), (no_noise,1))
    noise_z = np.random.normal(np.mean(pcd[:,2]), np.std(pcd[:,2]), (no_noise,1))
    noise = np.hstack((noise_x, noise_y, noise_z))
    noisy_pcd = np.vstack((pcd, noise))

    return (noisy_pcd)







def create_buchets(norm_pcd):

    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    z = np.linspace(-1, 1, 5)
    table = defaultdict(list)
    for i, point in enumerate(norm_pcd):
        done = False
        if math.isnan(point[0]) or math.isnan(point[1]) or math.isnan(point[2]):
            continue
        for xx in x:
            if done:
                break
            for yy in y:
                if done:
                    break
                for zz in z:
                    if done:
                        break
                    if point[0] < xx and point[1] < yy and point[2] < zz:
                        table[f'x{xx}y{yy}z{zz}'].append(i)
                        done = True
                        break

    return table

def test_icp():
    Path("../results/icp_test/").mkdir(parents=True, exist_ok=True)
    base, base_norm = read_pcd('../data/data/0000000000.pcd', ARGS.noise_treshold, '../data/data/0000000000_normal.pcd')
    target, target_norm = read_pcd('../data/data/0000000001.pcd',  ARGS.noise_treshold, '../data/data/0000000001_normal.pcd')

    #base = loadmat('../Data/source.mat')['source'].T
    #target = loadmat('../Data/target.mat')['target'].T
    stacked = np.zeros((1,3)) # stacked matrix of zeros for testing the icp



    sampling_methods = ['normal', 'none', 'uniform', 'random']
    for m in sampling_methods:
        print('--- {} ---'.format(m))
        ARGS.sampling_method = m

        # Test noise
        Path("../results/icp_test/added_noise").mkdir(parents=True, exist_ok=True)
        if m == 'none':
            ARGS.sampling_r = 1
        else:
            ARGS.sampling_r = 0.5
        noise_ratios = [0,0.25,0.5,0.75,1]
        for r in noise_ratios:
            base_noisy = add_noise(base, r)
            #visualize_pcd(base_noisy)
            target_noisy = add_noise(target, r)
            iters, all_RMS = iterative_closest_point(base_noisy, target_noisy, stacked, base_norm, target_norm)
            plt.plot(iters, all_RMS)

        plt.xlabel('iterations')
        plt.ylabel('RMS')
        plt.legend(noise_ratios, loc='upper right', title="percentage of added noise")
        dir = '../results/icp_test/added_noise/{}_sampling_ratio_{}.png'.format(m, ARGS.sampling_r)
        plt.savefig(dir)
        plt.clf()


        # Test sampling ratios
        Path("../results/icp_test/sampling_ratios").mkdir(parents=True, exist_ok=True)
        sampling_ratios = [1, 0.75,0.5,0.25,0.1]
        if m != 'none':
            for r in sampling_ratios:
                ARGS.sampling_r = r
                iters, all_RMS = iterative_closest_point(base, target, stacked, base_norm, target_norm)
                plt.plot(iters, all_RMS)

            plt.xlabel('iterations')
            plt.ylabel('RMS')
            plt.legend(sampling_ratios, loc='upper right', title="sampling ratio")
            dir = '../results/icp_test/sampling_ratios/{}.png'.format(m)
            plt.savefig(dir)
            plt.clf()
        else:
            ARGS.sampling_r = 1
            iters, all_RMS = iterative_closest_point(base, target, stacked, base_norm, target_norm)
            plt.plot(iters, all_RMS)
            plt.xlabel('iterations')
            plt.ylabel('RMS')
            dir = '../results/icp_test/sampling_ratios/{}.png'.format(m)
            plt.savefig(dir)
            plt.clf()





def iterative_closest_point_alt(base, target):
    all_RMS = []
    iters = []
    no_points = int(target.shape[0]*ARGS.sampling_r)
    target  = sampling(target, no_points)
    #base_sub = sampling(base, no_points) # make base and target same shape

    # Optimize R and t using the EM-algorithm
    RMS = math.inf
    for iter in range(ARGS.max_icp_iters):
        iters.append(iter)
        ind, new_RMS = find_closest_points(base, target) # E-step
        RMS = new_RMS
        all_RMS.append(RMS)
        R, t = compute_R_t(base, target[ind,:]) # M-step


        #base = np.dot(R, base.T).T - t.T
        #base_sub = np.dot(R, base_sub.T).T - t.T
        base = np.dot(R, base.T).T - t.T

        if iter > ARGS.icp_treshold_w:
            std = np.std(all_RMS[-ARGS.icp_treshold_w:])
            if (std < ARGS.icp_treshold):
                break

    return base, RMS















def merge_pcds():
    Path("../results/merge_pcds").mkdir(parents=True, exist_ok=True)
    base, base_norm = read_pcd(f'../data/data/00000000{ARGS.start:02}.pcd', ARGS.noise_treshold, f'../data/data/00000000{ARGS.start:02}_normal.pcd')

    stacked = base
    RMSs = []
    iters = []
    start_time = time.time()

    for i in tqdm(range(ARGS.start, ARGS.end, ARGS.step_size)):
        target, target_norm = read_pcd(f'../data/data/00000000{i + 1:02}.pcd', ARGS.noise_treshold, f'../data/data/00000000{i + 1:02}_normal.pcd')

        if ARGS.merge_method == '3.1':
            stacked, RMS = iterative_closest_point(base, target, stacked, base_norm, target_norm) #transform stacked
            #print(RMS)
            stacked = np.vstack((stacked, target))
            base = target
            base_norm = target_norm

        else:
            stacked, RMS = iterative_closest_point_alt(stacked, target) #transform stacked
            stacked = np.vstack((stacked, target))

        RMSs.append(RMS)
        iters.append(i)

    av_RMS = round(np.mean(RMSs), 5)
    seconds = int(time.time() - start_time)

    print('average RMS: {}'.format(av_RMS))
    print('execution time: {} seconds'.format(seconds))


    # Save stacked point cloud as pickle
    pkl.dump(stacked, open('../results/merge_pcds/{}_sampling_ratio_{}_step_size_{}.pkl'.format(ARGS.sampling_method,
            ARGS.sampling_r,
            ARGS.step_size), "wb"))

    # Plot RMS scores
    plt.plot(iters, RMSs)
    plt.xlabel('point cloud pair')
    plt.ylabel('RMS after convergence')
    plt.savefig('../results/merge_pcds/{}_sampling_ratio_{}_step_size_{}_average_RMS_{}_seconds_{}.png'.format(
        ARGS.sampling_method, ARGS.sampling_r,ARGS.step_size, av_RMS, seconds))
    plt.clf()

    if ARGS.visualize:
        visualize_pcd(stacked)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    # PCD files
    PARSER.add_argument('--start', default=0, type=int,
                        help='first pcd')
    PARSER.add_argument('--end', default=99, type=int,
                        help='final pcd')
    PARSER.add_argument('--step_size', default=1, type=int, # TO TEST: 1,2,4,10
                        help='step size between the pcds')


    # Point selection
    PARSER.add_argument('--sampling_method', default='uniform', type=str,
                        help='method for sub sampling pcd rows', choices=['uniform', 'random', 'normal'])
    PARSER.add_argument('--sampling_r', default=1, type=float,
                        help='ratio for sub sampling')


    # ICP algorithm parameters
    PARSER.add_argument('--max_icp_iters', default=100, type=int,
                        help='max number of iterations for icp algorithm')
    PARSER.add_argument('--icp_treshold', default=0.00001, type=float,
                        help='treshold for early stopping icp algorithm')
    PARSER.add_argument('--icp_treshold_w', default=10, type=int,
                        help='window for treshold for icp algorithm')

    # Noise 
    PARSER.add_argument('--noise_treshold', default=2, type=float,
                        help='keep points up to this distance')
    
    # Visualisation of pcd
    PARSER.add_argument('--visualize', default=True, type=bool,
                        help='whether to visualize the result')

    # Merging
    PARSER.add_argument('--merge_method', default='3.1', type=str,
                        help='Method for merging, always use uniform sampling', choices=['3.1','3.2'])

    # Distance measure
    PARSER.add_argument('--dist_measure', default='nn', type=str,
                        help='Method for measuring distance, nearest neighbour or kd trees', choices=['nn','kd'])


    ARGS = PARSER.parse_args()




#test_icp()
merge_pcds()
