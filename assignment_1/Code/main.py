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
plt.style.use('seaborn')
from collections import defaultdict

from data import read_pcd, read_normal_pcd

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



def iterative_closest_point(base, target, stacked):
    all_RMS = []
    iters = []
    print(base.shape[0])
    no_points = int(target.shape[0]*ARGS.sampling_r)
    target  = sampling(target, target[0:no_points])

    if ARGS.sampling_method in ['none','uniform']:
        base_sub = sampling(base, target) # make base and target same shape

    # Optimize R and t using the EM-algorithm
    RMS = math.inf
    for iter in range(ARGS.max_icp_iters): #todo: maybe chance to a while loop.
        iters.append(iter)
        if ARGS.sampling_method == 'random':
            base_sub = sampling(base, target)

        ind, new_RMS = find_closest_points(base_sub, target) # E-step
        RMS = new_RMS
        all_RMS.append(RMS)
        R, t = compute_R_t(base_sub, target[ind,:]) # M-step


        base = np.dot(R, base.T).T - t.T

        if ARGS.sampling_method in ['none','uniform']:
            base_sub = np.dot(R, base_sub.T).T - t.T

        if stacked.shape[0] > 1:
            stacked = np.dot(R, stacked.T).T - t.T


        if iter > ARGS.icp_threshold_w:
            std = np.std(all_RMS[-ARGS.icp_threshold_w:])
            if (std < ARGS.icp_threshold):
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




def sampling(in_array, out_array):
    ind = np.random.randint(0, in_array.shape[0], out_array.shape[0])
    return(in_array[ind])



def test_icp(file):
    Path("../results/icp_test").mkdir(parents=True, exist_ok=True)
    base = read_pcd(f'../data/data/00000000{file:02}.pcd', ARGS.noise_threshold)
    target = read_pcd(f'../data/data/00000000{file + 1:02}.pcd', ARGS.noise_threshold)
    stacked = np.zeros((1,3))

    #ARGS.max_icp_iters = 30
    sampling_methods = ['none', 'uniform', 'random']
    for m in sampling_methods:
        print(f'--- {m} ---')
        ARGS.sampling_method = m

        # Test noise
        if m == 'none':
            ARGS.sampling_r = 1
        else:
            ARGS.sampling_r = 0.5
        noise_ratios = [0,0.25,0.5,0.75]
        for r in noise_ratios:
            base_noisy = add_noise(base, r)
            target_noisy = add_noise(target, r)
            iters, all_RMS = iterative_closest_point(base_noisy, target_noisy, stacked)
            plt.plot(iters, all_RMS)

        plt.xlabel('iterations')
        plt.ylabel('RMS')
        plt.legend(noise_ratios, loc='upper right', title="percentage of added noise")
        Path("../results/icp_test/added_noise").mkdir(parents=True, exist_ok=True)
        dir = f'../results/icp_test/added_noise/{m}_sampling_ratio_{ARGS.sampling_r}.png'
        plt.savefig(dir)
        plt.clf()

        # Test sampling ratios
        sampling_ratios = [1, 0.75,0.5,0.25]
        if m != 'none':
            for r in sampling_ratios:
                ARGS.sampling_r = r
                iters, all_RMS = iterative_closest_point(base, target, stacked)
                plt.plot(iters, all_RMS)

            plt.xlabel('iterations')
            plt.ylabel('RMS')
            plt.legend(sampling_ratios, loc='upper right', title="sampling ratio")
            Path("../results/icp_test/sampling_ratios").mkdir(parents=True, exist_ok=True)
            dir = '../results/icp_test/sampling_ratios/{}.png'.format(m)
            plt.savefig(dir)
            plt.clf()
        else:
            ARGS.sampling_r = 1
            iters, all_RMS = iterative_closest_point(base, target, stacked)
            plt.plot(iters, all_RMS)
            plt.xlabel('iterations')
            plt.ylabel('RMS')
            dir = '../results/icp_test/sampling_ratios/{}.png'.format(m)
            plt.savefig(dir)
            plt.clf()


def normal_sampling():
    norm_pcd = read_normal_pcd('../Data/data/0000000000_normal.pcd')
    target = read_pcd('../Data/data/0000000000.pcd', ARGS.noise_threshold)
    buckets = create_buchets(norm_pcd)
    samps = []
    keys = buckets.keys()
    no_points = int(norm_pcd.shape[0]*ARGS.sampling_r)
   
    for buck in buckets.values():
        samps.extend(np.random.choice(buck, int(no_points / len(keys))))
        
    print(len(samps))
    return target[samps]

def create_buchets(norm_pcd):

    buckets = np.linspace(-1, 1, 10)
    table = defaultdict(list)
    for i, point in enumerate(norm_pcd):
        for buck in buckets:
            if math.isnan(point[0]):
                table['nan'].append(i)
                break
            if point.mean() < buck:
                table[buck].append(i)
                break 

    return table

    
def merge_pcds():
    Path("../results/merge_pcds").mkdir(parents=True, exist_ok=True)
    base = read_pcd(f'../data/data/00000000{ARGS.start:02}.pcd', ARGS.noise_threshold) #load first base
    stacked = base
    RMSs = []
    iters = []
    start_time = time.time()

    for i in tqdm(range(ARGS.start, ARGS.end, ARGS.step_size)):
        target = read_pcd(f'../data/data/00000000{i + 1:02}.pcd', ARGS.noise_threshold) #load target
        stacked, RMS = iterative_closest_point(base, target, stacked) #transform stacked
        stacked = np.vstack((stacked, target))
        base = target
        RMSs.append(RMS)
        iters.append(i)

    av_RMS = round(np.mean(RMSs), 5)
    seconds = int(time.time() - start_time)

    print(f'average RMS: {av_RMS}')
    print(f'execution time: {seconds} seconds')


    # Save stacked point cloud as pickle
    pkl.dump(stacked, open(f'../results/merge_pcds/{ARGS.sampling_method}_sampling_ratio_{ARGS.sampling_r}_step_size_{ARGS.step_size}.pkl', "wb"))

    # Plot RMS scores
    plt.plot(iters, RMSs)
    plt.xlabel('point cloud pair')
    plt.ylabel('RMS after convergence')
    plt.savefig(f'../results/merge_pcds/{ARGS.sampling_method}_sampling_ratio_{ARGS.sampling_r}_step_size_{ARGS.step_size}_average_RMS_{av_RMS}_seconds_{seconds}.png')
    plt.clf()

    #visualize_pcd(stacked)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--start', default=0, type=int,
                        help='first pcd')
    PARSER.add_argument('--end', default=99, type=int,
                        help='final pcd')
    PARSER.add_argument('--step_size', default=1, type=int,
                        help='step size between the pcds')

    PARSER.add_argument('--sampling_method', default='uniform', type=str,
                        help='method for sub sampling pcd rows, uniform or random')
    PARSER.add_argument('--sampling_r', default=0.5, type=float,
                        help='ratio for sub sampling')
    PARSER.add_argument('--test_sampling_methods', default=['uniform','random','no'], type=list,
                        help='list of methods to be tested')

    PARSER.add_argument('--max_icp_iters', default=10, type=int,
                        help='max number of iterations for icp algorithm')
    PARSER.add_argument('--icp_threshold', default=0.00001, type=float,
                        help='threshold for early stopping icp algorithm')
    PARSER.add_argument('--icp_threshold_w', default=10, type=int,
                        help='window for threshold for icp algorithm')

    PARSER.add_argument('--noise_threshold', default=2, type=float,
                        help='keep points up to this distance')

    ARGS = PARSER.parse_args()




#test_icp()
# merge_pcds()
normal_sampling()