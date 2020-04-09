import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import open3d as o3d
import math
import matplotlib.pyplot as plt

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

def iterative_closest_point(base, target, iters):
    #Todo. Maybe something to make them the same shape, if nessecay
    all_rms = []
    threshold = 0.0001
    padded_target = np.zeros(base.shape)
    padded_target[:target.shape[0], :target.shape[1]] = target

    # Initialize rotation matrix R and translation vector t
    # R = np.identity(base.shape[1])
    # t = np.zeros(base.shape[1]).shape

    # visualize_pcd(target)

    # Optimize R and t using the EM-algorithm
    RMS = math.inf
    print(RMS)
    for iter in range(iters): #todo: maybe chance to a while loop.
        ind, new_RMS = find_closest_points(base, padded_target) # E-step

        #if new_RMS < RMS:
        RMS = new_RMS
        R, t = compute_R_t(base, padded_target[ind,:]) # M-step
        base = np.dot(R, base.T).T - t.T # update the base point cloud

        print(RMS)
        all_rms.append(RMS)

        if iter > 2 and abs(all_rms[-1] - all_rms[-2]) < threshold:
            break

    stacked = np.vstack((base, target))
    return stacked


base = read_pcd('../Data/data/0000000000.pcd')
for i in range(1):
    target = read_pcd(f'../Data/data/000000000{str(i + 1)}.pcd')
    base = iterative_closest_point(base, target, 35)




visualize_pcd(base)
