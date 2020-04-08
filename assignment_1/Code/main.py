import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import open3d as o3d
import math


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

    # convert Open3D.o3d.geometry.PointCloud to numpy array
    pcd_array = np.asarray(pcd_load.points)

    # save z_norm as an image (change [0,1] range to [0,255] range with uint8 type)
    img = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
    o3d.io.write_image("sync.png", img)
    o3d.visualization.draw_geometries([img])


# EMMA. I have two methods here. Do you have a way to check which one is fastest?
def find_closest_points(base, target):
    # Nearest neighbors method
    nnb_model = NearestNeighbors(n_neighbors=1).fit(target)
    dis, ind = nnb_model.kneighbors(base, return_distance=True)
    dis = list(dis.squeeze())
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
    R = np.dot(V, U) # Not sure about the transpose here. See if the RMS goes down, then it should be fine.

    # Compute t
    t = target_centroid - np.dot(R, base_centroid)

    return R, t

def iterative_closest_point(base, target, iters):
    #Todo. Maybe something to make them the same shape, if nessecay

    # Initialize rotation matrix R and translation vector t
    R = np.identity(base.shape[1])
    t = np.zeros(base.shape[1]).shape

    # Optimize R and t using the EM-algorithm
    RMS = math.inf
    for iter in range(iters): #todo: maybe chance to a while loop.
        ind, new_RMS = find_closest_points(base, target) # E-step

        #if new_RMS < RMS:
        RMS = new_RMS
        R, t = compute_R_t(base, target[ind,:]) # M-step
        base = np.dot(base, R) - t # update the base point cloud
        print(RMS)

        #else:
        #    break






# sample base matrix (from 03d package)
x = np.linspace(0, 6, 50)
mesh_x, mesh_y = np.meshgrid(x, x)
z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
z_norm = (z - z.min()) / (z.max() - z.min())
base = np.zeros((np.size(mesh_x), 3))
base[:, 0] = np.reshape(mesh_x, -1)
base[:, 1] = np.reshape(mesh_y, -1)
base[:, 2] = np.reshape(z_norm, -1)

# sample target matrix (from 03d package)
x = np.linspace(-3, 3, 50)
mesh_x, mesh_y = np.meshgrid(x, x)
z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
z_norm = (z - z.min()) / (z.max() - z.min())
target = np.zeros((np.size(mesh_x), 3))
target[:, 0] = np.reshape(mesh_x, -1)
target[:, 1] = np.reshape(mesh_y, -1)
target[:, 2] = np.reshape(z_norm, -1)

iterative_closest_point(base, target, 1000)
