import numpy as np
from tqdm import tqdm

def read_pcd(fname, noise_threshold, fname_normal=None):
    '''
    Read PCD data

    args: fname - Path to the PCD file
    returns: data - Nx6 matrix where each row is a point, with fields x y z rgb
                    imX imY. x, y, z are the 3D coordinates of the point, rgb is
                    the color of the point packed into a float (unpack using
                    unpackRGBFloat), imX and imY are the horizontal and vertical
                    pixel locations of the point in the original Kinect image.

    Matlab Author: Kevin Lai
    '''
    data = []
    version = 0
    width = 0
    height = 0
    points = 0

    final = []

    with open(fname, 'r') as f, open(fname_normal, 'r') as fn:
        lines = f.readlines()
        norm_lines = fn.readlines()
        for i, l in enumerate(lines):
            # clean line
            l = l.split(' ')
            ln = norm_lines[i].split(' ')
            l[-1] = l[-1].strip('\n')
            ln[-1] = ln[-1].strip('\n')

            # if l[0].isalpha():
            #     if l[0] == "VERSION":
            #         version = float(l[1])
            #     elif l[0] == 'WIDTH':
            #         width = int(l[1])
            #     elif l[0] == 'HEIGHT':
            #         height = int(l[1])
            #     elif l[0] == 'POINTS':
            #         points = int(l[1])

            if not l[0].isalpha() and not l[0] == '#':
                l = [float(i) for i in l]
                ln = [float(j) for j in ln]
                data.append(l[:-1])
                data[-1].extend(ln[:-1])



    pcd = np.array(data)
    pcd = pcd[pcd[:, 2] < noise_threshold]
    norm = pcd[:, 3:]
    pcd = pcd[:, :3]
    return pcd, norm

def read_normal_pcd(fname):
    data = []
    with open(fname) as f:
        
        lines = f.readlines()
        for i, l in enumerate(lines):
            l = l.split(' ')
            l[-1] = l[-1].strip('\n')
            if i > 10:
                l = [float(k) for k in l]
                data.append(l[:-1])

    return np.array(data)

if __name__ == '__main__':
    print(read_pcd('../Data/data/0000000000.pcd'))
