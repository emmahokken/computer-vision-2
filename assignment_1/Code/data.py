import numpy as np
from tqdm import tqdm

def read_pcd(fname):
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

    with open(fname, 'r') as f:
        lines = f.readlines()
        for l in lines:
            # clean line
            l = l.split(' ')
            l[-1] = l[-1].strip('\n')

            #
            if l[0].isalpha():
                if l[0] == "VERSION":
                    version = float(l[1])
                elif l[0] == 'WIDTH':
                    width = int(l[1])
                elif l[0] == 'HEIGHT':
                    height = int(l[1])
                elif l[0] == 'POINTS':
                    points = int(l[1])

            elif l[0] != '#':
                l = [float(i) for i in l]
                data.append(l[:-1])



    pcd = np.array(data)
    pcd = pcd[pcd[:, -1] < 2]
    return(pcd)


if __name__ == '__main__':
    print(read_pcd('../Data/data/0000000000.pcd'))
