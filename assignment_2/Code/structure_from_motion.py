import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
import pandas as pd

from chaining import chaining

def sfm():

    pvm = pd.read_csv('Data/pvm.csv', index_col=0).values

    # TODO: get dense block
    dense_block = pvm

    # normalize by translating to mean
    dense_norm = translate_to_mean(dense_block)

    # apply SVD to

    U, W, Vt = np.linalg.svd(dense_norm)
    U3 = U[:,:3]
    W3 = W[:3, :3] # maybe should be np.diag
    Vt3 = Vt[:3, :]

    M = U3.dot(W3)
    S = W3.dot(Vt3)

def translate_to_mean(pvm):
    ''' Normalize the point coordinates by translating them to the mean of the
        points in each view in the dense block. '''
    translation = pvm.T - np.nanmean(pvm, axis=1)
    return translation




if __name__ == '__main__':
    sfm()
