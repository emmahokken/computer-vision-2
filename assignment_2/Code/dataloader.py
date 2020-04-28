import numpy as np 

def read_jpg(fname):
    with open(fname, 'r') as f:
        rea = f.readlines()

    print(rea)
