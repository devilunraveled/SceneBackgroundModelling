import numpy as np

def averageImg(imgArr):
    """
    Calculate the average of an image array.
    """
    return np.mean(imgArr, axis=0)