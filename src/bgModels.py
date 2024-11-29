import numpy as np

def average(imgArr):
    """
    Calculate the average of an image array.
    """
    return np.mean(imgArr, axis=0)

def median(imgArr):
    """
    Calculate the median of an image array.
    """
    return np.median(imgArr, axis=0)

def mostFreq(imgArr):
    """
    Calculate the mode of an image array.
    """
    # mostFreq = np.zeros(imgArr[0].shape, dtype=np.uint8)

    # for i in range(imgArr[0].shape[0]):
    #     for j in range(imgArr[0].shape[1]):
    #         for k in range(imgArr[0].shape[2]):
    #             freq = np.bincount(imgArr[:, i, j, k])
    #             mostFreq[i, j, k] = np.argmax(freq)
    
    mostFreq = 3 * median(imgArr) - 2 * average(imgArr)

    return mostFreq