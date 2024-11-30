import numpy as np
from sklearn.cluster import KMeans

def average(imgArr):
    """
    Calculate the average of an image array.
    """
    imgArr = imgArr.astype(np.float32)
    return np.mean(imgArr, axis=0).astype(np.uint8)

def median(imgArr):
    """
    Calculate the median of an image array.
    """
    return np.median(imgArr, axis=0).astype(np.uint8)

def mode(imgArr):
    """
    Calculate the mode of an image array.
    """
    modeImg = 3 * median(imgArr) - 2 * average(imgArr)
    modeImg = np.clip(modeImg, 0, 255)
    return modeImg.astype(np.uint8)

def mostFreq(imgArr):
    """
    Calculate the most frequent pixel values of an image array.
    """
    mostFreq = np.zeros(imgArr[0].shape, dtype=np.uint8)

    for i in range(imgArr[0].shape[0]):
        for j in range(imgArr[0].shape[1]):
            for k in range(imgArr[0].shape[2]):
                freq = np.bincount(imgArr[:, i, j, k])
                mostFreq[i, j, k] = np.argmax(freq)

    return mostFreq

def percentileAverage(imgArr, percentile=50):
    """
    Calculate the percentile average of an image array.
    """
    removePercent = (100 - percentile) / 2
    removeCount = int(imgArr.shape[0] * removePercent / 100)

    sortedImgArr = np.sort(imgArr, axis=0)

    percentileAvg = np.mean(sortedImgArr[removeCount:-removeCount], axis=0)

    return percentileAvg.astype(np.uint8)