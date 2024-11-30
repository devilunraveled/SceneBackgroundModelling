from collections.abc import Callable
import numpy as np
import cv2

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

def frameDifferenceAvg(imgArr, diffThreshold=30):
    """
    Mask the foreground using frame difference of an image array and take average.
    """
    bgMaskSum = np.zeros(imgArr[0].shape)
    maskedImgSum = np.zeros(imgArr[0].shape)

    for i in range(1, imgArr.shape[0]):
        diffImg = cv2.absdiff(imgArr[i], imgArr[i - 1])
        diffImg = cv2.cvtColor(diffImg, cv2.COLOR_BGR2GRAY)
        _, diffImg = cv2.threshold(diffImg, diffThreshold, 255, cv2.THRESH_BINARY)
        bgMask = np.where(diffImg == 255, 0, 1)
        bgMask = np.expand_dims(bgMask, axis=-1)
        bgMaskSum += bgMask
        maskedImgSum += imgArr[i] * bgMask
    
    maskedAvg = maskedImgSum / bgMaskSum
    maskedAvg = np.nan_to_num(maskedAvg, nan=0).astype(np.uint8)

    return maskedAvg.astype(np.uint8)

def frameDifferenceMedian(imgArr, diffThreshold=30):
    """
    Mask the foreground using frame difference of an image array and take median.
    """
    maskedImgArr = imgArr.copy()

    for i in range(1, imgArr.shape[0]):
        diffImg = cv2.absdiff(imgArr[i], imgArr[i - 1])
        diffImg = cv2.cvtColor(diffImg, cv2.COLOR_BGR2GRAY)
        _, diffImg = cv2.threshold(diffImg, diffThreshold, 255, cv2.THRESH_BINARY)
        bgMask = np.where(diffImg == 255, 0, 1)
        bgMask = np.expand_dims(bgMask, axis=-1)
        maskedImgArr[i] = imgArr[i] * bgMask

    maskedMedian = np.zeros(imgArr[0].shape)
    for i in range(imgArr[0].shape[0]):
        for j in range(imgArr[0].shape[1]):
            for k in range(imgArr[0].shape[2]):
                maskedValues = maskedImgArr[:, i, j, k][maskedImgArr[:, i, j, k] != 0]
                if len(maskedValues) == 0:
                    maskedMedian[i, j, k] = 0
                else:
                    maskedMedian[i, j, k] = np.median(maskedValues)

    return maskedMedian.astype(np.uint8)

def frameDifferenceMostFreq(imgArr, diffThreshold=30):
    """
    Mask the foreground using frame difference of an image array and take most frequent pixel values.
    """
    mostFreq = np.zeros(imgArr[0].shape, dtype=np.uint8)
    maskedImgArr = imgArr.copy()

    for i in range(1, imgArr.shape[0]):
        diffImg = cv2.absdiff(imgArr[i], imgArr[i - 1])
        diffImg = cv2.cvtColor(diffImg, cv2.COLOR_BGR2GRAY)
        _, diffImg = cv2.threshold(diffImg, diffThreshold, 255, cv2.THRESH_BINARY)
        bgMask = np.where(diffImg == 255, 0, 1)
        bgMask = np.expand_dims(bgMask, axis=-1)
        maskedImgArr[i] = imgArr[i] * bgMask

    for i in range(imgArr[0].shape[0]):
        for j in range(imgArr[0].shape[1]):
            for k in range(imgArr[0].shape[2]):
                freq = np.bincount(maskedImgArr[:, i, j, k][maskedImgArr[:, i, j, k] != 0])
                if len(freq) == 0:
                    mostFreq[i, j, k] = 0
                else:
                    mostFreq[i, j, k] = np.argmax(freq)

    return mostFreq

def gmm(imgArr):
    """
    Apply Gaussian Mixture Model (GMM) for background subtraction.
    
    Parameters:
        imgArr (numpy.ndarray): Array of images (frames) to process.
        
    Returns:
        numpy.ndarray: The background model image.
    """
    # Create a Background Subtractor object using MOG2
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=10, detectShadows=True)

    # Initialize an empty image to accumulate the foreground mask
    fgmask = np.zeros(imgArr[0].shape, dtype=np.uint8)

    # Process each frame in the image array
    for frame in imgArr:
        # Apply Gaussian blur to reduce noise and improve segmentation
        blurFrame = cv2.GaussianBlur(frame, (9, 9), 0)
        
        # Apply the background subtractor to get the foreground mask
        mask = fgbg.apply(blurFrame)
        mask = np.stack((mask,) * 3, axis=-1)
        
        # print(mask.shape, fgmask.shape)
        # Accumulate the foreground mask
        fgmask = cv2.bitwise_or(fgmask, mask)

    # Optionally apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    return fgmask

def opticalFlowAvg(imgArr):
    """
    Apply optical flow to an image array.
    """
    bgMaskSum = np.zeros(imgArr[0].shape)
    maskedImgSum = np.zeros(imgArr[0].shape)

    for i in range(1, imgArr.shape[0]):
        prev = cv2.cvtColor(imgArr[i - 1], cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(imgArr[i], cv2.COLOR_BGR2GRAY)

        flowImg = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flowImg = np.linalg.norm(flowImg, axis=-1)
        flowImg = cv2.normalize(flowImg, None, 0, 255, cv2.NORM_MINMAX)
        bgMask = np.where(flowImg == 0, 0, 1)
        bgMask = np.expand_dims(bgMask, axis=-1)
        bgMaskSum += bgMask
        maskedImgSum += imgArr[i] * bgMask

    maskedAvg = maskedImgSum / bgMaskSum
    maskedAvg = np.nan_to_num(maskedAvg, nan=0).astype(np.uint8)

    return maskedAvg.astype(np.uint8)

def opticalFlowMedian(imgArr):
    """
    Apply optical flow to an image array.
    """
    maskedImgArr = imgArr.copy()

    for i in range(1, imgArr.shape[0]):
        prev = cv2.cvtColor(imgArr[i - 1], cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(imgArr[i], cv2.COLOR_BGR2GRAY)

        flowImg = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flowImg = np.linalg.norm(flowImg, axis=-1)
        flowImg = cv2.normalize(flowImg, None, 0, 255, cv2.NORM_MINMAX)
        bgMask = np.where(flowImg == 0, 0, 1)
        bgMask = np.expand_dims(bgMask, axis=-1)
        maskedImgArr[i] = imgArr[i] * bgMask

    maskedMedian = np.zeros(imgArr[0].shape)
    for i in range(imgArr[0].shape[0]):
        for j in range(imgArr[0].shape[1]):
            for k in range(imgArr[0].shape[2]):
                maskedValues = maskedImgArr[:, i, j, k][maskedImgArr[:, i, j, k] != 0]
                if len(maskedValues) == 0:
                    maskedMedian[i, j, k] = 0
                else:
                    maskedMedian[i, j, k] = np.median(maskedValues)

    return maskedMedian.astype(np.uint8)

def pool(imgArr, candiadateFuncs, pooling_method='mean'):
    """
    Apply various background modeling techniques and pool their results.

    Parameters:
        imgArr (numpy.ndarray): Array of images (frames) to process.
        pooling_method (str): The pooling method to use ('average', 'max').

    Returns:
        numpy.ndarray: The pooled background model image.
    """
    # Stack all results into a single array for pooling
    results = np.array([func(imgArr) for func in candiadateFuncs])

    # Apply the specified pooling method
    if pooling_method == 'mean':
        pooled_result = np.mean(results, axis=0).astype(np.uint8)
    elif pooling_method == 'max':
        pooled_result = np.max(results, axis=0).astype(np.uint8)
    else:
        raise ValueError("Invalid pooling method. Choose 'average' or 'max'.")

    return pooled_result
