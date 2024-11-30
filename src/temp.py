import cv2
import os
import numpy as np
import bgModels
from alive_progress import alive_bar
import sys

dataPath = 'data/SBMnet_dataset'
# category = 'illuminationChanges'
# video = 'CameraParameter'
category = 'backgroundMotion'
video = 'fall'

if __name__ == '__main__':
    """
        python src/temp.py <method>

        <method>: func name from bgModels.py
    """
    method = sys.argv[1]
    bgFunc = getattr(bgModels, method)

    imgArr = []
    videoPath = os.path.join(dataPath, category, video, 'input')

    for img in os.listdir(videoPath):
        imgArr.append(cv2.imread(os.path.join(videoPath, img)))

    imgArr = np.array(imgArr)

    imgBg = bgFunc(imgArr)
    cv2.imwrite('result.jpg', imgBg)