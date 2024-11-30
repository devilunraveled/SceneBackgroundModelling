import cv2
import os
import numpy as np
import bgModels
from alive_progress import alive_bar
import sys

dataPath = 'data/SBMnet_dataset'
resultsPath = 'results'

if __name__ == '__main__':
    """
        python src/main.py <method>

        <method>: func name from bgModels.py
    """
    method = sys.argv[1]
    bgFunc = getattr(bgModels, method)

    for category in os.listdir(dataPath):
        print(f'Processing {category}...')
        videos = os.listdir(os.path.join(dataPath, category))
        with alive_bar(len(videos)) as bar:
            for video in videos:
                imgArr = []
                videoPath = os.path.join(dataPath, category, video, 'input')

                for img in os.listdir(videoPath):
                    imgArr.append(cv2.imread(os.path.join(videoPath, img)))

                imgArr = np.array(imgArr)

                imgBg = bgFunc(imgArr)
                os.makedirs(os.path.join(resultsPath, method, category, video), exist_ok=True)
                cv2.imwrite(os.path.join(resultsPath, method, category, video, 'result.jpg'), imgBg)
                bar()
