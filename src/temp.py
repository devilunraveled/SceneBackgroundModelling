import cv2
import os
import numpy as np
from bgModels import averageImg
from alive_progress import alive_bar

dataPath = 'data/SBMnet_dataset'
# category = 'backgroundMotion'
# video = 'canoe'

resultsPath = 'results'
method = 'averageImg'

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

            imgBg = averageImg(imgArr).astype(np.uint8)
            os.makedirs(os.path.join(resultsPath, method, category, video), exist_ok=True)
            cv2.imwrite(os.path.join(resultsPath, method, category, video, 'result.jpg'), imgBg)
            bar()