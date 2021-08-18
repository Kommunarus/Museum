import pandas as pd
from PIL import Image
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle as pkl
import cv2


df = pd.read_csv('./data/train_url_xs_2.csv')
# df = pd.read_csv('./data/train_url_only.csv')

paths = df['guid'].values.tolist()


def to_pkl(gui):

    image_raw = cv2.imread(os.path.join('/home/alex/PycharmProjects/dataForMC/images_little_2/', f'{gui}.jpg'))
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)


    with open(os.path.join('/home/alex/PycharmProjects/dataForMC/pkl', f'{gui}.pkl'), 'wb') as f:
        pkl.dump(image, f)


with ThreadPoolExecutor(max_workers=15) as executor:
    executor.map(to_pkl, paths)