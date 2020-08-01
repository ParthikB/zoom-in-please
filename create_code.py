from __helper__ import get_dominant_color
from PARAMETERS import *
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob


code       = {}
code_color = []
code_img   = []

total_images = len(os.listdir(CODE_IMG_DIR))

print('[INFO] Importing Images from :', CODE_IMG_DIR)
files = []
[files.extend(glob.glob(CODE_IMG_DIR + '/*.' + e)) for e in IMG_EXTENSIONS]
print('[INFO] Total Images found    :', len(files))


for idx, fname in tqdm(enumerate(files), desc='Processing '):   
    img   = plt.imread(fname)    
    dominant_color, img = get_dominant_color(img, SLICE_SIZE)
    
    code_color.append(dominant_color)
    code_img.append(img)
    
code['color'] = code_color
code['img']   = code_img

np.save('data/codes/'+CODE_NAME, code)
print(f'[INFO] Code saved : data/codes/{CODE_NAME}.npy')