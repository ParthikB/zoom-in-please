import cv2
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from __helper__ import *

import scipy
import scipy.misc
import scipy.cluster
from PIL import Image
import matplotlib.pyplot as plt
import os, numpy as np, cv2
from tqdm import tqdm
from multiprocessing import Pool
import gc

import time
from itertools import *


def show_dominant_color(img):
    code, img = get_dominant_color(img)
    
    code = [[list(map(int, code))]]
    print('RGB :', code)
   
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(code)
    
    
def show_single_color(code):
    code = [[list(map(int, code))]]
    plt.imshow(code)
    
    
def get_dominant_color(img, RESIZE=10):
   
    NUM_CLUSTERS = 3
    orig_img  = cv2.resize(img, (85, 85))
    
    img       = cv2.resize(img, (RESIZE, RESIZE))
    shape     = img.shape
    ar        = img.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
    codes, _  = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    vecs, _   = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, _ = scipy.histogram(vecs, len(codes))      # count occurrences
    index_max = scipy.argmax(counts)                   # find most frequent
    dominant_color      = codes[index_max]
    return dominant_color, orig_img



# Getting a single slice according to the current coordinates
def __get_slice(img, start_row, end_row, start_col, end_col):
    return img[start_row:end_row, start_col:end_col]

# Dividing the Image into smaller slices
def get_slice_list(img, slice_size):

    w, h, c = img.shape
    global nr, nc
    nr, nc = w//slice_size, h//slice_size

    slices = []
    for i in tqdm(range(nr)):
        for j in range(nc):
            start_row = slice_size*i 
            start_col = slice_size*j
            end_row   = start_row + slice_size 
            end_col   = start_col + slice_size

            _slice = __get_slice(img, start_row, end_row, start_col, end_col)
#             slices.append(_slice)
            yield _slice
#     return slices, nc, nr


def get_distance(x, y):
    return pow(sum((x - y)**2), 0.5)








code = np.load('data/vectorcode75.npy', allow_pickle=True).tolist()


# code = {}
# code_color = []
# code_img   = []

# DIR = 'images'
# total_images = len(os.listdir(DIR))

# for idx, fname in tqdm(enumerate(os.listdir(DIR))):    
# #     print(idx, fname)
#     fname = os.path.join(DIR, fname)
#     img = plt.imread(fname)
    
#     dominant_color, img = get_dominant_color(img)
    
#     code_color.append(dominant_color)
#     code_img.append(img)
    
# code['color'] = code_color
# code['img'] = code_img

# np.save('data/vectorcode75', code)





def get_new_shape(img, slice_size):
    w, h, c = img.shape
    nr, nc = w//slice_size, h//slice_size
    return nr, nc

## GENERATOR
def generate_slice_list(master_img, SLICE_SIZE):
    return get_slice_list(master_img, slice_size=SLICE_SIZE)







def find_match(dominant_color):
    dominant_color = dominant_color[0]
    
    min_idx = np.argmin(np.sum((dominant_color-color_arr)**2, axis=1))
    best_img = code['img'][min_idx]
    return best_img


# s = time.time()

def generate_maps(SLICE_SIZE):
    global color_arr
    color_arr = np.array(code['color'])
    slice_list = generate_slice_list(SLICE_SIZE)
    
    dominant_color_map = map(get_dominant_color, slice_list)
    similar_img_map    = map(find_match, dominant_color_map)

    return similar_img_map


# print(time.time()-s)





def multi(P, FUNC, ITER):
    p = Pool(processes=P)
    data = p.map(FUNC, ITER)
    p.close()
    gc.collect()
    return data

# CPU = 2

# s = time.time()

def generate_maps_parallel(master_img, slice_size, cpu=2):
    global color_arr
    color_arr = np.array(code['color'])
    a = multi(cpu, get_dominant_color, generate_slice_list(master_img, slice_size))
    similar_img_list = multi(cpu, find_match, a)
    return similar_img_list

# print(time.time()-s)




def stich_it_back(master_img, rows, cols, slice_size, cpu=2, size=75):
    ROWS = size*rows
    COLS = size*cols
    mask_im = Image.new("RGB", (COLS, ROWS), 0)
        
    # slices = generate_maps(SLICE_SIZE)
    start = time.time()
    print('Parallelizing...')
    slices = generate_maps_parallel(master_img, slice_size, cpu=cpu)
    print('Time taken :', round(time.time()-start), 'sec.')

    c = list(product(range(max(nr, nc)), repeat=2))
    
    i, row, col = 0, 0, 0
    for row, col in tqdm(c):
    
        if row < nr and col < nc:
            top  = col*size
            left = row*size
            
            # slice_ = next(slices, None)            
            slice_ = slices[i]
#             import pdb
#             pdb.set_trace()
            
            s = Image.fromarray(slice_)
            mask_im.paste(s, box=(top, left))
            i += 1

    gc.collect()
    del slices, s

    return mask_im



def save_img(fname, final_img, master_img, alpha=0.3):
    r = nr*100
    c = nc*100
    
    overlaying_back_img = Image.fromarray(cv2.resize(master_img, (c, r)))
    overlaying_back_img.putalpha(int(alpha*255))
    
    final_img.paste(overlaying_back_img, (0, 0), overlaying_back_img)
    del overlaying_back_img
    
    final_img.save(fname + '.jpg')
    del final_img


def reduce(img, new_row):
    r, c, _ = img.shape
    scale_factor = r/new_row

    new_r = int(r/scale_factor)
    new_c = int(c/scale_factor)
    
    return cv2.resize(img, (new_c, new_r))