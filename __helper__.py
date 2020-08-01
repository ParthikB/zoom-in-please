def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from PARAMETERS import *
import scipy
import scipy.misc
import scipy.cluster
from PIL import Image
import os, numpy as np, cv2, gc, matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool


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
    
    
def get_dominant_color(img, RESIZE=15):
   
    NUM_CLUSTERS = 3
    orig_img  = cv2.resize(img, (CODE_IMG_SIZE, CODE_IMG_SIZE))
    
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
            yield _slice


def get_distance(x, y):
    return pow(sum((x - y)**2), 0.5)

def get_new_shape(img, slice_size):
    w, h, c = img.shape
    nr, nc = w//slice_size, h//slice_size
    return nr, nc

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
    similar_slices     = map(find_match, dominant_color_map)

    return similar_slices


def multi(P, FUNC, ITER):
    p = Pool(processes=P)
    data = p.map(FUNC, ITER)
    p.close()
    gc.collect()
    return data

def generate_maps_parallel(master_img, code_name, slice_size, cpu=2):
    global color_arr, code
    code = np.load('data/codes/'+CODE_NAME+'.npy', allow_pickle=True).tolist()
    color_arr = np.array(code['color'])
    
    print('Parallelizing...')
    start = time.time()
    a = multi(cpu, get_dominant_color, generate_slice_list(master_img, slice_size))
    similar_slices = multi(cpu, find_match, a)
    print('[INFO] Time taken :', round(time.time()-start), 'sec.')
    return similar_slices


def stich_it_back(master_img, similar_slices, size):
    ROWS = size*nr
    COLS = size*nc
    mask_im = Image.new("RGB", (COLS, ROWS), 0)
        
    start = time.time()
    
    c = list(product(range(max(nr, nc)), repeat=2))
    
    i, row, col = 0, 0, 0
    for row, col in tqdm(c):
        if row < nr and col < nc:
            top  = col*size
            left = row*size
            
            # slice_ = next(similar_img, None)            
            slice_ = similar_slices[i]

            # if slice_.shape[0] != REBUILD_SLICE_SIZE:
            #     slice_ = cv2.resize(slice_, (REBUILD_SLICE_SIZE, REBUILD_SLICE_SIZE))

            try: s = Image.fromarray(slice_)
            except: s = Image.fromarray(slice_.astype('uint8'))
            # s = Image.fromarray((slice_ * 255).astype(np.uint8))
    

            mask_im.paste(s, box=(top, left))
            i += 1

    gc.collect()
    del s

    return mask_im



def save_img(fname, final_img, original_img, REBUILD_SLICE_SIZE, alpha=0.3):
    r = nr*REBUILD_SLICE_SIZE
    c = nc*REBUILD_SLICE_SIZE
    
    overlaying_back_img = Image.fromarray(cv2.resize(original_img, (c, r)))
    overlaying_back_img.putalpha(int(alpha*255))
    
    final_img.paste(overlaying_back_img, (0, 0), overlaying_back_img)
    del overlaying_back_img
    
    final_img.save('data/output_images/'+fname)
    print('[INFO] Image saved at : data/output_images/'+fname)
    del final_img
