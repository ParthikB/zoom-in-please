from __helper__ import *
from PARAMETERS import *
import matplotlib.pyplot as plt
import cv2

# Importing Image
original_img = plt.imread(INPUT_IMG_NAME)

# Resizing Image
img = cv2.resize(original_img, (RESIZE_TO, RESIZE_TO))

# Slicing the Image + Dominant Colour for Every Slice + Finding the Match
# similar_slices = generate_maps(SLICE_SIZE)
print('[INFO] Mapping Images')
similar_slices = generate_maps_parallel(img, CODE_NAME, SLICE_SIZE, cpu=CPU)

# Stiching back
print('[INFO] Stiching it back')
final_img = stich_it_back(img, 
                          similar_slices, 
                          size=REBUILD_SLICE_SIZE)

# Saving the Final Output
print('[INFO] Saving..')
save_img(fname=OUTPUT_IMG_NAME, 
    final_img=final_img, 
    original_img=original_img,
    REBUILD_SLICE_SIZE=REBUILD_SLICE_SIZE,
    alpha=ALPHA)

print('[INFO] Yay, all done! Terminating..!')
