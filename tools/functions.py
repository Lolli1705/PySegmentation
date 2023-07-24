from scipy.ndimage import label, generate_binary_structure, convolve
import typing
import skimage
from skimage import io
import cv2
import numpy as np
import os

def conn_segment(img,kernel, threshold):

    label_array, num_features = label(img,structure=kernel)

    label_size = [(label_array==lab).sum() for lab in range(num_features+1)]

    for label_, size in enumerate(label_size):

        if size < threshold:

            img[label_array==label_]=0
    
    return img

def save_files(output_path, file_names, threshold,
               kernel, iterations, conn_kernel, conn_threshold):
    
    for num,file in enumerate(file_names):

        img = io.imread(file)

        mask = img > threshold

        bin_img = np.ones((img.shape[0], img.shape[1]))*mask

        erod_img = cv2.erode(bin_img, kernel, iterations = iterations)

        img_final = conn_segment(erod_img, conn_kernel, conn_threshold)

        dil_img = cv2.dilate(img_final, kernel, iterations=iterations)
        
        out = os.path.join(output_path,"slice_%04d" %num)
        io.imsave(f"{out}" + ".tif", skimage.img_as_uint(dil_img))
        

        