import os
import cv2
import json
from itertools import chain
from glob import glob
import random
import numpy as np

import imageio

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import tensorflow as tf
from patchify import patchify, unpatchify

from model_UNet import UNetCompiled
from metrics import feature_f_score
from map_mask import map_mask


def build_patch(filename, legend, patch_dims = (256,256)):
    """
    filename = 'VA_Lahore_bm.tif'   
    legend = 'SOa_poly'
    """

    filePath = os.path.join(working_dir, filename)
    segmentation_file = filePath.split('.')[0]+'_'+legend+'.tif' 
    patch_dims = patch_dims
    
    map_patches_dir = os.path.join(working_dir, filename.split('.')[0]+'_map_patches')        
        
    map_im =  cv2.imread(filePath)

    map_im_dims = map_im.shape
    patch_overlap = 32
    patch_step = patch_dims[1]-patch_overlap
#     map_im =  cv2.imread(filePath)

    # To patchify, the (width - patch_width) mod step_size = 0
    shift_x = (map_im.shape[0]-patch_dims[0])%patch_step
    shift_y = (map_im.shape[1]-patch_dims[1])%patch_step
    shift_x_left = shift_x//2
    shift_x_right = shift_x - shift_x_left
    shift_y_left = shift_y//2
    shift_y_right = shift_y - shift_y_left

    shift_coord =  [shift_x_left, shift_x_right, shift_y_left, shift_y_right]

    map_im_cut = map_im[shift_x_left:map_im.shape[0]-shift_x_right, shift_y_left:map_im.shape[1]-shift_y_right,:]
    map_patchs = patchify(map_im_cut, (*patch_dims,3), patch_step)
    
    if not os.path.exists(map_patches_dir): 
        os.mkdir(map_patches_dir)
        for i in range(map_patchs.shape[0]):
            for j in range(map_patchs.shape[1]):
                imageio.imwrite(os.path.join(map_patches_dir, '{0:02d}_{1:02d}.png'.format(i,j)), (map_patchs[i][j][
0]).astype(np.uint8))

    ## work on cropping the legend and save it to a subfolder "filename_legend"
    legend_dir = os.path.join(working_dir, filename.split('.')[0]+'_legend')
    
#     if os.path.exists(os.path.join(legend_dir, legend+'.png')):
#         return shift_coord, map_im_cut.shape, map_patchs.shape
    
    if not os.path.exists(legend_dir):
        os.mkdir(legend_dir)

    json_file = filePath.split('.')[0]+'.json'
    with open(json_file, 'r') as f:
        jsonData = json.load(f)
    point_coord = []
    for label_dict in jsonData['shapes']:
        if label_dict['label'] == legend:
            point_coord = label_dict['points']
    if not point_coord: raise Exception("The provided legend does not exist: ", filename)
    flatten_list = list(chain.from_iterable(point_coord))
    
    if point_coord[0][0] >= point_coord[1][0] or point_coord[0][1] >= point_coord[1][1]:
        # print("Coordinate right is less than left:  ", filename, legend, point_coord)
        x_low = min(int(point_coord[0][0]), int(point_coord[1][0]))
        x_hi = max(int(point_coord[0][0]), int(point_coord[1][0]))
        y_low = min(int(point_coord[0][1]), int(point_coord[1][1]))
        y_hi = max(int(point_coord[0][1]), int(point_coord[1][1]))
    elif (len(flatten_list)!=4):
        x_coord = [x[0] for x in point_coord]
        y_coord = [x[1] for x in point_coord]
        x_low, y_low, x_hi, y_hi = int(min(x_coord)), int(min(y_coord)), int(max(x_coord)), int(max(y_coord))
        # print("Point Coordinates number is not 4: ", filename, legend)
    else: x_low, y_low, x_hi, y_hi = [int(x) for x in flatten_list]
    legend_coor =  [(x_low, y_low), (x_hi, y_hi)]
    shift_pixel  = 4
    im_crop = map_im[y_low+shift_pixel:y_hi-shift_pixel, x_low+shift_pixel:x_hi-shift_pixel] # need to resize
    #print(y_low+shift_pixel, y_hi-shift_pixel, x_low+shift_pixel, x_hi-shift_pixel)
    im_crop_resize = cv2.resize(im_crop, dsize=patch_dims, interpolation=cv2.INTER_CUBIC)

    imageio.imwrite(os.path.join(legend_dir, legend+'.png'), (im_crop_resize).astype(np.uint8))
    
    return legend_coor, shift_coord, map_im_cut.shape, map_patchs.shape

def test_dataGenerator(filename, legend, patch_dims = (256,256)):
    """
    filename = 'VA_Lahore_bm.tif'
    legend = 'Oc_poly'
    """
    def load_img(patchName):
        map_img = tf.io.read_file(patchName) # Read image file
        map_img = tf.cast(tf.io.decode_png(map_img), dtype=tf.float32) / 255.0
#         map_img = map_img[:,:,0:3]
        
        legendPath = os.path.join(working_dir, filename.split('.')[0]+'_legend', legend+'.png') 

        legend_img = tf.io.read_file(legendPath) # Read image file
        legend_img = tf.cast(tf.io.decode_png(legend_img), dtype=tf.float32) / 255.0
#         legend_img = legend_img[:,:,0:3]
        img = tf.concat(axis=2, values = [map_img, legend_img])
        img = img*2.0 - 1.0 # range(-1.0,1.0)
        resize_image = tf.image.resize(img, [256,256])
        return resize_image
    
    patch_dims = patch_dims
    patchNames = sorted(glob(os.path.join(working_dir, filename.split('.')[0]+'_map_patches/*')))
    test_dataset = tf.data.Dataset.from_tensor_slices(patchNames)
    test_dataset = test_dataset.map(load_img)   
    
    return test_dataset


def map_mask(filename, pad_unpatch_predicted_threshold):
    filePath = os.path.join(working_dir, filename)
    imarray = cv2.imread(filePath)
    gray = cv2.cvtColor(imarray, cv2.COLOR_BGR2GRAY)  # greyscale image
    # Detect Background Color
    pix_hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    background_pix_value = np.argmax(pix_hist, axis=None)

    # Flood fill borders
    height, width = gray.shape[:2]
    corners = [[0,0],[0,height-1],[width-1, 0],[width-1, height-1]]
    for c in corners:
        cv2.floodFill(gray, None, (c[0],c[1]), 255)

    # AdaptiveThreshold to remove noise
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    # Edge Detection
    thresh_blur = cv2.GaussianBlur(thresh, (11, 11), 0)
    canny = cv2.Canny(thresh_blur, 0, 200)
    canny_dilate = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Keeping only the largest detected contour.
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    wid, hight = pad_unpatch_predicted_threshold.shape[0], pad_unpatch_predicted_threshold.shape[1]
    mask = np.zeros([wid, hight])
    mask = cv2.fillPoly(mask, pts=[contour], color=(1,1,1)).astype(int)
    masked_img = cv2.bitwise_and(pad_unpatch_predicted_threshold, mask)
    
    return masked_img


def getInferenceForEachLegend(filename, legend):
    """
    filename = 'VA_Lahore_bm.tif'
    legend = 'CZsum_poly'
    """
    
    # <basemap_name>_<feature_name>.tif
    write_filename = filename.split('.')[0]+'_'+legend+'.tif'
    write_filePath = os.path.join(working_dir, 'Inference', write_filename)
    
    if os.path.exists(write_filePath):
        return
    
    legend_coor, shift_coord, map_im_cut_dims, map_patchs_dims = build_patch(filename, legend, patch_dims = (256,256
))
    
    test_dataset = test_dataGenerator(filename = filename, legend = legend, patch_dims = (256,256))
    
    test_dataset = test_dataset.batch(1)
    predicted = unet.predict(test_dataset)

    patched_predicted = np.reshape(predicted, (map_patchs_dims[0], map_patchs_dims[1], 1, 256, 256, 1))
    unpatch_predicted = unpatchify(patched_predicted, (map_im_cut_dims[0], map_im_cut_dims[1], 1))
    pad_unpatch_predicted = np.pad(unpatch_predicted, [(shift_coord[0], shift_coord[1]), (shift_coord[2], shift_coor
d[3]), (0,0)], mode='constant')
    pad_unpatch_predicted_threshold = np.where(pad_unpatch_predicted>0.5, 1, 0)
    
    masked_img = map_mask(filename, pad_unpatch_predicted_threshold)

    # expand one more dimension and repeat the pixel value in the third axis
    final_raster = np.repeat(masked_img[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

    cv2.imwrite(write_filePath, final_raster)

if __name__ == '__main__': 

    working_dir = '/home/shared/DARPA/eval_data_perfomer_shirui'
    patch_dims = (256,256)
    unet = UNetCompiled(input_size=(*patch_dims,6), n_filters=16, n_classes=1)
    unet.load_weights("./model_poly/saved_best_train_validation_model_16_filter.hdf5")

    tifPaths = glob(working_dir+'/*.tif')
    sorted(tifPaths)

    if not os.path.exists(os.path.join(working_dir, 'Inference')):
        os.mkdir(os.path.join(working_dir, 'Inference'))

    for tifPath in tifPaths:
        tifFile = tifPath.split('/')[-1]
        jsonFile = tifFile.split('.')[0]+'.json'

        with open(os.path.join(working_dir, jsonFile), 'r') as f:
            jsonData = json.load(f)

        for label_dir in jsonData['shapes']:
            legend = label_dir['label']
            if legend.endswith('_poly'):
                print(tifFile, legend)
                getInferenceForEachLegend(tifFile, legend)
