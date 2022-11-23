import os
import numpy as np
import json 
import glob
from itertools import chain

import matplotlib.pyplot as plt
from PIL import Image
import imageio

import cv2 
from patchify import patchify, unpatchify

def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

def create_patched_image(mapName, HE=False):
    # mapName = 'CA_Dubakella.tif' # a file name 

    mapPath = os.path.join(input_filePath, mapName)
    jsonPath = os.path.join(input_filePath, mapName[0:-4]+'.json')
    
    map_img = cv2.imread(mapPath)
    
    # histogram equilibrium the image
    if HE:
        map_img = hisEqulColor(map_img)

    # this is for training, no worry to unpatchify
    patch_dims = (256,256)
    map_im_dims = map_img.shape
    patch_overlap = 32
    patch_step = patch_dims[1]-patch_overlap

    map_patchs = patchify(map_img, (*patch_dims,3), patch_step)

    # to cut all the poly legend and save as image
    # read-in json legend
    with open(jsonPath, 'r') as f:
        jsonData = json.load(f)

    LegendList = [x['label'] for x in jsonData['shapes']]

    for label_dict in jsonData['shapes']:

        point_coord = label_dict['points']
        flatten_list = list(chain.from_iterable(point_coord))

        if point_coord[0][0] >= point_coord[1][0] or point_coord[0][1] >= point_coord[1][1] or (len(flatten_list)!=4
):
            # print("Coordinate that has problem:  ", mapPath, label_dict['label'], point_coord)
            x_coord = [x[0] for x in point_coord]
            y_coord = [x[1] for x in point_coord]
            x_low, y_low, x_hi, y_hi = int(min(x_coord)), int(min(y_coord)), int(max(x_coord)), int(max(y_coord))

        else: x_low, y_low, x_hi, y_hi = [int(x) for x in flatten_list]

        legend_coor =  [(x_low, y_low), (x_hi, y_hi)]
        shift_pixel  = 4
        im_crop = map_img[y_low+shift_pixel:y_hi-shift_pixel, x_low+shift_pixel:x_hi-shift_pixel] # need to resize

        im_crop_resize = cv2.resize(im_crop, dsize=patch_dims, interpolation=cv2.INTER_CUBIC)

        writefile = mapName.split('.')[0]+'_'+label_dict['label']+'.png'
        
        if label_dict['label'].endswith('_poly'):
             imageio.imwrite(os.path.join(write_filePath, 'poly', 'legend', writefile), im_crop_resize.astype(np.ui
nt8))
        elif label_dict['label'].endswith('_line'):
             imageio.imwrite(os.path.join(write_filePath, 'line', 'legend', writefile), im_crop_resize.astype(np.ui
nt8))
        if label_dict['label'].endswith('_pt'):
            imageio.imwrite(os.path.join(write_filePath, 'point', 'legend', writefile), im_crop_resize.astype(np.uin
t8))
            
    # keep patches that only when np.sum > 100
    for Legend in LegendList:
        
        segTif = mapPath.split('.')[0]+'_'+Legend+'.tif'
        seg_img = cv2.imread(segTif)
        seg_patchs = patchify(seg_img, (*patch_dims,3), patch_step)
        
        for i in range(seg_patchs.shape[0]):
            for j in range(seg_patchs.shape[1]):
                
                filename =mapPath.split('.')[0].split('/')[-1]
                writefile = '_'.join([filename, Legend, str(i), str(j)])+'.png'
                
                if Legend.endswith('_poly') and np.sum(seg_patchs[i][j][0]) > 100:
                    write_seg = os.path.join(write_filePath, 'poly', 'seg_patches', writefile)
                    write_map = os.path.join(write_filePath, 'poly', 'map_patches', writefile)
                    imageio.imwrite(write_seg, (seg_patchs[i][j][0][:,:,0]).astype(np.uint8))
                    imageio.imwrite(write_map, (map_patchs[i][j][0]).astype(np.uint8))

                elif Legend.endswith('_line') and np.sum(seg_patchs[i][j][0]) > 5:
                    write_seg = os.path.join(write_filePath, 'line', 'seg_patches', writefile)
                    write_map = os.path.join(write_filePath, 'line', 'map_patches', writefile)
                    imageio.imwrite(write_seg, (seg_patchs[i][j][0][:,:,0]).astype(np.uint8))
                    imageio.imwrite(write_map, (map_patchs[i][j][0]).astype(np.uint8))
                    
                if Legend.endswith('_pt') and np.sum(seg_patchs[i][j][0]) > 0:
                    write_seg = os.path.join(write_filePath, 'point', 'seg_patches', writefile)
                    write_map = os.path.join(write_filePath, 'point', 'map_patches', writefile)                    
                    imageio.imwrite(write_seg, (seg_patchs[i][j][0][:,:,0]).astype(np.uint8))
                    imageio.imwrite(write_map, (map_patchs[i][j][0]).astype(np.uint8))                    

if __name__ == "__main__":
    ## define file path
    input_filePath = '/home/shared/DARPA/validation_merged'
    write_filePath = '/home/shared/DARPA/all_patched_data/validation'
#     input_filePath = '/home/shared/DARPA/training'
#     write_filePath = '/home/shared/DARPA/all_patched_data/training'    

    jsonFiles = [x.split('/')[-1] for x in glob.glob(input_filePath+'/'+'*.json')]

    for jsonFile in jsonFiles:
        print(jsonFile)
        if os.path.exists(os.path.join(write_filePath, 'finished', jsonFile)):
            continue
        else:
        with open(os.path.join(write_filePath, 'finished', jsonFile), 'w') as fp: pass
        try:
            mapName = jsonFile[0:-5]+'.tif'
            create_patched_image(mapName)
        except:
            print("A file has something wrong with its legend: ", jsonFile)
