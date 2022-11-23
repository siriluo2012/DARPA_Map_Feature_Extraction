import os
import cv2
import json
from glob import glob
import csv

from itertools import chain

import numpy as np

from metrics import feature_f_score

if __name__ == "__main__":

    Pred_Folder = '/home/shared/DARPA/validation_shirui_HE/Inference'
    True_Folder = '/home/shared/DARPA/validation_rasters'
    
    write_csv_file = '/home/shirui/DARPA/UNet/validation_shirui_HE_performance_poly.csv'

    True_FileList = os.listdir(True_Folder)

    polyScore = {}

    for polyfileName in True_FileList:
        print('polyfileName:  ', polyfileName)
        
        if '_poly.tif' in polyfileName:
            trueSegPath = os.path.join(True_Folder, polyfileName)
            predicted_path = os.path.join(Pred_Folder, polyfileName)

            mapName = '_'.join(polyfileName.split('_')[0:-2])+'.tif'
            jsonName = '_'.join(polyfileName.split('_')[0:-2])+'.json'

            legend = '_'.join(polyfileName.split('_')[-2:]).split('.')[0]

            mapPath = os.path.join('/home/shared/DARPA/validation', mapName)
            jsonPath = os.path.join('/home/shared/DARPA/validation', jsonName)

            try:
                with open(jsonPath, 'r') as f:
                    jsonData = json.load(f)
            
                point_coord = []
                for label_dict in jsonData['shapes']:
                    if label_dict['label'] == legend:
                        point_coord = label_dict['points']

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
                            x_low, y_low, x_hi, y_hi = int(min(x_coord)), int(min(y_coord)), int(max(x_coord)), int(
max(y_coord))
                            # print("Point Coordinates number is not 4: ", filename, legend)
                        else: x_low, y_low, x_hi, y_hi = [int(x) for x in flatten_list]
                        legend_coor =  [(x_low, y_low), (x_hi, y_hi)]

                if len(point_coord) == 0:
                    print ('poly not found  ', polyfileName)
                    continue

                truSeg_im = cv2.imread(trueSegPath)
                
                predicted_seg_im = cv2.imread(predicted_path)

                map_im = cv2.imread(mapPath)

                precision, recall, f_score = feature_f_score(map_im, predicted_seg_im[:,:,0], truSeg_im[:,:,0], feat
ure_type ='poly',
                                                     legend_coor=legend_coor, min_valid_range=.25,
                                                     difficult_weight=.7, set_false_as='hard')

                polyScore[polyfileName] = (precision, recall, f_score)
            
            except:
                print('poly not found in the prediction folder:   ', polyfileName)

    with open(write_csv_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in polyScore.items():
            writer.writerow([key, value])
