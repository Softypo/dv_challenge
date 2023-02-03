# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:48:00 2023

@author: garetm
"""
import numpy as np
from PIL import Image
import cv2 as cv
from matplotlib import pyplot as plt
import open3d as o3d

# read in the array and shapen to volume
rootpath = "F:/Dropbox (DarkVision)/Analysis/Projects/EOG/DVT794_2023-02_FF_Codigo-102H/Renders/Vol/"
filename = rootpath+"volume_export_768x768x1280_uint8_t.raw"
outname = rootpath+"processed_volume_export_768x768x1280_uint8_t.raw"

array = np.fromfile(filename, dtype= 'uint8')
vol = array.reshape(1280,768,768)

for i in range(vol.shape[0]):
    slice = vol[i,:,:]
    result = cv.fastNlMeansDenoising(slice,h=20,templateWindowSize=10,searchWindowSize=21)
    edge = cv.Canny(result, 100, 200)

    vol[i,:,:] = edge
    
vol.astype('int8').tofile(outname)