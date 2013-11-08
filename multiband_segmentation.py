'''
Created on Nov 6, 2013

@author: daniele
'''

import os, sys
import string
import osgeo.gdal,gdal
from osgeo.gdalconst import *
from gdalconst import *
import numpy as np
from library_08_11_2013 import WriteOutputImage,Read_Image,SLIC,FELZENSZWALB,QUICKSHIFT,WATERSHED,Pixel2world,BAATZ,REGION_GROWING
import time

### Parameters #########################################################################################
input_image = 'C:\workspace\Sensum\Izmir\Applications\multiband_segmentation\\clipped_merged_new.tif'
Folder_output = 'C:\\workspace\\Sensum\\Izmir\\Applications\\multiband_segmentation\\'
temp_Folder = 'C:\workspace\Sensum\Izmir\Applications\\tmpfolder'
exe_folder = 'C:\workspace\Sensum\Izmir\Applications\seg_exec'
########################################################################################################

osgeo.gdal.AllRegister()
band_list = []
#read input image and all the parameters
rows,cols,nbands,band_list,geo_transform,projection = Read_Image(input_image,np.uint16)

img = np.dstack((band_list[2],band_list[1],band_list[0])) #stack RGB, segmentation algorithms are limited to 3 bands
print img.shape
#rows,cols,nbands,band_list_ws,geo_transform,projection = Read_Image(input_image,np.uint8)
#img_ws = np.dstack((band_list_ws[2],band_list_ws[1],band_list_ws[0]))


#SLIC segmentation, input as unsigned integer 16
#SLIC( Input_Image,ratio/compactness, n_segments, sigma, multiband_option) #0 for default values
print '--- SLIC'
start_time = time.time()
segments_slic = SLIC(img,0,300,0,True) #SLIC segmentation is like a k-mean classification, True in case of multichannel option
output_list = []
output_list.append(segments_slic)    
WriteOutputImage(input_image,Folder_output,'','Segmentation_slic.TIF',0,0,0,len(output_list),output_list)
end_time = time.time()
print '--- Time: ' + str(end_time-start_time)

#FELZENSZWALB segmentation
#FELZENSZWALB(Input_Image, scale, sigma, min_size)
print '--- FELZENSZWALB'
start_time = time.time()
segments_fz = FELZENSZWALB(img,0,0,0)
output_list = []
output_list.append(segments_fz)    
WriteOutputImage(input_image,Folder_output,'','Segmentation_fz.TIF',0,0,GDT_Float32,len(output_list),output_list)
end_time = time.time()
print '--- Time: ' + str(end_time-start_time)

#QUICKSHIFT segmentation
#QUICKSHIFT(Input_Image,kernel_size, max_distance, ratio)
print '--- QUICKSHIFT'
start_time = time.time()
segments_quick = QUICKSHIFT(img,0,0,0)
output_list = []
output_list.append(segments_quick)
WriteOutputImage(input_image,Folder_output,'','Segmentation_quick.TIF',0,0,0,len(output_list),output_list)
end_time = time.time()
print '--- Time: ' + str(end_time-start_time)
'''
'''
#BAATZ segmentation
#BAATZ(Input,Folder,exe,euc_threshold,compactness,baatz_color,scale,multiband_option)
print '--- BAATZ'
start_time = time.time()
segments_baatz = BAATZ(input_image ,temp_Folder, exe_folder,0,0,0,0,True)
output_list = []
output_list.append(segments_baatz)
WriteOutputImage(input_image,Folder_output,'','Segmentation_baatz.TIF',0,0,0,len(output_list),output_list)
end_time = time.time()
print '--- Time: ' + str(end_time-start_time)

#REGION GROWING
#REGION_GROWING(Input,Folder,exe,euc_threshold,compactness,baatz_color,scale,multiband_option)
print '--- REGION GROWING'
start_time = time.time()
segments_regiongrowing = REGION_GROWING(input_image,temp_Folder, exe_folder,0,0,0,0,True)
output_list = []
output_list.append(segments_regiongrowing)
WriteOutputImage(input_image,Folder_output,'','Segmentation_regiongrowing.TIF',0,0,0,len(output_list),output_list)
end_time = time.time()
print '--- Time: ' + str(end_time-start_time)

'''
#WATERSHED
#WATERSHED(Input_Image)
print '--- WATERSHED'
start_time = time.time()
segments_watershed = WATERSHED(img_ws)
output_list = []
output_list.append(segments_watershed)
WriteOutputImage(input_file,Folder_output,'','Segmentation_watershed.TIF',0,0,0,len(output_list),output_list)
end_time = time.time()
print '--- Time: ' + str(end_time-start_time)
'''