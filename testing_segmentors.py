'''
--------------------------------------------------------------------------
                                testing code
--------------------------------------------------------------------------                                
Created on Oct 4, 2013

Authors: Mostapha Harb - Daniele De Vecchi
         SENSUM Project
         University of Pavia - Remote Sensing Laboratory / EUCENTRE Foundation
         
In case of bugs or questions please contact: 
daniele.devecchi03@universitadipavia.it
mostapha.harb@eucentre.it
--------------------------------------------------------------------------

The file contains 6  wide spread segmentation for the different tasks in image analysis: graphbase, KNN, quick shift,watershed, Baatz, and Region growing 
to run the code for a normal use, the user has to define 4 main inputs 3 folders + input image
the four inputs needed to run the code 
1. temp_folder = temporary folder for the direct raw data product of baatz and regiongrowing segmentors 
2. exe_folder = a small folder containing the excutive files of baatz and regiongrowing segmentors.
3. Input_image is a pancromatic band 
4. Folder_output
'''
import numpy as np
import cv2,os, time
import gdal,sys
from gdalconst import *
from skimage.segmentation import felzenszwalb, slic, quickshift
from Library_04_10_2013 import SLIC, FELZENSZWALB,QUICKSHIFT,WATERSHED,Pixel2world,BAATZ,REGION_GROWING,WriteOutputImage

temp_Folder = 'C:\Users\mostapha\Desktop\intimgexe2\\tmpfolder'
exe_folder = 'C:\Users\mostapha\Documents\Sensum\Izmir\exe'
Input_Image = 'C:\\Users\\mostapha\\Desktop\\intimgexe2\\Izmir.tif'



#calling the seg. functions
segments_SLIC = SLIC( Input_Image,0,0,0)  
segments_fz = FELZENSZWALB(Input_Image,0,0,0) 
segments_quick = QUICKSHIFT(Input_Image,0,0,0) 
segments_watershed = WATERSHED(Input_Image)
segments_baatz = BAATZ(Input_Image ,temp_Folder, exe_folder,0,0,0,0)     
segments_regiongrowing = REGION_GROWING(Input_Image ,temp_Folder, exe_folder,0,0,0,0)     


# attach the seg masks  to a list of bands
ax_list = []
ax_list.append(segments_SLIC)
ax_list.append(segments_fz)
ax_list.append(segments_quick)
ax_list.append(segments_watershed)
ax_list.append(segments_baatz)
ax_list.append(segments_regiongrowing)


#output folder
Folder_output = 'C://Users//mostapha//Desktop//images//'

#delete the result if existed 
if os.path.exists(Folder_output + 'Segmentation.TIF'):
        os.remove( Folder_output + 'Segmentation.TIF')

#write the results as tif file
WriteOutputImage('Izmir.tif',Folder_output,'','Segmentation.TIF',0,0,0,6,ax_list)