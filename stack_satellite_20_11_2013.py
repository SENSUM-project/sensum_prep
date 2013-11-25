
'''
--------------------------------------------------------------------------
    Stack satellite - Works on a stack of Landsat 5 or 7 images
--------------------------------------------------------------------------                                
Created on May 13, 2013

Authors: Mostapha Harb - Daniele De Vecchi
         SENSUM Project
         University of Pavia - Remote Sensing Laboratory / EUCENTRE Foundation
         
In case of bugs or questions please contact: 
daniele.devecchi03@universitadipavia.it
mostapha.harb@eucentre.it

Notes: 
The input files are supposed to be landsat files with STANDARD NAMES (example "LT51800331991183XXX01_B1.TIF").
This procedure has been selected in order to facilitate the user.
--------------------------------------------------------------------------
'''

################# Parameters to set #################

##Fundamental
#sat_folder = 'C:\\workspace\\Sensum\\Izmir\\MR\\'   ##path of the folder containing satellite images
#shapefile = 'C:\\workspace\\Sensum\\Izmir\\MR\\sensum_TK_utm.shp' #path of the shapefile
sat_folder = 'C:\\workspace\\Sensum\\Bishkek\\'   ##path of the folder containing satellite images
shapefile = 'C:\\workspace\\Sensum\\Bishkek\\Bishkek.shp' #path of the shapefile
##Optional
#ref_dir = '/Users/daniele/Documents/Sensum/Izmir/Landsat5/LT51800331984164XXX04/'

################# End Parameters #################


import os,time
import shutil
import osgeo.gdal
from osgeo.gdalconst import *
import numpy as np
from library_25_11_2013 import clip,urban_development_landsat,WriteOutputImage,pca,Read_Image,Extraction,Offset_Comp

starttime=time.time()
#os.chdir(sat_folder)
dirs = os.listdir(sat_folder) #list of folders inside the satellite folder

print 'List of files and folders: ' + str(dirs)

mask_PCA_list = []
mask_BANDS_list = []
time_pca_avg = 0
time_urbandev_avg = 0
time_shift_avg = 0
time_classification_avg = 0
time_year_avg = 0

#Define the type of separator differentiating between windows and unix like systems
if os.name == 'posix':
    separator = '/'
else:
    separator = '\\'

#reference image - if not defined the first in alphabetic order is chosen

ref_dir = None
c = 0
if ref_dir is None: #if a reference image is not provided, the first in alphabetic order is chosen
    print 'Reference directory not specified - The first folder in alphabetical order will be chosen'
    while (os.path.isfile(dirs[c]) == True):### to avoid taking the files in the dirs as a reference folder so, the program will search for the first folder
        c=c+1
    else:
        reference_dir = dirs[c]
    ref_dir = sat_folder + reference_dir + separator #first directory assumed to be the reference
ref_files = os.listdir(ref_dir)
ref_list = [s for s in ref_files if ".TIF" in s and not "_city" in s]
for j in range(0,len(ref_list)):
    clip(ref_dir,ref_list[j],shapefile)
ref_files = os.listdir(ref_dir)
ref_list_city = [s for s in ref_files if "_city.TIF" in s]
rows,cols,nbands,band1_ref,geo_transform,projection = Read_Image(ref_dir+ref_list_city[0],np.uint8)
rows,cols,nbands,band2_ref,geo_transform,projection = Read_Image(ref_dir+ref_list_city[1],np.uint8)
rows,cols,nbands,band3_ref,geo_transform,projection = Read_Image(ref_dir+ref_list_city[2],np.uint8)
rows,cols,nbands,band4_ref,geo_transform,projection = Read_Image(ref_dir+ref_list_city[3],np.uint8)
rows,cols,nbands,band5_ref,geo_transform_ref,projection = Read_Image(ref_dir+ref_list_city[4],np.uint8)
if len(ref_list_city)==7: #landsat5 case
    rows,cols,nbands,band7_ref,geo_transform,projection = Read_Image(ref_dir+ref_list_city[6],np.uint8) 
else: #landsat7 case 
    rows,cols,nbands,band7_ref,geo_transform,projection = Read_Image(ref_dir+ref_list_city[7],np.uint8)

for i in range(0,len(dirs)):
    if (os.path.isfile(sat_folder+dirs[i]) == False) and ((ref_dir!=sat_folder+dirs[i]+separator)):
        img_files = os.listdir(sat_folder+dirs[i]+separator)
        image_list = [s for s in img_files if ".TIF" in s and not "_city" in s]
        for j in range(0,len(image_list)):
            clip(sat_folder+dirs[i]+separator,image_list[j],shapefile)
        
        #Read 3 bands
        img_files = os.listdir(sat_folder+dirs[i]+separator)
        image_list_city = [s for s in img_files if "_city.TIF" in s and not "aux.xml" in s]
        print image_list_city
        #Read 3 bands
        rows,cols,nbands,band1,geo_transform,projection = Read_Image(sat_folder+dirs[i]+separator+image_list_city[0],np.uint8)
        rows,cols,nbands,band2,geo_transform,projection = Read_Image(sat_folder+dirs[i]+separator+image_list_city[1],np.uint8)
        rows,cols,nbands,band3,geo_transform,projection = Read_Image(sat_folder+dirs[i]+separator+image_list_city[2],np.uint8)
        
        k1 = Extraction(band1_ref[0],band1[0])
        k2 = Extraction(band2_ref[0],band2[0])
        k3 = Extraction(band3_ref[0],band3[0])
        
        xoff,yoff = Offset_Comp(k1,k2,k3)
        
        #creation of the adjusted file
        geotransform_shift = list(geo_transform_ref)
        geotransform_shift[0] = float(geo_transform_ref[0]-geo_transform[1]*xoff)
        geotransform_shift[1] = geo_transform[1]
        geotransform_shift[5] = geo_transform[5]
        geotransform_shift[3] = float(geo_transform_ref[3]-geo_transform[5]*yoff)
        
        for k in range(0,len(image_list_city)):
            shutil.copyfile(sat_folder+dirs[i]+separator+image_list_city[k], sat_folder+dirs[i]+separator+image_list_city[k][:-4]+'_adj.TIF')
            output = osgeo.gdal.Open(sat_folder+dirs[i]+separator+image_list_city[k][:-4]+'_adj.TIF',GA_Update) #open the image
            
            output.SetGeoTransform(geotransform_shift) #set the transformation
            output.SetProjection(projection)   #set the projection
            output=None

            print 'Output: ' + image_list_city[k][:-4] + '_adj.TIF created' #output file created
        
        #clean memory
        del band1[0:len(band1)]
        del band2[0:len(band2)]
        del band3[0:len(band3)]  
        
        band6_1 = []
        band6_2 = []
        band6 = []
        #search for adj files 
        img_files = os.listdir(sat_folder+dirs[i]+separator)
        image_list_city = [s for s in img_files if "_city_adj.TIF" in s]
        rows,cols,nbands,band1,geo_transform,projection = Read_Image(sat_folder+dirs[i]+separator+image_list_city[0],np.uint8)  
        rows,cols,nbands,band2,geo_transform,projection = Read_Image(sat_folder+dirs[i]+separator+image_list_city[1],np.uint8)
        rows,cols,nbands,band3,geo_transform,projection = Read_Image(sat_folder+dirs[i]+separator+image_list_city[2],np.uint8)
        rows,cols,nbands,band4,geo_transform,projection = Read_Image(sat_folder+dirs[i]+separator+image_list_city[3],np.uint8)
        rows,cols,nbands,band5,geo_transform,projection = Read_Image(sat_folder+dirs[i]+separator+image_list_city[4],np.uint8)
        if len(image_list_city)==7: #landsat5 case
            rows,cols,nbands,band6,geo_transform,projection = Read_Image(sat_folder+dirs[i]+separator+image_list_city[5],np.uint8)
            rows,cols,nbands,band7,geo_transform,projection = Read_Image(sat_folder+dirs[i]+separator+image_list_city[6],np.uint8)
            band_list = (band1[0],band2[0],band3[0],band4[0],band5[0],band6[0],band7[0]) 
        else: #landsat7 case 
            rows,cols,nbands,band6_1,geo_transform,projection = Read_Image(sat_folder+dirs[i]+separator+image_list_city[5],np.uint8)
            rows,cols,nbands,band6_2,geo_transform,projection = Read_Image(sat_folder+dirs[i]+separator+image_list_city[6],np.uint8)
            rows,cols,nbands,band7,geo_transform,projection = Read_Image(sat_folder+dirs[i]+separator+image_list_city[7],np.uint8) 
            band_list = (band1[0],band2[0],band3[0],band4[0],band5[0],band6_1[0],band6_2[0],band7[0])
        
        SAVI,NDBI,MNDWI,Built_up = urban_development_landsat(band2[0],band3[0],band4[0],band5[0],band7[0])
        
        #clean memory
        del band1[0:len(band1)]
        del band2[0:len(band2)]
        del band3[0:len(band3)] 
        del band4[0:len(band4)]
        del band5[0:len(band5)]
        del band6[0:len(band6)] 
        del band6_1[0:len(band6_1)]
        del band6_2[0:len(band6_2)]
        del band7[0:len(band7)]
        
        mask_1 = np.greater( NDBI-SAVI, 0) 
        mask01_1 = np.choose(mask_1, (0,1))
        mask_11 = np.less(MNDWI-NDBI,0)
        mask01_11 = np.choose(mask_11, (0,1))
        mask_111 = np.greater(Built_up,0)
        mask01_111 = np.choose(mask_111, (0,1))
        mask_BANDS = mask01_1*mask01_11*mask01_111 
        mask_BANDS_list.append(mask_BANDS) 
        
        mean,first_mode,second_mode,third_mode,new_indicator = pca(band_list)
        
        mask_2 = np.less(second_mode- mean,0)        ### watermask
        mask01_2 = np.choose(mask_2,(0,1))
        mask_22 = np.less(new_indicator,2.45)
        mask01_22 = np.choose(mask_22,(0,1))
        mask_PCA = mask01_2 * mask01_22  
        mask_PCA_list.append(mask_PCA) 
        
WriteOutputImage(ref_dir+ref_list_city[0],sat_folder,'','time_evolution_pca.TIF',0,0,0,len(mask_PCA_list),mask_PCA_list) #time evolution written to file
WriteOutputImage(ref_dir+ref_list_city[0],sat_folder,'','time_evolution_BANDS.TIF',0,0,0,len(mask_BANDS_list),mask_BANDS_list) #time evolution written to file

endtime=time.time()
time_total = endtime-starttime
print '-----------------------------------------------------------------------------------------'
print 'Total time= ' + str(time_total)
#print 'Average time year= ' + str(time_year_avg/len(year_list))
#print 'Average shift compensation= ' + str(time_shift_avg/len(year_list)-1)
#print 'Average urban development= ' + str(time_urbandev_avg/len(year_list))
#print 'Average pca= ' + str(time_pca_avg/len(year_list))
#print 'Average classification= ' + str(time_classification_avg/len(year_list))
print '-----------------------------------------------------------------------------------------'