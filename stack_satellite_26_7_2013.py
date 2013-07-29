
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
######tested on the image stack 1984-1987-1995-2000-2003   28.17 sec on my windows 

import os,time
import osgeo.gdal
from osgeo.gdalconst import *
import numpy as np
from library_26_7_2013 import clip,shift_comp,urban_development_landsat,classification,WriteOutputImage,pca

starttime=time.time()
sat_folder = '/Users/daniele/Documents/Sensum/Izmir/Landsat5_2/'   ##path of the folder containing satellite images
shapefile = '/Users/daniele/Documents/Sensum/Izmir/sensum_tk_utm.shp' #path of the shapefile
os.chdir(sat_folder)
dirs = os.listdir(sat_folder) #list of folders inside the satellite folder
print 'List of files and folders: ' + str(dirs)

#MNDWI_list = []
#NDBI_list = []
#SAVI_list = []
#Built_up_list = []
year_list = []

mask_PCA_list = []
mask_BANDS_list = []
time_pca_avg = 0
time_urbandev_avg = 0
time_shift_avg = 0
time_classification_avg = 0
time_year_avg = 0
#Indices_list =[]

#pca_immean_list = []
#pca_mode_list = []
#pca_second_order_list = []
#pca_third_order_list = []

#Define the type of separator differentiating between windows and unix like systems
if os.name == 'posix':
    separator = '/'
else:
    separator = '\\'

#reference image - if not defined the first in alphabetic order is chosen
#ref_dir = '/Users/daniele/Documents/Sensum/Izmir/Landsat5/LT51800331984164XXX04/'
ref_dir = None
c = 0
if ref_dir is None: #if a reference image is not provided, the first in alphabetic order is chosen
    print 'Reference directory not specified - The first folder in alphabetical order will be chosen'
    while (os.path.isfile(dirs[c]) == True):### to avoid taking the files in the dirs as a reference folder so, the program will search for the first folder
        c=c+1
    else:
        reference_dir = dirs[c]
    ref_dir = sat_folder + reference_dir + separator #first directory assumed to be the reference
    #print ('as you did not gave a reference folder the ref_dir will be:', ref_dir)
ref_files = os.listdir(ref_dir)
#print ('the files in the reference directory are:', ref_files)


#reference image loop
for j in range(1,9): #1 to 8
        #print j
        refcity_file = [s for s in ref_files if "B"+str(j)+"_city" in s]
        #print ('the refcity_file', refcity_file)
        if refcity_file:
            os.remove(ref_dir+refcity_file[0])
        ref_file = [s for s in ref_files if "B"+str(j) in s] #search for files like B1, B2, etc
        #print ('the ref_file', ref_file)

        if ref_file:
            clip(ref_dir,ref_file[0],shapefile) #clip all the reference files
        if (j==1):    ### this might not be the clipped image we have to fix it
            reference = ref_file[0]####refcity_file[0] might suits better
            #print ('the reference is', reference)
    
for i in range(0,len(dirs)):
    #print i
    if os.path.isdir(dirs[i]):
        year = dirs[i][9:13]
        year_list.append(year)
        year_list.sort()
print 'The year_list includes: ' + str(year_list)

#main loop
for i in range(0,len(year_list)): 
    classification_list=[]
    starttime_year = time.time()
    print 'Year: ' + str(year_list[i])
    directory = [s for s in dirs if year_list[i] in s]
    #print 'The directory[0]  is: ' + str(directory)  # 2 images from the same year appeared, the problem is that it will take one single image from each year as the loop is on the directory[0] always, Moreover, two images from the same year in the year_list lead to repeating the same calculation on the same same image

    if os.path.isdir(directory[0]):   
        #print ('now the directory[0] is:', directory[0])
        
        if ((sat_folder+directory[0])!= ref_dir[:-1]): #avoid to compute the feature extraction between the reference folder and itself
            band_files = os.listdir(sat_folder + directory[0] + separator)
            #print ('the band_files are:', band_files)
            
            for j in range(1,9):## looping through bands
                maincity_file = [s for s in band_files if "B"+str(j)+"_city" in s]
                #print ('maincity_file are:',maincity_file)
                if maincity_file:
                    os.remove(directory[0]+separator+maincity_file[0])
                band_file = [s for s in band_files if "B"+str(j) in s] #search for files like B1, B2 etc 
                #print ('band_file are:',band_file)
                if band_file:
                    clip(directory[0]+separator,band_file[0],shapefile) #clip all the input files
            starttime_shift=time.time()
            shift_comp(sat_folder,ref_dir,directory[0]+separator,shapefile) #shift compensation function  
            endtime_shift = time.time()
            time_shift = endtime_shift - starttime_shift
            time_shift_avg = time_shift_avg + time_shift
            print '--- Shift Compensation - Total time= ' + str(time_shift) + '\n'
        
        #urban area extraction
        starttime_classification = time.time()
        classification(sat_folder,directory[0],5,'Turkey','Izmir',4,5) #unsupervised classification for urban area extraction  ### but is it going to use the full set of images in that folder, (city,adj city and the full image)
        endtime_classification = time.time()
        time_classification = endtime_classification - starttime_classification
        time_classification_avg = time_classification_avg + time_classification
        print '--- Classification - Total time= ' + str(time_classification)
        
        #if os.path.isfile(directory[0]+'/'+'uclasspy_'+directory[0]+'_5_city.TIF'):
            #os.remove(directory[0]+'/'+'uclasspy_'+directory[0]+'_5_city.TIF')
        #clip(directory[0]+'/','uclasspy_'+directory[0]+'_5.TIF',shapefile)
        
        starttime_urbandev=time.time()
        SAVI, NDBI, MNDWI, Built_up = urban_development_landsat(sat_folder,directory[0]+separator)
        endtime_urbandev = time.time()
        time_urbandev = endtime_urbandev - starttime_urbandev
        time_urbandev_avg = time_urbandev_avg + time_urbandev
        print '--- Urban development - Total time= ' + str(time_urbandev) + '\n'
        
        starttime_pca=time.time()
        mean,first_mode,second_mode,third_mode, new_indicator = pca(sat_folder,directory[0]+separator) #pca for urban area extraction#### big mistake it will consider lots of unrelated files
        endtime_pca = time.time()
        time_pca = endtime_pca - starttime_pca
        time_pca_avg = time_pca_avg + time_pca
        print '--- PCA - Total time= ' + str(time_pca) + '\n'
        
        #output_pca_list.append(third_order)  ##what is the point of combining them, why the script cannat work when I aviod this step, and y should I append in the first place
        
        #WriteOutputImage(reference_dir+'/'+reference[:-4]+'_city.TIF',sat_folder,'',directory[0]+'/pca_mode_list.TIF',0,0,0,1,pca_mode_list)

        mask_1 = np.greater( NDBI -SAVI, 0) 
        mask01_1 = np.choose(mask_1, (0,1))
       
        mask_11 = np.less(MNDWI-NDBI,0)
        mask01_11 = np.choose(mask_11, (0,1))
        
        mask_111 = np.greater(Built_up,0)
        mask01_111 = np.choose(mask_111, (0,1))
        
        mask_BANDS = mask01_1*mask01_11*mask01_111 
        mask_BANDS_list.append( mask_BANDS)    
    
        #new_indicator = ((4*mode)+immean)     /(immean + mode+sec_order+third_order+0.0001)### isolate the built-up and the water
        mask_2 = np.less(  second_mode- mean  ,0)        ### watermask
        mask01_2 = np.choose(mask_2, (0,1))

        mask_22 = np.less(  new_indicator   ,2.45)
        mask01_22 = np.choose(mask_22, (0,1))
        
        mask_PCA = mask01_2 * mask01_22  
        
        mask_PCA_list.append( mask_PCA) 
        endtime_year = time.time()
        time_year = endtime_year - starttime_year
        time_year_avg = time_year + time_year_avg
#mask_list.append(evolution)
WriteOutputImage(reference_dir+separator+reference[:-4]+'_city.TIF',sat_folder,'','time_evolution_pca.TIF',0,0,0,len(year_list),mask_PCA_list) #time evolution written to file
WriteOutputImage(reference_dir+separator+reference[:-4]+'_city.TIF',sat_folder,'','time_evolution_BANDS.TIF',0,0,0,len(year_list),mask_BANDS_list) #time evolution written to file

endtime=time.time()
time_total = endtime-starttime
print '-----------------------------------------------------------------------------------------'
print 'Total time= ' + str(time_total)
print 'Average time year= ' + str(time_year_avg/len(year_list))
print 'Average shift compensation= ' + str(time_shift_avg/len(year_list)-1)
print 'Average urban development= ' + str(time_urbandev_avg/len(year_list))
print 'Average pca= ' + str(time_pca_avg/len(year_list))
print 'Average classification= ' + str(time_classification_avg/len(year_list))
print '-----------------------------------------------------------------------------------------'