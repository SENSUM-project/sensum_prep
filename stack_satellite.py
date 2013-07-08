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

import os,time
import numpy as np
from library import clip,shift_comp,urban_development_landsat,classification,WriteOutputImage,pca

starttime=time.time()
sat_folder = '/Users/daniele/Documents/Sensum/Izmir/Landsat5/' #path of the folder containing satellite images
shapefile = '/Users/daniele/Documents/Sensum/Izmir/sensum_TK_utm.shp' #path of the shapefile
os.chdir(sat_folder)
dirs = os.listdir(sat_folder) #list of folders inside the satellite folder
thr_min = 276
thr_max = 330

mask_list = []
year_list = []
output_pca_list=[]
#reference image - if not defined the first in alphabetic order is chosen
#ref_dir = '/Users/daniele/Documents/Sensum/Izmir/Landsat5/LT51800331984164XXX04/'
ref_dir = None
if ref_dir is None: #if a reference image is not provided, the first in alphabetic order is chosen
    print 'ref_dir empty'
    reference_dir = dirs[1]
    ref_dir = sat_folder + dirs[1] + '/' #first directory assumed to be the reference
ref_files = os.listdir(ref_dir)

#reference image loop
for j in range(1,9): 
        refcity_file = [s for s in ref_files if "B"+str(j)+"_city" in s]
        if refcity_file:
            os.remove(ref_dir+refcity_file[0])
        ref_file = [s for s in ref_files if "B"+str(j) in s] #search for files like B1, B2, etc
        if ref_file:
            clip(ref_dir,ref_file[0],shapefile) #clip all the reference files
            reference = ref_file[0]
print reference            
for i in range(1,len(dirs)):
    if os.path.isdir(dirs[i]):
        year = dirs[i][9:13]
        year_list.append(year)
        year_list.sort()
print year_list
        
#main loop
for i in range(0,len(year_list)): 
    directory = [s for s in dirs if year_list[i] in s]
    if os.path.isdir(directory[0]):
        if ((sat_folder+directory[0])!= ref_dir[:-1]): #avoid to compute the feature extraction between the reference folder and itself
            band_files = os.listdir(sat_folder + directory[0] + '/')
            for j in range(1,9):
                maincity_file = [s for s in band_files if "B"+str(j)+"_city" in s]
                if maincity_file:
                    os.remove(directory[0]+'/'+maincity_file[0])
                band_file = [s for s in band_files if "B"+str(j) in s] #search for files like B1, B2 etc 
                if band_file:
                    clip(directory[0]+'/',band_file[0],shapefile) #clip all the input files
            shift_comp(sat_folder,ref_dir,directory[0]+'/',shapefile) #shift compensation function
             
        
        #urban area extraction
        classification(sat_folder,directory[0],5,'Turkey','Izmir') #unsupervised classification for urban area extraction
        if os.path.isfile(directory[0]+'/'+'uclasspy_'+directory[0]+'_5_city.TIF'):
            os.remove(directory[0]+'/'+'uclasspy_'+directory[0]+'_5_city.TIF')
        clip(directory[0]+'/','uclasspy_'+directory[0]+'_5.TIF',shapefile)
        urban_index = urban_development_landsat(sat_folder,directory[0]+'/')
        immean,mode,second_order,third_order = pca(sat_folder,directory[0]+'/') #pca for urban area extraction
        output = mode+mode+mode+mode #increase the mode values
        output_pca_list.append(output)
        WriteOutputImage(reference_dir+'/'+reference[:-4]+'_city.TIF',sat_folder,'',directory[0]+'/urban_area_pca.TIF',0,0,0,1,output_pca_list)
        output_pca_list.remove(output)
        mask1 = np.greater(output-thr_min, 0) 
        mask2 = np.less(output-thr_max,0)
        mask = mask1*mask2 #combination of the 2 masks

        #compute the time evolution
        if i == 0:
            mask_ref = mask
            evolution = np.choose(mask_ref, (0,mask*int(year_list[0])))
        else:
            pos_values = np.greater(mask-mask_ref,0) #select just pixels changing from non-urban to urban (not viceversa)
            mask_ref = mask_ref + np.choose(pos_values, (0,mask)) #new mask reference
            evolution = evolution + np.choose(pos_values, (0,mask*int(year_list[i]))) #every level of temporal change has a different value assigned (10, 20, ...)
        mask_list.append(mask)    
evolution = np.choose(mask, (0,evolution))
mask_list.append(evolution)
WriteOutputImage(reference_dir+'/'+reference[:-4]+'_city.TIF',sat_folder,'','time_evolution_pca.TIF',0,0,0,len(year_list)+1,mask_list) #time evolution written to file
    
endtime=time.time()
print 'Total time= ' + str(endtime-starttime)