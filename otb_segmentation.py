'''
--------------------------------------------------------------------------
            testing code for orfeo toolbox segmentation algorithms
--------------------------------------------------------------------------                                
Created on Oct 18, 2013

Authors: Marc Wieland
         SENSUM Project
         GFZ German Research Centre For Geosciences - Centre for Early Warning
         
In case of bugs or questions please contact: 
mwieland@gfz-potsdam.de
--------------------------------------------------------------------------
Description: 4 segmentation algorithms implemented in the orfeo toolbox are tested.
             Default parameters are set in this example code. 
             To use individual parameters uncomment the ".SetParameterString" lines.
Input: multi-spectral satellite image
Output: image segmentation in vector format
Requirements: Orfeo Toolbox 3.18.1
Installation: http://www.orfeo-toolbox.org/SoftwareGuide/SoftwareGuidech2.html (otb-bin, otb-python, otb-wrapping)

'''

##############################################
#Segmentation tests for Orfeo Toolbox Library#
##############################################
import time
import otbApplication

starttime=time.time()
in_img = "/home/marc/eclipse_data/Test/landsat8.tif"
out_folder = "/home/marc/eclipse_data/Test/"

#Watershed algorithm
Segmentation = otbApplication.Registry.CreateApplication("Segmentation") 
Segmentation.SetParameterString("in", in_img)  
Segmentation.SetParameterString("mode","vector") 
Segmentation.SetParameterString("mode.vector.out", out_folder + "watershed_default.shp") 
Segmentation.SetParameterString("filter","watershed") 
#Segmentation.SetParameterString("filter.watershed.threshold","5")
#Segmentation.SetParameterString("filter.watershed.level","5")
Segmentation.ExecuteAndWriteOutput()

#Meanshift algorithm
Segmentation = otbApplication.Registry.CreateApplication("Segmentation") 
Segmentation.SetParameterString("in", in_img)  
Segmentation.SetParameterString("mode","vector") 
Segmentation.SetParameterString("mode.vector.out", out_folder + "meanshift_default.shp") 
Segmentation.SetParameterString("filter","meanshift") 
#Segmentation.SetParameterString("filter.meanshift.spatialr","5")
#Segmentation.SetParameterString("filter.meanshift.ranger","5")
#Segmentation.SetParameterString("filter.meanshift.thres","5")
#Segmentation.SetParameterString("filter.meanshift.maxiter","5")
#Segmentation.SetParameterString("filter.meanshift.minsize","50")
Segmentation.ExecuteAndWriteOutput()

#Edison mean-shift algorithm
Segmentation = otbApplication.Registry.CreateApplication("Segmentation") 
Segmentation.SetParameterString("in", in_img)  
Segmentation.SetParameterString("mode","vector") 
Segmentation.SetParameterString("mode.vector.out", out_folder + "edison_default.shp") 
Segmentation.SetParameterString("filter","edison") 
#Segmentation.SetParameterString("filter.edison.spatialr","5")
#Segmentation.SetParameterString("filter.edison.ranger","5")
#Segmentation.SetParameterString("filter.edison.minsize","50")
#Segmentation.SetParameterString("filter.edison.scale","5")
Segmentation.ExecuteAndWriteOutput()

#Morphological profiles algorithm
Segmentation = otbApplication.Registry.CreateApplication("Segmentation") 
Segmentation.SetParameterString("in", in_img)  
Segmentation.SetParameterString("mode","vector") 
Segmentation.SetParameterString("mode.vector.out", out_folder + "mprofiles_default.shp") 
Segmentation.SetParameterString("filter","mprofiles")
#Segmentation.SetParameterString("filter.mprofiles.size","5")
#Segmentation.SetParameterString("filter.mprofiles.start","5")
#Segmentation.SetParameterString("filter.mprofiles.step","5")
#Segmentation.SetParameterString("filter.mprofiles.sigma","5")
Segmentation.ExecuteAndWriteOutput()

endtime=time.time()
time_total = endtime-starttime
print '-------------------------------------'
print 'Computation time= ' + str(time_total)
print '-------------------------------------'
