'''
--------------------------------------------------------------------------
                                Library
--------------------------------------------------------------------------                                
Created on May 13, 2013

Authors: Mostapha Harb - Daniele De Vecchi
         SENSUM Project
         University of Pavia - Remote Sensing Laboratory / EUCENTRE Foundation
         
In case of bugs or questions please contact: 
daniele.devecchi03@universitadipavia.it
mostapha.harb@eucentre.it
--------------------------------------------------------------------------
'''

import os, sys
import string
import osgeo.gdal
from osgeo.gdalconst import *
import cv2
from cv2 import cv
import scipy as sp
import numpy as np
from numpy import unravel_index
import osgeo.osr
import osgeo.ogr
from collections import defaultdict
import grass.script.setup as gsetup
import grass.script as grass

if os.name == 'posix':
    separator = '/'
else:
    separator = '\\'

def shp_conversion(path,name_input,name_output,epsg):
    
    '''
    ###################################################################################################################
    Conversion from KML to SHP file using EPSG value as projection - Used to convert the drawn polygon around the city in GE to a SHP
    
     Input:
     - path: contains the folder path of the original file; the output file is going to be created into the same folder
     - name_input: name of the kml input file
     - name_output: name of shp output file
     - epsg: epsg projection code
     
     Output:
     SHP file is saved into the same folder of the original KML file
    ###################################################################################################################
    '''
    
    #conversion from kml to shapefile
    os.system("ogr2ogr -f 'ESRI Shapefile' " + path + name_output + ' ' + path + name_input)
    # set the working directory
    os.chdir(path)
    # get the shapefile driver
    driver = osgeo.ogr.GetDriverByName('ESRI Shapefile')
    # create the input SpatialReference, 4326 is the default one
    inSpatialRef = osgeo.osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(4326)
    # create the output SpatialReference
    outSpatialRef = osgeo.osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(epsg)
    # create the CoordinateTransformation
    coordTrans = osgeo.osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # open the input data source and get the layer
    inDS = driver.Open(name_output, 0)
    if inDS is None:
        print 'Could not open file'
        sys.exit(1)
    inLayer = inDS.GetLayer()
    # create a new data source and layer
    if os.path.exists(name_output):
        driver.DeleteDataSource(name_output)
    outDS = driver.CreateDataSource(name_output)
    if outDS is None:
        print 'Could not create file'
        sys.exit(1)
    outLayer = outDS.CreateLayer('City', geom_type=osgeo.ogr.wkbPoint)
    # get the FieldDefn for the name field
    feature = inLayer.GetFeature(0)
    fieldDefn = feature.GetFieldDefnRef('name')
    # add the field to the output shapefile
    outLayer.CreateField(fieldDefn)
    # get the FeatureDefn for the output shapefile
    featureDefn = outLayer.GetLayerDefn()
    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = osgeo.ogr.Feature(featureDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        outFeature.SetField('name', inFeature.GetField('name'))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # destroy the features and get the next input feature
        outFeature.Destroy
        inFeature.Destroy
        inFeature = inLayer.GetNextFeature()
    # close the shapefiles
    inDS.Destroy()
    outDS.Destroy()
    # create the *.prj file
    outSpatialRef.MorphToESRI()
    file = open(name_output[:-4]+'.prj', 'w')
    file.write(outSpatialRef.ExportToWkt())
    file.close()
    print 'Conversion finished!'
    

def world2Pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate 
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / xDist)
    return (pixel, line) 


def clip(path,name,shapefile):
    
    '''
    ###################################################################################################################
    Clip an image using a shapefile
    
    Input:
     - path: path to the images location in your pc
     - name: name of the input file
     - shapefile: path of the shapefile to be used
     
    Output:
    New file is saved into the same folder as "original_name_city.TIF"
    ###################################################################################################################  
    '''
    
    #os.system('gdalwarp -q -cutline ' + shapefile + ' -crop_to_cutline -of GTiff ' + path + name +' '+ path + name[:-4] + '_city.TIF')
    #new command working on fwtools, used just / for every file
    #print 'Clipped file: ' + name[:-4] + '_city.TIF'
    x_list = []
    y_list = []
    # get the shapefile driver
    driver = osgeo.ogr.GetDriverByName('ESRI Shapefile')
    
    # open the data source
    datasource = driver.Open(shapefile, 0)
    if datasource is None:
        print 'Could not open shapefile'
        sys.exit(1)

    layer = datasource.GetLayer() #get the shapefile layer
    inb = osgeo.gdal.Open(path+name, GA_ReadOnly)
    if inb is None:
        print 'Could not open'
        sys.exit(1)
        
    geoMatrix = inb.GetGeoTransform()
    driver = inb.GetDriver()
    cols = inb.RasterXSize
    rows = inb.RasterYSize    
    inband = inb.GetRasterBand(1)
    data = inband.ReadAsArray()
    # loop through the features in the layer
    feature = layer.GetNextFeature()
    while feature:
        # get the x,y coordinates for the point
        geom = feature.GetGeometryRef()
        #print geom
        ring = geom.GetGeometryRef(0)
        n_vertex = ring.GetPointCount()
        for i in range(0,n_vertex-1):
            lon,lat,z = ring.GetPoint(i)
            x_matrix,y_matrix = world2Pixel(inb.GetGeoTransform(),lon,lat)
            x_list.append(x_matrix)
            y_list.append(y_matrix)
        # destroy the feature and get a new one
        feature.Destroy()
        feature = layer.GetNextFeature()
    
    x_list.sort()
    x_min = x_list[0]
    y_list.sort()
    y_min = y_list[0]
    x_list.sort(None, None, True)
    x_max = x_list[0]
    y_list.sort(None, None, True)
    y_max = y_list[0]
    
    #compute the new starting coordinates
    lon_min = float(x_min*30.0+geoMatrix[0]) 
    lat_min = float(geoMatrix[3]-y_min*30.0)
    #print lon_min
    #print lat_min
    
    geotransform = [lon_min,30.0,0.0,lat_min,0.0,-30.0]
    #print x_min,x_max
    #print y_min,y_max
    out=data[int(y_min):int(y_max),int(x_min):int(x_max)]
    cols_out = x_max-x_min
    rows_out = y_max-y_min
    output=driver.Create(path+name[:-4]+'_city.TIF',cols_out,rows_out,1)
    inprj=inb.GetProjection()
    #WriteOutputImage('/Users/daniele/Documents/Sensum/Izmir/Landsat5/LT51800331984164XXX04/LT51800331984164XXX04_B1.TIF','','','/Users/daniele/Documents/Sensum/Izmir/Landsat5/LT51800331984164XXX04/test.TIF',cols_out,rows_out,GDT_Float32,1,list_out)
    outband=output.GetRasterBand(1)
    outband.WriteArray(out,0,0) #write to output image
    output.SetGeoTransform(geotransform) #set the transformation
    output.SetProjection(inprj)
    # close the data source and text file
    datasource.Destroy()
    #print 'Clipped file: ' + name[:-4] + '_city.TIF'
    

def merge(path,output,name):
    
    '''
    ###################################################################################################################
    Merge different band-related files into a multi-band file
    
    Input:
     - path: folder path of the original files
     - output: name of the output file
     - name: input files to be merged
     
    Output:
    New file is created in the same folder
    ###################################################################################################################
    '''
    
    #function to extract single file names
    instring = name.split()
    num = len(instring)
    #os command to merge files into separate bands
    com = 'gdal_merge.py -separate -of GTiff -o ' + path + output
    for i in range(0,num):
        com = com + path + instring[i] + ' '
    os.system(com)
    print 'Output file: ' + output
    
    
def split(path,name,option):
    
    '''
    ###################################################################################################################
    Split the multi-band input image into different band-related files
    
    Input:
     - path: folder path of the image files
     - name: name of the input file to be split
     - option: specifies the band to extract, if equal to 0 all the bands are going to be extracted
    
    Output:
    Output file name contains the number of the extracted band - example: B1.TIF for band number 1
    ###################################################################################################################
    '''
    
    osgeo.gdal.AllRegister()
    #open the input file
    inputimg = osgeo.gdal.Open(path+name,GA_ReadOnly)
    if inputimg is None:
        print 'Could not open ' + name
        sys.exit(1)
    #extraction of columns, rows and bands from the input image    
    cols=inputimg.RasterXSize
    rows=inputimg.RasterYSize
    bands=inputimg.RasterCount
    
    if (option!=0):
        #extraction of just one band to a file
        inband=inputimg.GetRasterBand(option)
        driver=inputimg.GetDriver()
        output=driver.Create(path+'B'+str(option)+'.TIF',cols,rows,1)
        outband=output.GetRasterBand(1)
        data = inband.ReadAsArray()
        outband.WriteArray(data,0,0)
        print 'Output file: B' + str(option) + '.TIF'
    else:
        #extraction of all the bands to different files
        for i in range(1,bands+1):
            inband=inputimg.GetRasterBand(i)
    
            driver=inputimg.GetDriver()
            output=driver.Create(path+'B'+str(i)+'.TIF',cols,rows,1)
            outband=output.GetRasterBand(1)
    
            data = inband.ReadAsArray()
            outband.WriteArray(data,0,0)
            print 'Output file: B' + str(i) + '.TIF'
    inputimg=None   
    

def Extraction(image1,image2):
    
    '''
    ###################################################################################################################
    Feature Extraction using the SURF algorithm
    
    Input:
     - image1: path to the reference image - each following image is going to be matched with this reference
     - image2: path to the image to be corrected
    
    Output:
    Returns a matrix with x,y coordinates of matching points
    ###################################################################################################################
    '''
    
    #print 'Reference: ' + str(image1)
    #print 'Target: ' + str(image2)
    img1 = cv2.imread(image1, cv2.CV_LOAD_IMAGE_GRAYSCALE) #read the reference image
    if img1 is None: 
        print 'File not found ' + image1
        sys.exit(1)
    img2 = cv2.imread(image2, cv2.CV_LOAD_IMAGE_GRAYSCALE) #read the image to correct
    if img2 is None:
        print 'File not found ' + image2
        sys.exit(1)
    
    detector = cv2.FeatureDetector_create("SURF") 
    descriptor = cv2.DescriptorExtractor_create("BRIEF")
    matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    
    # detect keypoints
    kp1 = detector.detect(img1)
    kp2 = detector.detect(img2)
    
    # descriptors
    k1, d1 = descriptor.compute(img1, kp1)
    k2, d2 = descriptor.compute(img2, kp2)
    
    # match the keypoints
    matches = matcher.match(d1, d2)
    
    # visualize the matches
    dist = [m.distance for m in matches] #extract the distances
    a=sorted(dist) #order the distances
    fildist=np.zeros(1) #use 1 in order to select the most reliable matches
    
    for i in range(0,1):
        fildist[i]=a[i]
    thres_dist = max(fildist)
    # keep only the reasonable matches
    sel_matches = [m for m in matches if m.distance <= thres_dist] 
    
    i=0
    points=np.zeros(shape=(len(sel_matches),4))
    for m in sel_matches:
        #matrix containing coordinates of the matching points
        points[i][:]= [int(k1[m.queryIdx].pt[0]),int(k1[m.queryIdx].pt[1]),int(k2[m.trainIdx].pt[0]),int(k2[m.trainIdx].pt[1])]
        i=i+1
    #print 'Feature Extraction - Done'
    return points 


def shift_comp(path,folder1,folder2,shapefile):
    
    '''
    ###################################################################################################################
    Calculation of shift using 3 different bands for each acquisition; the feature extraction algorithm is used to extract features
    
    Input:
     - path: path to the files
     - folder1: name of the folder containing the first acquisition (reference images)
     - folder2: name of the folder containing the second acquisition
     - shapefile: path to the shapefile used to clip the image
     
    Output: 
    Output file is in the same folder as the original file and is called "original_name_adj.TIF"
    
    Notes: 
    The input files are supposed to be landsat files with STANDARD NAMES (example "LT51800331991183XXX01_B1.TIF") modified by OUR CLIP ALGORITHM (example "LT51800331991183XXX01_B1_city.TIF").
    This procedure has been selected in order to facilitate the user.
    ###################################################################################################################
    '''
    
    osgeo.gdal.AllRegister()
    os.chdir(path)
    ref_files = os.listdir(folder1)
    #print ref_files
    b1_ref = [s for s in ref_files if "B1_city" in s]
    b2_ref = [s for s in ref_files if "B2_city" in s]
    b3_ref = [s for s in ref_files if "B3_city" in s]
    
    band_files = os.listdir(path + folder2) #list files inside the directory
    #print band_files
    b1_file = [s for s in band_files if "B1_city" in s]
    b2_file = [s for s in band_files if "B2_city" in s]
    b3_file = [s for s in band_files if "B3_city" in s]
    
    if b1_file: #if it exists
        k1 = Extraction(folder1+b1_ref[0],path+folder2+b1_file[0]) #comparison band1
    if b2_file:
        k2 = Extraction(folder1+b2_ref[0],path+folder2+b2_file[0]) #comparison band2
    if b3_file:
        k3 = Extraction(folder1+b3_ref[0],path+folder2+b3_file[0]) #comparison band3
            
    xoff1=np.zeros(len(k1)) 
    xoff2=np.zeros(len(k2))
    xoff3=np.zeros(len(k3))
    
    yoff1=np.zeros(len(k1))
    yoff2=np.zeros(len(k2))
    yoff3=np.zeros(len(k3))
    
    #Offset calculation band1
    for l in range(0,len(k1)):
        xoff1[l]=k1[l][2]-k1[l][0]
        yoff1[l]=k1[l][3]-k1[l][1]
   
    #Offset calculation band2
    for l in range(0,len(k2)):
        xoff2[l]=k2[l][2]-k2[l][0]
        yoff2[l]=k2[l][3]-k2[l][1]
    
    #Offset calculation band3
    for l in range(0,len(k3)):
        xoff3[l]=k3[l][2]-k3[l][0]
        yoff3[l]=k3[l][3]-k3[l][1]
        
    #Final offset calculation - mean of calculated offsets
    xoff=round((xoff1.mean()+xoff2.mean()+xoff3.mean())/3)
    yoff=round((yoff1.mean()+yoff2.mean()+yoff3.mean())/3)
    
    print 'Offset: ' + str(xoff) + ', ' + str(yoff)
    
    '''
    Computing initial and final pixel for submatrix extraction.
    For each band a new matrix is created with rows+2*yoff and cols+2*xoff, filled with zeros where no original pixel values are available.
    The algorithm extracts a submatrix with same dimensions as the original image but changing the starting point using the calculated offset:
    - in case of negative offset the submatrix is going to start from (0,0)
    - in case of positive index the starting point is (2*off,2*yoff) because of the new dimensions
    '''
    
    if (xoff<=0):
        xstart=0
    else:
        xstart=2*xoff
        
    if (yoff<=0):
        ystart=0
    else:
        ystart=2*yoff
    
    band_files = os.listdir(path + folder2) #list files inside the directory
    #print band_files
    for j in range(1,9):
        band_file = [s for s in band_files if "B"+str(j)+"_city" in s]
        if band_file:
            inputimg2 = osgeo.gdal.Open(folder2+band_file[0],GA_ReadOnly) #open the image
            #print inputimg2
            if inputimg2 is None:
                print 'Could not open ' + band_file[0]
                sys.exit(1)
            cols2=inputimg2.RasterXSize #number of columns
            rows2=inputimg2.RasterYSize #number of rows
            band2 = inputimg2.RasterCount #number of bands
            geotransform=inputimg2.GetGeoTransform() #get geotransformation from the original image
            inprj=inputimg2.GetProjection() #get projection from the original image
            out=np.zeros(shape=(rows2,cols2)) #empty matrix
            driver=inputimg2.GetDriver()
            if os.path.isfile(folder2+band_file[0][:-4]+'_adj.TIF') == True:
                os.remove(folder2+band_file[0][:-4]+'_adj.TIF')
            output=driver.Create(folder2+band_file[0][:-4]+'_adj.TIF',cols2,rows2,band2) #create the output multispectral image
            inband2=inputimg2.GetRasterBand(1)
            outband=output.GetRasterBand(1)
            data2 = inband2.ReadAsArray()
            if j==8: #panchromatic band, dimensions of the panchromatic are different
                xoff = xoff*2
                yoff = yoff*2
                if (xoff<=0):
                    xstart=0
                else:
                    xstart=2*xoff
    
                if (yoff<=0):
                    ystart=0
                else:
                    ystart=2*yoff
            xend=xstart+cols2
            yend=ystart+rows2
    
            data2=np.c_[np.zeros((rows2,np.abs(xoff))),data2,np.zeros((rows2,np.abs(xoff)))] #add columns of zeros depending on the value of xoff around the original data
            data2=np.r_[np.zeros((np.abs(yoff),cols2+2*np.abs(xoff))),data2,np.zeros((np.abs(yoff),cols2+2*np.abs(xoff)))] #add rows of zeros depending on the value of yoff around the original data
            out=data2[int(ystart):int(yend),int(xstart):int(xend)] #submatrix extraction
            outband.WriteArray(out,0,0) #write to output image
            output.SetGeoTransform(geotransform) #set the transformation
            output.SetProjection(inprj)   #set the projection
            print 'Output: ' + band_file[0][:-4] + '_adj.TIF created' #output file created
        #inputimg2=None
        #output=None
        
        
def classification(path,folder,n_class,location,mapset,min_class,max_class):
    
    '''
    ###################################################################################################################
    Unsupervised classification of an image using GRASS - MAPSET HAS TO BE CREATED BEFORE EXECUTING THIS FUNCTION
    
    Input:
    - path: path to input folder
    - folder: name of the input folder, is used as part of the output name
    - n_class: number of classes to extract
    - location: grass parameter about the desired location to use
    - mapset: grass parameter about the mapset to be used
    - min_class: minimum value of the classes to select as a mask
    - max_class: maximum value of the classes to select as a mask
    
    Output:
    A file called uclasspy_folder_n_classes is added to the mapset - example "uclasspy_LT51800331984164XXX04_4"
    A file called uclasspy_folder_n_classes.TIF is added to the original folder - "uclasspy_LT51800331984164XXX04_4.TIF"
    ###################################################################################################################
    '''
    
    #GRASS details, mapset has to be created before
    if os.name == 'posix':
        #GRASS details, mapset has to be created before
        gisbase = '/Applications/GRASS-6.4.app/Contents/MacOS'
        gisdbase = '/Users/daniele/Documents/Test'
    else:
        gisbase = 'C:\Program Files (x86)\GRASS GIS 6.4.3RC2'
        gisdbase = 'C:\grassdata'
    
    #Initialize GRASS session
    gsetup.init(gisbase, gisdbase, location, mapset)
    #Print list of mapsets in location
    m = grass.mapsets(False)
    
    #Set GRASS region to DEFAULT and print GRASS region extent
    grass.run_command("g.region", flags = 'd')
    r = grass.read_command("g.region", flags = 'p')
    filename = 'urban_area'
    
    #define the output file name
    output_name = 'uclasspy_' + folder + '_' + str(n_class)
    #define the group name
    group_name = 'g_'+folder
    #define the subgroup name
    subgroup_name = 'sub' + group_name
    #define the signature file name
    sigfile_name = group_name + '_sign_'  + str(n_class)
    band_files = os.listdir(path + folder) #list files inside the directory
    b1_file = [s for s in band_files if "B1_city_adj" in s]
    b2_file = [s for s in band_files if "B2_city_adj" in s]
    b3_file = [s for s in band_files if "B3_city_adj" in s]
    b4_file = [s for s in band_files if "B4_city_adj" in s]
    b5_file = [s for s in band_files if "B5_city_adj" in s]
    b6_file = [s for s in band_files if "B6_city_adj" in s]
    b7_file = [s for s in band_files if "B7_city_adj" in s]
    
    if not b1_file or not b2_file or not b3_file or not b4_file or not b5_file or not b6_file or not b7_file:
        b1_file = [s for s in band_files if "B1_city" in s]
        b2_file = [s for s in band_files if "B2_city" in s]
        b3_file = [s for s in band_files if "B3_city" in s]
        b4_file = [s for s in band_files if "B4_city" in s]
        b5_file = [s for s in band_files if "B5_city" in s]
        b6_file = [s for s in band_files if "B6_city" in s]
        b7_file = [s for s in band_files if "B7_city" in s]
        
    #remove mapsets with the same if existing
    grass.run_command("g.remove", rast = output_name)
    grass.run_command("g.remove", rast = sigfile_name)
    grass.run_command("g.remove", rast = folder+'_B1_city_adj')
    grass.run_command("g.remove", rast = folder+'_B2_city_adj')
    grass.run_command("g.remove", rast = folder+'_B3_city_adj')
    grass.run_command("g.remove", rast = folder+'_B4_city_adj')
    grass.run_command("g.remove", rast = folder+'_B5_city_adj')
    grass.run_command("g.remove", rast = folder+'_B6_city_adj')
    grass.run_command("g.remove", rast = folder+'_B7_city_adj')
    
    
    #add raster files to mapset
    grass.run_command("r.in.gdal", input = path+folder+separator+b1_file[0], output = folder+'_B1_city_adj', flags = 'k')
    grass.run_command("r.in.gdal", input = path+folder+separator+b2_file[0], output = folder+'_B2_city_adj', flags = 'k')
    grass.run_command("r.in.gdal", input = path+folder+separator+b3_file[0], output = folder+'_B3_city_adj', flags = 'k')
    grass.run_command("r.in.gdal", input = path+folder+separator+b4_file[0], output = folder+'_B4_city_adj', flags = 'k')
    grass.run_command("r.in.gdal", input = path+folder+separator+b5_file[0], output = folder+'_B5_city_adj', flags = 'k')
    grass.run_command("r.in.gdal", input = path+folder+separator+b6_file[0], output = folder+'_B6_city_adj', flags = 'k')
    grass.run_command("r.in.gdal", input = path+folder+separator+b7_file[0], output = folder+'_B7_city_adj', flags = 'k')
        
    #Set current grass region to input
    grass.run_command("g.region", rast = folder+'_B1_city_adj')
    #create a group
    #grass.run_command("i.group", group = group_name, subgroup = subgroup_name, input = 'mul_'+folder+'.1@'+mapset+',mul_'+folder+'.2@'+mapset+',mul_'+folder+'.3@'+mapset)
    grass.run_command("i.group", group = group_name, subgroup = subgroup_name, input = folder+'_B1_city_adj@'+mapset+','+folder+'_B2_city_adj@'+mapset+','+folder+'_B3_city_adj@'+mapset+','+folder+'_B4_city_adj@'+mapset+','+folder+'_B5_city_adj@'+mapset+','+folder+'_B6_city_adj@'+mapset+','+folder+'_B7_city_adj@'+mapset)
    
    #cluster function
    grass.run_command("i.cluster", group = group_name, subgroup = subgroup_name, sigfile = sigfile_name, classes = n_class)
    
    #maximum likelihood function, os.system is used because of the python conflict with the class name
    os.system("i.maxlik group="+ group_name + ' subgroup=' + subgroup_name + ' class=' + output_name + ' sigfile=' + sigfile_name + ' --v')
    grass.run_command("r.out.gdal", input=output_name, output=path+folder+separator+output_name+'.TIF', format='GTiff') #create the output image
    osgeo.gdal.AllRegister()
    inb = osgeo.gdal.Open(path+folder+separator+output_name+'.TIF', GA_ReadOnly)
    print str(path+folder+separator+output_name+'.TIF')
    inband = inb.GetRasterBand(1)
    data = inband.ReadAsArray()
    mask_class1 = np.greater_equal(data,min_class)
    mask_class2 = np.less_equal(data,max_class)
    mask_class = mask_class1*mask_class2
    classification_list=[]
    classification_mask = np.choose(mask_class,(0,1))
    classification_list.append(classification_mask)
    WriteOutputImage(folder+separator+output_name+'.TIF',path,'',folder+separator+'classification_mask_' + folder[9:13]+'.TIF',0,0,0,1,classification_list)
    del classification_list

def urban_development_landsat(path,folder):
    
    '''
    ###################################################################################################################
    Calculates the urban_index index helping the user to define a threshold
    
    Input:
    - path: path to the input file folder
    - folder: name of the input folder
    
    Output:
    Returns a matrix containing the urban_index values
    ###################################################################################################################
    '''
    
    print 'Urban Area Extraction'
    osgeo.gdal.AllRegister()
    os.chdir(path)
    ref_files = os.listdir(folder)
    #search for adjusted band-files inside the folder
    b1_file = [s for s in ref_files if "B1_city_adj" in s]
    b2_file = [s for s in ref_files if "B2_city_adj" in s]
    b3_file = [s for s in ref_files if "B3_city_adj" in s]
    b4_file = [s for s in ref_files if "B4_city_adj" in s]
    b5_file = [s for s in ref_files if "B5_city_adj" in s]
    b6_file = [s for s in ref_files if "B6_city_adj" in s]
    b7_file = [s for s in ref_files if "B7_city_adj" in s]

    #in case of non-adjusted files inside the folder 
    if not b1_file or not b2_file or not b3_file or not b4_file or not b5_file or not b6_file:
        b1_file = [s for s in ref_files if "B1_city" in s]
        b2_file = [s for s in ref_files if "B2_city" in s]
        b3_file = [s for s in ref_files if "B3_city" in s]
        b4_file = [s for s in ref_files if "B4_city" in s]
        b5_file = [s for s in ref_files if "B5_city" in s]
        b6_file = [s for s in ref_files if "B6_city" in s]
        b7_file = [s for s in ref_files if "B7_city" in s]

    # open the images
    inb1 = osgeo.gdal.Open(folder+b1_file[0], GA_ReadOnly)
    if inb1 is None:
        print 'Could not open ' + b1_file[0]
        sys.exit(1)
    inb2 = osgeo.gdal.Open(folder+b2_file[0], GA_ReadOnly)
    if inb2 is None:
        print 'Could not open ' + b2_file[0]
        sys.exit(1)
    inb3 = osgeo.gdal.Open(folder+b3_file[0], GA_ReadOnly)
    if inb3 is None:
        print 'Could not open ' + b3_file[0]
        sys.exit(1)
    inb4 = osgeo.gdal.Open(folder+b4_file[0], GA_ReadOnly)
    if inb4 is None:
        print 'Could not open ' + b4_file[0]
        sys.exit(1)
    inb5 = osgeo.gdal.Open(folder+b5_file[0], GA_ReadOnly)
    if inb5 is None:
        print 'Could not open ' + b5_file[0]
        sys.exit(1)
    inb6 = osgeo.gdal.Open(folder+b6_file[0], GA_ReadOnly)
    if inb6 is None:
        print 'Could not open ' + b6_file[0]
        sys.exit(1)
    inb7 = osgeo.gdal.Open(folder+b7_file[0], GA_ReadOnly)
    if inb7 is None:
        print 'Could not open ' + b7_file[0]
        sys.exit(1)
    # get image size
    rows = inb2.RasterYSize
    cols = inb2.RasterXSize
    
    # get the values
    inBand1 = inb1.GetRasterBand(1)
    inBand2 = inb2.GetRasterBand(1)
    inBand3 = inb3.GetRasterBand(1)
    inBand4 = inb4.GetRasterBand(1)
    inBand5 = inb5.GetRasterBand(1)
    inBand6 = inb6.GetRasterBand(1)
    inBand7 = inb6.GetRasterBand(1)

    mat_data1 = inBand1.ReadAsArray().astype(np.float16) 
    mat_data2 = inBand2.ReadAsArray().astype(np.float16) 
    mat_data3 = inBand3.ReadAsArray().astype(np.float16) 
    mat_data4 = inBand4.ReadAsArray().astype(np.float16) 
    mat_data5 = inBand5.ReadAsArray().astype(np.float16) 
    mat_data6 = inBand6.ReadAsArray().astype(np.float16)
    mat_data7 = inBand7.ReadAsArray().astype(np.float16)

    MNDWI = ((mat_data2-mat_data5) / (mat_data2+mat_data5+0.0001)) #compute the urban_index
    NDBI = ((mat_data5 - mat_data4) / (mat_data5 + mat_data4+0.0001))
    #NBI = ((mat_data3 * mat_data5) / (mat_data4+0.0001))
    #NDVI = ((mat_data4 - mat_data3) / (mat_data4 + mat_data3+0.0001))
    SAVI = (((mat_data4 - mat_data3)*(8+0.5)) / (mat_data3 + mat_data4+0.0001+0.5))
    #NDISI = ((mat_data6 - ((mat_data1 + mat_data4 + mat_data5)/3)) / (mat_data6 + ((mat_data1 + mat_data4 + mat_data5)/3)))
    Built_up = ((mat_data7+mat_data2 - 1.5*mat_data5) / (mat_data2 + mat_data5 + mat_data7+0.0001))# my built up indicator positive for builtup and negative for mountains 

        
    inb2 = None
    inb3 = None
    inb4 = None
    inb5 = None
    inb6 = None
    
    return  SAVI, NDBI, MNDWI, Built_up 
    
    
def pca(path,folder):
    
    '''
    ###################################################################################################################
    Computes the Principal Component Analysis - Used in urban area extraction, good results with mode and third order component
    
    Input:
    - path: path to the input file folder
    - folder: input file folder
    
    Output:
    - mean: mean of all the bands
    - first_mode: first order component
    - second_mode: second order component
    - third_mode: third order component
    - new_indicator: urban development indicator
    ###################################################################################################################
    '''

    osgeo.gdal.AllRegister()
    os.chdir(path)
    ref_files = os.listdir(folder)
    bandList = []
    
    for i in range(1,8):
    #loop through band files
        b_file = [s for s in ref_files if "B"+str(i)+"_city_adj" in s]
        if not b_file:
            b_file = [s for s in ref_files if "B"+str(i)+"_city" in s]
        print b_file[0]    
        inb = osgeo.gdal.Open(folder+b_file[0], GA_ReadOnly)
        if inb is None:
            print 'Could not open ' + b_file[0]
            sys.exit(1)
            
        rows = inb.RasterYSize
        cols = inb.RasterXSize
        inband = inb.GetRasterBand(1)
        
        #read the values
        data = inband.ReadAsArray(0, 0, cols, rows)
        #print i,data
        bandList.append(data)
        
    #expand the list
    immatrix = np.array([np.array(bandList[i]).flatten() for i in range(1,7)],'f')
 
    #get dimensions
    num_data,dim = immatrix.shape

    #center data
    img_mean = immatrix.mean(axis=0)
    
    for i in range(num_data):
        immatrix[i] -= img_mean
    
    if dim>100:
        print 'PCA - compact trick used'
        M = np.dot(immatrix,immatrix.T) #covariance matrix
        e,EV = np.linalg.eigh(M) #eigenvalues and eigenvectors
        tmp = np.dot(immatrix.T,EV).T #this is the compact trick
        V = tmp[::-1] #reverse since last eigenvectors are the ones we want
        S = np.sqrt(e)[::-1] #reverse since eigenvalues are in increasing order
    else:
        print 'PCA - SVD used'
        U,S,V = np.linalg.svd(immatrix)
        V = V[:num_data] #only makes sense to return the first num_data    

    mean = img_mean.reshape(rows,cols)
    first_mode = V[0].reshape(rows,cols)
    second_mode = V[1].reshape(rows,cols)
    third_mode = V[2].reshape(rows,cols)       
    new_indicator = ((4*first_mode)+mean)    /(mean + first_mode+second_mode+third_mode+0.0001)
    
    
    
    return mean,first_mode,second_mode,third_mode, new_indicator


def WriteOutputImage(projection_reference,path,folder,output_name,cols,rows,type,nbands,array_list):
    
    '''
    ###################################################################################################################
    Writes one or more matrixes to an image file setting the projection
    
    Input:
    - projection_reference: reference image used to get the projection
    - path: path to the input folder
    - folder: input file folder
    - output_name: name of the output image
    - cols: number of columns, in case set to 0 the number of columns is taken from the reference image
    - rows: number of rows, in case set to 0 the number of rows is taken from the reference image
    - type: type of data to be written into the output file, if 0 the default is GDT_FLoat32
    - nbands: number of bands to be written to the output file
    - array_list: list containing all the data to be written; each element of the list should be a matrix
    
    Output:
    Output file is created into the same folder of the reference
    ###################################################################################################################
    '''
    # create the output image using a reference image for the projection
    # type is the type of data
    # array_list is a list containing all the data matrixes; a list is used because could be more than one matrix (more than one band)
    # if cols and rows are not provided, the algorithm uses values from the reference image
    # nbands contains the number of bands in the output image
    if type == 0:
        type = GDT_Float32
    inb = osgeo.gdal.Open(path+folder+projection_reference, GA_ReadOnly)
    driver = inb.GetDriver()
    if rows == 0 or cols == 0:
        rows = inb.RasterYSize
        cols = inb.RasterXSize
    
    outDs = driver.Create(path+folder+output_name, cols, rows,nbands, type)
    if outDs is None:
        print 'Could not create ' + output_name
        sys.exit(1)
    for i in range(nbands): 
        outBand = outDs.GetRasterBand(i+1)
        outmatrix = array_list[i].reshape(rows,cols)
        outBand.WriteArray(outmatrix, 0, 0)
        
    # georeference the image and set the projection
    outDs.SetGeoTransform(inb.GetGeoTransform())
    outDs.SetProjection(inb.GetProjection())
