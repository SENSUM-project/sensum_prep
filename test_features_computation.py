'''
Created on Oct 21, 2013

@author: daniele
'''

import osgeo.gdal
from osgeo.gdalconst import *
import osgeo.ogr,osgeo.osr
import os,sys
import multiprocessing
from multiprocessing import Pool
import numpy as np
import time
import skimage
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from library_05_11_2013 import spectral_features,multispectral_features,textural_features,call_multiprocess

# Parameters to set ########################################################################################################

path = '/Users/daniele/Documents/Sensum/Izmir/HR/Applications/'    #path of the folder containing the input files
#path = '/Users/daniele/Documents/Sensum/Izmir/HR/Applications/Features/'
input_file = 'clipped_merged_new.TIF'    #name of the input file
#input_file = 'Izmir_multi.TIF'
#segmentation_file = 'seg_baatz_raster_clipped_new.TIF'    #name of the raster file containing segmentation results
segmentation_file = 'Segmentation_baatz.TIF'
segmentation_vector_file = 'Segmentation_baatz_shape.shp'   #name of the shapefile containing segmentation results
#segmentation_vector_file = 'seg_baatz_raster_clipped_new.shp'
output_type = 'table'    #type of the produced output (table for vector, segment for raster); default is table
output_vector_file = 'segments_features_multi.shp' #name of the output shapefile

#optional in case of 'table' as output_type, necessary in case of 'segment' as output_type
output_image_spectral = 'spectral_features.TIF' #output name for raster with single band features
output_image_multispectral = 'multispectral_features.TIF'   #output name for raster with multiband features
output_image_textural = 'textural_features.TIF' #output name for raster with textural features

############################################################################################################################

start_time = time.time()
result_single = []
result_multi = []
result_texture = []
input_data_list = []
result_single_shape = []
result_single_shape_bands = []
result_texture_shape = []
result_texture_shape_bands = []
result_multi_shape = []

processors = multiprocessing.cpu_count()
inb = osgeo.gdal.Open(path+input_file, GA_ReadOnly)
inb_seg = osgeo.gdal.Open(path+segmentation_file, GA_ReadOnly)
if inb is None:
    print 'Could not open ' + input_file
    sys.exit(1)
nbands = inb.RasterCount
driver=inb.GetDriver()
rows = inb.RasterYSize
cols = inb.RasterXSize

#inband = inb.GetRasterBand(1)
#input_data = inband.ReadAsArray().astype(np.uint8) #read as uint8 to build the glcm after
#input_data2 = inband.ReadAsArray() #standard format

inband_seg = inb_seg.GetRasterBand(1) #read raster product of the segmentation
seg_data = inband_seg.ReadAsArray()
end_seg = np.amax(seg_data) 
start_seg = np.amin(seg_data)

#start_seg = 100
#end_seg = 400

for k in range(1,nbands+1): #read all the bands of the input file for the multispectral features
    inband = inb.GetRasterBand(k)
    data_mat = inband.ReadAsArray()
    data_mat2 = inband.ReadAsArray().astype(np.uint8)
    input_data_list.append(data_mat)
    print '--- Band: ' + str(k)
    print 'Spectral Features'
    result_single.append(call_multiprocess(spectral_features,(output_type,seg_data,data_mat),start_seg,end_seg))
    print 'Textural Features'
    result_texture.append(call_multiprocess(textural_features,(output_type,seg_data,data_mat2),start_seg,end_seg))
#result_single = call_multiprocess(spectral_features,(output_type,seg_data,input_data2),start_seg,end_seg)
#result_texture = call_multiprocess(textural_features,(output_type,seg_data,input_data),start_seg,end_seg)
#print len(result_single),len(result_single[0]),len(result_single[0][0])
#print len(result_texture),len(result_texture[0]),len(result_texture[0][0])

if (nbands>1):
    print 'Multispectral Features'
    result_multi = call_multiprocess(multispectral_features,('',seg_data,input_data_list),start_seg,end_seg)

#merge results from multiprocessing
'''
for p in range(0,processors):
    print p
    for b in range(0,nbands):
        print b
        #print result_single[b][p]
        if p == 0:
            result_single_shape = result_single[b][p] + result_single_shape
        else:
            result_single_shape = result_single_shape + result_single[b][p]
    
        #result_texture_shape[b] = result_texture_shape[b] + result_texture[b][p]
    #if (nbands>1):
     #   result_multi_shape = result_multi_shape + result_multi[p]
'''
for b in range(0,nbands):
    #print b
    result_single_shape = []
    result_texture_shape = []
    for p in range(0,processors):
        result_single_shape = result_single_shape + result_single[b][p]
        result_texture_shape = result_texture_shape + result_texture[b][p]
        if (nbands>1) and (b==0):
            result_multi_shape = result_multi_shape + result_multi[p]
    result_single_shape_bands.append(result_single_shape) 
    result_texture_shape_bands.append(result_texture_shape)

#output as a shapefile    
if (output_type == 'table'):
    driver_shape=osgeo.ogr.GetDriverByName('ESRI Shapefile')
    epsg_input = 4326
    epsg_output = 32635
    inprj=osgeo.osr.SpatialReference()
    inprj.ImportFromEPSG(epsg_input)

    outprj=osgeo.osr.SpatialReference()
    outprj.ImportFromEPSG(epsg_output)

    newcoord=osgeo.osr.CoordinateTransformation(inprj,outprj)
    infile=driver_shape.Open(path+segmentation_vector_file,0)
    inlayer=infile.GetLayer()
    
    outfile=driver_shape.CreateDataSource(path+output_vector_file)
    outlayer=outfile.CreateLayer('Features',geom_type=osgeo.ogr.wkbPolygon)
    
    #infeature=inlayer.GetFeature(0)
    layer_defn = inlayer.GetLayerDefn()
    feature_def = outlayer.GetLayerDefn()
    
    infeature = inlayer.GetNextFeature()

    dn_def = osgeo.ogr.FieldDefn('DN', osgeo.ogr.OFTInteger)
    outlayer.CreateField(dn_def)
    for b in range(1,nbands+1):
        mean_def = osgeo.ogr.FieldDefn('Mean' + str(b), osgeo.ogr.OFTReal)
        outlayer.CreateField(mean_def)
        std_def = osgeo.ogr.FieldDefn('Std' + str(b),osgeo.ogr.OFTReal)
        outlayer.CreateField(std_def)
        maxbr_def = osgeo.ogr.FieldDefn('MaxBr' + str(b),osgeo.ogr.OFTReal)
        outlayer.CreateField(maxbr_def)
        minbr_def = osgeo.ogr.FieldDefn('MinBr' + str(b),osgeo.ogr.OFTReal)
        outlayer.CreateField(minbr_def)
        mode_def = osgeo.ogr.FieldDefn('Mode' + str(b),osgeo.ogr.OFTReal)
        outlayer.CreateField(mode_def)
        
        contrast_def = osgeo.ogr.FieldDefn('Contr' + str(b), osgeo.ogr.OFTReal)
        outlayer.CreateField(contrast_def)
        energy_def = osgeo.ogr.FieldDefn('Energy' + str(b),osgeo.ogr.OFTReal)
        outlayer.CreateField(energy_def)
        homogeneity_def = osgeo.ogr.FieldDefn('Homoge' + str(b),osgeo.ogr.OFTReal)
        outlayer.CreateField(homogeneity_def)
        correlation_def = osgeo.ogr.FieldDefn('Correl' + str(b),osgeo.ogr.OFTReal)
        outlayer.CreateField(correlation_def)
        dissimilarity_def = osgeo.ogr.FieldDefn('Dissi' + str(b),osgeo.ogr.OFTReal)
        outlayer.CreateField(dissimilarity_def)
        asm_def = osgeo.ogr.FieldDefn('Asm' + str(b),osgeo.ogr.OFTReal)
        outlayer.CreateField(asm_def)
    
    if (nbands>1):
        ndvi_mean_def = osgeo.ogr.FieldDefn('Ndvi_Mean',osgeo.ogr.OFTReal)
        outlayer.CreateField(ndvi_mean_def)
        ndvi_std_def = osgeo.ogr.FieldDefn('Ndvi_Std',osgeo.ogr.OFTReal)
        outlayer.CreateField(ndvi_std_def)
        wb_def = osgeo.ogr.FieldDefn('Wb',osgeo.ogr.OFTReal)
        outlayer.CreateField(wb_def)
        
    feature_def = outlayer.GetLayerDefn()
    
    while infeature:
        # get the input geometry
        geom = infeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(newcoord)
        # create a new feature
        outfeature = osgeo.ogr.Feature(feature_def)
        # set the geometry and attribute
        outfeature.SetGeometry(geom)
        #field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
        dn = infeature.GetField('DN')
        outfeature.SetField('DN',dn)
        #print 'dn: ' + str(dn)
        index = (dn-1)*5
        index2 = (dn-1)*6
        for b in range(0,nbands):
            #print b
            #print len(result_single_shape_bands[b])
            mean = result_single_shape_bands[b][index]
            std = result_single_shape_bands[b][index+1]
            maxbr = result_single_shape_bands[b][index+2]
            minbr = result_single_shape_bands[b][index+3]
            mode = result_single_shape_bands[b][index+4]
            outfeature.SetField('Mean' + str(b+1),mean)
            outfeature.SetField('Std' + str(b+1),std)
            outfeature.SetField('MaxBr' + str(b+1),maxbr)
            outfeature.SetField('MinBr' + str(b+1),minbr)
            outfeature.SetField('Mode' + str(b+1),mode)
        
        
            contrast = result_texture_shape_bands[b][index2]
            energy = result_texture_shape_bands[b][index2+1]
            homogeneity = result_texture_shape_bands[b][index2+2]
            correlation = result_texture_shape_bands[b][index2+3]
            dissimilarity = result_texture_shape_bands[b][index2+4]
            asm = result_texture_shape_bands[b][index2+5]
            outfeature.SetField('Contr' + str(b+1),contrast)
            outfeature.SetField('Energy' + str(b+1),energy)
            outfeature.SetField('Homoge' + str(b+1),homogeneity)
            outfeature.SetField('Correl' + str(b+1),correlation)
            outfeature.SetField('Dissi' + str(b+1),dissimilarity)
            outfeature.SetField('Asm' + str(b+1),asm)
        
        if (nbands>1):
            index3 = (dn-1)*3
            ndvi_mean = result_multi_shape[index3]
            ndvi_std = result_multi_shape[index3+1]
            wb = result_multi_shape[index3+2]
            outfeature.SetField('Ndvi_Mean',ndvi_mean)
            outfeature.SetField('Ndvi_Std',ndvi_std)
            outfeature.SetField('Wb',wb)
            
        outlayer.CreateFeature(outfeature)
        infeature = inlayer.GetNextFeature()
    outprj.MorphToESRI()
    file_prj = open(path+output_vector_file[:-4]+'.prj', 'w')
    file_prj.write(outprj.ExportToWkt())
    file_prj.close()
    print 'Output created: ' + output_vector_file
    # close the shapefiles
    infile.Destroy()
    outfile.Destroy()

#output as a raster    
if (output_type == 'segment'):
    outmask_mean = np.zeros((rows,cols))
    outmask_std = np.zeros((rows,cols))
    outmask_maxbr = np.zeros((rows,cols))
    outmask_minbr = np.zeros((rows,cols))
    outmask_mode = np.zeros((rows,cols))
    
    outmask_contrast = np.zeros((rows,cols))
    outmask_energy = np.zeros((rows,cols))
    outmask_homogeneity = np.zeros((rows,cols))
    outmask_correlation = np.zeros((rows,cols))
    outmask_dissimilarity = np.zeros((rows,cols))
    outmask_asm = np.zeros((rows,cols))
    
    outmask_ndvi_mean = np.zeros((rows,cols))
    outmask_ndvi_std = np.zeros((rows,cols))
    outmask_wb = np.zeros((rows,cols))
    
    out_img_single = driver.Create(path+output_image_spectral,cols,rows,5,GDT_Float32)
    out_img_multi = driver.Create(path+output_image_multispectral,cols,rows,3,GDT_Float32)
    out_img_textural = driver.Create(path+output_image_textural,cols,rows,6,GDT_Float32)
    
    out_img_single.SetGeoTransform(inb.GetGeoTransform())
    out_img_single.SetProjection(inb.GetProjection())
    out_img_multi.SetGeoTransform(inb.GetGeoTransform())
    out_img_multi.SetProjection(inb.GetProjection())
    out_img_textural.SetGeoTransform(inb.GetGeoTransform())
    out_img_textural.SetProjection(inb.GetProjection())
    
    for p in range(0,processors):
        outmask_mean = outmask_mean + result_single[p][0]
        outmask_std = outmask_std + result_single[p][1]
        outmask_maxbr = outmask_maxbr + result_single[p][2]
        outmask_minbr = outmask_minbr + result_single[p][3]
        outmask_mode = outmask_mode + result_single[p][4]
        
        outmask_contrast = outmask_contrast + result_texture[p][0]
        outmask_energy = outmask_energy + result_texture[p][1]
        outmask_homogeneity = outmask_homogeneity + result_texture[p][2]
        outmask_correlation = outmask_correlation + result_texture[p][3]
        outmask_dissimilarity = outmask_dissimilarity + result_texture[p][4]
        outmask_asm = outmask_asm + result_texture[p][5]
        
        if (nbands>1):
            outmask_ndvi_mean = outmask_ndvi_mean + result_multi[p][0]
            outmask_ndvi_std = outmask_ndvi_std + result_multi[p][1]
            outmask_wb = outmask_wb + result_multi[p][2]
        
    outband_single=out_img_single.GetRasterBand(1)
    outband_single.WriteArray(outmask_mean,0,0)
    outband2_single = out_img_single.GetRasterBand(2)
    outband2_single.WriteArray(outmask_std,0,0)
    outband3_single = out_img_single.GetRasterBand(3)
    outband3_single.WriteArray(outmask_maxbr,0,0)
    outband4_single = out_img_single.GetRasterBand(4)
    outband4_single.WriteArray(outmask_minbr,0,0)
    outband5_single = out_img_single.GetRasterBand(5)
    outband5_single.WriteArray(outmask_mode,0,0)
    
    outband_textural = out_img_textural.GetRasterBand(1)
    outband_textural.WriteArray(outmask_contrast,0,0)
    outband2_textural = out_img_textural.GetRasterBand(2)
    outband2_textural.WriteArray(outmask_energy,0,0)
    outband3_textural = out_img_textural.GetRasterBand(3)
    outband3_textural.WriteArray(outmask_homogeneity,0,0)
    outband4_textural = out_img_textural.GetRasterBand(4)
    outband4_textural.WriteArray(outmask_correlation,0,0)
    outband5_textural = out_img_textural.GetRasterBand(5)
    outband5_textural.WriteArray(outmask_dissimilarity,0,0)
    outband6_textural = out_img_textural.GetRasterBand(6)
    outband6_textural.WriteArray(outmask_asm,0,0)
    
    if (nbands>1):
        outband_multi=out_img_multi.GetRasterBand(1)
        outband_multi.WriteArray(outmask_ndvi_mean,0,0)
        outband2_multi = out_img_multi.GetRasterBand(2)
        outband2_multi.WriteArray(outmask_ndvi_std,0,0)
        outband3_multi = out_img_multi.GetRasterBand(3)
        outband3_multi.WriteArray(outmask_wb,0,0)
        
end_time = time.time()
print 'Total time = ' + str(end_time-start_time)    
    