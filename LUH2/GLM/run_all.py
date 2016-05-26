import numpy
import os
import pdb

import constants
import plot
import preprocess_GCAM, preprocess_IMAGE

import pygeoutil.util as util


def plot_GCAM_annual_diffs(nc_file, tme='', var='', legend='', ylabel=''):
    """
    :return:
    """
    arr_diff = util.max_diff_netcdf(nc_file, var = var)
    arr_diff.insert(0,0.0)
    plot.plot_np_ts(arr_diff, tme, constants.out_dir+os.sep+'Annual_diff_GCAM_'+os.path.basename(nc_file)[:-3]+'.png',
                    leg_name=legend, ylabel = ylabel, col='k')

def iterate_plot_GCAM_annual_diffs():
    """
    :return:
    """
    obj = preprocess_GCAM.GCAM(constants.gcam_dir+constants.PASTURE[0])

    for i in range(len(constants.CROPS)):
        plot_GCAM_annual_diffs(constants.gcam_dir+constants.CROPS[i], tme=obj.time, var='landcoverpercentage',
                              legend=constants.CROPS[i][:-3], ylabel = 'Fraction of gridcell area')

def process_GCAM():

    obj = preprocess_GCAM.GCAM(constants.gcam_dir+constants.PASTURE[0])
    yrs = len(obj.time)
    ny  = len(obj.lat)
    nx  = len(obj.lon)

    # Calculate sum of all crop classes and create new GCAM file
    sum_perc = numpy.zeros((yrs,ny,nx),dtype=float)
    sum_perc = obj.create_GCAM_croplands(sum_perc)

    shape = yrs, ny, nx
    obj.write_GCAM_nc(sum_perc, shape)
    del sum_perc

    # Create netCDF with each crop category represented as fraction of cropland
    # area and not total grid cell area
    obj.create_nc_perc_croplands(obj.gcam_out_fl, shape)

    # Copy global metadata
    #obj.copy_global_metadata_nc(obj.path_pas, obj.gcam_out_fl)
    
    # Plot maps
    plot.plot_map_from_nc(obj.perc_crops_fl,
                          os.path.dirname(obj.perc_crops_fl) + constants.CROPS[0][:-3] + '.png',
                          constants.CROPS[0], xlabel='Land cover fraction', grid = True)

def process_IMAGE():
    path_pas = constants.gcam_dir+constants.PASTURE[0]

    obj = preprocess_IMAGE.IMAGE(path_pas)

if __name__ == '__main__':
    if constants.CONVERT_WH:
        obj = preprocess_GCAM.GCAM(constants.gcam_dir+constants.PASTURE[0])
        obj.AEZ_to_national_GCAM(data_source = 'wh', out_nc_name = 'woodharvest')

    if constants.PREPROCESS_GCAM:
        process_GCAM()
        iterate_plot_GCAM_annual_diffs()

    if constants.PREPROCESS_IMAG:
        process_IMAGE()
