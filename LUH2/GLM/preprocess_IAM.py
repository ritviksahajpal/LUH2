import os
import logging

import numpy

import constants
import pygeoutil.util as util


class IAM:

    def __init__(self, iam, path_nc, lon_name = 'lon', lat_name = 'lat', tme_name = 'time'):
        """
        Constructor
        """
        self.iam  = iam
        self.lon_name = lon_name
        self.lat_name = lat_name
        self.tme_name = tme_name

        # Open up netCDF and get dimensions
        ds        = util.open_or_die(path_nc)
        self.lat  = ds.variables[lat_name][:]
        self.lon  = ds.variables[lon_name][:]
        self.time = ds.variables[tme_name][:]
        ds.close()

    def get_shape_data(self, path_nc, var):
        """
        :param path_nc: Path to netCDF dataset
        :return Shape of netCDF dataset iyrs: time-dimension, iny: y-dimension, inx: x-dimension
        """
        ds = util.open_or_die(path_nc)
        iyrs, iny, inx = numpy.shape(ds.variables[var])
        ds.close()

        return iyrs, iny, inx

    def copy_var_metadata(self, path_src, path_dest, vars):
        """
        Copy all the variable attributes from original file
        :param src_nc: Source netcdf file from which to copy metadata
        :param dest_nc: Target netCDF file
        :param vars: Variables for which to copy metadata
        :return: Nothing (side-effect: target netCDF has modified metadata)
        """
        # @TODO: Empirical testing showing that it is failing
        src_ds   = util.open_or_die(path_dest)
        dest_ds  = util.open_or_die(path_dest, 'r+')

        for var in [vars[0], vars[1], vars[2]]:
            for att in src_ds.variables[var].ncattrs():
                setattr(dest_ds.variables[var],att,getattr(src_ds.variables[var],att))

        src_ds.close()
        dest_ds.close()


    def copy_global_metadata_nc(self, path_src, path_dest):
        """
        :param path_src: Source netcdf file from which to copy metadata
        :param path_dest: Target netCDF file
        :return: Nothing (side-effect: target netCDF has modified metadata)
        """
        src_ds   = util.open_or_die(path_src)
        dest_ds  = util.open_or_die(path_dest, 'r+')

        # copy Global attributes from original file
        for att in src_ds.ncattrs():
            setattr(dest_ds,att,getattr(src_ds,att))

        src_ds.close()
        dest_ds.close()

if __name__ == '__main__':
    pass