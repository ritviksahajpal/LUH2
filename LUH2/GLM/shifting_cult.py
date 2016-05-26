import os
import pdb
import numpy
import netCDF4
import sys
import pandas
import logging

import plot
import constants
import pygeoutil.util as util
import pygeoutil.rgeo as rgeo
import tempfile


# TODO: Clean up output paths, currently windows specific
# TODO: Check if creating butler map is working (not high priority at all)

class ShftCult:
    """

    """
    def __init__(self, use_andreas=False, file_sc=constants.ASC_BUTLER, default_rate=0.067, start_yr=850, end_yr=2015,
                 skiprows=0):
        """

        :return:
        """
        # Shifting cultivation constants
        self.start_yr = start_yr
        self.end_yr = end_yr
        self.default_rate = default_rate

        # Properties of ascii file
        self.asc_prop = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'NODATA_value']

        # Constants
        self.dict_cont = {0: 'Antartica', 1: 'North_America', 2: 'South_America', 3: 'Europe', 4: 'Asia', 5: 'Africa',
                          6: 'Australia'}
        self.CCODES = 'country_codes'
        self.CONT_CODES = 'continent_codes'

        # Lats and lons
        self.num_lats = constants.NUM_LATS
        self.num_lons = constants.NUM_LONS

        # continent and country code files
        self.ccodes_file = constants.ccodes_file
        self.contcodes_file = constants.contcodes_file
        self.ccodes_asc = numpy.genfromtxt(constants.CNTRY_CODES, skip_header=skiprows, delimiter=' ')

        self.file_raw_sc = file_sc
        if use_andreas:
            self.data_source = 'Andreas'
        else:
            self.data_source = 'Butler'

    def get_ascii_properties(self, ascii_fl, name_property='ncols'):
        """
        Get value for property of ascii file.
        Valid property names are: ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value
        :param ascii_fl:
        :param name_property:
        :return:
        """
        # From https://en.wikipedia.org/wiki/Esri_grid, this is ESRI ascii format
        # ncols and nrows are the numbers of rows and columns, respectively (represented as integers);
        # xllcorner and yllcorner are the western (left) x-coordinate and southern (bottom) y-coordinates, such as
        # easting and northing (represented as real numbers with an optional decimal point)
        # cellsize is the length of one side of a square cell (a real number); and,
        # nodata_value is the value that is regarded as "missing" or "not applicable"; this line is optional, but highly
        # recommended as some programs expect this line to be declared (a real number).
        asc_data = numpy.genfromtxt(ascii_fl, max_rows=6, dtype=None)

        try:
            val = [item for item in asc_data if item[0] == name_property][0][1]
        except:
            logging.info(name_property + ' is not a valid property for GIS ascii file')

        return val

    def do_andreas(self, to_field='SC2010Cat'):
        """

        Args:
            to_field: can be 'SC2090Cat'

        Returns:

        """
        path_asc_andreas = os.path.dirname(constants.TIF_ANDREAS) + os.sep + \
                           os.path.basename(os.path.splitext(constants.TIF_ANDREAS)[0]) + '.asc'

        # Get 2010 shifting cultivation map
        path_out_lup = tempfile.mkstemp(suffix='.tif')[1]
        rgeo.lookup(self.file_raw_sc, path_out_ds=path_out_lup, from_field='Value', to_field=to_field)

        # Call convert_raster_to_ascii with appropriate output file name.
        rgeo.convert_raster_to_ascii(path_input_raster=path_out_lup, path_ascii_output=path_asc_andreas)

        asc_sc = self.create_global_sc_from_andreas(path_andreas_asc=path_asc_andreas, new_res=0.25)

        asc_binary_sc = asc_sc[:]
        asc_binary_sc[(asc_binary_sc > 0.0) & (asc_binary_sc < 65536.0)] = 1.0

        return asc_binary_sc

    def create_global_sc_from_andreas(self, path_andreas_asc, new_res=0.25):
        """

        :param path_andreas_asc: input andreas ascii file that is missing rows
        :param new_res:
        :return:
        """
        if self.data_source != 'Andreas':
            logging.error('Incorrect data source. Should be Andreas')
            sys.exit(0)

        # Get properties of Andreas ascii file
        ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value = \
            [self.get_ascii_properties(path_andreas_asc, prop) for prop in self.asc_prop]

        andreas_data = numpy.genfromtxt(path_andreas_asc, skip_header=6, delimiter=' ')

        # We want to create a file with same resolution as Andreas's file, but global and not regional like Andreas
        global_xll = -180.0  # x lower left
        global_yll = -90.0  # y lower left
        global_xlr = 180.0  # x lower right
        global_yur = 90.0  # y upper right

        andreas_end_row = numpy.ceil(global_yur - yllcorner).astype(int)
        andreas_start_row = numpy.ceil(andreas_end_row - nrows).astype(int)
        andreas_start_col = numpy.ceil(xllcorner - global_xll).astype(int)
        andreas_end_col = numpy.ceil(andreas_start_col + ncols).astype(int)

        # Create numpy 2D array,
        global_sc = numpy.zeros((int(180//cellsize), int(360//cellsize)))
        global_sc.fill(65536.0)

        # Replace global sc array with andreas's data
        global_sc[andreas_start_row:andreas_end_row, andreas_start_col:andreas_end_col] = andreas_data

        # Create new global sc file
        new_res_global_sc = global_sc.repeat(1.0/new_res, 0).repeat(1.0/new_res, 1)

        return new_res_global_sc

    def combine_country_continent(self):
        """

        :return:
        """
        ccodes = pandas.read_csv(self.ccodes_file, header=None)
        ccodes.columns = [self.CCODES]

        contcodes = pandas.read_csv(self.contcodes_file, header=None)
        contcodes.columns = [self.CONT_CODES]

        return pandas.concat([ccodes, contcodes], axis=1)

    def replace_country_by_continent(self, arr, lup_codes):
        """
        stackoverflow.com/questions/34321025/replace-values-in-numpy-2d-array-based-on-pandas-dataframe/34328891#34328891
        :param arr:
        :return:
        """
        old_val = numpy.array(lup_codes[self.CCODES])
        new_val = numpy.array(lup_codes[self.CONT_CODES])

        mask = numpy.in1d(arr, old_val)
        idx = numpy.searchsorted(old_val, arr.ravel()[mask])
        arr.ravel()[mask] = new_val[idx]

        return arr

    def make_shifting_cult_nc(self, out_path, ccode=0.0, new_rate=0.15, desc='netCDF'):
        """

        Args:
            out_path:
            ccode:
            new_rate:
            desc:

        Returns:

        """
        # Compute dimensions of nc file based on # rows/cols in ascii file
        fl_res = self.num_lats/self.asc_sc.shape[0]
        if fl_res != self.num_lons/self.asc_sc.shape[1]:
            print('Incorrect dimensions in ascii file')
            sys.exit(0)

        # Initialize nc file
        out_nc = out_path+os.sep + os.path.basename(self.file_raw_sc)[:-4] + '.nc'
        nc_data = netCDF4.Dataset(out_nc, 'w', format='NETCDF4')
        nc_data.description = desc

        # dimensions
        nc_data.createDimension('lon', self.asc_sc.shape[1])
        nc_data.createDimension('lat', self.asc_sc.shape[0])
        tme = numpy.arange(self.start_yr, self.end_yr + 1)
        nc_data.createDimension('time', numpy.shape(tme)[0])

        # Populate and output nc file
        longitudes = nc_data.createVariable('longitude', 'f4', ('lon',))
        latitudes = nc_data.createVariable('latitude', 'f4', ('lat',))
        time = nc_data.createVariable('time', 'i4', ('time',))

        data = nc_data.createVariable('shift_cult', 'f4', ('time', 'lat', 'lon',), fill_value=numpy.nan)
        cntr_codes = nc_data.createVariable('cntry_codes', 'f4', ('lat', 'lon',), fill_value=0.0)

        data.units = 'fraction of gridcell area'
        data.long_name = 'shifting cultivation fraction of gridcell area'

        # Assign values to dimensions and data
        latitudes[:] = numpy.arange(90.0 - fl_res/2.0, -90.0, -fl_res)
        longitudes[:] = numpy.arange(-180.0 + fl_res/2.0, 180.0,  fl_res)
        time[:] = tme
        cntr_codes[:] = self.ccodes_asc[:, :]  # Read in the country codes data

        # Assign default shifting cultivation rate
        self.asc_sc[self.asc_sc > 0.0] = self.default_rate

        if constants.shft_by_country:
            # Store data into netCDF file
            for idx, j in enumerate(tme):
                print idx, j
                if j > 1970 and j <= 2015:
                    self.asc_sc[self.asc_sc > 0.0] = self.default_rate - \
                                                     ((self.default_rate - new_rate) * (j - 1970) / 45.0)
                data[idx, :, :] = self.asc_sc[:, :]
        else:
            # Replace country codes by respective continent codes
            lup_codes = self.combine_country_continent()
            if lup_codes.empty:
                logging.error('Empty lookup table (country-continent)')
                sys.exit(0)

            cont_shftCult = self.replace_country_by_continent(self.ccodes_asc, lup_codes)

            # Replace default shifting cultivation rate for some continents
            for idx, j in enumerate(tme):
                if j > 1970 and j <= 2015:
                    # After 1970, Asia declines by 90%
                    new_rate = self.default_rate * 0.10
                    tmp_arr = self.asc_sc[self.asc_sc > 0.0][cont_shftCult[:] == 4]
                    tmp_arr = self.default_rate - ((self.default_rate - new_rate) * (j - 1970)/45.0)

                    # After 1970, S America declines by 70%
                    new_rate = self.default_rate * 0.30
                    tmp_arr = self.asc_sc[self.asc_sc > 0.0][cont_shftCult[:] == 2]
                    tmp_arr = self.default_rate - ((self.default_rate - new_rate) * (j - 1970)/45.0)

            data[idx, :, :] = tmp_arr

        # Change default rate of shifting cultivation for some countries
        if ccode > 0.0:
            self.asc_sc[cntr_codes[:] == ccode] = new_rate

        nc_data.close()
        return out_nc

    def interpolate_asc(self, start_asc, end_asc, start_pt, current_pt, end_pt):
        """

        Args:
            start_asc:
            end_asc:
            start_pt:
            current_pt:
            end_pt:

        Returns:

        """
        if end_pt <= start_pt:
            print('End point should be less than starting point')
        interp_asc = end_asc + (start_asc - end_asc) * ((end_pt - current_pt) / (end_pt - start_pt))

        return interp_asc

    def create_andreas_nc(self):
        """

        Returns:

        """
        path_out = constants.input_dir + os.sep + 'shift_cult'

        asc_2010 = self.do_andreas(to_field='SC2010Cat')
        asc_2090 = self.do_andreas(to_field='SC2090Cat')

        static_map = numpy.copy(asc_2010)
        sc_map_2090 = numpy.copy(asc_2090)

        # Frequency of occurrence is used as a proxy for the fraction of cropland area in each grid cell that is
        # associated with shifting cultivation. Assuming a 1 year cultivation period, and re-clearing from secondary
        # land unless secondary is less than 10*(cropland in SC).

        # From Heineman et al.
        # Each grid cell was classified into: None, very low, low, moderate, high (shifting cultivation)
        # This corresponds to ranges of area share of shifting cultivation (cultivated fields plus fallow) within an
        # entire one-degree cell
        # 0 (none): < 1%
        # 1 (very-low): 1 - 9%
        # 2 (low): 10 - 19%
        # 3 (moderate): 20 - 39%
        # 4. (high): >= 40%
        # 5. (historic): 70%
        static_map[static_map == 1.0] = 0.05
        static_map[static_map == 0.0] = 0.05
        static_map[static_map == 2.0] = 0.05
        static_map[static_map == 3.0] = 0.15
        static_map[static_map == 4.0] = 0.3
        static_map[static_map == 5.0] = 0.7
        static_map = static_map * 3.0  # Back in 1850, move each category 3 levels up
        static_map[static_map >= 0.7] = 0.7  # cap
        # static_map[static_map >= 0.0] = 0.7  # Constant rate of SC rate in 1850 approach 0

        rate_map = numpy.copy(asc_2010)
        rate_map[rate_map == 1.0] = 0.05
        rate_map[rate_map == 0.0] = 0.05
        rate_map[rate_map == 2.0] = 0.05
        rate_map[rate_map == 3.0] = 0.15
        rate_map[rate_map == 4.0] = 0.3
        rate_map[rate_map == 5.0] = 0.7

        sc_map_2090[sc_map_2090 == 1.0] = 0.05
        sc_map_2090[sc_map_2090 == 0.0] = 0.05
        sc_map_2090[sc_map_2090 == 2.0] = 0.05
        sc_map_2090[sc_map_2090 == 3.0] = 0.15
        sc_map_2090[sc_map_2090 == 4.0] = 0.3
        sc_map_2090[sc_map_2090 == 5.0] = 0.7

        rate_map[rate_map == -9999.0] = 0.0
        sc_map_2090[sc_map_2090 <= -9999.0] = 0.0
        static_map[static_map <= -9999.0] = 0.0  # approach 0

        # Compute dimensions of nc file based on # rows/cols in ascii file
        fl_res = self.num_lats/asc_2010.shape[0]
        if fl_res != self.num_lons/asc_2010.shape[1]:
            print('Incorrect dimensions in ascii file')
            sys.exit(0)

        # Initialize nc file
        out_nc = path_out + os.sep + 'andreas_approach1.nc'
        nc_data = netCDF4.Dataset(out_nc, 'w', format='NETCDF4')
        nc_data.description = ''

        # dimensions
        nc_data.createDimension('lon', asc_2010.shape[1])
        nc_data.createDimension('lat', asc_2010.shape[0])
        tme = numpy.arange(self.start_yr, 2100 + 1)
        nc_data.createDimension('time', numpy.shape(tme)[0])

        # Populate and output nc file
        longitudes = nc_data.createVariable('longitude', 'f4', ('lon',))
        latitudes = nc_data.createVariable('latitude', 'f4', ('lat',))
        time = nc_data.createVariable('time', 'i4', ('time',))

        data = nc_data.createVariable('shift_cult', 'f4', ('time', 'lat', 'lon',), fill_value=numpy.nan)
        cntr_codes = nc_data.createVariable('cntry_codes', 'f4', ('lat', 'lon',), fill_value=0.0)

        data.units = 'fraction of gridcell area'
        data.long_name = 'shifting cultivation fraction of gridcell area'

        # Assign values to dimensions and data
        latitudes[:] = numpy.arange(90.0 - fl_res/2.0, -90.0, -fl_res)
        longitudes[:] = numpy.arange(-180.0 + fl_res/2.0, 180.0,  fl_res)
        time[:] = tme
        cntr_codes[:] = self.ccodes_asc[:, :]  # Read in the country codes data

        for idx, j in enumerate(time):
            print j
            if j <= 1850:
                data[idx, :, :] = static_map
            elif j <= 2015:
                # SC rate is same as static_map in 1850, it is same as rate_map in 2015
                mod_rate = self.interpolate_asc(static_map, rate_map, 1850, j, 2015)
                data[idx, :, :] = mod_rate
            else:  # Upto 2100
                # SC rate is same as rate_map in 2015 and sc_ma_2100 in 2100
                mod_rate = self.interpolate_asc(static_map, rate_map, 2015, j, 2100)
                data[idx, :, :] = mod_rate

        nc_data.close()
        return out_nc


def use_butler_map():
    obj = ShftCult(use_andreas=False, file_sc=constants.ASC_BUTLER, default_rate=0.067, start_yr=850, end_yr=2015,
                   skiprows=0)
    # Read ASCII File of shifting cultivation (butler ascii map)
    asc_sc = numpy.genfromtxt(self.file_raw_sc, skip_header=skiprows, delimiter=' ')
    asc_binary_sc = asc_sc[:]

    if not constants.shft_by_country:
        obj.combine_country_continent()

    shift_nc_file = obj.make_shifting_cult_nc(ccode=0.0, new_rate=0.033, out_path=constants.out_dir)

    # Shifting cultivation netCDF
    # 1. Read in ascii file of continent and country codes
    # 2. For our 3 key continents, combine with butler map
    # 3. Change shft cult rates

    # Get list of unique countries from country code 2D ascii file
    # Merge continent code and country code (both 1-D) files
    # Double check whether unique countries in 2D file and 1D list match
    # Replace country codes in 2D ascii file by continent codes
    # Apply shifting cultivation algorithm


def use_andreas_map():
    obj = ShftCult(use_andreas=True, file_sc=constants.TIF_ANDREAS, skiprows=0)
    path_nc = obj.create_andreas_nc()
    pdb.set_trace()
    # Plot maps

    ds = util.open_or_die(path_nc)
    lat = ds.variables['latitude'][:]
    lon = ds.variables['longitude'][:]
    imgs_for_movie = plot.plot_maps_ts(path_nc,
                                       'shift_cult', lon, lat,
                                       out_path='C:\\Users\\ritvik\\Documents\\PhD\\Projects\\GLM\\Input\\shift_cult\\andreas\\',
                                       save_name='shift_cult', xlabel='Shifting cultivation frequency on croplands',
                                       title='', land_bg=False, grid=True)
    plot.make_movie(imgs_for_movie, 'C:\\Users\\ritvik\\Documents\\PhD\\Projects\\GLM\\Input\\shift_cult\\andreas\\', out_fname='shift_cult.gif')

    ncc = util.open_or_die('C:\\Users\\ritvik\\Documents\\PhD\\Projects\\GLM\\Input\\shift_cult\\andreas\\andreas.nc')
    crop_data = util.open_or_die('C:\\Users\\ritvik\\Documents\\PhD\\Projects\\GLM\\Input\\LUH\\v0.3_historical\\states.nc')

    tme = numpy.arange(0, 1165 + 1)
    mult = numpy.zeros(len(tme))
    mult_one = numpy.zeros(len(tme))
    cell_area = util.open_or_die(constants.path_GLM_carea)

    nc_data = ncc.variables['shift_cult'][int(1), :, :]
    all_one = numpy.copy(nc_data)
    all_one[all_one > 0.0] = 1.0
    for idx, t in enumerate(tme):
        print t
        nc_data = ncc.variables['shift_cult'][int(t), :, :]

        all_crops = crop_data.variables['c3ann'][int(t), :, :] + crop_data.variables['c4ann'][int(t), :, :] +\
        crop_data.variables['c3per'][int(t), :, :] + crop_data.variables['c4per'][int(t), :, :] +\
        crop_data.variables['c3nfx'][int(t), :, :]

        mult[idx] = numpy.ma.sum(all_crops * nc_data * cell_area)
        mult_one[idx] = numpy.ma.sum(all_crops * all_one * cell_area)
    pdb.set_trace()
    import matplotlib.pyplot as plt
    ax = plt.plot(mult_one, label='butler')
    plt.plot(mult, label='andreas')
    plt.show()


if __name__ == '__main__':
    use_andreas_map()
