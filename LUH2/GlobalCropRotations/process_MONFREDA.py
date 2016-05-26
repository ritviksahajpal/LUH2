from __future__ import print_function, division
import GLM.constants, os, pdb, pandas, numpy, logging, crop_stats
import pygeoutil.util as util


class CropFunctionalTypes:
    """

    """
    def __init__(self, res='q'):
        """
        :param res: Resolution of output dataset: q=quarter, h=half, o=one
        :return:
        """
        # Dictionary of crop functional types
        self.cft = {'C4Annual': ['maize.asc', 'millet.asc', 'sorghum.asc'],
                    'C4Perren': ['sugarcane.asc'],
                    'C3Perren': ['banana.asc', 'berry.asc', 'citrus.asc', 'fruittree.asc', 'grape.asc', 'palm.asc', 'tropevrgrn.asc'],
                    'Ntfixing': ['alfalfa.asc', 'bean.asc', 'legumehay.asc', 'peanut.asc', 'soybean.asc'],
                    'C3Annual': ['beet.asc', 'cassava.asc', 'cotton.asc', 'flax.asc', 'hops.asc', 'mixedcover.asc',
                                 'nursflower.asc', 'oat.asc', 'potato.asc', 'rapeseed.asc', 'rice.asc', 'rye.asc',
                                 'safflower.asc', 'sunflower.asc', 'tobacco.asc', 'vegetable.asc', 'wheat.asc'],
                    'TotlRice': ['rice.asc', 'xrice.asc']}

        # Get shape of file
        self.skiprows = 6
        self.res = res
        self.tmpdata = util.open_or_die(path_file=GLM.constants.MFD_DATA_DIR + os.sep + 'maize.asc',
                                            skiprows=self.skiprows, delimiter=' ')
        self.asc_hdr = util.get_ascii_header(path_file=GLM.constants.MFD_DATA_DIR + os.sep + 'maize.asc',
                                                 getrows=self.skiprows)
        self.yshape = self.tmpdata.shape[0]
        self.xshape = self.tmpdata.shape[1]

        # Create empty numpy arrays
        self.c4annual = numpy.zeros(shape=(self.yshape, self.xshape))
        self.c4perren = numpy.zeros(shape=(self.yshape, self.xshape))
        self.c3perren = numpy.zeros(shape=(self.yshape, self.xshape))
        self.ntfixing = numpy.zeros(shape=(self.yshape, self.xshape))
        self.c3annual = numpy.zeros(shape=(self.yshape, self.xshape))
        self.totlrice = numpy.zeros(shape=(self.yshape, self.xshape))
        self.totlcrop = numpy.zeros(shape=(self.yshape, self.xshape))

        self.c4anarea = numpy.zeros(shape=(self.yshape, self.xshape))
        self.c4prarea = numpy.zeros(shape=(self.yshape, self.xshape))
        self.c3prarea = numpy.zeros(shape=(self.yshape, self.xshape))
        self.ntfxarea = numpy.zeros(shape=(self.yshape, self.xshape))
        self.c3anarea = numpy.zeros(shape=(self.yshape, self.xshape))
        self.croparea = numpy.zeros(shape=(self.yshape, self.xshape))

        # Area of each cell in Monfreda dataset
        self.mfd_area = numpy.zeros(shape=(self.yshape, self.xshape))

        # Ice-water fraction and other static data
        self.icwtr = util.open_or_die(GLM.constants.path_GLM_stat)

        # Read in area file based on res
        if res == 'q':
            self.area_data = util.open_or_die(path_file=GLM.constants.CELL_AREA_Q)
        elif res == 'h':
            self.area_data = util.open_or_die(path_file=GLM.constants.CELL_AREA_H)
        elif res == 'o':
            self.area_data = util.open_or_die(path_file=GLM.constants.CELL_AREA_O)
        else:
            logging.info('Incorrect resolution for output of Monfreda')

        # Compute cell area (excluding ice-water fraction)
        self.cell_area = util.open_or_die(GLM.constants.path_GLM_carea)
        self.land_area = self.cell_area * (1.0 - self.icwtr.variables[GLM.constants.ice_water_frac][:, :])

        # Get FAO country concordance list
        self.fao_id = pandas.read_csv(GLM.constants.FAO_CONCOR)[['Country_FAO', 'ISO']]

        # Output path
        self.out_path = GLM.constants.out_dir + os.sep + 'Monfreda'
        util.make_dir_if_missing(self.out_path)

    def read_monfreda(self):
        # Loop over crop functional types
        for key, value in self.cft.iteritems():
            for val in value:
                logging.info('Processing ' + key + ' ' + val)
                tmp_asc = util.open_or_die(path_file=GLM.constants.MFD_DATA_DIR + os.sep + val,
                                               skiprows=self.skiprows, delimiter=' ')

                if key == 'C4Annual':
                    self.c4annual = self.c4annual + tmp_asc
                elif key == 'C4Perren':
                    self.c4perren = self.c4perren + tmp_asc
                elif key == 'C3Perren':
                    self.c3perren = self.c3perren + tmp_asc
                elif key == 'Ntfixing':
                    self.ntfixing = self.ntfixing + tmp_asc
                elif key == 'C3Annual':
                    self.c3annual = self.c3annual + tmp_asc
                elif key == 'TotlRice':
                    self.totlrice = self.totlrice + tmp_asc
                else:
                    logging.info('Wrong key')

                # Add to total crop fraction of grid cell area
                self.totlcrop = self.totlcrop + tmp_asc

        # Aggregate MONFREDA data from 5' to 0.25 degree
        self.totlcrop = util.avg_np_arr(self.totlcrop, block_size=3)
        self.croparea = self.totlcrop * self.land_area

        # Aggregate MONFREDA data for each CFT from 5' to 0.25 degree
        self.c4anarea = util.avg_np_arr(self.c4annual, block_size=3) * self.land_area
        self.c4prarea = util.avg_np_arr(self.c4perren, block_size=3) * self.land_area
        self.c3prarea = util.avg_np_arr(self.c3perren, block_size=3) * self.land_area
        self.ntfxarea = util.avg_np_arr(self.ntfixing, block_size=3) * self.land_area
        self.c3anarea = util.avg_np_arr(self.c3annual, block_size=3) * self.land_area

    def read_HYDE(self):
        pass

    def output_ascii_to_file(self, fl_name, data, delim=' '):
        asc_file = open(fl_name, 'w+')

        if self.res == 'q':
            ncols = 720
            nrows = 360
            cell_size = 0.25
        elif self.res == 'h':
            # @TODO
            pass
        elif self.res == 'o':
            # @TODO
            pass

        asc_file.write('ncols         %s\n' % ncols)
        asc_file.write('nrows         %s\n' % nrows)
        asc_file.write('xllcorner     -180\n')
        asc_file.write('yllcorner     -90\n')
        asc_file.write('cellsize         %s\n' % cell_size)
        asc_file.write('NODATA_value  -9999\n')

        # Write numpy array
        numpy.savetxt(asc_file, data, delimiter=delim)
        asc_file.close()

    def compute_area_by_country(self):
        """
        Output area of each CFT by country
        :return:
        """
        df = []
        # Read ASCII file of country codes
        ccodes_fl = numpy.genfromtxt(GLM.constants.CNTRY_CODES, skip_header=0, delimiter=' ')

        # Get list of unique countries, remove 0.0 as it is not a country ID
        list_cntrs = numpy.unique(ccodes_fl)
        list_cntrs = list_cntrs[list_cntrs > 0.0]

        # For each country:
        for cnt in list_cntrs:
            # Get area of cropland in country based on MONFREDA data
            area_cnt = self.croparea[ccodes_fl[:] == cnt].sum()
            area_c4a = self.c4anarea[ccodes_fl[:] == cnt].sum()
            area_c4p = self.c4prarea[ccodes_fl[:] == cnt].sum()
            area_c3p = self.c3prarea[ccodes_fl[:] == cnt].sum()
            area_ntf = self.ntfxarea[ccodes_fl[:] == cnt].sum()
            area_c3a = self.c3anarea[ccodes_fl[:] == cnt].sum()

            # Get country name from concordance table
            cnt_name = self.fao_id[self.fao_id['ISO'] == int(cnt)]['Country_FAO'].iloc[0]
            df.append({'ISO': int(cnt),
                       'Country_Monfreda': cnt_name,
                       'Area_Monfreda': area_cnt,
                       'Monfreda_c4annual': area_c4a,
                       'Monfreda_c4perren': area_c4p,
                       'Monfreda_c3perren': area_c3p,
                       'Monfreda_ntfixing': area_ntf,
                       'Monfreda_c3annual': area_c3a})

        return pandas.DataFrame(df)

if __name__ == '__main__':
    # Logging
    LOG_FILENAME = constants.log_dir + os.sep + 'Log_' + constants.TAG + '.txt'
    logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO, filemode='w',\
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',\
                    datefmt="%m-%d %H:%M") # Logging levels are DEBUG, INFO, WARNING, ERROR, and CRITICAL
    # Add a rotating handler
    logging.getLogger().addHandler(logging.handlers.RotatingFileHandler(LOG_FILENAME, maxBytes=50000, backupCount=5))
    # Output to screen
    logging.getLogger().addHandler(logging.StreamHandler())

    obj = CropFunctionalTypes(res='q')
    obj.read_monfreda()
    df = obj.compute_area_by_country()
    df.to_csv(obj.out_path + os.sep + 'Monfreda_area_crops.csv')

    # Output crop area to file
    obj.output_ascii_to_file(fl_name=obj.out_path + os.sep + 'MONFREDA_crop_area.asc', data=obj.croparea, delim=' ')

    # Read in FAO data for entire time period
    fao_obj = crop_stats.CropStats()
    fao_obj.process_crop_stats()

    # Compare FAO and Monfreda
    # fao_df.FAO_perc_all_df

    # Convert from ascii files to our crop functional types

    # Aggregate ascii files to quarter degree

    # Produce netCDF file as output
    # Currently, we compute crop type ratio's for the year 2000 based on Monfreda data crop type ratios and ag area
    # from HYDE. This gives us a static number per CFT per grid cell. However, sometimes, the crop type ratio might be 0
    # because Monfreda says that no crop exists, but HYDE says that crops do exist.
    # New approach will:
    # 1. Compute CFT fraction per grid cell in 2000
    # 2. Aggregate this fractional value by country
    # 3. Apply this fractional value for grid cells in the past where Monfreda says that no data exists. The values are
    #    assigned by country

    # 1. Read in country codes file
    # 2. Determine list of unique countries
    # 3. For each country:
    # 3a.   Compute fraction of each CFT for that particular country in 2000 by averaging
    # 4. Use the HYDE gcrop dataset so that for each grid cell where a crop is present, we provide either:
    #    a. Fraction as determined through Monfreda
    #    b. Country-specific fraction for grid cells where Monfreda says that there is no cropping
    # 5. Output is a netCDF file with dimensions: time x lat x lon which provides CFT fraction for 5 CFTs across time
    #    and for each grid cell

    # 1) We need to replace that 1/5th for each crop type with something more reasonable from the FAO data. So, we need
    # to get FAO national ratios for each crop type for years 1961-2014 and then perhaps hold constant for 2015 and use
    # 1961-1965 averages for years prior to 1961. This will give us a 199 x 5 x 516 file (could be netcdf or other.

    # 2) The next step is a bit more complicated. In this step we need to use the FAO data to modify the Monfreda maps
    # for years before/after 2000. I think the first step here is to compare Monfreda and FAO national crop type ratios
    # in year 2000. We can then 'normalize' the FAO values so that in the year 2000 FAO national ratios are the same as
    # Monfreda. Using these normalized values we can modify the Monfreda map forwards and backwards in time (throughout
    # the FAO period) by changing all grid-cells within a country using the new FAO ratios. Prior to 1961 we would hold
    # the map constant. The input to GLM would be a lat x lon x time x 5 file.

    # This second step is definitely more complicated so we just need to break it down into manageable chunks. I would
    # not worry about forming a data cube at this point. We first need to be looking at the data and seeing how we can
    # connect the FAO and Monfreda data together.