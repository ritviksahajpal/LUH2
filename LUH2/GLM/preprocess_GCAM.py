import logging
import numpy as np

import pandas

import constants
import pygeoutil.util as util
from preprocess_IAM import IAM

class GCAM(IAM):
    """
    Class for GCAM
    """
    def __init__(self, path_nc):
        IAM.__init__(self, 'GCAM', path_nc)

        self.land_var = 'landcoverpercentage'

        # Input file paths
        self.path_pas = constants.gcam_dir + constants.PASTURE[0]
        self.path_frt = constants.gcam_dir + constants.FOREST[0]
        self.path_urb = constants.gcam_dir + constants.URBAN[0]
        self.path_wh  = constants.gcam_dir + constants.WOOD_HARVEST
        self.path_ftz = constants.gcam_dir + constants.FERT_DATA

        # Output directories
        self.gcam_out_fl   = constants.out_dir+constants.GCAM_OUT
        self.perc_crops_fl = constants.out_dir+constants.GCAM_CROPS

    def AEZ_to_national_GCAM(self, data_source = 'wh', out_nc_name = ''):
        """
        :param data_source:
        :param out_nc_name:
        :return:
        """
        # Create a dictionary mapping GCAM AEZ's to regions
        if data_source == 'wh':
            df = util.open_or_die(self.path_wh)
        elif data_source == 'ftz': # fertilizer data
            df = util.open_or_die(self.path_ftz)

        # Insert columns so that we have data for each year
        idx = 1
        for yr in xrange(constants.GCAM_START_YR + 1, constants.GCAM_END_YR):
            # Skip years for which we already have data i.e. multiples of constants.GCAM_STEP_YR
            if yr%constants.GCAM_STEP_YR != 0:
                df.insert(constants.SKIP_GCAM_COLS + idx, str(yr), np.nan)
            idx += 1

        # Extract columns with information on GCAM regions
        gcam_df = df[['region', 'subsector']]

        # Fill in missing values, note that using interpolate
        df = df.ix[:, str(constants.GCAM_START_YR):str(constants.GCAM_END_YR)]
        df = df.interpolate(axis = 1)

        # Concatenate
        df = pandas.concat([gcam_df, df], axis=1, join='inner')

        # Extract "Russia", "Central Asia", "EU-12", and "Europe-Eastern"  into a single larger region with region code 33
        merg_df = df.loc[df['region'].isin(['Russia', 'Central Asia', 'EU-12', 'Europe-Eastern'])]

        # Create a new row with data for USSR or region code 33
        new_row = ['USSR', 'subsector']
        new_row.extend(merg_df.ix[:, 2:].sum().tolist())

        # Add newly created row to dataframe
        df.loc[len(df.index)] = np.array(new_row)
        # Group dataframe by region
        df = df.groupby('region').sum()
        # Remove the subsector column since it interferes with netCDF creation later
        df.drop('subsector', axis=1, inplace=True)

        # Read in GCAM region country mapping
        xdf = util.open_or_die(constants.gcam_dir + constants.GCAM_MAPPING)
        map_xdf = xdf.parse("Sheet1")
        df_dict = dict((z[0],list(z[1:])) for z in zip(map_xdf['country ISO code'], map_xdf['Modified GCAM Regions'],
                                                       map_xdf['GCAM REGION NAME'], map_xdf['country-to-region WH ratios']))

        # Create WH output netCDF
        onc = util.open_or_die(constants.out_dir + out_nc_name + '_' + str(constants.GCAM_START_YR) + '_' +
                               str(constants.GCAM_END_YR) + '.nc', 'w')

        # dimensions
        onc.createDimension('country_code', len(df_dict.keys()))
        onc.createDimension('time', constants.GCAM_END_YR - constants.GCAM_START_YR + 1)

        # variables
        country_code = onc.createVariable('country_code', 'i4', ('country_code',))
        time         = onc.createVariable('time', 'i4', ('time',))
        data         = onc.createVariable(out_nc_name, 'f4', ('country_code', 'time',))

        # Metadata
        country_code.long_name     = 'country_code'
        country_code.units         = 'index'
        country_code.standard_name = 'country_code'

        time.units    = 'year as %Y.%f'
        time.calendar = 'proleptic_gregorian'

        if data_source == 'wh':
            data.units     = 'MgC'
            data.long_name = 'wood harvest carbon'
        elif data_source == 'ftz':
            print 'TODO!!'

        # Assign data
        time[:]         = np.arange(constants.GCAM_START_YR, constants.GCAM_END_YR + 1)
        country_code[:] = sorted(df_dict.keys())

        for idx, ctr in enumerate(country_code[:]):
            # Get GCAM region corresponding to country
            gcam_reg = df_dict.get(ctr)[1] # GCAM region identifier
            gcam_mul = df_dict.get(ctr)[2] # GCAM country-to-region WH ratios

            try:
                # @TODO: Need to finalize woodharvest calculation
                # @TODO: Generalize for data other than wood harvest
                data[idx, :] = df.ix[gcam_reg].values.astype(float) * 0.225 * constants.BILLION * gcam_mul
                # @TODO: Multiply by 1.3 to account for slash fraction
            except:
                data[idx, :] = np.zeros(len(time[:]))

        onc.close()

    def create_GCAM_croplands(self, nc):
        """
        :param nc: Empty 3D numpy array (yrs,ny,nx)
        :return nc: 3D numpy array containing SUM of all GCAM cropland percentages
        """
        # Iterate over all crop categories and add the self.land_var data
        for i in range(len(constants.CROPS)):
            print('Processing: ' + constants.CROPS[i])
            logging.info('Processing: ' + constants.CROPS[i])

            ds = util.open_or_die(constants.gcam_dir+constants.CROPS[i])
            for j in range(len(self.time)):
                nc[j,:,:] += ds.variables[self.land_var][j,:,:].data
            ds.close()

        # @TODO: Test whether sum of all self.land_var in a given year is <= 1.0

        return nc

    def create_nc_perc_croplands(self, sum_nc, shape):
        """
        Create netcdf file with each crop category represented as fraction of cropland
        area and not total grid cell area

        :param sum_nc: netCDF file containing 'croplands' which is fraction of area
                       of cell occupied by all croplands
        :param shape: Tuple containing dimensions of netCDF (yrs, ny, nx)
        :return: None
        """
        print 'Creating cropland nc'
        logging.info('Creating cropland nc')

        inc = util.open_or_die(sum_nc)
        onc = util.open_or_die(self.perc_crops_fl, 'w')

        onc.description = 'crops_as_fraction_of_croplands'

        # dimensions
        onc.createDimension('time',shape[0])
        onc.createDimension('lat', shape[1])
        onc.createDimension('lon', shape[2])


        # variables
        time       = onc.createVariable('time', 'i4', ('time',))
        latitudes  = onc.createVariable('lat', 'f4', ('lat',))
        longitudes = onc.createVariable('lon', 'f4', ('lon',))

        # Metadata
        latitudes.units          = 'degrees_north'
        latitudes.standard_name  = 'latitude'
        longitudes.units         = 'degrees_east'
        longitudes.standard_name = 'longitude'

        # Assign time
        time[:] = self.time

        # Assign lats/lons
        latitudes[:]  = self.lat
        longitudes[:] = self.lon

        # Assign data
        for i in range(len(constants.CROPS)):
            print '\t'+constants.CROPS[i]
            onc_var = onc.createVariable(constants.CROPS[i], 'f4', ('time', 'lat', 'lon',),fill_value=np.nan)
            onc_var.units = 'percentage'

            ds = util.open_or_die(constants.gcam_dir+constants.CROPS[i])
            # Iterate over all years
            for j in range(shape[0]):
                onc_var[j,:,:] = ds.variables[self.land_var][j,:,:].data / inc.variables['cropland'][j,:,:]

            ds.close()

        # @TODO: Copy metadata from original GCAM netcdf
        onc.close()

    def write_GCAM_nc(self, isum_perc, shape):
        """
        :param isum_perc: Sum of self.land_var values for all crop classes
        :param shape: Tuple containing dimensions of netCDF (yrs, ny, nx)
        :return: Nothing, side-effect is to create a netCDF file with each crop category
                represented as fraction of cropland area and not total grid cell area
        """
        print 'Creating GCAM file'
        logging.info('Creating GCAM file')

        # Read in netCDF datasets
        ids_pas  = util.open_or_die(self.path_pas)
        ids_frt  = util.open_or_die(self.path_frt)
        ids_urb  = util.open_or_die(self.path_urb)

        iam_nc = util.open_or_die(self.gcam_out_fl, perm = 'w')
        iam_nc.description = 'GCAM'

        # dimensions
        iam_nc.createDimension('time',shape[0])
        iam_nc.createDimension('lat', shape[1])
        iam_nc.createDimension('lon', shape[2])

        # variables
        time       = iam_nc.createVariable('time', 'i4', ('time',))
        latitudes  = iam_nc.createVariable('lat', 'f4', ('lat',))
        longitudes = iam_nc.createVariable('lon', 'f4', ('lon',))
        crp        = iam_nc.createVariable('cropland', 'f4', ('time', 'lat', 'lon',),fill_value=np.nan)
        pas        = iam_nc.createVariable('pasture', 'f4', ('time', 'lat', 'lon',),fill_value=np.nan)
        frt        = iam_nc.createVariable('forest', 'f4', ('time', 'lat', 'lon',),fill_value=np.nan)
        urb        = iam_nc.createVariable('urban', 'f4', ('time', 'lat', 'lon',),fill_value=np.nan)

        # Metadata
        crp.units = 'percentage'
        pas.units = 'percentage'
        frt.units = 'percentage'
        urb.units = 'percentage'

        latitudes.units          = 'degrees_north'
        latitudes.standard_name  = 'latitude'
        longitudes.units         = 'degrees_east'
        longitudes.standard_name = 'longitude'

        # Assign time
        time[:] = self.time

        # Assign lats/lons
        latitudes[:]  = self.lat
        longitudes[:] = self.lon

        # Assign data to new netCDF file
        for i in range(len(self.time)):
            crp[i,:,:] = isum_perc[i,:,:]
            pas[i,:,:] = ids_pas.variables[self.land_var][i,:,:].data
            frt[i,:,:] = ids_frt.variables[self.land_var][i,:,:].data
            urb[i,:,:] = ids_urb.variables[self.land_var][i,:,:].data

        # @TODO: Copy metadata from original GCAM netcdf

        ids_pas.close()
        ids_frt.close()
        ids_urb.close()
        iam_nc.close()


