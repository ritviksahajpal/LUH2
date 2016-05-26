# create new static data file.
import os
import numpy as np
import pandas as pd
import pdb
import datetime
import math
import palettable

import constants
import GLM.constants as constants_glm
import pygeoutil.util as util
import netCDF4

from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

label_size = 10
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size


def plot_miami_3D(max_temp=40.5, max_prcp=3000.):
    step = 0.5

    temp = np.arange(0.0, max_temp, step)
    prcp = np.arange(0.0, max_prcp, (max_prcp - 0)/(max_temp/step))
    x_surf, y_surf = np.meshgrid(temp, prcp / 10.)

    all = np.transpose([np.tile(temp, len(prcp)), np.repeat(prcp, len(temp))])

    temp = all[:, 0]
    prcp = all[:, 1]
    nppt = 3000. / (1 + np.exp(1.315 - 0.119 * temp))
    nppp = 3000. * (1 - np.exp(-0.000664 * prcp))

    npp = np.minimum(nppt, nppp) * 0.45/1000.

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #ax.scatter(temp, prcp/10., npp, alpha=0.5, color=None)
    #ax.plot(temp, np.repeat(max(prcp / 10.), len(temp)), nppt * 0.45/1000., color='black')
    #ax.plot(np.repeat(max(temp), len(npp)), prcp/10., nppp * 0.45/1000., color='black')

    surf = ax.plot_surface(x_surf, y_surf, np.reshape(npp, (max_temp/step, max_temp/step)), linewidth=0.2,
                           cmap=palettable.colorbrewer.sequential.Greens_9.mpl_colormap)

    ax.set_xlabel('MAT ($^\circ$C)', fontsize=10)
    ax.set_ylabel('MAP ($cm\ yr^{-1}$)', fontsize=10)
    ax.set_zlabel('NPP $(kg\ C\ m^{-2}\ yr^{-1})$', fontsize=10)

    ax.set_xlim3d(0, 40.)
    ax.set_ylim3d(0, max(prcp/10.))
    ax.set_zlim3d(0, 3. * 0.45)

    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, orientation='horizontal')
    cbar.ax.set_xlabel('NPP $(kg\ C\ m^{-2}\ yr^{-1})$')

    for label in cbar.ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    plt.show()

def create_tillage_nc(fao_tillage_yr=1973):
    if os.name == 'nt':
        path_hyde_crop = 'C:\\Users\\ritvik\\Documents\\PhD\\Projects\\GLM\\Input\\lumip_data\\hyde_3.2\\quarter_deg_grids_incl_urban\\gcrop_850_2015_quarterdeg_incl_urban.nc'
        path_till = 'C:\\Users\\ritvik\\Documents\\PhD\\Projects\\GLM\\Input\\Management\\Tillage\\aquastat.xlsx'
        path_cft_frac = 'C:\\Users\\ritvik\\Documents\\PhD\\Projects\\GLM\\Input\\lumip_data\\other\\croptypes\\FAO_CFT_fraction.nc'
    elif os.name == 'posix' or os.name == 'mac':
        # Read in tillage dataset
        path_hyde_crop = '/Volumes/gel1/pukeko_restore/data/hyde3.2_june_23_2015/feb26_2016/hyde32_baseline/processed/gcrop_850_2015_quarterdeg_incl_urban.nc'
        path_till = '/Users/ritvik/Documents/Projects/GLM/Input/Management/Tillage/aquastat.xlsx'
        path_cft_frac = '/Volumes/gel1/data/glm_data/lumip_data/other/croptypes/FAO_CFT_fraction.nc'

    hndl_hyde_crop = util.open_or_die(path_hyde_crop)
    hndl_till = util.open_or_die(path_till)
    hndl_cft_frac = util.open_or_die(path_cft_frac)

    # Country code map
    map_ccodes = np.genfromtxt(constants.CNTRY_CODES, skip_header=0, delimiter=' ')
    carea = util.open_or_die(constants.path_glm_carea)

    df = hndl_till.parse('processed_tillage')

    # Create dataframe with columns from 850 to 1973 with all 0's
    cols = np.arange(850, fao_tillage_yr + 1)
    df_ = pd.DataFrame(index=df.index, columns=cols)
    df_ = df_.fillna(0)  # with 0s rather than NaNs

    # Concatenate all years from 850 to 2015
    df = pd.concat([df_, df], axis=1)

    # Interpolate across the years
    df = pd.concat([df[['country code', 'country name']],
                    df.filter(regex='^8|9|1|2').interpolate(axis=1)], axis=1)
    pdb.set_trace()
    out_nc = constants_glm.path_glm_output + os.sep + 'national_tillage_data_850_2015_new.nc'
    nc_data = util.open_or_die(out_nc, perm='w', format='NETCDF4_CLASSIC')
    nc_data.description = ''

    # dimensions
    tme = np.arange(850, 2015 + 1)

    # country codes
    ccodes = pd.read_csv(constants_glm.ccodes_file, header=None)
    ccodes.columns = ['country code']
    contcodes = pd.read_csv(constants_glm.contcodes_file, header=None)
    contcodes.columns = ['continent code']
    lup_codes = pd.concat([ccodes, contcodes], axis=1)

    nc_data.createDimension('time', np.shape(tme)[0])
    nc_data.createDimension('country', len(ccodes))

    # Populate and output nc file
    time = nc_data.createVariable('time', 'i4', ('time',), fill_value=0.0)
    country = nc_data.createVariable('country', 'i4', ('country',), fill_value=0.0)

    # Assign units and other metadata
    time.units = 'year as %Y.%f'
    time.calendar = 'proleptic_gregorian'
    country.units = 'ISO country code'

    # Assign values to dimensions and data
    time[:] = tme
    country[:] = ccodes.values

    tillage = nc_data.createVariable('tillage', 'f4', ('time', 'country',))
    tillage[:, :] = 0.0  # Assign all values to 0.0
    tillage.units = 'fraction of cropland area'

    # Loop over all countries
    for index, row in lup_codes.iterrows():
        # Find row containing country in df
        row_country = df[df['country code'] == row['country code']]

        if len(row_country):
            cntr = row_country.values[0][0]
            mask_cntr = np.where(map_ccodes == cntr, 1.0, 0.0)
            idx_cntr = np.where(ccodes == cntr)[0][0]

            # Iterate over years
            for idx, yr in enumerate(range(fao_tillage_yr, 2015 + 1)):
                # Get fraction of cell area that is cropland
                crop_frac = hndl_hyde_crop.variables['cropland'][fao_tillage_yr - 850 + idx, :, :]

                # Get fraction of cell area that is (C4 Annual + C3 Annual + N-fixing)
                cft_frac = crop_frac * \
                           (hndl_cft_frac.variables['C4annual'][fao_tillage_yr - 850 + idx, idx_cntr].data +
                            hndl_cft_frac.variables['C3annual'][fao_tillage_yr - 850 + idx, idx_cntr].data +
                            hndl_cft_frac.variables['N-fixing'][fao_tillage_yr - 850 + idx, idx_cntr].data)

                # Subset of cropland area (C4 Annual + C3 Annual + N-fixing) * mask applied to country * cell area
                sum_area = np.ma.sum(cft_frac * carea * mask_cntr)

                # Multiply by 100 to convert numerator from ha to km2
                frac = row_country.values[0][2 + fao_tillage_yr + idx - 850]/(100 * sum_area)
                if frac > 1.0:
                    tillage[idx + fao_tillage_yr - 850, idx_cntr] = 1.0
                else:
                    tillage[idx + fao_tillage_yr - 850, idx_cntr] = frac

    nc_data.close()


def create_biofuel_nc():
    ##########################################################################################################
    # Creating new file on biofuel cft
    ##########################################################################################################
    if os.name == 'nt':
        path_hndl_biof = 'C:\\Users\\ritvik\\Documents\\PhD\\Projects\\GLM\\Input\\Biofuel\\biofuels.xls'
        path_hyde_crop = 'C:\\Users\\ritvik\\Documents\\PhD\\Projects\\GLM\\Input\\lumip_data\\hyde_3.2\\quarter_deg_grids_incl_urban\\gcrop_850_2015_quarterdeg_incl_urban.nc'
        path_cft_frac = 'C:\\Users\\ritvik\\Documents\\PhD\\Projects\\GLM\\Input\\lumip_data\\other\\croptypes\\FAO_CFT_fraction.nc'
    elif os.name == 'posix':
        path_hndl_biof = '/Users/ritvik/Documents/Projects/GLM/Input/Biofuel/biofuels.xls'
        path_hyde_crop = '/Volumes/gel1/pukeko_restore/data/hyde3.2_june_23_2015/feb26_2016/hyde32_baseline/processed/gcrop_850_2015_quarterdeg_incl_urban.nc'
        path_cft_frac = '/Volumes/gel1/data/glm_data/lumip_data/other/croptypes/FAO_CFT_fraction.nc'

    hndl_biof = util.open_or_die(path_hndl_biof)
    hndl_hyde_crop = util.open_or_die(path_hyde_crop)
    hndl_cft_frac = util.open_or_die(path_cft_frac)

    # Determine biofuel area/crop land area per grid cell over time
    sheets_biof = hndl_biof.sheet_names

    # Country code map
    map_ccodes = np.genfromtxt(constants.CNTRY_CODES, skip_header=0, delimiter=' ')

    # List of country codes
    ccodes = pd.read_csv(constants.ccodes_file, header=None)[0].values

    start_cbio_yr = 2000

    # Loop through all crop types
    # If crop type is not present, then set value to nan for all countries in the globe
    # If crop type is present, then set value using excel table for specific countries, others get nan
    # Read in HYDE file or some other file that has cropland information, this will be used to compute cropland fraction
    # Initialize nc file
    out_nc = constants.path_glm_output + os.sep + 'biofuel.nc'
    nc_data = netCDF4.Dataset(out_nc, 'w', 'NETCDF4_CLASSIC')
    nc_data.description = ''

    # dimensions
    START_YR = 850
    tme = np.arange(START_YR, 2015 + 1)

    nc_data.createDimension('time', np.shape(tme)[0])
    nc_data.createDimension('country', len(ccodes))

    # Populate and output nc file
    time = nc_data.createVariable('time', 'i4', ('time',), fill_value=0.0)
    country = nc_data.createVariable('country', 'i4', ('country',), fill_value=0.0)

    # Assign units and other metadata
    time.units = 'year as %Y.%f'
    time.calendar = 'proleptic_gregorian'
    country.units = 'ISO country code'

    # Assign values to dimensions and data
    time[:] = tme
    country[:] = ccodes

    c3ann_cbio_frac = nc_data.createVariable('c3ann_cbio_frac', 'f4', ('time', 'country',))
    c4ann_cbio_frac = nc_data.createVariable('c4ann_cbio_frac', 'f4', ('time', 'country',))
    c3per_cbio_frac = nc_data.createVariable('c3per_cbio_frac', 'f4', ('time', 'country',))
    c4per_cbio_frac = nc_data.createVariable('c4per_cbio_frac', 'f4', ('time', 'country',))
    c3nfx_cbio_frac = nc_data.createVariable('c3nfx_cbio_frac', 'f4', ('time', 'country',))

    c3ann_cbio_frac.units = 'fraction of crop type area occupied by biofuel crops'
    c3ann_cbio_frac.long_name = 'C3 annual crops grown as biofuels'

    c4ann_cbio_frac.units = 'fraction of crop type area occupied by biofuel crops'
    c4ann_cbio_frac.long_name = 'C4 annual crops grown as biofuels'

    c3per_cbio_frac.units = 'fraction of crop type area occupied by biofuel crops'
    c3per_cbio_frac.long_name = 'C3 perennial crops grown as biofuels'

    c4per_cbio_frac.units = 'fraction of crop type area occupied by biofuel crops'
    c4per_cbio_frac.long_name = 'C4 perennial crops grown as biofuels'

    c3nfx_cbio_frac.units = 'fraction of crop type area occupied by biofuel crops'
    c3nfx_cbio_frac.long_name = 'C3 nitrogen-fixing crops grown as biofuels'

    carea = util.open_or_die(constants.path_glm_carea)

    # Assign all values to 0.0
    c4ann_cbio_frac[:, :] = 0.0
    c4per_cbio_frac[:, :] = 0.0
    c3nfx_cbio_frac[:, :] = 0.0
    c3ann_cbio_frac[:, :] = 0.0
    c3per_cbio_frac[:, :] = 0.0

    # [u'country code', u'country name', 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
    # C4ANN
    print 'C4ANN'
    df_c4ann = hndl_biof.parse('c4ann')
    for row in df_c4ann.iterrows():
        for idx, yr in enumerate(range(start_cbio_yr, 2015 + 1)):
            crop_frac = hndl_hyde_crop.variables['cropland'][start_cbio_yr - 850 + idx, :, :]
            for idy, cntr in enumerate(ccodes):
                if cntr == row[1]['country code']:
                    global_cft_frac = crop_frac * hndl_cft_frac.variables['C4annual'][start_cbio_yr - 850 + idx, np.where(ccodes == cntr)[0][0]].data
                    mask_cntr = np.where(map_ccodes == cntr, 1.0, 0.0)

                    sum_area = (global_cft_frac * carea * mask_cntr).sum()
                    frac = row[1][yr]/sum_area
                    if frac <= 1.0:
                        c4ann_cbio_frac[idx + start_cbio_yr - START_YR, np.where(ccodes == cntr)[0][0]] = row[1][yr]/sum_area  # maize
                    else:
                        if row[1][yr] / (crop_frac * carea * mask_cntr).sum() <= 1.0:
                            c4ann_cbio_frac[idx + start_cbio_yr - START_YR, np.where(ccodes == cntr)[0][0]] = row[1][yr]/(crop_frac * carea * mask_cntr).sum()
                        else:
                            c4ann_cbio_frac[idx + start_cbio_yr - START_YR, np.where(ccodes == cntr)[0][0]] = 1.0

    # C4PER
    print 'C4PER'
    df_c4per = hndl_biof.parse('c4per')
    for row in df_c4per.iterrows():
        for idx, yr in enumerate(range(start_cbio_yr, 2015 + 1)):
            crop_frac = hndl_hyde_crop.variables['cropland'][start_cbio_yr - 850 + idx, :, :]
            for idy, cntr in enumerate(ccodes):
                if cntr == row[1]['country code']:
                    global_cft_frac = crop_frac * hndl_cft_frac.variables['C4perennial'][start_cbio_yr - 850 + idx, np.where(ccodes == cntr)[0][0]].data
                    mask_cntr = np.where(map_ccodes == cntr, 1.0, 0.0)

                    sum_area = (global_cft_frac * carea * mask_cntr).sum()
                    frac = row[1][yr]/sum_area

                    if frac <= 1.0:
                        c4per_cbio_frac[idx + start_cbio_yr - START_YR, np.where(ccodes == cntr)[0][0]] = row[1][yr]/sum_area  # sugarcane
                    else:
                        if row[1][yr]/(crop_frac * carea * mask_cntr).sum() <= 1.0:
                            c4per_cbio_frac[idx + start_cbio_yr - START_YR, np.where(ccodes == cntr)[0][0]] = row[1][yr]/(crop_frac * carea * mask_cntr).sum()
                        else:
                            c4per_cbio_frac[idx + start_cbio_yr - START_YR, np.where(ccodes == cntr)[0][0]] = 1.0
    # C3NFX
    df_c3nfx = hndl_biof.parse('c3nfx')
    print 'C3NFX'
    for row in df_c3nfx.iterrows():
        for idx, yr in enumerate(range(start_cbio_yr, 2015 + 1)):
            crop_frac = hndl_hyde_crop.variables['cropland'][start_cbio_yr - 850 + idx, :, :]
            for idy, cntr in enumerate(ccodes):
                if cntr == row[1]['country code']:
                    global_cft_frac = crop_frac * hndl_cft_frac.variables['N-fixing'][start_cbio_yr - 850 + idx, np.where(ccodes == cntr)[0][0]].data
                    mask_cntr = np.where(map_ccodes == cntr, 1.0, 0.0)

                    sum_area = (global_cft_frac * carea * mask_cntr).sum()
                    frac = row[1][yr]/sum_area
                    if frac <= 1.0:
                        c3nfx_cbio_frac[idx + start_cbio_yr - START_YR, np.where(ccodes == cntr)[0][0]] = frac  # soybean
                    else:
                        if row[1][yr] / (crop_frac * carea * mask_cntr).sum() <= 1.0:
                            c3nfx_cbio_frac[idx + start_cbio_yr - START_YR, np.where(ccodes == cntr)[0][0]] = row[1][yr]/(crop_frac * carea * mask_cntr).sum()
                        else:
                            c3nfx_cbio_frac[idx + start_cbio_yr - START_YR, np.where(ccodes == cntr)[0][0]] = 1.0

    nc_data.close()


def add_bounds_to_nc(path_inp_nc):
    """

    :param path_inp_nc:
    :return:
    """
    # input file
    print path_inp_nc
    src = netCDF4.Dataset(path_inp_nc, mode='r+')

    # copy the data for lats and lons
    lats = src.variables['lat'][:]
    lons = src.variables['lon'][:]

    # Create bounds dimension
    src.createDimension('bounds', 2)
    out_var = src.createVariable('lat_bounds', 'f8', ('lat', 'bounds',))
    out_var[:] = np.vstack((lats - 0.5 * (lats[1] - lats[0]), lats + 0.5 * (lats[1] - lats[0]))).T
    out_var = src.createVariable('lon_bounds', 'f8', ('lon', 'bounds',))
    out_var[:] = np.vstack((lons - 0.5 * (lons[1] - lons[0]), lons + 0.5 * (lons[1] - lons[0]))).T

    # close the file
    src.close()


def create_static_info_nc():
    ##########################################################################################################
    # Creating new static info file
    ##########################################################################################################
    if os.name == 'nt':
        path_inp_file = 'C:\\Users\\ritvik\\Documents\\PhD\\Projects\\GLM\\Input\\staticData_quarterdeg.nc'
        path_out_file = 'C:\\Users\\ritvik\\Documents\\PhD\\Projects\\GLM\\Input\\staticData_quarterdeg_out.nc'
    elif os.name == 'posix' or os.name == 'mac':
        path_inp_file = '/Users/ritvik/Documents/Projects/GLM/Input/LUH/v0.1/staticData_quarterdeg.nc'
        path_out_file = '/Users/ritvik/Documents/Projects/GLM/Input/LUH/v0.1/staticData_quarterdeg_out.nc'

    vba = util.open_or_die(constants.path_glm_vba)
    vba = vba / constants.AGB_TO_BIOMASS
    vba[vba < 0.0] = 0.0

    asc_vba = np.copy(vba)
    asc_vba[asc_vba < 2.0] = 0.0  # Biomass defn of forest: > 2.0 kg C/m^2
    asc_vba[asc_vba > 0.0] = 1.0
    fnf = asc_vba  # Boolean ascii file indicating whether it is forest(1.0)/non-forest(0.0)

    # copy netCDF
    # http://guziy.blogspot.com/2014/01/netcdf4-python-copying-variables-from.html
    now = datetime.datetime.now()

    # input file
    dsin = netCDF4.Dataset(path_inp_file)

    # output file
    dsout = netCDF4.Dataset(path_out_file, 'w')

    # Copy dimensions
    for dname, the_dim in dsin.dimensions.iteritems():
        dsout.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)

    # Copy variables
    for v_name, varin in dsin.variables.iteritems():
        print v_name
        if v_name == 'lat' or v_name == 'lon':
            outVar = dsout.createVariable(v_name, 'f8', varin.dimensions)
        else:
            outVar = dsout.createVariable(v_name, 'f4', varin.dimensions)

        # Copy variable attributes
        # outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})

        if v_name == 'ptbio':
            outVar[:] = vba[:]
            outVar.missing_value = 1e20
            outVar._fillvalue = 1e20
            outVar.standard_name = 'vegetation_carbon_content'
            outVar.long_name = 'potential biomass carbon content'
            outVar.units = 'kg m-2'
        elif v_name == 'fstnf':
            outVar[:] = fnf[:]
            outVar.missing_value = 1e20
            outVar._fillvalue = 1e20
            outVar.standard_name = ''
            outVar.long_name = 'mask denoting forest (1) or non-forest (0)'
            outVar.units = '1'
        elif v_name == 'carea':
            outVar[:] = varin[:]
            outVar.missing_value = 1e20
            outVar._fillvalue = 1e20
            outVar.standard_name = ''
            outVar.long_name = 'area of grid cell'
            outVar.units = 'km2'
        elif v_name == 'icwtr':
            outVar[:] = varin[:]
            outVar.missing_value = 1e20
            outVar._fillvalue = 1e20
            outVar.standard_name = 'area_fraction'
            outVar.long_name = 'ice/water fraction'
            outVar.units = '1'
        elif v_name == 'ccode':
            outVar[:] = varin[:]
            outVar._fillvalue = 1e20
            outVar.missing_value = 1e20
            outVar.standard_name = ''
            outVar.long_name = 'country codes'
            outVar.units = '1'
        elif v_name == 'lon':
            outVar[:] = varin[:]
            outVar.missing_value = 1e20
            outVar._fillvalue = 1e20
            outVar.standard_name = 'longitude'
            outVar.long_name = 'longitude'
            outVar.units = 'degrees_east'
            outVar.axis = 'X'
        elif v_name == 'lat':
            outVar[:] = varin[:]
            outVar.missing_value = 1e20
            outVar._fillvalue = 1e20
            outVar.standard_name = 'latitude'
            outVar.long_name = 'latitude'
            outVar.units = 'degrees_north'
            outVar.axis = 'Y'
        else:
            print '***'
            outVar[:] = varin[:]

    # Write global variables
    dsout.history = 'Processed: ' + str(now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    dsout.host = 'UMD College Park'
    dsout.comment = 'LUH2'
    dsout.contact = 'gchurtt@umd.edu, lchini@umd.edu, steve.frolking@unh.edu, ritvik@umd.edu'
    dsout.creation_date = now.strftime("%Y %m %d %H:%M")
    dsout.title = 'Land Use Data Sets'
    dsout.activity_id = 'input4MIPs'
    dsout.Conventions = 'CF-1.6'
    dsout.data_structure = 'grid'
    dsout.source = 'LUH2-0-1: Land-Use Harmonization Data Set'
    dsout.source_id = 'LUH2-0-1'
    dsout.license = 'MIT'
    dsout.further_info_url = 'http://luh.umd.edu'
    dsout.frequency = 'yr'
    dsout.instituition = 'University of Maryland College Park'
    dsout.realm = 'land'
    dsout.references = 'Hurtt, Chini et al. 2011'

    # close the output file
    dsout.close()

if __name__ == '__main__':
    plot_miami_3D()
    if os.name == 'nt':
        path = 'C:\\Users\\ritvik\\Documents\\PhD\\Projects\\GLM\\Input\\LUH\\test_double_backup\\'
    elif os.name == 'mac' or os.name == 'posix':
        path = '/Users/ritvik/Documents/Projects/GLM/Input/LUH/test_double_backup/'
    add_bounds_to_nc(path + os.sep + 'transitions.nc')
    #add_bounds_to_nc(path + os.sep + 'management.nc')
    #add_bounds_to_nc(path + os.sep + 'transition.nc')
    #create_static_info_nc()

