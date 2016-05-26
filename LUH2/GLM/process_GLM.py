import collections
import datetime
import itertools
import logging
import os
import pdb
import sys
import subprocess
import numpy as np
import pandas
import palettable
import time
import deco

from collections import OrderedDict
from enum import Enum
from tqdm import tqdm
import matplotlib.pyplot as plt

import constants
import pygeoutil.util as util
import plot
import process_HYDE

import warnings
from tempfile import mkdtemp

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    from joblib import Memory
    memory = Memory(cachedir=mkdtemp(), verbose=0)

# Logging.
cur_flname = os.path.splitext(os.path.basename(__file__))[0]
LOG_FILENAME = constants.log_dir + os.sep + 'Log_' + cur_flname + '.txt'
util.make_dir_if_missing(constants.log_dir)
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, filemode='w',
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt="%m-%d %H:%M")  # Logging levels are DEBUG, INFO, WARNING, ERROR, and CRITICAL
# Output to screen
logger = logging.getLogger(cur_flname)
logger.addHandler(logging.StreamHandler(sys.stdout))


class LU(Enum):
    """
    Land-use types (main).
    """
    __order__ = 'vrgn scnd urban crop pastr all'
    vrgn, scnd, urban, crop, pastr, all = range(6)


class SubLU(Enum):
    """
    Sub land-use types
    """
    __order__ = 'primf primn secdf secdn urban c3ann c4ann c3per c4per c3nfx pastr range'
    primf, primn, secdf, secdn, urban, c3ann, c4ann, c3per, c4per, c3nfx, pastr, range = range(12)


class glm:

    def __init__(self, name_exp='', path_input='', time_period='all', out_path='', lon_name='lon', lat_name='lat',
                 tme_name='time', start_year=-1):
        """
        Constructor
        :param name_exp
        :param path_input
        :param time_period:
        :param out_path: Output path where outputs are stored
        :param lon_name: Name of longitude dimension
        :param lat_name: Name of latitude dimension
        :param tme_name: Name of time dimension
        :param start_year: Starting year to extract from netCDF data
        """
        # Land use variables
        # NOTE!!! Do not change dictionary key names, it will cause errors in plot_glm_time_series
        self.lus = OrderedDict([(SubLU.primf.name, 'Forested primary'),
                                (SubLU.primn.name, 'Non-forested primary'),
                                (SubLU.secdf.name, 'Forested secondary'),
                                (SubLU.secdn.name, 'Non-forested secondary'),
                                (SubLU.c3ann.name, 'C3 annual'),
                                (SubLU.c4ann.name, 'C4 annual'),
                                (SubLU.c4per.name, 'C4 perennial'),
                                (SubLU.c3per.name, 'C3 perennial'),
                                (SubLU.c3nfx.name, 'C3 N-fixing'),
                                (SubLU.pastr.name, 'Managed pasture'),
                                (SubLU.range.name, 'Rangeland'),
                                (SubLU.urban.name, 'Urban')])

        self.glm = collections.OrderedDict()
        self.glm[SubLU.urban.name] = 'Urban'
        self.glm[LU.crop.name] = 'Crop'
        self.glm[SubLU.pastr.name] = 'Pasture'
        self.glm[SubLU.range.name] = 'Rangeland'
        self.glm[SubLU.secdn.name] = 'Secondary Non-Forest'
        self.glm[SubLU.secdf.name] = 'Secondary Forest'
        self.glm[SubLU.primn.name] = 'Primary Non-Forest'
        self.glm[SubLU.primf.name] = 'Primary Forest'

        self.ignore_nc_vars = ['time', 'lat', 'lon', 'lat_bounds', 'lon_bounds']

        self.glm_colors = ['saddlebrown', 'darkturquoise', 'palegreen', 'darkolivegreen', 'mediumorchid', 'firebrick',
                           'darkgreen', 'forestgreen']

        # Names of all Land-uses
        self.all_columns = [SubLU.primf.name, SubLU.primn.name, SubLU.secdf.name, SubLU.secdn.name, SubLU.c3ann.name,
                            SubLU.c4ann.name, SubLU.c4per.name, SubLU.c3per.name, SubLU.c3nfx.name, SubLU.pastr.name,
                            SubLU.range.name, SubLU.urban.name]
        # All land-uses, with CFTs collapsed into crops
        self.no_cft_columns = [SubLU.primf.name, SubLU.primn.name, SubLU.secdf.name, SubLU.secdn.name, 'crop',
                               SubLU.pastr.name, SubLU.range.name, SubLU.urban.name]

        self.crop = [SubLU.c3ann.name, SubLU.c3nfx.name, SubLU.c4ann.name, SubLU.c4per.name, SubLU.c3per.name]
        self.pastr = [SubLU.pastr.name, SubLU.range.name]

        self.sec_diag = {'secmb': 'secondary mean biomass density',
                         'secma': 'secondary mean age'}

        # Diagnostics for N fertilization
        self.nmgt = {'fertl_' + SubLU.c3ann.name: 'C3 annual',
                     'fertl_' + SubLU.c3nfx.name: 'C3 nitrogen-fixing',
                     'fertl_' + SubLU.c3per.name: 'C3 perennial',
                     'fertl_' + SubLU.c4ann.name: 'C4 annual',
                     'fertl_' + SubLU.c4per.name: 'C4 perennial'}

        # Diagnostics for Irrigation
        self.irrg = {'irrig_' + SubLU.c3ann.name: 'C3 annual',
                     'irrig_' + SubLU.c3nfx.name: 'C3 nitrogen-fixing',
                     'irrig_' + SubLU.c3per.name: 'C3 perennial',
                     'irrig_' + SubLU.c4ann.name: 'C4 annual',
                     'irrig_' + SubLU.c4per.name: 'C4 perennial'}

        # Wood-harvest
        self.whrv = {'primf_harv': 'wood harvest area from primary forest',
                     'primn_harv': 'wood harvest area from primary non-forest',
                     'secmf_harv': 'wood harvest area from secondary mature forest',
                     'secnf_harv': 'wood harvest area from secondary non-forest',
                     'secyf_harv': 'wood harvest area from secondary young forest'}

        # Wood-harvest biomass
        self.wbio = {'primf_bioh': 'wood harvest biomass from primary forest',
                     'primn_bioh': 'wood harvest biomass from primary non-forest',
                     'secmf_bioh': 'wood harvest biomass from secondary mature forest',
                     'secnf_bioh': 'wood harvest biomass from secondary non-forest',
                     'secyf_bioh': 'wood harvest biomass from secondary young forest'}

        # Management (wood harvest)
        self.wood = {'combf': 'Commercial biofuels fraction of wood harvest',
                     'fulwd': 'Fuelwood fraction of wood harvest',
                     'rndwd': 'Industrial roundwood fraction of wood harvest'}

        # Management (biofuels)
        self.biof = {'crpbf_' + SubLU.c3ann.name: 'C3 annual crops grown as biofuels',
                     'crpbf_' + SubLU.c3nfx.name: 'C3 nitrogen-fixing crops grown as biofuels',
                     'crpbf_' + SubLU.c3per.name: 'C3 perennial crops grown as biofuels',
                     'crpbf_' + SubLU.c4ann.name: 'C4 annual crops grown as biofuels',
                     'crpbf_' + SubLU.c4per.name: 'C4 perennial crops grown as biofuels'}

        # Management (crops harvested annually)
        self.harv = {'fharv_' + SubLU.c3per.name: 'C3 perennial crops harvested annually',
                     'fharv_' + SubLU.c4per.name: 'C4 perennial crops harvested annually'}

        # Tillage
        self.tillg = {'tillg': 'tilled fraction of cropland area'}

        self.name_exp = name_exp  # name of experiment (ideally same as name of glm output folder)
        self.do_LUH1 = constants.do_LUH1  # Compare with LUH1?

        # glm static data
        self.path_glm_stat = constants.path_glm_stat  # Static data, contains grid cell area (carea)
        self.path_glm_carea = constants.path_glm_carea
        self.path_glm_vba = constants.path_glm_vba  # MIAMI-LU biomass

        # Dataset #1: historical data
        self.path_nc_states = path_input + os.sep + constants.path_nc_states
        self.path_nc_mgt = path_input + os.sep + constants.path_nc_mgt
        self.path_nc_trans = path_input + os.sep + constants.path_nc_trans

        # Dataset #1: future data
        self.path_nc_futr_states = path_input + os.sep + constants.path_nc_futr_states
        self.path_nc_futr_mgt = path_input + os.sep + constants.path_nc_futr_mgt

        # Dataset #1: historical + future data
        self.path_nc_all_states = path_input + os.sep + constants.path_nc_all_states
        self.path_nc_all_mgt = path_input + os.sep + constants.path_nc_all_mgt

        # Dataset #2
        self.path_nc_alt_state = path_input + os.sep + constants.path_nc_alt_state
        self.path_nc_alt_trans = path_input + os.sep + constants.path_nc_alt_trans

        # Legend names
        self.legend_glm = constants.legend_glm
        # Should additional datasets be plotted?
        self.do_alternate = constants.do_alternate
        self.legend_alt_glm = constants.legend_alt_glm

        self.path_data = self.path_nc_states if time_period == 'past' else self.path_nc_futr_states \
            if time_period == 'future' else self.path_nc_all_states

        self.path_mgt_data = self.path_nc_mgt if time_period == 'past' else self.path_nc_futr_mgt \
            if time_period == 'future' else self.path_nc_all_mgt

        # Get colors for plotting
        self.cols = plot.get_colors(palette='tableau')

        # Lat, Lon and time dimensions
        self.time_period = time_period
        self.lon_name = lon_name
        self.lat_name = lat_name
        self.tme_name = tme_name

        # Open up netCDF and get dimensions
        ds = util.open_or_die(self.path_data)
        self.lat = ds.variables[lat_name][:]
        self.lon = ds.variables[lon_name][:]
        self.time = ds.variables[tme_name][:].astype(int)
        self.start_yr = self.time[0] if start_year == -1 else start_year
        self.iyr = self.time - self.start_yr
        ds.close()

        # Get resolution
        self.resolution = constants.NUM_LATS/len(self.lat)

        # Get cell area (after subtracting ice/water fraction)
        icwtr_nc = util.open_or_die(self.path_glm_stat)
        icwtr = icwtr_nc.variables[constants.ice_water_frac][:, :]
        self.carea = util.open_or_die(self.path_glm_carea)
        self.carea_wo_wtr = util.open_or_die(self.path_glm_carea) * (1.0 - icwtr)

        # Get country names corresponding to FAO IDs
        self.df_fao = pandas.read_csv(constants.FAO_CONCOR)

        # Get continent codes
        import shifting_cult
        if constants.SHFT_MAP == 'Andreas':
            shft_obj = shifting_cult.ShftCult(file_sc=constants.TIF_ANDREAS, use_andreas=True)
        elif constants.SHFT_MAP == 'Butler':
            shft_obj = shifting_cult.ShftCult(file_sc=constants.ASC_BUTLER)
        else:
            logger.error('Unrecognized shifting cultivation map')
        self.lup_codes = shft_obj.combine_country_continent()
        self.asc_sc = shft_obj.asc_binary_sc

        # Country code map
        self.map_ccodes = np.genfromtxt(constants.CNTRY_CODES, skip_header=0, delimiter=' ')

        # List of country codes
        self.ccodes = pandas.read_csv(constants.ccodes_file, header=None)[0].values

        # Look-up country codes (840, USA)
        self.lup_cntr_codes = pandas.read_csv('../countries.csv').set_index('ID')['Name'].to_dict()

        # Get names of countries
        self.names_cntr = []
        for cntr in self.ccodes:
            self.names_cntr.append(self.get_country_from_FAO_ID(cntr))

        # Replace country codes by respective continent codes
        self.map_cont_codes = shft_obj.replace_country_by_continent(shft_obj.ccodes_asc, self.lup_codes)
        self.lup_cont_codes = shft_obj.dict_cont  # Look-up country codes: (1, North America)

        # Miami-lu biomass
        self.path_vba = constants.miami_vba

        # remote sensing forest loss data (Hansen et al. 2013)
        self.rs_forest = constants.rs_forest

        # Get virgin biomass and fnf constraint
        self.vba = util.open_or_die(self.path_glm_vba)
        self.vba *= 1.0/constants.AGB_TO_BIOMASS

        asc_vba = np.copy(self.vba)
        asc_vba[asc_vba < 2.0] = 0.0  # Biomass defn of forest: > 2.0 kg C/m^2
        asc_vba[asc_vba > 0.0] = 1.0
        self.fnf = asc_vba  # Boolean ascii file indicating whether it is forest(1.0)/non-forest(0.0)

        # Ascii file of global biodiversity hotspots
        self.arr_hotspots = util.upscale_np_arr(util.open_or_die(constants.file_hotspots), block_size=2.)
        asc_hotspots = np.copy(self.arr_hotspots)
        asc_hotspots[asc_hotspots > 1.0] = 1.0
        self.hnh = asc_hotspots  # hotspot/not-hotspot

        # Set up directories and create them
        self.out_path = out_path
        self.movies_path = self.out_path + os.sep + 'movies'
        self.movie_imgs_path = self.movies_path + os.sep + 'images'
        util.make_dir_if_missing(self.out_path)
        util.make_dir_if_missing(self.movies_path)
        util.make_dir_if_missing(self.movie_imgs_path)

        # Set plotting preferences
        plot.set_matplotlib_params()

        # If plotting both past and future data, combine netCDF files first
        if self.time_period == 'all':
            self.combine_glm()

        # Compare LUH1 and LUH2 in diagnostics
        if self.do_LUH1:
            import process_LUH1
            logger.info('Reading in LUH1 data for comparison with LUH2')
            self.cum_net_C_focal, self.gross_trans_focal, self.net_trans_focal, self.sec_area_focal, \
            self.sec_age_focal, self.wh_focal = process_LUH1.diag_LUH1()

    def get_continent_from_ID(self, id_continent):
        """
        Return continent name corresponding to id_continent
        :param id_continent:
        :return:
        """
        return self.lup_cont_codes[id_continent]

    def get_country_from_FAO_ID(self, id_country):
        """
        Get country name from FAO ID
        :param id_country:
        :return:
        """
        val = self.df_fao[self.df_fao.ISO == int(id_country)]['Country_FAO'].values[0]
        return val

    @staticmethod
    def return_subLU(lu_type):
        """
        Return sub land-uses corresponding to a land-use.
        :param lu_type:
        :return:
        """
        lu = lu_type.name

        if lu == LU.crop.name:
            return [SubLU.c3ann.name, SubLU.c3nfx.name, SubLU.c4ann.name, SubLU.c4per.name, SubLU.c3per.name]
        elif lu == LU.pastr.name:
            return [SubLU.pastr.name, SubLU.range.name]
        elif lu == LU.vrgn.name:
            return [SubLU.primf.name, SubLU.primn.name]
        elif lu == LU.scnd.name:
            return [SubLU.secdf.name, SubLU.secdn.name]
        elif lu == LU.urban.name:
            return [SubLU.urban.name]
        elif lu == LU.all.name:
            return [SubLU.urban.name,
                    SubLU.c3ann.name, SubLU.c3nfx.name, SubLU.c4ann.name, SubLU.c4per.name, SubLU.c3per.name,
                    SubLU.pastr.name, SubLU.range.name,
                    SubLU.secdf.name, SubLU.secdn.name,
                    SubLU.primf.name, SubLU.primn.name]
        else:
            logger.error('land use ' + lu + ' does not exist')
            sys.exit(0)

    def combine_glm(self):
        """
        Combine glm past data with glm future data
        :return: nothing, side-efftct: outputs a netCDF file containing past+future glm data combined
        """
        # ncrcat nc1.nc nc2.nc -O nc3.nc
        # Combine states.nc
        if not(os.path.isfile(self.path_nc_all_states)):
            with open(os.devnull, "w") as f:
                logger.info('Creating combined netCDF ' + self.path_nc_all_states)
                subprocess.call('ncrcat ' + self.path_nc_states + ' ' + self.path_nc_futr_states + ' -O ' +
                                self.path_nc_all_states, stdout=f, stderr=f)

        # Combine management.nc
        if not(os.path.isfile(self.path_nc_all_mgt)):
            with open(os.devnull, "w") as f:
                logger.info('Creating combined netCDF ' + self.path_nc_all_mgt)
                subprocess.call('ncrcat ' + self.path_nc_mgt + ' ' + self.path_nc_futr_mgt + ' -O ' + self.path_nc_all_mgt,
                                stdout=f, stderr=f)

    @staticmethod
    def get_transition(path_nc, source, target, year, subset_arr=None):
        """
        Add up all transitions between source and target for given year
        :param path_nc:
        :param source:
        :param target:
        :param year:
        :param subset_arr:
        :return:
        """
        src_lus = []
        tgt_lus = []

        hndl_nc = util.open_or_die(path_nc)

        # Check if source is main or sub LU
        if isinstance(source, LU):
            src_lus = glm.return_subLU(source)
        else:
            src_lus.append(source.name)

        if isinstance(target, LU):
            tgt_lus = glm.return_subLU(target)
        else:
            tgt_lus.append(target.name)

        sum_trans = np.zeros_like(hndl_nc.variables[src_lus[0] + '_to_' + tgt_lus[0]][year, :, :])

        # Add up all transition values for year
        for trans in list(itertools.product(src_lus, tgt_lus)):
            name_trans = trans[0] + '_to_' + trans[1]
            sum_trans = sum_trans + util.get_nc_var3d(hndl_nc, name_trans, year, subset_arr=subset_arr)

        np.ma.set_fill_value(sum_trans, constants.FILL_VALUE)

        hndl_nc.close()

        return sum_trans

    def get_region_area(self, subset_arr=None):
        # Get area of region/country/continent
        aa = np.ma.sum(self.carea * subset_arr)
        pdb.set_trace()
        pass

    def movie_glm_LU(self, start_yr=850):
        """
        1. Make movie of glm land-uses for all years
        2. Output kml of land-uses (average for all years)
        :param start_yr:
        :return:
        """
        logger.info('movie_glm_LU')
        hndl_nc = util.open_or_die(self.path_data)

        cmap = palettable.colorbrewer.sequential.PuRd_9.mpl_colormap
        cmap = plot.truncate_colormap(cmap, 0.1, 1.0)

        for name, lname in self.lus.iteritems():
            logger.info('Making movie map for glm variable ' + name)
            # Create maps of glm past
            imgs_all = plot.plot_maps_ts_from_path(self.path_data, name, self.lon, self.lat,
                                                   out_path=self.movie_imgs_path, save_name=name, do_jenks=False,
                                                   xlabel='Fraction of grid cell area', cmap=cmap,
                                                   start_movie_yr=start_yr, title=name, land_bg=True, grid=True)

            plot.make_movie(imgs_all, self.movies_path, out_fname='GLM_' + name + '.gif')

        # Output kml of land-uses (first year)
        for name, lname in self.lus.iteritems():
            logger.info('Making kml, glm variable ' + name + ' for year ' + str(self.iyr[0] + self.start_yr))
            # Get area for first year of simulation
            arr = util.get_nc_var3d(hndl_nc, var=name, year=self.iyr[0]) * self.carea

            plot.output_kml(arr, self.lon, self.lat, path_out=self.out_path,
                            xmin=0.0, xmax=np.nanmax(arr) * 1.1, step=np.nanmax(arr) / 10.0,
                            cmap=cmap, do_log_cb=False,
                            fname_out='GLM_' + name + '_' + str(self.iyr[0]),
                            name_legend=name + ' (' + str(self.iyr[0]) + ')',
                            label=r'$Area\ (km^{2})$')

        # Output kml of land-uses (last year)
        for name, lname in self.lus.iteritems():
            logger.info('Making kml, glm variable ' + name + ' for year ' + str(self.iyr[0] + self.start_yr))
            # Get area for first year of simulation
            arr = util.get_nc_var3d(hndl_nc, var=name, year=self.iyr[-1]) * self.carea

            plot.output_kml(arr, self.lon, self.lat, path_out=self.out_path,
                            xmin=0.0, xmax=np.nanmax(arr) * 1.1, step=np.nanmax(arr) / 10.0,
                            cmap=cmap, do_log_cb=False,
                            fname_out='GLM_' + name + '_' + str(self.iyr[-1]),
                            name_legend=name + ' (' + str(self.iyr[-1]) + ')',
                            label=r'$Area\ (km^{2})$')

    def get_ts_as_df(self, path_nc, var=LU.crop, freq_yr=50, include_last_yr=True, subset_arr=None):
        """
        Extract time-series data for LU state and return area as dataframe
        :param path_nc:
        :param var:
        :param freq_yr:
        :param include_last_yr:
        :param subset_arr:
        :return:
        """
        hndl_nc = util.open_or_die(path_nc)

        # Get the time array
        arr_time = util.get_nc_var1d(hndl_nc, self.tme_name)

        src_lus = []
        # Check if source is main or sub LU
        if isinstance(var, LU):
            src_lus = self.return_subLU(var)
        else:
            src_lus.append(var.name)

        # Create dataframe containing year column and area column
        df = pandas.DataFrame(columns=[var.name + '_year', var.name + '_area'])
        for yr in arr_time[::freq_yr]:
            sum_area = 0.0
            for v in src_lus:
                sum_area += np.ma.sum(util.get_nc_var3d(hndl_nc, var=v, year=yr - self.start_yr, subset_arr=subset_arr) *
                                      self.carea)
            df.loc[len(df)] = [yr, sum_area * constants.TO_MILLION]

        # Include the last year in the dataframe
        if include_last_yr and yr != arr_time[-1]:
            sum_area = 0.0
            for v in src_lus:
                sum_area += np.ma.sum(util.get_nc_var3d(hndl_nc, var=v, year=arr_time[-1] - self.start_yr,
                                                        subset_arr=subset_arr) * self.carea)
            df.loc[len(df)] = [arr_time[-1], sum_area * constants.TO_MILLION]

        return df

    def maps_glm_crop_mgt(self, year, do_till=False):
        """
        Map glm management (fertilizer/irrigation)
        :param year:
        :param do_till:
        :return:
        """
        logger.info('Plotting crop management map for given year')
        hndl_mgt = util.open_or_die(self.path_mgt_data)
        hndl_crp = util.open_or_die(self.path_data)

        cmap = palettable.colorbrewer.sequential.PuRd_9.mpl_colormap
        cmap = plot.truncate_colormap(cmap, 0.1, 1.0)

        # Plot Tillage
        if do_till:
            arr = np.zeros([len(self.lat), len(self.lon)])
            for count, (name, lname) in enumerate(self.tillg.iteritems(), 1):
                logger.info('tillage: ' + name)

                # Tillage is performed on C3ANN, C4ANN and C3NFX crops
                arr = arr + (hndl_mgt.variables[self.tillg.keys()[0]][int(year), :, :] *
                             (hndl_crp.variables[SubLU.c3ann.name][int(year), :, :] +
                              hndl_crp.variables[SubLU.c3nfx.name][int(year), :, :] +
                              hndl_crp.variables[SubLU.c4ann.name][int(year), :, :]))

            plot.plot_arr_to_map(arr, self.lon, self.lat, out_path=self.out_path + os.sep + 'mgt_tillage.png',
                                 var_name='tillg', plot_type='sequential', annotate_date=True,
                                 yr=int(year + self.start_yr),
                                 xlabel='Tilled fraction of cropland area', title='Tillage',
                                 cmap=cmap,
                                 any_time_data=False, land_bg=True, grid=True)

            plot.output_kml(arr, self.lon, self.lat, path_out=self.out_path,
                            xmin=np.nanmin(arr), xmax=np.nanmax(arr), step=(np.nanmax(arr) - np.nanmin(arr)) / 10.,
                            cmap=cmap,
                            fname_out='mgt_tillage', name_legend='mgt_tillage ' + str(year),
                            label='Tilled fraction of cropland area')

        # N fertilization map
        arr = np.zeros([len(self.lat), len(self.lon)])
        for count, (name, lname) in enumerate(self.nmgt.iteritems(), 1):
            logger.info('fertilization: ' + name)
            arr = arr + (hndl_mgt.variables[name][int(year), :, :] *
                         hndl_crp.variables[name.split('_')[1]][int(year), :, :])

        # Plot fertilizer
        plot.plot_arr_to_map(arr, self.lon, self.lat, out_path=self.out_path + os.sep + 'mgt_N.png',
                             var_name='N', plot_type='sequential', annotate_date=True,
                             xaxis_min=0.0, xaxis_max=np.nanmax(arr), xaxis_step=np.nanmax(arr)/10.0,
                             yr=int(year + self.start_yr),
                             xlabel='Fertilization rate (kg N/ha/yr)',
                             title='Fertilization',
                             cmap=cmap,
                             any_time_data=False, land_bg=True, grid=True)
        plot.output_kml(arr, self.lon, self.lat, path_out=self.out_path,
                        xmin=np.nanmin(arr), xmax=np.nanmax(arr), step=(np.nanmax(arr) - np.nanmin(arr)) / 10.,
                        cmap=cmap,
                        fname_out='mgt_N', name_legend='mgt_N ' + str(year), label='Fertilization rate (kg N/ha/yr)')

        # Irrigation map
        arr = np.zeros([len(self.lat), len(self.lon)])
        for count, (name, lname) in enumerate(self.irrg.iteritems(), 1):
            logger.info('irrigation: ' + name)
            arr = arr + (hndl_mgt.variables[name][int(year), :, :] *
                         hndl_crp.variables[name.split('_')[1]][int(year), :, :])

        # Plot irrigation
        plot.plot_arr_to_map(arr, self.lon, self.lat, out_path=self.out_path + os.sep + 'mgt_irrigation.png',
                             var_name='Irr', plot_type='sequential', annotate_date=True,
                             yr=int(year + self.start_yr),
                             xlabel='Irrigated fraction of cropland area', title='Irrigation',
                             cmap=cmap,
                             any_time_data=False, land_bg=True, grid=True)

        plot.output_kml(arr, self.lon, self.lat, path_out=self.out_path,
                        xmin=np.nanmin(arr), xmax=np.nanmax(arr), step=(np.nanmax(arr) - np.nanmin(arr)) / 10.,
                        cmap=cmap,
                        fname_out='mgt_irrigation', name_legend='mgt_irrigation ' + str(year),
                        label='Irrigated fraction of cropland area')

    def maps_glm_mgt(self, year):
        """

        :param year:
        :return:
        """
        logger.info('Plotting map for management for given year')
        hndl_mgt = util.open_or_die(self.path_mgt_data)
        hndl_crp = util.open_or_die(self.path_data)
        hndl_trans = util.open_or_die(self.path_nc_trans)

        cmap = palettable.colorbrewer.sequential.PuRd_9.mpl_colormap
        cmap = plot.truncate_colormap(cmap, 0.1, 1.0)

        # biof
        for count, (name, lname) in enumerate(self.biof.iteritems(), 1):
            logger.info('biof: ' + name)
            arr = hndl_mgt.variables[name][int(year), :, :] * hndl_crp.variables[name.split('_')[1]][int(year), :, :]

            plot.plot_arr_to_map(arr, self.lon, self.lat,
                                 out_path=self.out_path + os.sep + 'mgt_' + name.split('_')[1] + '_biof.png',
                                 xaxis_min=np.nanmin(arr), xaxis_max=np.nanmax(arr) * 1.1,
                                 xaxis_step=(np.nanmax(arr) * 1.1 - np.nanmin(arr))/10.0,
                                 var_name='mgt_' + name.split('_')[1] + '_biof', plot_type='sequential',
                                 annotate_date=True, yr=self.time[-1],
                                 xlabel='Fraction of grid cell area occupied by biofuels',
                                 title=lname,
                                 cmap=cmap,
                                 any_time_data=False, land_bg=True, grid=True)

            plot.output_kml(arr, self.lon, self.lat, path_out=self.out_path,
                            xmin=np.nanmin(arr), xmax=np.nanmax(arr) * 1.1,
                            step=(np.nanmax(arr) * 1.1 - np.nanmin(arr)) / 10.0,
                            cmap=cmap,
                            fname_out='mgt_' + name.split('_')[1] + '_biof', name_legend='mgt_biof ' +
                                                                                         str(year + self.start_yr),
                            label='Fraction of grid cell area occupied by ' + name + ' biofuels\n')

        # Crops harvested annually
        for count, (name, lname) in enumerate(self.harv.iteritems(), 1):
            logger.info('management (crops harvested annually): ' + name)
            arr = hndl_mgt.variables[name][int(year), :, :] * hndl_crp.variables[name.split('_')[1]][int(year), :, :]

            plot.plot_arr_to_map(arr, self.lon, self.lat,
                                 out_path=self.out_path + os.sep + 'mgt_' + name.split('_')[1] + '_harvest_annual.png',
                                 xaxis_min=np.nanmin(arr), xaxis_max=np.nanmax(arr) * 1.1,
                                 xaxis_step=(np.nanmax(arr) * 1.1 - np.nanmin(arr))/10.0,
                                 var_name='mgt_' + name.split('_')[1] + '_annual', plot_type='sequential',
                                 annotate_date=True, yr=self.time[-1],
                                 xlabel='Fraction of grid cell harvested annually',
                                 title=lname,
                                 cmap=cmap,
                                 any_time_data=False, land_bg=True, grid=True)

            plot.output_kml(arr, self.lon, self.lat, path_out=self.out_path,
                            xmin=np.nanmin(arr), xmax=np.nanmax(arr) * 1.1,
                            step=(np.nanmax(arr) * 1.1 - np.nanmin(arr)) / 10.0,
                            cmap=cmap,
                            fname_out='mgt_' + name.split('_')[1] + '_harvest_annual',
                            name_legend='mgt_' + name.split('_')[1] + '_annual ' + str(year + self.start_yr),
                            label='Fraction of grid cell harvested annually\n')

        # wood
        arr_wh = np.zeros([len(self.lat), len(self.lon)])
        # wood harvest
        for count, (name, lname) in enumerate(self.whrv.iteritems(), 1):
            arr_wh = arr_wh + hndl_trans.variables[name][int(year - 1), :, :]

        for count, (name, lname) in enumerate(self.wood.iteritems(), 1):
            logger.info('biof: ' + name)
            arr = hndl_mgt.variables[name][int(year), :, :] * arr_wh * self.carea

            plot.plot_arr_to_map(arr, self.lon, self.lat,
                                 out_path=self.out_path + os.sep + 'mgt_wood.png',
                                 xaxis_min=np.nanmin(arr), xaxis_max=np.nanmax(arr) * 1.1,
                                 xaxis_step=(np.nanmax(arr) * 1.1 - np.nanmin(arr))/10.0,
                                 var_name='mgt_wood', plot_type='sequential',
                                 annotate_date=True, yr=self.time[-1],
                                 xlabel='Wood harvest area' + r'$\ (km^{2})$',
                                 title=lname,
                                 cmap=cmap,
                                 any_time_data=False, land_bg=True, grid=True)

            plot.output_kml(arr, self.lon, self.lat, path_out=self.out_path,
                            xmin=np.nanmin(arr), xmax=np.nanmax(arr) * 1.1,
                            step=(np.nanmax(arr) * 1.1 - np.nanmin(arr)) / 10.0,
                            cmap=cmap,
                            fname_out='mgt_wood', name_legend='mgt_wood ' + str(year + self.start_yr),
                            label='Wood harvest area' + r'$\ (km^{2})$\n')

        hndl_mgt.close()
        hndl_crp.close()
        hndl_trans.close()

    def plot_glm_time_series(self, lu=LU.all):
        """
        glm time-series of land-use classes
        :param lu: which landuse to plot e.g. lu='crop' plots: 'c3ann', 'c3nfx', 'c4ann', 'c4per'
        :return: nothing, side-effect: time-series plot is saved to self.out_path
        """
        # Create time-series plot
        logger.info('Create time-series plot ' + lu.name + ' ' + os.path.basename(self.path_data))

        # Create figure
        fig, ax = plt.subplots()

        # Get lu type to plot their time-series
        dict_lus = {k: self.lus.get(k, None) for k in glm.return_subLU(lu)}

        total = np.zeros_like(self.time)

        for count, (name, lname) in enumerate(dict_lus.iteritems(), 1):
            logger.info(name)

            arr_sum = util.sum_netcdf(self.path_data, name, do_area_wt=True, arr_area=self.carea)
            total = total + arr_sum

            # Determine if time series being plotted is first or last
            pos = 'first' if count == 1 else 'last' if count == len(dict_lus) + 1 else 'mid'

            plot.plot_multiple_ts(ax, arr_sum, self.time, self.out_path + os.sep + 'ts_' + lu.name + '_GLM.png',
                                  leg_name=lname.title(), ylabel=r'$Area\ (km^{2})$', pos=pos, fill_between=False,
                                  title='')

        # Plot total of time-series
        plot.plot_multiple_ts(ax, total, self.time, self.out_path + os.sep + 'ts_' + lu.name + '_GLM.png',
                              leg_name='Total', col='k', ylabel=r'$Area\ (km^{2})$', pos='last', fill_between=False,
                              title='')
        plt.close(fig)

    def plot_glm_hovmoller(self, lu=LU.all):
        """
        glm hovmoller plots of land-use classes
        :param lu:
        :return:
        """
        # Get lu type to plot their time-series
        dict_lus = {k: self.lus.get(k, None) for k in glm.return_subLU(lu)}

        for name, lname in tqdm(dict_lus.iteritems(), desc='hovmoller', disable=(len(dict_lus) < 10)):
            logging.info('Hovmoller ' + lname)
            long_name = util.open_or_die(self.path_data).variables[name].long_name

            plot.plot_hovmoller(self.path_data, long_name, self.out_path + os.sep + 'hovmoller_lat_' + name + '.png',
                                do_latitude=False, xlabel='Latitude ($^\circ$)', ylabel='Years', title='GLM: ' + lname,
                                cbar='Fraction of gridcell area')

            plot.plot_hovmoller(self.path_data, long_name, self.out_path + os.sep + 'hovmoller_lon_' + name + '.png',
                                do_latitude=True, xlabel='Longitude ($^\circ$)', ylabel='Years', title='GLM: ' + lname,
                                cbar='Fraction of gridcell area')

    def plot_natural_veg_in_hotspots(self, year):
        """

        :param year:
        :return:
        """
        logger.info('plot_natural_veg_in_hotspots')
        hndl_nc = util.open_or_die(self.path_data)

        # Current global land surface covered by natural vegetation in the biodiversity hotspots
        arr_primary = util.get_nc_var3d(hndl_nc, SubLU.primf.name, year - self.start_yr) + \
                      util.get_nc_var3d(hndl_nc, SubLU.primn.name, year - self.start_yr)

        arr_primary_in_hotspot = (self.hnh * self.carea * arr_primary) * 100./self.carea

        # Plot array to map
        cmap = palettable.colorbrewer.sequential.YlGnBu_9.mpl_colormap

        plot.plot_arr_to_map(arr_primary_in_hotspot, self.lon, self.lat,
                             out_path=self.out_path + os.sep + 'natural_veg_in_hotspots_' + str(year) + '.png',
                             var_name='hotspots', xaxis_min=0.0, xaxis_max=101.0, xaxis_step=10.0,
                             cmap=cmap, plot_type='sequential', annotate_date=True, yr=int(year),
                             xlabel='% of land surface',
                             title='Natural vegetation in hotspots', any_time_data=False,
                             land_bg=True, grid=True)

    def plot_glm_annual_diffs(self):
        """
        Plot difference maps for all glm land-uses showing difference in land-use in consecutive years
        :return:
        """
        logger.info('plot_glm_annual_diffs')

        for name, lname in self.lus.iteritems():
            logger.info(name)
            fig, ax = plt.subplots()

            arr_sum = util.sum_netcdf(self.path_data, name, do_area_wt=True, arr_area=self.carea)

            # Difference in area between consecutive years
            arr_sum = [t - s for s, t in zip(arr_sum, arr_sum[1:])]
            arr_sum.insert(0, 0.0)

            plot.plot_np_ts(ax, arr_sum, self.time, self.out_path + os.sep + 'annual_diff_GLM_' + name + '.png',
                            col=self.cols[0], vert_yr=[1961], leg_name=lname.title(),
                            ylabel=r'$Annual\ difference\ in\ area\ (km^{2})$')
            plt.close(fig)

    def plot_glm_scnd_diagnostics(self):
        """
        Diagnostic plots
        :return:
        """
        logger.info('plot_glm_scnd_diagnostics')
        for name, lname in tqdm(self.sec_diag.iteritems(), desc='GLM secondary diagnostics',
                                disable=(len(self.sec_diag) < 2)):
            logger.info(name)

            if name == 'secma':
                label = 'Age (years)'
                bins = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 1250]
            elif name == 'secmb':
                label = r'kg C'
                bins = []

            hist, bin_edges = util.avg_hist_netcdf(self.path_data, name, subset_asc=self.fnf, do_area_wt=True,
                                                   bins=bins, area_data=self.carea, date=2015)
            plot.plot_hist(hist, bin_edges, out_path=self.out_path + os.sep + 'hist_GLM_' + name + '.png', do_per=False,
                           do_log=True, xlabel=label, ylabel=r'$Area\ (km^{2})$')
            plot.plot_hist(hist, bin_edges, out_path=self.out_path + os.sep + 'percentage_hist_GLM_' + name + '.png',
                           do_per=True, xlabel=label, ylabel=r'Percentage of secondary land area')

            # Plot maps (subset by forest/non-forest)
            hndl_nc = util.open_or_die(self.path_data)
            arr = hndl_nc.variables[name][2015 - self.start_yr, :, :]
            arr[self.fnf <= 0.0] = np.nan

            xaxis_min = np.nanmin(arr)
            xaxis_max = np.nanmax(arr)
            xaxis_step = (xaxis_max * 1.1 - xaxis_min)/10.

            cmap = palettable.colorbrewer.sequential.YlGnBu_9.mpl_colormap
            cmap = plot.truncate_colormap(cmap, 0.1, 1.0)

            plot.plot_arr_to_map(arr, self.lon, self.lat, out_path=self.out_path + os.sep + 'map_' + name + '_2015.png',
                                 var_name=name, xaxis_min=xaxis_min, xaxis_max=xaxis_max, xaxis_step=xaxis_step,
                                 cmap=cmap, plot_type='sequential', annotate_date=True, yr=int(2015), xlabel=label,
                                 title=lname, any_time_data=False, land_bg=True, grid=True)
            hndl_nc.close()

    def plot_glm_N_diagnostics(self, do_global=True):
        """
        :param do_global:
        :return:
        """
        logger.info('plot_glm_N_diagnostics, global: ' + str(do_global))
        hndl_crp = util.open_or_die(self.path_data)
        hndl_mgt = util.open_or_die(self.path_mgt_data)

        # N fertilization plots
        fig, ax = plt.subplots()

        for count, (name, lname) in enumerate(self.nmgt.iteritems(), 1):
            logger.info(name)
            arr_mgt = []
            name_crop = name.split('_')[1]

            if do_global:
                for yr in self.iyr:
                    area_crp = hndl_crp.variables[name_crop][int(yr), :, :] * self.carea
                    frac_mgt = hndl_mgt.variables[name][int(yr), :, :]
                    arr_mgt.extend([np.ma.sum(area_crp * frac_mgt) * constants.KM2_TO_HA * constants.KG_TO_TG])

                fname = 'global_N_GLM.png'
                title = 'Global fertilizer use by crop type'
                ylabel = 'Tg N'
            else:
                for yr in self.iyr:
                    frac_mgt = hndl_mgt.variables[name][int(yr), :, :]
                    arr_mgt.extend([np.ma.mean(frac_mgt)])

                fname = 'average_N_GLM.png'
                title = 'Mean fertilizer rate by crop type'
                ylabel = r'Fertilizer rate (kgN/ha/yr)'

            # Determine if time series being plotted is first or last
            pos = 'first' if count == 1 else 'last' if count == len(self.nmgt) else 'mid'
            plot.plot_multiple_ts(ax, arr_mgt, self.time, self.out_path + os.sep + fname, title=title, vert_yr=[1961],
                              leg_name=lname.title(), ylabel=ylabel, col=self.cols[count-1], pos=pos)

        plt.close(fig)

    def plot_glm_Irr_diagnostics(self, do_global=True):
        """
        :param do_global
        :return:
        """
        logger.info('plot_glm_Irr_diagnostics, global: ' + str(do_global))
        hndl_crp = util.open_or_die(self.path_data)
        hndl_mgt = util.open_or_die(self.path_mgt_data)

        fig, ax = plt.subplots()
        for count, (name, lname) in enumerate(self.irrg.iteritems(), 1):
            logger.info(name)
            arr_mgt = []

            if do_global:
                for yr in self.iyr:
                    area_crp = hndl_crp.variables[name.split('_')[1]][int(yr), :, :] * self.carea
                    frac_mgt = hndl_mgt.variables[name][int(yr), :, :]
                    arr_mgt.extend([np.ma.sum(area_crp * frac_mgt)])

                fname = 'irrigation_global_GLM.png'
                title = 'Global irrigation area by crop type'
                ylabel = r'$Area\ (km^{2})$'
            else:
                for yr in self.iyr:
                    frac_mgt = hndl_mgt.variables[name][int(yr), :, :]
                    arr_mgt.extend([np.ma.mean(frac_mgt)])

                fname = 'irrigation_average_GLM.png'
                title = 'Mean irrigated fraction of crop type area'
                ylabel = r'Fraction of grid cell area'

            # Determine if time series being plotted is first or last
            pos = 'first' if count == 1 else 'last' if count == len(self.nmgt) else 'mid'

            plot.plot_multiple_ts(ax, arr_mgt, self.time, self.out_path + os.sep + fname, title=title, vert_yr=[1961],
                                  leg_name=lname.title(), ylabel=ylabel, col=self.cols[count-1], pos=pos)
        plt.close(fig)

    def plot_ts_wood_harvest(self, years):
        """
        Plot time-series of biomass in wood-harvest
        :param years:
        :return:
        """
        logger.info('plot_ts_wood_harvest')

        fig, ax = plt.subplots()
        # TODO Hack plot multiple scenarios
        # Get wood-harvest time-series
        arr_wh = self.compute_wh(self.path_nc_trans, years)

        plot.plot_multiple_ts(ax, arr_wh, years + self.start_yr, self.out_path + os.sep + 'ts_wh.png', vert_yr=[1961],
                              title='Wood harvest (Pg C)', leg_name=constants.glm_experiments[0], ylabel='Pg C',
                              col=self.cols[0], pos='first')

        obj = glm(name_exp=constants.glm_experiments[1],
                  path_input=constants.path_glm_input + os.sep + constants.glm_experiments[1],
                  time_period='past',
                  out_path=constants.path_glm_output)
        arr_wh = obj.compute_wh(obj.path_nc_trans, years)

        plot.plot_multiple_ts(ax, arr_wh, years + obj.start_yr, obj.out_path + os.sep + 'ts_wh.png', vert_yr=[1961],
                              title='Wood harvest (Pg C)', leg_name=constants.glm_experiments[1], ylabel='Pg C',
                              col=obj.cols[1], pos='mid')

        obj = glm(name_exp=constants.glm_experiments[2],
                  path_input=constants.path_glm_input + os.sep + constants.glm_experiments[1],
                  time_period='past',
                  out_path=constants.path_glm_output)
        arr_wh = obj.compute_wh(obj.path_nc_trans, years)

        plot.plot_multiple_ts(ax, arr_wh, years + obj.start_yr, obj.out_path + os.sep + 'ts_wh.png', vert_yr=[1961],
                              title='Wood harvest (Pg C)', leg_name=constants.glm_experiments[2], ylabel='Pg C',
                              col=obj.cols[2], pos='last')

        # Plot Hurtt 2011
        if self.do_LUH1:
            plot.plot_multiple_ts(ax, self.wh_focal, years + self.start_yr, self.out_path + os.sep + 'ts_wh.png',
                                  vert_yr=[1961], title='Wood harvest (Pg C)', leg_name='Hurtt 2011', ylabel='Pg C',
                                  col=self.cols[3], pos='last')

        plt.close(fig)

    def plot_ts_mgt(self):
        """
        Plot time-series for:
        1. glm_global_wood_harvest_area
        2. glm_global_biofuels_area
        3. glm_global_crops_harvested_area
        :return:
        """
        ######################################
        # management (biofuels)
        ######################################
        fig, ax = plt.subplots()
        logger.info('plot_ts_mgt: management (biofuels)')

        for count, (name, lname) in enumerate(self.biof.iteritems(), 1):
            # crpbf_c3ann: C3 annual crops grown as biofuels
            # crpbf_c3nfx: C3 nitrogen-fixing crops grown as biofuels
            # crpbf_c3per: C3 perennial crops grown as biofuels
            # crpbf_c4ann: C4 annual crops grown as biofuels
            # crpbf_c4per: C4 perennial crops grown as biofuels
            fname = 'global_biofuels_area_GLM.png'
            title = 'Crop type area occupied by biofuels'
            ylabel = r'$Area\ (km^{2})$'

            # Fraction of crops harvested annually * Cropland fraction * cell area
            arr_sum = []
            hndl_crop = util.open_or_die(self.path_data)
            hndl_mgt = util.open_or_die(self.path_mgt_data)
            for yr in self.iyr:
                arr_sum.append(np.ma.sum(hndl_crop.variables[name.split('_')[1]][int(yr), :, :] *
                                            hndl_mgt.variables[name][int(yr), :, :] * self.carea))

            # Determine if time series being plotted is first or last
            pos = 'first' if count == 1 else 'last' if count == len(self.biof) else 'mid'

            plot.plot_multiple_ts(ax, arr_sum, self.time, self.out_path + os.sep + fname, title=title, vert_yr=[1961],
                                  leg_name=lname.title(), ylabel=ylabel, col=self.cols[count-1], pos=pos)
        plt.close(fig)

        ######################################
        # management (crops harvested annually)
        ######################################
        logger.info('plot_ts_mgt: management (crops harvested annually)')
        fig, ax = plt.subplots()

        for count, (name, lname) in enumerate(self.harv.iteritems(), 1):
            # fharv_c3per: Fraction of C3 perennial crops harvested annually
            # fharv_c4per: 'Fraction of C4 perennial crops harvested annually
            fname = 'global_crops_harvested_area_GLM.png'
            title = 'Crop type area harvested annually'
            ylabel = r'$Area\ (km^{2})$'

            # Fraction of crops harvested annually * Cropland fraction * cell area
            arr_sum = []
            hndl_crop = util.open_or_die(self.path_data)
            hndl_mgt = util.open_or_die(self.path_mgt_data)
            for yr in self.iyr:
                arr_sum.append(np.ma.sum(hndl_crop.variables[name.split('_')[1]][int(yr), :, :] *
                                            hndl_mgt.variables[name][int(yr), :, :] * self.carea))

            # Determine if time series being plotted is first or last
            pos = 'first' if count == 1 else 'last' if count == len(self.harv) else 'mid'

            plot.plot_multiple_ts(ax, arr_sum, self.time, self.out_path + os.sep + fname, title=title, pos=pos,
                                  vert_yr=[1961], leg_name=lname.title(), ylabel=ylabel, col=self.cols[count-1])
        plt.close(fig)

        ######################################
        # management (wood)
        ######################################
        fig, ax = plt.subplots()
        logger.info('plot_ts_mgt: management (wood)')

        for count, (name, lname) in enumerate(self.wood.iteritems(), 1):
            # combf: Commercial biofuels fraction of wood harvest
            # fulwd: Fuelwood fraction of wood harvest
            # rndwd: Industrial roundwood fraction of wood harvest
            fname = 'global_wood_harvest_area_GLM.png'
            title = 'Wood harvest area'
            ylabel = r'$Area\ (km^{2})$'

            arr = util.sum_netcdf(self.path_mgt_data, name, do_area_wt=True, arr_area=self.carea)

            # Determine if time series being plotted is first or last
            pos = 'first' if count == 1 else 'last' if count == len(self.wood) else 'mid'

            plot.plot_multiple_ts(ax, arr, self.time, self.out_path + os.sep + fname, title=title, vert_yr=[1961],
                                  leg_name=lname.title(), ylabel=ylabel, col=self.cols[count-1], pos=pos)
        plt.close(fig)

    def plot_stacked_ts(self, nc_path, dict_lus, time_var='time'):
        """

        :param nc_path:
        :param dict_lus:
        :param time_var:
        :return:
        """
        logger.info('plot_stacked_ts')
        nc_hndl = util.open_or_die(nc_path)

        # Get x-axis
        ts = nc_hndl.variables[time_var][:]
        stk_arr = np.zeros((len(dict_lus), len(ts)))
        sum_arr = np.zeros(len(ts))

        # Compute cumulative stacked Y-axis
        for index, (key, value) in tqdm(enumerate(dict_lus.iteritems()), desc='stacked time-series',
                                        disable=(len(dict_lus) < 2)):
            logger.info(key)
            vals = np.zeros((len(ts)))

            if key == LU.crop.name:
                for idx, crp in enumerate(self.crop):
                    tmp = util.avg_netcdf(nc_path, crp, do_area_wt=True, area_data=self.carea)
                    vals = [x + y for x, y in zip(tmp, vals)]
            else:  # All other LUs
                vals = util.avg_netcdf(nc_path, key, do_area_wt=True, area_data=self.carea)

            sum_arr += vals
            stk_arr[index, :] = vals

        stk_arr = stk_arr/sum_arr

        # Plot stacked plot
        import matplotlib.patches as mpatches
        from matplotlib import pyplot as plt
        import matplotlib.ticker as tkr

        fig, ax = plt.subplots()
        ax.stackplot(ts, stk_arr, linewidth=0.0, colors=self.glm_colors)

        # Format
        plt.gca().yaxis.grid(True)
        plt.gca().yaxis.set_major_locator(tkr.MaxNLocator(10))
        plt.xlim(xmin=min(ts), xmax=max(ts))
        plt.ylim(0.0, 1.0)
        plt.ylabel('Fraction of global land area')

        # Plot labels. This is very sensitive to position of keys in self.glm dictionary
        patches = []
        for idx, key in enumerate(self.glm.keys()):
            patches.append(mpatches.Patch(color=self.glm_colors[idx], label=self.glm.values()[idx]))

        # Legend
        leg = plt.legend(handles=patches, loc='best', ncol=2, prop={'size':10}, frameon=True)
        leg.get_frame().set_linewidth(0.0)

        plt.savefig(self.out_path + os.sep + 'stacked_global_LU_GLM.png', dpi = constants.DPI)
        plt.close('all')
        nc_hndl.close()

    def plot_glm_stacked(self):
        """

        :return:
        """
        self.plot_stacked_ts(self.path_data, self.glm)

    def movie_glm_diff_maps(self, time_var='time'):
        """
        Movie from maps showing spatial differences between glm versions
        :param time_var
        :return:
        """
        logger.info('movie_glm_diff_maps')
        nc_hndl = util.open_or_die(self.path_data)

        # Get x-axis
        ts = nc_hndl.variables[time_var][:]

        for lu, lu_name in self.lus.iteritems():
            imgs_diff_movie = []

            # Create difference map and movie of difference maps
            for yr in ts.tolist()[::constants.MOVIE_SEP]:
                logger.info('Create difference map for year ' + str(int(yr)) + ' for variable ' + lu)
                diff_nc = util.subtract_netcdf(self.path_nc_alt_state, self.path_data, left_var=lu, right_var=lu,
                                               date=int(yr))

                # Convert netCDF file to map
                plot.plot_arr_to_map(diff_nc, self.lon, self.lat, out_path=self.movie_imgs_path,
                                     var_name='GLM_Diff', plot_type='diverging', annotate_date=True, yr=int(yr),
                                     xlabel='Difference in fraction of grid cell area: ' + lu,
                                     any_time_data=False, land_bg=True, grid=True)
            plot.make_movie(imgs_diff_movie, self.movies_path, out_fname='Diff_GLM_' + lu + '.gif')

        nc_hndl.close()

    def get_bool_map_region(self, map_region, id_region):
        """
        Get a binary map with a specific country/region in 1, and all other as 0
        :param map_region:
        :param id_region:
        :return:
        """
        # Get binary map of region
        bool_map_region = map_region.copy()
        bool_map_region[bool_map_region != id_region] = 0.0
        bool_map_region[bool_map_region == id_region] = 1.0

        return bool_map_region

    def get_map_region(self, type_region):
        """

        :param type_region: 'country', 'continent', 'SC'
        :return:
        """
        if type_region == 'country':
            map_region = self.map_ccodes
        elif type_region == 'continent':
            map_region = self.map_cont_codes
        elif type_region == 'SC':
            map_region = self.asc_sc

        return map_region

    def get_region_name_from_region_id(self, id_region, type_region):
        """

        :param id_region: numeric ID of region e.g. USA is 840
        :param type_region: 'country', 'continent', 'SC'
        :return:
        """
        # Get name of type_region
        if type_region == 'country':
            reg_name = self.get_country_from_FAO_ID(id_region)  # id_region[0] returns ID of country
        elif type_region == 'continent':
            reg_name = self.get_continent_from_ID(id_region)  # id_region[0] returns ID of continent
        elif type_region == 'SC':
            reg_name = 'SC type_region' if id_region else 'Outside SC type_region'  # It is binary, inside(1)/outside(0)

        return reg_name

    def glm_by_region(self, path_nc, var, type_region, id_region):
        """
        Time-series of LU state variable for region (country/continent)
        :param path_nc:
        :param var:
        :param type_region
        :param id_region
        :return:
        """
        vals = []
        hndl_nc = util.open_or_die(self.path_data)

        map_region = self.get_map_region(type_region)
        bool_map_region = self.get_bool_map_region(map_region, id_region)

        ts = util.get_nc_var1d(hndl_nc, var=self.tme_name)

        # Return time-series
        for yr in ts - self.start_yr:
            arr = util.get_nc_var3d(hndl_nc, var=var, year=yr) * self.carea
            vals.append(np.ma.sum(arr * bool_map_region))

        hndl_nc.close()
        return vals

    def get_top_regions_for_LUstate(self, region, LUstate, num_regions, year=2015):
        """
        Get list of top 'num_regions' regions by area for a given LUstate in a given year
        :param region:
        :param LUstate:
        :param num_regions:
        :param year:
        :return:
        """
        reg_codes = self.get_map_region(region)

        # Get list of region codes by removing 0
        if region != 'SC':
            rcodes = np.unique(reg_codes[reg_codes > 0.0])
        else:
            # In case of butler, a value of 0 means the no shifting cultivation region
            # Andreas SC ascii has 65536.0 as missing value, remove it
            rcodes = np.unique(reg_codes[~reg_codes.mask]).data

        reg_data = np.zeros((len(rcodes), 2))

        # For region, extract data from states.nc for 'year'
        nc_data = util.open_or_die(self.path_data)
        # Create empty array
        arr_data = np.zeros_like(nc_data.variables[self.crop[0]][int(year), :, :])

        if LUstate != LU.crop.name:
            arr_data = nc_data.variables[LUstate][int(year), :, :] * self.carea
        else:
            # Aggregate areas for all croplands
            for crop in self.crop:
                arr_data = arr_data + nc_data.variables[crop][int(year), :, :] * self.carea

        # For each country extract LU_state data for 'year'
        for idx, reg in enumerate(rcodes):
            reg_data[idx, 0] = int(reg)  # country code
            reg_data[idx, 1] = np.ma.sum(arr_data[reg_codes[:] == reg][:])  # Sum of LUstate

        # Sort by the 1st column which contains LUstate data
        reg_data = reg_data[np.argsort(reg_data[:, 1])][-num_regions:, :]

        nc_data.close()
        return reg_data

    def plot_top_regions_for_LU_state(self, region, LU_state, long_name='', num_regions=1, vert_yr=[]):
        """
        For the 'num_regions' regions with the highest LU_state value in the current year, plot time-series of LU_state
        :param region:
        :param LU_state:
        :param long_name:
        :param num_regions:
        :param vert_yr:
        :return:
        """
        logger.info('plot_top_regions_for_LU_state ' + region + ' ' + LU_state)

        out_path = self.out_path + os.sep + 'Plot_by_' + region
        util.make_dir_if_missing(out_path)

        vals = np.zeros(len(self.iyr))
        df_top_region = pandas.DataFrame(index=self.iyr + self.start_yr)
        top_region_data = self.get_top_regions_for_LUstate(region, LU_state, num_regions, year=int(max(self.iyr)))
        for idx, row in enumerate(top_region_data):
            # row[0] returns ID of country/continent/SC
            reg_name = self.get_region_name_from_region_id(int(row[0]), region)

            if LU_state != LU.crop.name:
                # Get time-series of LU_state data for region
                vals = self.glm_by_region(self.path_data, LU_state, region, int(row[0]))
            else:
                # Add time-series of cropland data for region (for all CFTs)
                for crop in self.crop:
                    vals = vals + self.glm_by_region(self.path_data, crop, region, int(row[0]))

            # Store output in pandas dataframe
            df_top_region[reg_name] = vals

        # Create plots
        plot.plot_LUstate_top_regions(df_top_region, xlabel='years', ylabel=r'$Area\ (km^{2})$',
                                      title=long_name, out_path=out_path, fname=region + '_' + LU_state + '.png',
                                      vert_yr=vert_yr)

    def glm_2011_diag_secd_area(self):
        """
        Plot time-series of secondary area as sum of secondary forested and secondary non-forested
        :return:
        """
        logger.info('glm_2011_diag_secd_area')
        sec_area_focal = []  # Secondary area is sum of secondary forested and secondary non-forested
        sec_area_focal_alt = []

        # Create figure
        fig, ax = plt.subplots()

        ds = util.open_or_die(self.path_data)
        ts = ds.variables['time'][:]
        ds.close()

        # From LUH v0.1: sec_area_focal(ind) = sum(sum(secd.data.*cellarea_non_netcdf));
        data = util.open_or_die(self.path_data)
        if self.do_alternate:
            alt_data = util.open_or_die(self.path_nc_alt_state)

        for yr in self.iyr:
            secd_nc = data.variables['secdf'][int(yr), :, :] + data.variables['secdn'][int(yr), :, :]
            sec_area_focal.append(np.ma.sum(secd_nc * self.carea))

            if self.do_alternate:
                secd_nc_alt = alt_data.variables['secdf'][int(yr), :, :] + alt_data.variables['secdn'][int(yr), :, :]
                sec_area_focal_alt.append(np.ma.sum(secd_nc_alt * self.carea))
                alt_data.close()

        # Plot
        if self.do_alternate or self.do_LUH1:
            plot.plot_multiple_ts(ax, sec_area_focal, ts, self.out_path + os.sep + 'Secondary_area.png',
                                  title='Secondary area', leg_name=self.legend_glm, ylabel=r'$Area\ (km^{2})$', col='k',
                                  pos='first')
        else:
            plot.plot_np_ts(ax, sec_area_focal, ts, self.out_path + os.sep + 'Secondary_area.png',
                            title='Secondary area', leg_name=self.legend_glm, ylabel=r'$Area\ (km^{2})$', col='k')

        if self.do_alternate:
            plot.plot_multiple_ts(ax, sec_area_focal_alt, ts, self.out_path + os.sep + 'Secondary_area.png',
                                  title='Secondary area', leg_name=self.legend_alt_glm, ylabel=r'$Area\ (km^{2})$',
                                  col='r', pos='last')

        if self.do_LUH1:
            plot.plot_multiple_ts(ax, self.sec_area_focal, ts, self.out_path + os.sep + 'Secondary_area.png',
                                  title='Secondary area', leg_name='LUH1', ylabel=r'$Area\ (km^{2})$', col='r',
                                  pos='last')
        data.close()
        plt.close(fig)

    def glm_2011_diag_secd_biom(self):
        """

        :return:
        """
        logger.info('glm_2011_diag_secd_biom')
        cum_net_C_focal = []
        cum_net_C_focal_alt = []

        # Create figure
        fig, ax = plt.subplots()

        # From LUH v0.1: cum_net_C_focal(ind) = sum(sum(vba.*(1-icew_non_netcdf).*carea))*1e6*1e3/1e15 -
        # sum(sum(secd.data.*carea.*ssmb.data+othr.data.*carea.*vba))*1e6*1e3/1e15;
        data = util.open_or_die(self.path_data)
        ts = data.variables['time'][:]  # time-series
        if self.do_alternate:
            alt_data = util.open_or_die(self.path_nc_alt_state)

        for yr in self.iyr:
            # Secondary biomass = secondary forested * cell_area * secondary biomass density +
            #                     secondary non-forested * cell_area * secondary biomass density
            tot_secd_nc = data[SubLU.secdf.name][int(yr), :, :] * self.carea * data['secmb'][int(yr), :, :] + \
                          data[SubLU.secdn.name][int(yr), :, :] * self.carea * data['secmb'][int(yr), :, :]

            # Primary biomass = primary forested * cell_area * primary biomass density +
            #                   primary non-forested * cell_area * secondary biomass density
            tot_prim_nc = data[SubLU.primf.name][int(yr), :, :] * self.carea * self.vba + \
                          data[SubLU.primn.name][int(yr), :, :] * self.carea * self.vba

            cum_net_C_focal.append(np.ma.sum(self.vba * self.carea_wo_wtr)*1e6*1e3/1e15 -
                                   np.ma.sum(tot_secd_nc + tot_prim_nc)*1e6*1e3/1e15)

            # Alternate
            if self.do_alternate:
                tot_secd_nc_alt = alt_data[SubLU.secdf.name][int(yr), :, :] * self.carea * alt_data['secmb'][int(yr), :, :] + \
                              alt_data[SubLU.secdn.name][int(yr), :, :] * self.carea * alt_data['secmb'][int(yr), :, :]

                # Primary biomass = primary forested * cell_area * primary biomass density +
                #                   primary non-forested * cell_area * secondary biomass density
                tot_prim_nc_alt = alt_data[SubLU.primf.name][int(yr), :, :] * self.carea * self.vba + \
                                  alt_data[SubLU.primn.name][int(yr), :, :] * self.carea * self.vba

                cum_net_C_focal_alt.append(np.ma.sum(self.vba * self.carea_wo_wtr)*1e6*1e3/1e15 -
                                           np.ma.sum(tot_secd_nc_alt + tot_prim_nc_alt)*1e6*1e3/1e15)

        if self.do_alternate or self.do_LUH1:
            plot.plot_multiple_ts(ax, cum_net_C_focal, ts, self.out_path + os.sep + 'Cumulative_GLM_C.png',
                                  title='Global cumulative net loss of AGB', leg_name=self.legend_glm, ylabel=r'PgC',
                                  col='k', pos='first')
        else:
            plot.plot_np_ts(ax, cum_net_C_focal, ts, self.out_path + os.sep + 'Cumulative_GLM_C.png',
                            title='Global cumulative net loss of AGB', leg_name=self.legend_glm, ylabel=r'PgC', col='k')

        if self.do_alternate:
            plot.plot_multiple_ts(ax, cum_net_C_focal_alt, ts, self.out_path + os.sep + 'Cumulative_GLM_C.png',
                                  title='Global cumulative net loss of AGB', leg_name=self.legend_alt_glm,
                                  ylabel=r'PgC', col='r', pos='last')
            alt_data.close()

        if self.do_LUH1:
            plot.plot_multiple_ts(ax, self.cum_net_C_focal, ts, self.out_path + os.sep + 'Cumulative_GLM_C.png',
                                  title='Global cumulative net loss of AGB', leg_name='LUH1', ylabel=r'PgC', col='r',
                                  pos='last')

        data.close()
        plt.close(fig)

    def net_transitions(self, path_nc, area, yrs, subset_arr=None, do_global=True):
        """
        Compute transitions
        Net transitions in v0.1 LU harmonization: Net transitions measure only net changes in land use
        (i.e., net transitions exclude wood harvest on secondary forests, and agricultural land abandonment that
        is offset by land conversions to agriculture - net transitions specifically exclude shifting cultivation
        but also other historical redistribution of agriculture across a region).
        Not doing 'secdf_to_secdn' or 'secdn_to_secdf'
        flsp: fraction of each gridcell that transitioned from secondary land to pasture
        flsc: fraction of each gridcell that transitioned from secondary land to cropland
        flvc: fraction of each gridcell that transitioned from primary land to cropland
        flvp: fraction of each gridcell that transitioned from primary land to pasture
        flps: fraction of each gridcell that transitioned from pasture to secondary land
        flcs: fraction of each gridcell that transitioned from cropland to secondary land
        fvh1: fraction of each gridcell that had wood harvested from primary forested land
        fvh2: fraction of each gridcell that had wood harvested from primary non-forested land
        Net tran: (flsp.data + flsc.data + flvc.data + flvp.data) - (flps.data + flcs.data) + fvh1.data + fvh2.data
        :param path_nc:
        :param area:
        :param yrs:
        :param subset_arr:
        :param do_global: Get a single value showing sum of gross transition over globe
        :return:
        """
        logger.info('net_transitions')
        hndl_nc = util.open_or_die(path_nc)

        if do_global:
            net_trans = []
        else:
            net_trans = np.zeros([len(self.lat), len(self.lon)])

        for yr in tqdm(yrs, desc='net transitions', disable=(len(yrs) < 2)):
            # cropland
            cropland_to_secondary = glm.get_transition(path_nc, LU.crop, LU.scnd, subset_arr=subset_arr, year=yr)

            # pasture
            pasture_to_secondary = glm.get_transition(path_nc, LU.pastr, LU.scnd, subset_arr=subset_arr, year=yr)

            # secondary
            secondary_to_cropland = glm.get_transition(path_nc, LU.scnd, LU.crop, subset_arr=subset_arr, year=yr)
            secondary_to_pasture = glm.get_transition(path_nc, LU.scnd, LU.pastr, subset_arr=subset_arr, year=yr)
            secondary_to_urban = glm.get_transition(path_nc, LU.scnd, LU.urban, subset_arr=subset_arr, year=yr)

            # primary
            primary_to_cropland = glm.get_transition(path_nc, LU.vrgn, LU.crop, subset_arr=subset_arr, year=yr)
            primary_to_pasture = glm.get_transition(path_nc, LU.vrgn, LU.pastr, subset_arr=subset_arr, year=yr)
            primary_to_urban = glm.get_transition(path_nc, LU.vrgn, LU.urban, subset_arr=subset_arr, year=yr)

            # urban
            urban_to_secondary = glm.get_transition(path_nc, LU.urban, LU.scnd, subset_arr=subset_arr, year=yr)

            primf_harv = util.get_nc_var3d(hndl_nc, var='primf_harv', year=yr, subset_arr=subset_arr)
            primn_harv = util.get_nc_var3d(hndl_nc, var='primn_harv', year=yr, subset_arr=subset_arr)

            if do_global:
                net_trans.append(np.ma.sum(
                    (secondary_to_cropland + secondary_to_pasture + secondary_to_urban +  # from secondary
                     primary_to_cropland + primary_to_pasture + primary_to_urban -  # from primary
                     pasture_to_secondary - cropland_to_secondary - urban_to_secondary +  # Subtracting already secd
                     primf_harv + primn_harv) * area))  # primary
            else:
                net_trans += (secondary_to_cropland + secondary_to_pasture + secondary_to_urban +  # from secondary
                              primary_to_cropland + primary_to_pasture + primary_to_urban -  # from primary
                              pasture_to_secondary - cropland_to_secondary - urban_to_secondary +  # minus prev secd
                              primf_harv + primn_harv)

        if not do_global:
            net_trans /= len(yrs)

        return net_trans

    def glm_2011_diag_net_trans(self):
        """
        :return:
        """
        logger.info('glm_2011_diag_net_trans')

        # Create figure
        fig, ax = plt.subplots()
        ts = self.time[:-1]

        # Transitions
        net_trans_focal = self.net_transitions(path_nc=self.path_nc_trans, area=self.carea, yrs=self.iyr[:-1])
        if self.do_alternate or self.do_LUH1:
            plot.plot_multiple_ts(ax, net_trans_focal, ts, self.out_path + os.sep + 'Net_transitions.png',
                                  title='Net transitions', leg_name=self.legend_glm, ylabel=r'$km^{2}/yr$', col='k',
                                  pos='first')
        else:
            plot.plot_np_ts(ax, net_trans_focal, ts, self.out_path + os.sep + 'Net_transitions.png',
                            title='Net transitions', leg_name=self.legend_glm, ylabel=r'$km^{2}/yr$', col='k')

        if self.do_alternate:
            net_trans_focal2 = self.net_transitions(path_nc=self.path_nc_alt_trans, area=self.carea, yrs=self.iyr[:-1])
            plot.plot_multiple_ts(ax, net_trans_focal2, ts, self.out_path + os.sep + 'Net_transitions.png',
                                  title='Net transitions', leg_name=self.legend_alt_glm, ylabel=r'$km^{2}/yr$', col='g',
                                  pos='last')

        if self.do_LUH1:
            plot.plot_multiple_ts(ax, self.net_trans_focal, ts, self.out_path + os.sep + 'Net_transitions.png',
                                  title='Net transitions', leg_name='LUH1', ylabel=r'$km^{2}/yr$', col='r', pos='last')

        plt.close(fig)

    def gross_transitions(self, path_nc, area, yrs, subset_arr=None, do_global=True):
        """
        Gross transitions are a measure of all land-use change activity; specifically, they are the sum of the
        absolute value of all land-use transitions
        flcp: fraction of each gridcell that transitioned from cropland to pasture
        flpc: fraction of each gridcell that transitioned from pasture to cropland
        flsp: fraction of each gridcell that transitioned from secondary land to pasture
        flps: fraction of each gridcell that transitioned from pasture to secondary land
        flsc: fraction of each gridcell that transitioned from secondary land to cropland
        flcs: fraction of each gridcell that transitioned from cropland to secondary land
        flvc: fraction of each gridcell that transitioned from primary land to cropland
        flvp: fraction of each gridcell that transitioned from primary land to pasture
        fvh1: fraction of each gridcell that had wood harvested from primary forested land
        fvh2: fraction of each gridcell that had wood harvested from primary non-forested land
        fsh1: fraction of each gridcell that had wood harvested from mature secondary forested land
        fsh2: fraction of each gridcell that had wood harvested from young secondary forested land
        fsh3: fraction of each gridcell that had wood harvested from secondary non-forested land
        abs(flcp.data)+abs(flpc.data)+abs(flsp.data)+abs(flps.data)+abs(flsc.data)+abs(flcs.data)+abs(flvc.data)+
        abs(flvp.data)+abs(fvh1.data)+abs(fvh2.data)+abs(fsh1.data)+abs(fsh2.data)+abs(fsh3.data)
        :param path_nc:
        :param area:
        :param yrs:
        :param subset_arr:
        :param do_global: Get a single value showing sum of gross transition over globe
        :return:
        """
        logger.info('gross_transitions')
        hndl_nc = util.open_or_die(path_nc)

        if do_global:
            gross_trans = []
        else:
            gross_trans = np.zeros([len(self.lat), len(self.lon)])

        for yr in tqdm(yrs, desc='gross_transitions', disable=(len(yrs) < 2)):
            cropland_to_pasture = glm.get_transition(path_nc, LU.crop, LU.pastr, subset_arr=subset_arr, year=yr)
            cropland_to_scnd = glm.get_transition(path_nc, LU.crop, LU.scnd, subset_arr=subset_arr, year=yr)
            cropland_to_urban = glm.get_transition(path_nc, LU.crop, LU.urban, subset_arr=subset_arr, year=yr)

            pasture_to_scnd = glm.get_transition(path_nc, LU.pastr, LU.scnd, subset_arr=subset_arr, year=yr)
            pasture_to_urban = glm.get_transition(path_nc, LU.pastr, LU.urban, subset_arr=subset_arr, year=yr)
            pasture_to_cropland = glm.get_transition(path_nc, LU.pastr, LU.crop, subset_arr=subset_arr, year=yr)

            scnd_to_pasture = glm.get_transition(path_nc, LU.scnd, LU.pastr, subset_arr=subset_arr, year=yr)
            scnd_to_urban = glm.get_transition(path_nc, LU.scnd, LU.urban, subset_arr=subset_arr, year=yr)
            scnd_to_cropland = glm.get_transition(path_nc, LU.scnd, LU.crop, subset_arr=subset_arr, year=yr)

            primary_to_pasture = glm.get_transition(path_nc, LU.vrgn, LU.pastr, subset_arr=subset_arr, year=yr)
            primary_to_urban = glm.get_transition(path_nc, LU.vrgn, LU.urban, subset_arr=subset_arr, year=yr)
            primary_to_cropland = glm.get_transition(path_nc, LU.vrgn, LU.crop, subset_arr=subset_arr, year=yr)

            urban_to_pasture = glm.get_transition(path_nc, LU.urban, LU.pastr, subset_arr=subset_arr, year=yr)
            urban_to_scnd = glm.get_transition(path_nc, LU.urban, LU.scnd, subset_arr=subset_arr, year=yr)
            urban_to_cropland = glm.get_transition(path_nc, LU.urban, LU.crop, subset_arr=subset_arr, year=yr)

            # fvh1: fraction of each gridcell that had wood harvested from primary forested land
            primf_harv = util.get_nc_var3d(hndl_nc, var='primf_harv', year=yr, subset_arr=subset_arr)

            # fvh2: fraction of each gridcell that had wood harvested from primary non-forested land
            primn_harv = util.get_nc_var3d(hndl_nc, var='primn_harv', year=yr, subset_arr=subset_arr)

            # fsh1: fraction of each gridcell that had wood harvested from mature secondary forested land
            secmf_harv = util.get_nc_var3d(hndl_nc, var='secmf_harv', year=yr, subset_arr=subset_arr)

            # fsh2: fraction of each gridcell that had wood harvested from young secondary forested land
            secyf_harv = util.get_nc_var3d(hndl_nc, var='secyf_harv', year=yr, subset_arr=subset_arr)

            # fsh3: fraction of each gridcell that had wood harvested from secondary non-forested land
            secnf_harv = util.get_nc_var3d(hndl_nc, var='secnf_harv', year=yr, subset_arr=subset_arr)

            if do_global:
                gross_trans.append(np.ma.sum((cropland_to_pasture + cropland_to_scnd + cropland_to_urban +
                                                 pasture_to_cropland + pasture_to_scnd + pasture_to_urban +
                                                 scnd_to_pasture + scnd_to_cropland + scnd_to_urban + primary_to_cropland +
                                                 primary_to_urban + primary_to_pasture +
                                                 urban_to_cropland + urban_to_pasture + urban_to_scnd + primf_harv +
                                                 primn_harv + secmf_harv + secyf_harv + secnf_harv) * area))
            else:
                gross_trans += (cropland_to_pasture + cropland_to_scnd + cropland_to_urban + pasture_to_cropland +
                               pasture_to_scnd + pasture_to_urban + scnd_to_pasture + scnd_to_cropland + scnd_to_urban +
                               primary_to_cropland + primary_to_urban + primary_to_pasture + urban_to_cropland +
                               urban_to_pasture + urban_to_scnd + primf_harv + primn_harv + secmf_harv + secyf_harv +
                               secnf_harv)

        if not do_global:
            gross_trans /= len(yrs)

        return gross_trans

    def glm_2011_diag_gross_trans(self):
        """
        Time-series of gross transitions
        :return:
        """
        logger.info('glm_2011_diag_gross_trans')

        # Create figure
        fig, ax = plt.subplots()
        ts = self.time[:-1]

        gross_trans_focal = self.gross_transitions(path_nc=self.path_nc_trans, area=self.carea, yrs=self.iyr[:-1])

        if self.do_alternate or self.do_LUH1:
            plot.plot_multiple_ts(ax, gross_trans_focal, ts, self.out_path + os.sep + 'gross_transitions.png',
                                  title='Gross transitions', leg_name=self.legend_glm, ylabel=r'$km^{2}/yr$', col='k',
                                  pos='first')
        else:
            plot.plot_np_ts(ax, gross_trans_focal, ts, self.out_path + os.sep + 'gross_transitions.png',
                            title='Gross transitions', leg_name=self.legend_glm, ylabel=r'$km^{2}/yr$', col='k')

        if self.do_alternate:
            gross_trans_focal2 = self.gross_transitions(path_nc=self.path_nc_alt_trans, area=self.carea, yrs=self.iyr[:-1])
            plot.plot_multiple_ts(ax, gross_trans_focal2, ts, self.out_path + os.sep + 'gross_transitions.png',
                                  title='Gross transitions', leg_name=self.legend_alt_glm, ylabel=r'$km^{2}/yr$',
                                  col='g', pos='last')

        if self.do_LUH1:
            plot.plot_multiple_ts(ax, self.gross_trans_focal, ts, self.out_path + os.sep + 'gross_transitions.png',
                                  title='Gross transitions', leg_name='LUH1', ylabel=r'$km^{2}/yr$', col='r',
                                  pos='last')
        plt.close(fig)

    def Hurtt_2011_diagnostics(self):
        """
        Fig. 7 from Hurtt et al. 2011
        :return:
        """

        logger.info('Hurtt_2011_diagnostics')

        self.glm_2011_diag_gross_trans()
        self.glm_2011_diag_net_trans()
        self.glm_2011_diag_secd_area()
        self.glm_2011_diag_secd_biom()
        self.plot_glm_scnd_diagnostics()

    def plot_regions(self, region, num_regions=2, vert_yr=[]):
        """

        :param region:
        :param num_regions:
        :param vert_yr:
        :return:
        """
        logger.info('plot_regions ' + region)

        self.plot_top_regions_for_LU_state(region, LU_state=LU.crop.name, long_name=LU.crop.name,
                                           num_regions=num_regions, vert_yr=vert_yr)

        for name, lname in self.lus.iteritems():
            self.plot_top_regions_for_LU_state(region, LU_state=name, long_name=lname, num_regions=num_regions,
                                               vert_yr=vert_yr)

    @staticmethod
    def compute_wh(path_nc, years):
        """
        Compute wood harvesting (sum of wood harvest biomass (Pg C) and time-series of wood-harvest biomass)
        :param path_nc:
        :param years:
        :return: sum of wood harvest biomass (Pg C) and time-series of wood-harvest biomass
        """
        # Get handle to transition data
        nc_file = util.open_or_die(path_nc)

        ts_wh = []
        for yr in years:
            # wood harvest biomass from primary forest
            primf_bioh = nc_file.variables['primf_bioh'][int(yr), :, :]

            # wood harvest biomass from primary non-forest
            primn_bioh = nc_file.variables['primn_bioh'][int(yr), :, :]

            # wood harvest biomass from secondary mature forest
            secmf_bioh = nc_file.variables['secmf_bioh'][int(yr), :, :]

            # wood harvest biomass from secondary non-forest
            secnf_bioh = nc_file.variables['secnf_bioh'][int(yr), :, :]

            # wood harvest biomass from secondary young forest
            secyf_bioh = nc_file.variables['secyf_bioh'][int(yr), :, :]

            val = np.ma.sum(primf_bioh + primn_bioh + secmf_bioh + secnf_bioh + secyf_bioh) * constants.KG_TO_PG

            ts_wh.extend([val])

        return ts_wh

    def get_lu_area(self, lu_type, year, subset_arr=None):
        """
        Get sum of global area of LU category
        :param lu_type:
        :param year:
        :param subset_arr:
        :return:
        """
        src_lus = []
        area_sum = 0

        # Check if land-use is main or sub LU
        if isinstance(lu_type, LU):
            src_lus = glm.return_subLU(lu_type)
        else:
            src_lus.extend([lu_type.name])

        for lu in src_lus:
            area_sum += util.sum_netcdf(self.path_data, lu, do_area_wt=True, arr_area=self.carea, date=year,
                                        subset_arr=subset_arr)[0]

        return area_sum

    def wood_clearing_for_ag(self, path_nc, years):
        """
         For each grid-cell this is the sum of transitions from primary to cropland and pasture, multiplied by the cell
         area and the potential biomass density PLUS the sum of transitions from secondary to cropland and pasture,
         multiplied by the cell area and secondary mean biomass density. To get a global total, sum over all grid-cells.
        :param path_nc
        :param area:
        :param years:
        :return:
        """
        logger.info('wood_clearing_for_ag')
        data = util.open_or_die(self.path_data)

        sum_wc_ag = 0.0
        wc_ag = []
        for yr in tqdm(years, desc='wood_clearing_for_ag', disable=(len(years) < 2)):
            primary_to_pasture = glm.get_transition(path_nc, LU.vrgn, LU.pastr, year=yr)  # primary land to pasture
            primary_to_cropland = glm.get_transition(path_nc, LU.vrgn, LU.crop, year=yr)  # primary to cropland

            scnd_to_pasture = glm.get_transition(path_nc, LU.scnd, LU.pastr, year=yr)  # Secondary land to pasture
            scnd_to_cropland = glm.get_transition(path_nc, LU.scnd, LU.crop, year=yr)  # secondary to cropland

            val = np.ma.sum(((primary_to_pasture + primary_to_cropland) * self.vba +
                             (scnd_to_pasture + scnd_to_cropland) * data['secmb'][int(yr), :, :]) * self.fnf *
                            self.carea)

            sum_wc_ag += val
            wc_ag.extend([val])

        return sum_wc_ag, wc_ag

    def ag_land_in_sc(self, nc_sc, years, subset_arr=None):
        """

        :param nc_sc:
        :param years:
        :param subset_arr:
        :return:
        """
        area_ag_to_sc = 0.0
        lus_crp = glm.return_subLU(LU.crop)
        hndl_sc = util.open_or_die(nc_sc)
        hndl_nc = util.open_or_die(self.path_data)

        for yr in years:
            area_ag_sum = 0.0  # Sum of all CFT area

            arr_sc = util.get_nc_var3d(hndl_sc, var='shift_cult', year=yr, subset_arr=subset_arr)

            for lu in lus_crp:
                area_ag_sum += util.get_nc_var3d(hndl_nc, var=lu, year=yr, subset_arr=subset_arr)

            area_ag_to_sc += np.ma.sum(arr_sc * area_ag_sum * self.carea)

        hndl_nc.close()
        hndl_sc.close()

        return area_ag_to_sc

    def plot_compare_sc(self, nc_andreas, fl_butler, years):
        """
        Line plot comparing time-series of ANDREAS vs BUTLER
        :param nc_andreas:
        :param fl_butler:
        :param years:
        :return:
        """
        logger.info('plot_compare_sc')
        arr_andreas = []
        arr_butler = []
        hndl_andreas = util.open_or_die(nc_andreas)
        hndl_nc = util.open_or_die(self.path_data)

        asc_butler = np.genfromtxt(fl_butler, skip_header=0, delimiter=' ')  # 2D butler map

        # Sub land-uses for crop and pasture
        lus_crops = glm.return_subLU(LU.crop)
        lus_pastr = glm.return_subLU(LU.pastr)

        cmap = palettable.colorbrewer.sequential.PuRd_9.mpl_colormap
        cmap = plot.truncate_colormap(cmap, 0.1, 1.0)

        # Plot Andreas map for multiple time-steps (1500, 1850, 1980, 2015)
        for yr in [1500, 1850, 1980, 2015, 2030, 2050, 2075, 2090, 2100]:
            asc_andreas = util.get_nc_var3d(hndl_andreas, var='shift_cult', year=yr - self.start_yr)
            plot.plot_arr_to_map(asc_andreas, self.lon, self.lat,
                                 out_path=self.out_path + os.sep + 'Andreas_' + str(yr) + '.png',
                                 var_name='Andreas', plot_type='sequential',
                                 annotate_date=True, yr=yr, xlabel='Fraction of cropland area in gridcell',
                                 title='Andreas shifting cultivation region',
                                 cmap=cmap,
                                 any_time_data=False, land_bg=True, grid=True)
        pdb.set_trace()
        # Plot Butler map (single time-step)
        area_ag_sum = 0.0
        area_pastr_sum = 0.0
        for lu_crop in lus_crops:
            area_ag_sum += util.get_nc_var3d(hndl_nc, var=lu_crop, year=1980 - self.start_yr)

        for lu_pastr in lus_pastr:
            area_pastr_sum += util.get_nc_var3d(hndl_nc, var=lu_pastr, year=1980 - self.start_yr)
        plot.plot_arr_to_map(asc_butler * (area_ag_sum + area_pastr_sum), self.lon, self.lat,
                             out_path=self.out_path + os.sep + 'Butler_1980.png',
                             var_name='Butler', plot_type='sequential',
                             annotate_date=True, yr=1980, xlabel='Fraction of cropland area in grid cell',
                             title='Butler shifting cultivation region',
                             cmap=cmap,
                             any_time_data=False, land_bg=True, grid=True)

        for yr in years:
            area_ag_sum = 0.0  # Sum of all CFT area
            area_pastr_sum = 0.0  # Sum of all pasture areas
            asc_andreas = util.get_nc_var3d(hndl_andreas, var='shift_cult', year=yr)

            for lu_crop in lus_crops:
                area_ag_sum += util.get_nc_var3d(hndl_nc, var=lu_crop, year=yr)

            for lu_pastr in lus_pastr:
                area_pastr_sum += util.get_nc_var3d(hndl_nc, var=lu_pastr, year=yr)

            # Andreas: cropland* SC_mask * cell area
            arr_andreas.append(np.ma.sum(asc_andreas * area_ag_sum * self.carea))
            arr_butler.append(np.ma.sum(asc_butler * (area_ag_sum + area_pastr_sum) * self.carea))

        # Butler:  1/15*(cropland + pasture)* SC_mask * cell area
        arr_butler = [x / 15. for x in arr_butler]

        fig, ax = plt.subplots()
        plot.plot_multiple_ts(ax, np.array(arr_andreas), np.asarray(years + self.start_yr), self.out_path + os.sep +
                              'compare_andreas_butler.png', vert_yr=[1850, 1980], leg_name='Andreas',
                              ylabel=r'$Area\ (km^{2})$', pos='first')
        plot.plot_multiple_ts(ax, np.array(arr_butler), np.asarray(years + self.start_yr), self.out_path + os.sep +
                              'compare_andreas_butler.png', vert_yr=[1850, 1980], leg_name='Hurtt, 2011',
                              title='Shifting cultivation area (Andreas, Hurtt 2011)', ylabel=r'$Area\ (km^{2})$',
                              pos='last')
        plt.close(fig)

    def human_impact_land(self, years, subset_arr=None):
        """

        :param years:
        :param subset_arr:
        :return:
        """
        logger.info('human_impact_land')

        area_global = 0.0
        area_impacted_human = 0.0

        for yr in tqdm(years, desc='human_impact_land', disable=(len(years) < 2)):
            # Total global land area
            area_global += self.get_lu_area(LU.vrgn, yr, subset_arr=subset_arr) + \
                           self.get_lu_area(LU.scnd, yr, subset_arr=subset_arr) + \
                           self.get_lu_area(LU.urban, yr, subset_arr=subset_arr) + \
                           self.get_lu_area(LU.pastr, yr, subset_arr=subset_arr) + \
                           self.get_lu_area(LU.crop, yr, subset_arr=subset_arr)

            area_impacted_human += self.get_lu_area(LU.crop, yr, subset_arr=subset_arr) + \
                                   self.get_lu_area(LU.pastr, yr, subset_arr=subset_arr) + \
                                   self.get_lu_area(LU.scnd, yr, subset_arr=subset_arr)

        return (area_impacted_human * 100.0)/area_global

    def get_rs_forest_loss(self, only_forest=False, subset_arr=None):
        """
        :param only_forest: Should only areas with biomass > 2kgC/m^2 be used in computing forest loss or all areas?
        :param subset_arr:
        :return: global_forest_loss, sc_forest_loss, no_sc_forest_loss
        """
        rs_data = util.open_or_die(self.rs_forest)
        fnf = 1.0 if only_forest is False else self.fnf
        subset = 1.0 if subset_arr is None else subset_arr

        # Global rs forest loss. Only use the areas which are classified as forest by glm
        global_forest_loss = np.ma.sum(rs_data * self.carea * fnf * subset)

        # Determine forest loss in sc areas
        sc_forest_loss = np.ma.sum(rs_data * self.asc_sc * fnf * self.carea * subset)

        # Determine forest loss outside of sc areas
        no_sc_forest_loss = global_forest_loss - sc_forest_loss

        return global_forest_loss, sc_forest_loss, no_sc_forest_loss

    def get_glm_forest_loss(self, path_nc, years, subset_arr=None):
        """
        glm transitions that we are currently including towards meeting the Landsat forest loss are:
        primary forest harvest
        primary forest to crop transitions
        primary forest to managed pasture transitions
        secondary mature forest harvest, secondary to crop transitions when secondary land is > 20 years old
        secondary to managed pasture transitions when secondary land is > 20 years old
        secondary immature forest harvest when secondary land is > 20 years old
        :param path_nc:
        :param years:
        :param subset_arr:
        :return:
        """
        df = pandas.DataFrame(columns=('transition', 'area'))
        hndl_nc = util.open_or_die(path_nc)
        hndl_data = util.open_or_die(self.path_data)

        # Just get the shape and fill with 0.0
        glm_forest_loss = np.zeros_like(util.get_nc_var3d(hndl_nc, var='primf_harv', year=0))

        primf_harv = np.zeros([len(self.lat), len(self.lon)])
        primary_to_cropland = np.zeros([len(self.lat), len(self.lon)])
        primary_to_managed = np.zeros([len(self.lat), len(self.lon)])
        secmf_harv = np.zeros([len(self.lat), len(self.lon)])
        secondary_to_crop = np.zeros([len(self.lat), len(self.lon)])
        secondary_to_managed_pasture = np.zeros([len(self.lat), len(self.lon)])
        secyf_harv = np.zeros([len(self.lat), len(self.lon)])

        for yr in tqdm(years, desc='get_glm_forest_loss', disable=(len(years) < 2)):
            # primary forest harvest, fvh1: fraction of each gridcell that had wood harvested from primary forested land
            primf_harv = primf_harv + util.get_nc_var3d(hndl_nc, var='primf_harv', year=yr, subset_arr=subset_arr)

            # primary forest to crop transitions
            primary_to_cropland = primary_to_cropland + glm.get_transition(path_nc, SubLU.primf, LU.crop, year=yr,
                                                                           subset_arr=subset_arr)

            # primary forest to managed pasture transitions
            primary_to_managed = primary_to_managed + glm.get_transition(path_nc, SubLU.primf, SubLU.pastr, year=yr,
                                                                         subset_arr=subset_arr)

            # secondary mature forest harvest
            secmf_harv = secmf_harv + util.get_nc_var3d(hndl_nc, var='secmf_harv', year=yr, subset_arr=subset_arr)

            # Select secondary land with age > 20 years
            age_secondary = util.get_nc_var3d(hndl_data, var='secma', year=yr, subset_arr=subset_arr)
            age_secondary[age_secondary <= constants.MATURITY_AGE] = 0.0
            age_secondary[age_secondary > 0.0] = 1.0
            mature_secondary = age_secondary

            # secondary to crop transitions when secondary land is > MATURITY_AGE years old
            secondary_to_crop = secondary_to_crop + glm.get_transition(path_nc, SubLU.secdf, LU.crop, year=yr) * \
                                                    mature_secondary

            # secondary to managed pasture transitions when secondary land is > MATURITY_AGE years old
            secondary_to_managed_pasture = secondary_to_managed_pasture + \
                                           glm.get_transition(path_nc, SubLU.secdf, SubLU.pastr, year=yr) * \
                                           mature_secondary

            # secondary immature forest harvest when secondary land is > MATURITY_AGE years old
            secyf_harv = secyf_harv + util.get_nc_var3d(hndl_nc, var='secyf_harv', year=yr, subset_arr=subset_arr) * \
                                      mature_secondary

        # Plot actvity matrix for Landsat transitions
        df.loc[len(df)] = ['primf_to_secdf', np.ma.sum(primf_harv * self.carea)]
        df.loc[len(df)] = ['primf_to_crop', np.ma.sum(primary_to_cropland * self.carea)]
        df.loc[len(df)] = ['primf_to_pasture', np.ma.sum(primary_to_managed * self.carea)]
        df.loc[len(df)] = ['secdf_to_secdf', np.ma.sum(secmf_harv * self.carea)]
        df.loc[len(df)] = ['secondary_to_crop', np.ma.sum(secondary_to_crop * self.carea)]
        df.loc[len(df)] = ['secondary_to_managed_pasture', np.ma.sum(secondary_to_managed_pasture * self.carea)]
        df.loc[len(df)] = ['secyf_to_secyf', np.ma.sum(secyf_harv * self.carea)]

        if len(years) > 1:
            title = 'Landsat transitions: ' + \
                    str(years[0] + self.start_yr) + ' - ' + str(years[-1] + self.start_yr) + '\n' + \
                    r'Area transitions $(km^{2}/yr)$: ' + '{:.2e}'.format(df['area'].sum())
            out_fname = 'hmap_landsat_' + str(years[0] + self.start_yr) + ' - ' + str(years[-1] + self.start_yr) + '.png'
        else:
            title = 'Landsat transitions: ' + str(years[0] + self.start_yr) + '\n' + \
                    r'Area transitions $(km^{2}/yr)$: ' + '{:.2e}'.format(df.sum().sum())
            out_fname = 'hmap_landsat_' + str(years[0] + self.start_yr) + '.png'

        df = df.set_index('transition').T.rename_axis(None, axis=1).reset_index(drop=True)
        df_trans = util.transitions_to_matrix(df)
        df_trans = df_trans.divide(len(years))

        import seaborn.apionly as sns
        cmap = sns.cubehelix_palette(start=2.8, rot=.1, light=0.9, as_cmap=True)
        cmap.set_under('gray')  # 0 values in activity matrix are shown in gray (inactive transitions)
        plot.plot_activity_matrix(df_trans, cmap, out_path=self.out_path + os.sep + out_fname, annotate=True,
                                  normalized=False, title=title)

        glm_forest_loss = glm_forest_loss + primf_harv + primary_to_cropland + primary_to_managed + secmf_harv + \
                          secondary_to_crop + secondary_to_managed_pasture + secyf_harv
        global_glm_forest_loss = np.ma.sum(glm_forest_loss * self.carea)

        # Determine forest loss in sc areas
        binary_sc = self.asc_sc[:]
        binary_sc[(binary_sc > 0.0) & (binary_sc < 65536.0)] = 1.0

        sc_glm_forest_loss = np.ma.sum(glm_forest_loss * binary_sc * self.carea)

        # Determine forest loss outside of sc areas.
        no_sc_forest_loss = global_glm_forest_loss - sc_glm_forest_loss

        return global_glm_forest_loss, sc_glm_forest_loss, no_sc_forest_loss

    def get_biomass(self, years, ulat=90.0, llat=-90.0, llon=-180.0, rlon=180.0, subset_arr=None):
        """

        :param years:
        :param ulat:
        :param llat:
        :param llon:
        :param rlon:
        :param subset_arr:
        :return:
        """
        data = util.open_or_die(self.path_data)

        sum_biom = 0.0
        for yr in years:
            # Secondary biomass = secondary forested * cell_area * secondary biomass density
            subset_secdf = util.extract_from_ascii(data['secdf'][int(yr), :, :], ulat=ulat, llat=llat, llon=llon,
                                                   rlon=rlon, res=self.resolution, subset_arr=subset_arr)
            subset_secmb = util.extract_from_ascii(data['secmb'][int(yr), :, :], ulat=ulat, llat=llat, llon=llon,
                                                   rlon=rlon, res=self.resolution, subset_arr=subset_arr)
            subset_area = util.extract_from_ascii(self.carea, ulat=ulat, llat=llat, llon=llon, rlon=rlon,
                                                  res=self.resolution, subset_arr=subset_arr)
            secdf_biom = np.ma.sum(subset_secdf * subset_area * subset_secmb)

            # Primary biomass = primary forested * cell_area * primary biomass density
            subset_primf = util.extract_from_ascii(data['primf'][int(yr), :, :], ulat=ulat, llat=llat, llon=llon,
                                                   rlon=rlon, res=self.resolution, subset_arr=subset_arr)
            subset_vba = util.extract_from_ascii(self.vba, ulat=ulat, llat=llat, llon=llon, rlon=rlon,
                                                 res=self.resolution, subset_arr=subset_arr)
            primf_biom = np.ma.sum(subset_primf * subset_area * subset_vba)

            sum_biom += primf_biom + secdf_biom

        return sum_biom

    def secnd_forest_increase(self, last_yr=2000, first_yr=1700):
        """
        :param last_yr:
        :param first_yr:
        :return:
        """

        increase_scnd = (self.get_lu_area(LU.scnd, last_yr) -
                         self.get_lu_area(LU.scnd, first_yr)) * constants.TO_MILLION

        frst_scnd = (self.get_lu_area(SubLU.secdf, last_yr) -
                     self.get_lu_area(SubLU.secdf, first_yr)) * constants.TO_MILLION

        return frst_scnd * 100.0 / increase_scnd

    def get_glm_forest_area(self, year, subset_arr=None):
        """
        Return forest area (secdf + primf) for given year
        :param year:
        :return:
        """
        return self.get_lu_area(SubLU.secdf, year, subset_arr=subset_arr) + \
               self.get_lu_area(SubLU.primf, year, subset_arr=subset_arr)

    def get_MLU_biomass(self, subset_arr=None):
        """
        Returns potential biomass on FOREST areas
        :param subset_arr:
        :return:
        """
        if subset_arr is not None:
            return np.ma.sum(self.vba * self.carea * self.fnf * subset_arr)
        else:
            return np.ma.sum(self.vba * self.carea * self.fnf)

    def get_ag_clearing_towards_wh(self, wh_start_yr=850, wh_rampdown_yr=1850, wh_cutoff_yr=1920):
        """

        :param wh_start_yr:
        :param wh_rampdown_yr:
        :param wh_cutoff_yr:
        :return:
        """
        if wh_start_yr > wh_cutoff_yr:
            return 0.0

        hndl_wh = util.open_or_die(constants.wh_file)
        hndl_trans = util.open_or_die(self.path_nc_trans)

        global_ag_clearing_used = np.zeros(wh_cutoff_yr - self.start_yr)

        data = util.open_or_die(self.path_data)

        # hndl_tran = xr.open_dataset(self.path_nc_trans, chunks={'time': 10})

        data_wh_hist = hndl_wh.variables['woodharvest'][:]

        for yr in tqdm(self.iyr[wh_start_yr - self.start_yr:wh_cutoff_yr - self.start_yr],
                       desc='get_ag_clearing_towards_wh',
                       disable=(len(self.iyr[wh_start_yr - self.start_yr:wh_cutoff_yr - self.start_yr]) < 2)):
            data_ssmb = data.variables['secmb'][int(yr), :, :]
            ag_clearing = np.zeros_like(data_ssmb)

            # Ag clearing contribution towards wh ramps down from 1.0 in 1850 to 0.0 in 1920
            ramp_factor = 1.0 if yr <= (wh_start_yr - self.start_yr) else 0.0 if yr > (wh_cutoff_yr - self.start_yr) \
                else (1. - float(yr + self.start_yr - wh_rampdown_yr)/(wh_cutoff_yr - wh_rampdown_yr))

            # Ramp factor should be between 0. and 1.
            ramp_factor = max(0.0, min(ramp_factor, 1.0))

            for lu_from in [SubLU.primf.name, SubLU.secdf.name]:
                for lu_to in [SubLU.c3ann.name, SubLU.c3per.name, SubLU.c4per.name, SubLU.c3nfx.name, SubLU.pastr.name,
                              SubLU.range.name]:

                    ag_flow_data = util.get_nc_var3d(hndl_trans, var=lu_from + '_to_' + lu_to, year=yr)

                    if lu_from == SubLU.primf.name:
                        ag_clearing = ag_clearing + ag_flow_data * self.carea * self.vba * ramp_factor
                    else:
                        ag_clearing = ag_clearing + ag_flow_data * self.carea * data_ssmb * ramp_factor

            for idx, cntr in enumerate(self.ccodes):
                national_wh_input = data_wh_hist[yr, idx]

                # Get mask for country
                cntr_map = self.get_bool_map_region(self.map_ccodes, self.ccodes[idx])

                national_ag_clearing = np.ma.sum(ag_clearing * cntr_map) * 1000

                if national_ag_clearing >= national_wh_input:
                    global_ag_clearing_used[yr] += national_wh_input
                else:
                    global_ag_clearing_used[yr] += national_ag_clearing

        return np.ma.sum(global_ag_clearing_used) * 1e-9

    def compute_global_fertilizer_use(self, years, subset_arr=None):
        """
        Total global fertilizer use = sum over world grids of sum over crop_types of
        (rate_by_crop_type [kgN/ha] X area_crop_type [ha])
        :param years:
        :param subset_arr:
        :return: Units: TgN/yr
        """
        logger.info('compute_global_fertilizer_use')
        hndl_nc = util.open_or_die(self.path_data)
        hndl_mgt = util.open_or_die(self.path_nc_mgt)

        global_fertilizer_use = 0.0

        # Sum over crop types x rate_by_crop_type x area_crop_type
        for key in self.nmgt:
            name_crop = key.split('_')[1]
            for yr in years:

                cft_N = util.get_nc_var3d(hndl_nc, var=name_crop, year=yr, subset_arr=subset_arr) * \
                        util.get_nc_var3d(hndl_mgt, var=key, year=yr, subset_arr=subset_arr)
                global_fertilizer_use += np.ma.sum(cft_N * self.carea)

        return global_fertilizer_use * constants.KM2_TO_HA * constants.KG_TO_TG

    def compute_global_irrigated_area(self, years, subset_arr=None):
        """
        Total global irrigated area = sum over world grid of sum over crop_types of
        (fraction_of_grid_cell_irrigated x area_crop_type)
        :param years:
        :param subset_arr:
        :return: Units: million ha
        """
        logger.info('compute_global_irrigated_area')
        hndl_nc = util.open_or_die(self.path_data)
        hndl_mgt = util.open_or_die(self.path_nc_mgt)

        global_irrigated_area = 0.0

        for key in self.irrg:
            name_crop = key.split('_')[1]
            for yr in years:
                cft_irr = util.get_nc_var3d(hndl_nc, var=name_crop, year=yr, subset_arr=subset_arr) * \
                          util.get_nc_var3d(hndl_mgt, var=key, year=yr, subset_arr=subset_arr)
                global_irrigated_area += np.ma.sum(cft_irr * self.carea)

        return global_irrigated_area * constants.TO_MILLION

    def get_forest_secma(self, var_name='secma', yr=2015, subset_arr=None):
        """

        :param var_name:
        :param yr:
        :param subset_arr:
        :return:
        """
        logger.info('get_forest_secma')
        hndl_nc = util.open_or_die(self.path_data)

        # Get secma/secmb data for given yr
        arr = util.get_nc_var3d(hndl_nc, var=var_name, year=yr - self.start_yr, subset_arr=subset_arr)

        # Subset by forest/non-forest map
        # TODO: Should we use forest/non-forest mask
        # arr = np.ma.masked_where(self.fnf <= 0.0, arr)

        return arr

    def biof_diagnostic(self, id_country=840.0, year=2015, name_crp='c4ann'):
        """

        :param id_country:
        :param year:
        :param name_crp:
        :return:
        """
        # 3.3 million ha (or 33000 km2) of corn was used for corn-based ethanol in U.S. in 2004 (Ref. Section III.A,
        # page 3 of http://science.sciencemag.org/content/sci/suppl/2008/02/06/1151861.DC1/Searchinger.SOM.pdf
        hndl_mgt = util.open_or_die(self.path_mgt_data)
        hndl_crp = util.open_or_die(self.path_data)

        for count, (name, lname) in enumerate(self.biof.iteritems(), 1):
            if name != 'crpbf_' + name_crp:
                continue

            arr = util.get_nc_var3d(hndl_mgt, var=name, year=year - self.start_yr) * \
                  util.get_nc_var3d(hndl_crp, var=name.split('_')[1], year=year - self.start_yr)

        # Get binary map of country
        map_country = self.get_bool_map_region(self.map_ccodes, id_country)

        return (arr * map_country * self.carea).sum()

    def compute_biodiv_area(self, year, subset_arr=None):
        """
        Compute % of # Current global land surface covered by natural vegetation in the biodiversity hotspots
        :param year:
        :param subset_arr:
        :return:
        """
        logger.info('compute_biodiv_area')
        hndl_nc = util.open_or_die(self.path_data)

        # Compute global area
        area_globe = np.ma.sum(self.carea)

        # Current global land surface covered by natural vegetation in the biodiversity hotspots
        arr_primary = util.get_nc_var3d(hndl_nc, SubLU.primf.name, year, subset_arr=subset_arr) + \
                      util.get_nc_var3d(hndl_nc, SubLU.primn.name, year, subset_arr=subset_arr)

        if subset_arr is None:
            area_primary_in_hotspot = np.ma.sum(self.hnh * self.carea * arr_primary)
        else:
            area_primary_in_hotspot = np.ma.sum(self.hnh * self.carea * arr_primary * subset_arr)

        hndl_nc.close()

        return area_primary_in_hotspot * 100./area_globe

    def area_US_forest_scnd(self):
        """

        :return:
        """
        logger.info('area_US_forest_scnd')

        map_usa = self.get_bool_map_region(self.map_ccodes, id_region=840)
        area_usa_secdf = self.get_lu_area(SubLU.secdf, 2000, subset_arr=map_usa)

        area_usa_forest = self.get_lu_area(SubLU.secdf, 2000, subset_arr=map_usa) + \
                          self.get_lu_area(SubLU.primf, 2000, subset_arr=map_usa)

        per_usa_forest_scnd = area_usa_secdf * 100./area_usa_forest

        return per_usa_forest_scnd

    def per_scnd_increase_forest(self, subset_arr=None):
        """
        % of secondary land increase that is forested (1700 - 2000)
        :return:
        """
        logger.info('per_scnd_increase_forest')

        forested_scnd_2000 = self.get_lu_area(SubLU.secdf, 2000, subset_arr=subset_arr)
        forested_scnd_1700 = self.get_lu_area(SubLU.secdf, 1700, subset_arr=subset_arr)

        scnd_2000 = self.get_lu_area(LU.scnd, 2000, subset_arr=subset_arr)
        scnd_1700 = self.get_lu_area(LU.scnd, 1700, subset_arr=subset_arr)

        per_increase_forested_scnd = (forested_scnd_2000 - forested_scnd_1700) * 100./(scnd_2000 - scnd_1700)

        return per_increase_forested_scnd

    def create_diagnostics_table(self, name_dataset='', region=None, type_region=None, name_region='Global_'):
        """
        TODO Fix indexes in the comments below
        Land use area
        1. * Cropland area (1990)
        2. * Pasture area (1990)
        3. * Primary land area (1990)

        Transitions
        4. Total gross transitions (2000)
        5. Total net transitions (2000)

        Human land use impacts
        3. * % land impacted by human land use (850)
        4. * % land impacted by human land use (1700)
        5. * % land impacted by human land use (2000)
        6. Secondary land increase (1700 - 2000)
        7. Secondary land increase (forest, 1700 - 2000)
        8. Secondary land increase (non-forest, 1700 - 2000)

        Wood harvest and agricultural clearing
        9. Wood harvest (1850 - 1990)
        10. * Wood harvest (850 - 1990)
        11. * Wood harvest (1500 - 1990)
        12. Wood harvest (1961 - 2010)
        13. Wood clearing for agriculture (1850 - 1990)
        14. * Wood clearing for agriculture (850 - 1990)
        15. * Agricultural clearing contribution to wood harvest (850 - 1850)
        16. * Agricultural clearing contribution to wood harvest (1500 - 1850)

        Shifting cultivation
        17. * Agricultural land undergoing shifting cultivation (2000)
        18. Agricultural land undergoing shifting cultivation (1980)
        19. * Agricultural land undergoing shifting cultivation (1850)

        Forest loss and area
        30. * Potential forest area (850)
        29. * Forest area (2015)
        23. * Forest loss (2000 - 2011)
        24. * Forest loss Inside Andreas shifting cultivation region (2000 - 2011)
        25. * Forest loss Outside Andreas shifting cultivation region (2000 - 2011)
        26. * Forest loss, GLM forest definition (2000 - 2011)
        27. * Forest loss, GLM forest definition, Inside Andreas shifting cultivation region (2000 - 2011)
        28. * Forest loss, GLM forest definition, Outside Andreas shifting cultivation region (2000 - 2011)

        Management
        31. * Fertilizer use (2012)
        32. * Irrigated area (2003)

        Biomass
        35. * Biomass estimate (2005)
        33. * Potential plant carbon
        34. * Pantropical above-ground biomass (2007 - 2008)

        36. % of secondary land increase that is forested (1700 - 2000)
        37. % of secondary land generated by wood harvest and shifting cultivation (850 - 2015)
        38. Percentage of US Forests that are secondary (2000)
        39. Mean age of Eastern US Secondary Forests (2000)
        40. Area of forested land in shifting cultivation fallow (2000)
        :param name_dataset:
        :param region: Default 'Global'.
        :return:
        """
        logger.info('create_diagnostics_table ' + name_dataset + ' ' + name_region[:-1])

        df_diag = pandas.DataFrame(columns=('Diagnostic', 'Time-period', 'Region', name_dataset, 'Hurtt 2011',
                                            'Hurtt 2006', 'Units', 'Reference data'))

        # Get 2D mask for continent/country/SC from global map
        if region is not None:
            type_region = self.get_map_region(type_region=type_region)
            arr_region = self.get_bool_map_region(type_region, id_region=region)
        else:
            logger.error('Incorrect/Unimplemented region')
            arr_region = None

        df_diag.loc[len(df_diag)] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = ['Land use area', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = [' * Cropland area',
                                     '1990',
                                     name_region,
                                     self.get_lu_area(LU.crop, 1990, subset_arr=arr_region) * constants.TO_MILLION,
                                     15.1,
                                     12.1,
                                     '10^6 km^2',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * Pasture area',
                                     '1990',
                                     name_region,
                                     self.get_lu_area(LU.pastr, 1990, subset_arr=arr_region) * constants.TO_MILLION,
                                     33.1,
                                     25.8,
                                     '10^6 km^2',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * Primary land area',
                                     '1990',
                                     name_region,
                                     self.get_lu_area(LU.vrgn, 1990, subset_arr=arr_region) * constants.TO_MILLION,
                                     58.4,
                                     57.7,
                                     '10^6 km^2',
                                     np.nan]

        df_diag.loc[len(df_diag)] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = ['Transitions', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = ['Total gross transitions',
                                     '2000',
                                     name_region,
                                     self.gross_transitions(path_nc=self.path_nc_trans, area=self.carea,
                                                            subset_arr=arr_region,
                                                            yrs=np.asarray([2000 - self.start_yr]))[0] *
                                     constants.TO_MILLION,
                                     2.9,
                                     '0.55-4.2',
                                     '10^6 km^2',
                                     np.nan]
        df_diag.loc[len(df_diag)] = ['Total net transitions',
                                     '2000',
                                     name_region,
                                     self.net_transitions(path_nc=self.path_nc_trans, area=self.carea,
                                                          subset_arr=arr_region,
                                                          yrs=np.asarray([2000 - self.start_yr]))[0] *
                                     constants.TO_MILLION,
                                     0.17,
                                     '0.0-0.17',
                                     '10^6 km^2',
                                     np.nan]

        df_diag.loc[len(df_diag)] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = ['Human land use impacts', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = ['% of secondary land increase that is forested',
                                     '1700 - 2000',
                                     name_region,
                                     self.per_scnd_increase_forest(subset_arr=arr_region),
                                     47.0,
                                     50.0,
                                     '%',
                                     np.nan]
        df_diag.loc[len(df_diag)] = ['Percentage of US Forests that are secondary',
                                     '2000',
                                     'USA',
                                     self.area_US_forest_scnd(),
                                     100.0,
                                     '94.0-99.0',
                                     '%',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * Global area covered by natural vegetation in biodiversity hotspots',
                                     '2005',
                                     name_region,
                                     self.compute_biodiv_area(year=2005 - self.start_yr),
                                     4.6,
                                     np.nan,
                                     '%',
                                     'Mittermeier et al. 2005: 2.3']
        df_diag.loc[len(df_diag)] = [' * Median secondary mean age',
                                     '2015',
                                     'Global',
                                     np.nanmedian(self.get_forest_secma(yr=2015, subset_arr=arr_region)),
                                     np.nan,
                                     np.nan,
                                     'years',
                                     '30 - 40 years, Ben Poulter, NACP 2013']
        df_diag.loc[len(df_diag)] = [' * 80th percentile secondary mean age',
                                     '2015',
                                     'Global',
                                     np.nanpercentile(self.get_forest_secma(yr=2015, subset_arr=arr_region), 80),
                                     np.nan,
                                     np.nan,
                                     'years',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * 90th percentile secondary mean age',
                                     '2015',
                                     'Global',
                                     np.nanpercentile(self.get_forest_secma(yr=2015, subset_arr=arr_region), 90),
                                     np.nan,
                                     np.nan,
                                     'years',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * % land impacted by human land use',
                                     '850',
                                     name_region,
                                     self.human_impact_land(years=[850], subset_arr=arr_region),
                                     np.nan,
                                     np.nan,
                                     '%',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * % land impacted by human land use',
                                     '1700',
                                     name_region,
                                     self.human_impact_land(years=[1700], subset_arr=arr_region),
                                     np.nan,
                                     np.nan,
                                     '%',
                                     np.nan]
        df_diag.loc[len(df_diag)] = ['% land impacted by human land use',
                                     '2000',
                                     name_region,
                                     self.human_impact_land(years=[2000], subset_arr=arr_region),
                                     60.0,
                                     '42-68',
                                     '%',
                                     np.nan]
        df_diag.loc[len(df_diag)] = ['Secondary land increase',
                                     '1700 - 2000',
                                     name_region,
                                     (self.get_lu_area(LU.scnd, 2000, subset_arr=arr_region) -
                                      self.get_lu_area(LU.scnd, 1700, subset_arr=arr_region)) * constants.TO_MILLION,
                                     28.9,
                                     '10.0-44.0',
                                     '10^6 km^2',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * Secondary land increase (forest)',
                                     '1700 - 2000',
                                     name_region,
                                     (self.get_lu_area(SubLU.secdf, 2000, subset_arr=arr_region) -
                                      self.get_lu_area(SubLU.secdf, 1700, subset_arr=arr_region)) * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * Secondary land increase (non-forest)',
                                     '1700 - 2000',
                                     name_region,
                                     (self.get_lu_area(SubLU.secdn, 2000, subset_arr=arr_region) -
                                      self.get_lu_area(SubLU.secdn, 1700, subset_arr=arr_region)) * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * Secondary land increase',
                                     '850 - 2015',
                                     name_region,
                                     (self.get_lu_area(LU.scnd, 2015, subset_arr=arr_region) -
                                      self.get_lu_area(LU.scnd, 850, subset_arr=arr_region)) * constants.TO_MILLION,
                                     28.9,
                                     '10.0-44.0',
                                     '10^6 km^2',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * Secondary land increase (forest)',
                                     '850 - 2015',
                                     name_region,
                                     (self.get_lu_area(SubLU.secdf, 2015, subset_arr=arr_region) -
                                      self.get_lu_area(SubLU.secdf, 850, subset_arr=arr_region)) * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * Secondary land increase (non-forest)',
                                     '850 - 2015',
                                     name_region,
                                     (self.get_lu_area(SubLU.secdn, 2015, subset_arr=arr_region) -
                                      self.get_lu_area(SubLU.secdn, 850, subset_arr=arr_region)) * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     np.nan]

        df_diag.loc[len(df_diag)] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = ['Wood harvest and agricultural clearing', np.nan, np.nan, np.nan, np.nan,
                                     np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = [' * Total wood clearing',
                                     '850 - 1990',
                                     'Global',
                                     self.wood_clearing_for_ag(path_nc=self.path_nc_trans,
                                                               years=range(850 - self.start_yr,
                                                                           1991 - self.start_yr))[0] *
                                     constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     'Pg C',
                                     'Direct wood harvest LUH1: 159.2, Kaplan high-case: 514, Kaplan low-case: 216.8']
        direct_wh = np.ma.sum(glm.compute_wh(self.path_nc_trans, range(850 - self.start_yr, 1991 - self.start_yr)))
        ag_to_wh = self.get_ag_clearing_towards_wh(wh_start_yr=850)
        df_diag.loc[len(df_diag)] = [' * Total wood harvest',
                                     '850 - 1990',
                                     'Global',
                                     direct_wh + ag_to_wh,
                                     np.nan,
                                     np.nan,
                                     'Pg C',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * Direct wood harvest',
                                     '850 - 1990',
                                     'Global',
                                     direct_wh,
                                     np.nan,
                                     np.nan,
                                     'Pg C',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * Agricultural clearing contribution to wood harvest',
                                     '850 - 1990',
                                     'Global',
                                     ag_to_wh,
                                     np.nan,
                                     np.nan,
                                     'Pg C',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * Total wood clearing',
                                     '1500 - 1990',
                                     'Global',
                                     self.wood_clearing_for_ag(path_nc=self.path_nc_trans,
                                                               years=range(1500 - self.start_yr,
                                                                           1991 - self.start_yr))[0] *
                                     constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     'Pg C',
                                     'Direct wood harvest LUH1: 121.9, Kaplan high-case: 356.3, Kaplan low-case: 167.9']
        direct_wh = np.ma.sum(glm.compute_wh(self.path_nc_trans, range(1500 - self.start_yr, 1991 - self.start_yr)))
        ag_to_wh = self.get_ag_clearing_towards_wh(wh_start_yr=1500)
        df_diag.loc[len(df_diag)] = [' * Total wood harvest',
                                     '1500 - 1990',
                                     'Global',
                                     direct_wh + ag_to_wh,
                                     np.nan,
                                     np.nan,
                                     'Pg C',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * Direct wood harvest',
                                     '1500 - 1990',
                                     'Global',
                                     direct_wh,
                                     np.nan,
                                     np.nan,
                                     'Pg C',
                                     np.nan]
        df_diag.loc[len(df_diag)] = [' * Agricultural clearing contribution to wood harvest',
                                     '1500 - 1990',
                                     'Global',
                                     ag_to_wh,
                                     np.nan,
                                     np.nan,
                                     'Pg C',
                                     np.nan]
        # df_diag.loc[len(df_diag)] = ['Direct wood harvest',
        #                              '1850 - 1990',
        #                              name_region,
        #                              np.ma.sum(glm.compute_wh(self.path_nc_trans, range(1850 - self.start_yr,
        #                                                                                  1991 - self.start_yr))),
        #                              84.0,
        #                              85.0,
        #                              'Pg C',
        #                              'Houghton, 1999: 106. Direct wood harvest LUH1: 87.8, Kaplan high-case: 123.4, '
        #                              'Kaplan low-case: 94.3']
        # df_diag.loc[len(df_diag)] = ['Direct wood harvest',
        #                              '1961 - 2010',
        #                              name_region,
        #                              np.ma.sum(glm.compute_wh(self.path_nc_trans, range(1961 - self.start_yr,
        #                                                                                  2011 - self.start_yr))),
        #                              np.nan,
        #                              np.nan,
        #                              'Pg C',
        #                              'FAO: 82. Direct wood harvest LUH1: 58.9, Kaplan high-case: 58.9, Kaplan low-case: 58.9']
        # df_diag.loc[len(df_diag)] = ['Total wood clearing',
        #                              '1850 - 1990',
        #                              name_region,
        #                              self.wood_clearing_for_ag(nc_file=self.path_nc_trans,
        #                                                        years=range(1850 - self.start_yr,
        #                                                                    1991 - self.start_yr))[0] *
        #                              constants.TO_MILLION,
        #                              170.0,
        #                              '105.0-158.0',
        #                              'Pg C',
        #                              'Houghton, 1999: 149']
        # df_diag.loc[len(df_diag)] = [' * Total wood clearing',
        #                              '1961 - 2010',
        #                              name_region,
        #                              self.wood_clearing_for_ag(nc_file=self.path_nc_trans,
        #                                                        years=range(1961 - self.start_yr,
        #                                                                    2011 - self.start_yr))[0] *
        #                              constants.TO_MILLION,
        #                              np.nan,
        #                              np.nan,
        #                              'Pg C',
        #                              np.nan]

        # df_diag.loc[len(df_diag)] = [' * Agricultural clearing contribution to wood harvest',
        #                              '1850 - 1990',
        #                              name_region,
        #                              self.get_ag_clearing_towards_wh(wh_start_yr=1850),
        #                              np.nan,
        #                              np.nan,
        #                              'Pg C',
        #                              np.nan]
        # df_diag.loc[len(df_diag)] = [' * Agricultural clearing contribution to wood harvest',
        #                              '1961 - 2010',
        #                              name_region,
        #                              self.get_ag_clearing_towards_wh(wh_start_yr=1960),
        #                              np.nan,
        #                              np.nan,
        #                              'Pg C',
        #                              np.nan]

        df_diag.loc[len(df_diag)] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = ['Shifting cultivation', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = ['Agricultural land undergoing shifting cultivation',
                                     '2000',
                                     name_region,
                                     self.ag_land_in_sc(constants.NC_ANDREAS, np.asarray([2000 - self.start_yr]),
                                                        subset_arr=arr_region) * constants.TO_MILLION,
                                     0.58,
                                     '0.48-0.65',
                                     '10^6 km^2/yr',
                                     'Andreas, Ole personal communication: 0.3']
        df_diag.loc[len(df_diag)] = ['* Agricultural land undergoing shifting cultivation',
                                     '1980',
                                     name_region,
                                     self.ag_land_in_sc(constants.NC_ANDREAS, np.asarray([1980 - self.start_yr]),
                                                        subset_arr=arr_region) * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2/yr',
                                     'Rojstaczer, 2001: 0.2-0.6']
        df_diag.loc[len(df_diag)] = [' * Agricultural land undergoing shifting cultivation',
                                     '1850',
                                     name_region,
                                     self.ag_land_in_sc(constants.NC_ANDREAS, np.asarray([1850 - self.start_yr]),
                                                        subset_arr=arr_region) * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2/yr',
                                     np.nan]

        df_diag.loc[len(df_diag)] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = ['Forest loss and area', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        # Forest area globally over time: http://www.sciencedirect.com/science/article/pii/S0378112715003400
        df_diag.loc[len(df_diag)] = [' * Potential forest area',
                                     '850',
                                     name_region,
                                     self.get_glm_forest_area(850, subset_arr=arr_region) * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     'Miami-LU potential forest area: ' + '%.2f' %
                                     (np.ma.sum(self.carea * self.fnf) * constants.TO_MILLION)]
        df_diag.loc[len(df_diag)] = [' * Forest area',
                                     '2015',
                                     name_region,
                                     self.get_glm_forest_area(2015, subset_arr=arr_region) * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     'Sexton, 2016: 32.1-41.4']
        df_diag.loc[len(df_diag)] = [' * Forest loss',
                                     '2000 - 2011',
                                     name_region,
                                     self.get_glm_forest_loss(path_nc=self.path_nc_trans,
                                                              years=range(2000 - self.start_yr,
                                                                          2012 - self.start_yr),
                                                              subset_arr=arr_region)[0]
                                     * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     self.get_rs_forest_loss(subset_arr=arr_region)[0] * constants.TO_MILLION]
        df_diag.loc[len(df_diag)] = [' * Forest loss',
                                     '2000 - 2011',
                                     'Inside Andreas shifting cultivation region: ' + name_region,
                                     self.get_glm_forest_loss(path_nc=self.path_nc_trans,
                                                              years=range(2000 - self.start_yr,
                                                                          2012 - self.start_yr),
                                                              subset_arr=arr_region)[1]
                                     * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     self.get_rs_forest_loss(subset_arr=arr_region)[1] * constants.TO_MILLION]
        df_diag.loc[len(df_diag)] = [' * Forest loss',
                                     '2000 - 2011',
                                     'Outside Andreas shifting cultivation region: ' + name_region,
                                     self.get_glm_forest_loss(path_nc=self.path_nc_trans,
                                                              years=range(2000 - self.start_yr,
                                                                          2012 - self.start_yr),
                                                              subset_arr=arr_region)[2]
                                     * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     self.get_rs_forest_loss(subset_arr=arr_region)[2] * constants.TO_MILLION]
        df_diag.loc[len(df_diag)] = [' * Forest loss, (GLM forest definition)',
                                     '2000 - 2011',
                                     name_region,
                                     self.get_glm_forest_loss(path_nc=self.path_nc_trans,
                                                              years=range(2000 - self.start_yr,
                                                                          2012 - self.start_yr),
                                                              subset_arr=arr_region)[0]
                                     * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     self.get_rs_forest_loss(only_forest=True, subset_arr=arr_region)[0] *
                                     constants.TO_MILLION]
        df_diag.loc[len(df_diag)] = [' * Forest loss, (GLM forest definition)',
                                     '2000 - 2011',
                                     'Inside Andreas shifting cultivation region: ' + name_region,
                                     self.get_glm_forest_loss(path_nc=self.path_nc_trans,
                                                              years=range(2000 - self.start_yr,
                                                                          2012 - self.start_yr),
                                                              subset_arr=arr_region)[1]
                                     * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     self.get_rs_forest_loss(only_forest=True, subset_arr=arr_region)[1] *
                                     constants.TO_MILLION]
        df_diag.loc[len(df_diag)] = [' * Forest loss, (GLM forest definition)',
                                     '2000 - 2011',
                                     'Outside Andreas shifting cultivation region: ' + name_region,
                                     self.get_glm_forest_loss(path_nc=self.path_nc_trans,
                                                              years=range(2000 - self.start_yr,
                                                                          2012 - self.start_yr),
                                                              subset_arr=arr_region)[2]
                                     * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     self.get_rs_forest_loss(only_forest=True, subset_arr=arr_region)[2] *
                                     constants.TO_MILLION]

        df_diag.loc[len(df_diag)] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = ['Management', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = [' * Fertilizer use',
                                     '2012',
                                     name_region,
                                     self.compute_global_fertilizer_use(years=[2012 - self.start_yr],
                                                                        subset_arr=arr_region),
                                     np.nan,
                                     np.nan,
                                     'Tg N/yr',
                                     'Zhang, 2016: 100']
        df_diag.loc[len(df_diag)] = [' * Irrigated area',
                                     '2003',
                                     name_region,
                                     self.compute_global_irrigated_area(years=[2003 - self.start_yr],
                                                                        subset_arr=arr_region),
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     'FAO: 2.77']
        # Searchinger, 2008: http://science.sciencemag.org/content/sci/suppl/2008/02/06/1151861.DC1/Searchinger.SOM.pdf
        df_diag.loc[len(df_diag)] = [' * Biofuel area (corn)',
                                     '2004',
                                     'USA',
                                     self.biof_diagnostic(id_country=840.0, year=2004, name_crp='c4ann') *
                                     constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     'Searchinger, 2008: 0.03']
        # Martinelli, 2008: http://onlinelibrary.wiley.com/doi/10.1890/07-1813.1/full
        df_diag.loc[len(df_diag)] = [' * Biofuel area (sugarcane)',
                                     '2007',
                                     'Brazil',
                                     self.biof_diagnostic(id_country=76.0, year=2007, name_crp='c4per') *
                                     constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     '10^6 km^2',
                                     'Martinelli, 2008: 0.07 (Total area of sugarcane)']
        # biof_diagnostic(self, id_country=840.0, year=2015, name_crp='c4ann')
        df_diag.loc[len(df_diag)] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = ['Biomass', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = [' * Potential plant carbon',
                                     np.nan,
                                     name_region,
                                     self.get_MLU_biomass(subset_arr=arr_region) * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     'Pg C',
                                     'Prentice, 2001: 466-654']
        df_diag.loc[len(df_diag)] = [' * Pantropical above-ground biomass',
                                     '2007 - 2008',
                                     name_region,
                                     self.get_biomass(years=[2007 - self.start_yr], ulat=23.0, llat=-23.0,
                                                      subset_arr=arr_region) * constants.TO_MILLION,
                                     np.nan,
                                     np.nan,
                                     'Pg C',
                                     'Saatchi, 2011: 206.5; Baccini, 2012: 228.7; Avitabile, 2016: 187.5']
        # Biomass estimate: http://www.annualreviews.org/doi/full/10.1146/annurev-ecolsys-110512-135914
        df_diag.loc[len(df_diag)] = [' * Biomass estimate',
                                     '2005',
                                     name_region,
                                     self.get_biomass(years=[2005 - self.start_yr], subset_arr=arr_region) *
                                     constants.TO_MILLION * constants.AGB_TO_BIOMASS,
                                     np.nan,
                                     np.nan,
                                     'Pg C',
                                     'Pan, 2013: 393.4, (annualreviews.org, Table 2)']

        df_diag.loc[len(df_diag)] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        df_diag.loc[len(df_diag)] = ['% of secondary land generated by wood harvest and shifting cultivation',
                                     '850 - 2015',
                                     'Global',
                                     np.nan,
                                     86.0,
                                     '70.0-90.0',
                                     '%',
                                     np.nan]
        df_diag.loc[len(df_diag)] = ['Mean age of Eastern US Secondary Forests',
                                     '2000',
                                     'Global',
                                     np.nan,
                                     63.0,
                                     71.0,
                                     'years',
                                     38.0]
        df_diag.loc[len(df_diag)] = ['Area of forested land in shifting cultivation fallow',
                                     '2000',
                                     'Global',
                                     np.nan,
                                     3.9,
                                     '4.56-6.19',
                                     '10^6 km^2',
                                     'Lanly, 1985: 4.0']

        path_diag_csv = self.out_path + os.sep + name_region + 'diagnostics_table_' + self.name_exp + '.xlsx'
        df_diag.to_excel(path_diag_csv, encoding='UTF-8', float_format='%11.2f')

        return df_diag

    def get_list_transitions(self, do_cfts=True):
        """

        :param do_cfts:
        :return:
        """
        names_matrix = []
        for lu_from in SubLU:
            for lu_to in SubLU:
                # Should the CFTs be tracked?
                if not do_cfts:
                    name_from = lu_from.name
                    name_to = lu_to.name

                    if self.is_cropland(lu_from.name):
                        name_from = 'crop'
                    if self.is_cropland(lu_to.name):
                        name_to = 'crop'
                else:
                    # Do cropland functional types
                    name_from = lu_from.name
                    name_to = lu_to.name

                names_matrix.extend([name_from + '_to_' + name_to])

        return names_matrix

    def is_cropland(self, cft):
        """
        Is cft a crop type or not (return True if it is)
        :param cft:
        :return:
        """
        if cft in self.crop:
            return True
        else:
            return False

    def plot_hmp_transitions(self, years, do_cfts=True, do_norm=False, do_circlize=True):
        """

        :param years:
        :param do_cfts:
        :param do_norm:
        :param do_circlize:
        :return:
        """
        logger.info('plot_hmp_transitions ' + str(years[0] + self.start_yr) + ' ' + str(years[-1] + self.start_yr))
        hndl_trans = util.open_or_die(self.path_nc_trans)

        names_matrix = self.get_list_transitions(do_cfts=do_cfts)
        name_cfts = 'cft_' if do_cfts else ''

        df_sum_trans = pandas.DataFrame(0, columns=names_matrix, index=np.arange(1))

        # Loop through all pairs of transitions and store in dataframe
        for lu_from in SubLU:
            for lu_to in SubLU:
                sum_trans = 0.0
                # Should the CFTs be plotted?
                if not do_cfts:
                    name_from = lu_from.name
                    name_to = lu_to.name

                    # crop rotations will not be computed
                    if self.is_cropland(lu_from.name) and self.is_cropland(lu_to.name):
                        continue

                    if self.is_cropland(lu_from.name):
                        name_from = 'crop'
                    if self.is_cropland(lu_to.name):
                        name_to = 'crop'
                else:
                    # Do cropland functional types
                    name_from = lu_from.name
                    name_to = lu_to.name

                if (lu_to == SubLU.primf) or (lu_to == SubLU.primn) or (lu_from == lu_to) or \
                        (lu_from == SubLU.primf and lu_to == SubLU.primn) or \
                        (lu_from == SubLU.primn and lu_to == SubLU.primf) and not \
                        (lu_from == SubLU.secdf and lu_to == SubLU.secdf) and not \
                        (lu_from == SubLU.secdn and lu_to == SubLU.secdn):
                    # Cannot transition to primary forested/non-forested
                    # Cannot transition to itself (except secdf to itself and secdn to itself)
                    sum_trans = 0.0
                else:
                    # for primf_to_secdf use primf_harv
                    # for primn_to_secdn use primn_harv
                    # for secdf_to_secdf use secyf_harv plus secmf_harv
                    # for secdn_to_secdn use secnf_harv
                    # (don't use any of the info already in primf_to_secdf or primn_to_secdn)
                    for yr in years:
                        if lu_from == SubLU.primf and lu_to == SubLU.secdf:
                            # fraction of each gridcell that had wood harvested from primary forested land
                            sum_trans += util.sum_area_nc(self.path_nc_trans, 'primf_harv', self.carea, yr)
                        elif lu_from == SubLU.primn and lu_to == SubLU.secdn:
                            # fraction of each gridcell that had wood harvested from primary non-forested land
                            sum_trans += util.sum_area_nc(self.path_nc_trans, 'primn_harv', self.carea, yr)
                        elif lu_from == SubLU.secdf and lu_to == SubLU.secdf:
                            # fraction of each gridcell that had wood harvested from mature secondary forested land +
                            # fraction of each gridcell that had wood harvested from young secondary forested land
                            sum_trans += np.ma.sum(self.carea * (util.get_nc_var3d(hndl_trans, 'secmf_harv', yr) +
                                                                 util.get_nc_var3d(hndl_trans, 'secyf_harv', yr)))
                        elif lu_from == SubLU.secdn and lu_to == SubLU.secdn:
                            # fraction of each gridcell that had wood harvested from secondary non-forested land
                            sum_trans += util.sum_area_nc(self.path_nc_trans, 'secnf_harv', self.carea, yr)
                        else:
                            sum_trans += np.ma.sum(self.carea * glm.get_transition(self.path_nc_trans, lu_from,
                                                                                   lu_to, year=yr))

                # Add previous transitions (important when CFTs are not tracked)
                df_sum_trans[name_from + '_to_' + name_to] = sum_trans + \
                                                             np.unique(df_sum_trans[name_from + '_to_' + name_to])[0]

        # convert pandas dataframe to matrix that can be used for heatmap
        df_trans = util.transitions_to_matrix(df_sum_trans)
        df_trans = df_trans.divide(len(years))
        if do_circlize:
            df_trans.to_csv(self.out_path + os.sep + 'df_trans_' + str(len(years)) + '_' +
                            str(years[0] + self.start_yr) + '.csv')

        # Reorder columns (solely to conform with transition matrices presented in Hurtt et al. 2011)
        reorder_columns = self.all_columns if do_cfts else self.no_cft_columns
        df_trans = df_trans.reindex(index=reorder_columns, columns=reorder_columns)

        if len(years) > 1:
            title = 'Year(s): ' + \
                    str(years[0] + self.start_yr) + ' - ' + str(years[-1] + self.start_yr) + '\n' + \
                    r'Area transitions $(km^{2}/yr)$: ' + '{:.2e}'.format(df_trans.sum().sum())
            out_fname = name_cfts + str(years[0] + self.start_yr) + ' - ' + str(years[-1] + self.start_yr)
        else:
            title = 'Year(s): ' + str(years[0] + self.start_yr) + '\n' + \
                    r'Area transitions $(km^{2}/yr)$: ' + '{:.2e}'.format(df_trans.sum().sum())
            out_fname = name_cfts + str(years[0] + self.start_yr)

        # Normalize matrix so that sum of all transitions is 1.0, do this after sum transitions in title has computed
        if do_norm:
            df_trans = df_trans/df_trans.sum().sum()

        import seaborn.apionly as sns
        cmap = sns.cubehelix_palette(start=2.8, rot=.1, light=0.9, as_cmap=True)
        cmap.set_under('gray')  # 0 values in activity matrix are shown in gray (inactive transitions)
        plot.plot_activity_matrix(df_trans, cmap, out_path=self.out_path + os.sep + 'hmap_' + out_fname + '.png',
                                  annotate=True, normalized=do_norm, title=title)

    def plot_transitions_map(self, years, do_gross=False):
        """
        Plot map of gross/net transitions for one or more years
        :param do_gross: Plot gross transitions or net transition on map (True/False)
        :param years: Transitions across years are averaged
        :return:
        """
        imgs_all = []
        type_trans = 'Gross' if do_gross else 'Net'

        if len(years) > 1:
            title = str(years[0] + self.start_yr) + ' - ' + str(years[-1] + self.start_yr)
            out_fname = type_trans + '_' + str(years[0] + self.start_yr) + ' - ' + str(years[-1] + self.start_yr)
        else:
            title = str(years[0] + self.start_yr)
            out_fname = type_trans + '_' + str(years[0] + self.start_yr)

        # Transitions (net/gross)
        if do_gross:
            trans = self.gross_transitions(path_nc=self.path_nc_trans, area=self.carea, yrs=years, do_global=False)
            # cmap = palettable.colorbrewer.sequential.PuRd_9.mpl_colormap
            # cmap = plot.truncate_colormap(cmap, 0.1, 1.0)
        else:
            trans = self.net_transitions(path_nc=self.path_nc_trans, area=self.carea, yrs=years, do_global=False)
        cmap = palettable.colorbrewer.diverging.RdYlBu_11_r.mpl_colormap

        # Replace 0 by nan
        trans[trans == 0.0] = np.nan

        # Output kml for cumulative gross transitions expressed as fraction of grid-cell area
        logger.info('plot_transitions_map:output_kml ' + type_trans + ' ' + title)
        plot.output_kml(trans * len(years), self.lon, self.lat, path_out=self.out_path, do_log_cb=True,
                        xmin=np.nanmin(trans * len(years)), xmax=np.nanmax(trans * len(years)),
                        step=(np.nanmax(trans * len(years)) - np.nanmin(trans * len(years)))/10.,
                        cmap=cmap,
                        fname_out='cumulative_fraction_' + out_fname, name_legend='cumulative_' + out_fname + ' ' +
                                                                                  title,
                        label='Fraction of grid cell area')

        # Output map for averaged transitions (per year) expressed in fraction of grid cell area
        plot.output_kml(trans, self.lon, self.lat, path_out=self.out_path, do_log_cb=True,
                        xmin=np.nanmin(trans), xmax=np.nanmax(trans), step=(np.nanmax(trans) - np.nanmin(trans)) / 10.,
                        cmap=cmap,
                        fname_out='averaged_fraction_' + out_fname, name_legend='averaged_' + out_fname + ' ' + title,
                        label='Fraction of grid cell area')

        # Multiply by area
        trans = trans * self.carea

        # Determine x axis limits of colorbar and step size
        xmin = np.nanmin(trans)
        xmax = np.nanmax(trans)
        step = (xmax - xmin) / 10.

        # Output kml for averaged transitions (per year) expressed in areal units
        plot.output_kml(trans, self.lon, self.lat, path_out=self.out_path, do_log_cb=True,
                        xmin=xmin, xmax=xmax, step=step, cmap=cmap,
                        fname_out='averaged_area_' + out_fname, name_legend='averaged_' + out_fname,
                        label=r'Area $(km^{2})$')

        # Output kml for cumulative gross transitions expressed in areal units
        plot.output_kml(trans, self.lon, self.lat, path_out=self.out_path, do_log_cb=True,
                        xmin=xmin, xmax=xmax, step=step, cmap=cmap,
                        fname_out='cumulative_area_' + out_fname, name_legend='cumulative_' + out_fname + ' ' + title,
                        label=r'Area $(km^{2})$')

        # Output map for averaged transitions (per year) expressed in areal units
        logger.info('plot_transitions_map:plot_arr_to_map ' + type_trans + ' ' + title)
        plot.plot_arr_to_map(trans, self.lon, self.lat, self.out_path + os.sep + out_fname,
                             xaxis_min=xmin, xaxis_max=xmax, xaxis_step=step, any_time_data=False,
                             xlabel=r'Area $(km^{2})$', title=type_trans + ' transitions: ' + title,
                             format='%.1f', land_bg=True, cmap=cmap, grid=True)

        # Create maps of glm past transitions
        logger.info('plot_transitions_map:make_movie ' + type_trans + ' ' + title)
        for yr in self.iyr[::constants.MOVIE_SEP]:
            # Transitions (net/gross)
            if do_gross:
                trans = self.gross_transitions(path_nc=self.path_nc_trans, area=self.carea, yrs=[yr], do_global=False)
                #cmap = palettable.colorbrewer.sequential.PuRd_9.mpl_colormap
            else:
                trans = self.net_transitions(path_nc=self.path_nc_trans, area=self.carea, yrs=[yr], do_global=False)
            cmap = palettable.colorbrewer.diverging.RdYlBu_11_r.mpl_colormap
            # Determine x axis limits of colorbar and step size
            xmin = np.nanmin(trans * self.carea)
            xmax = np.nanmax(trans * self.carea)
            step = (xmax - xmin) / 10.

            imgs_all.extend(plot.plot_maps_ts((trans * self.carea) + 0.1, [yr + self.start_yr], self.lon, self.lat,
                                              out_path=self.movie_imgs_path,
                                              xaxis_min=xmin, xaxis_max=xmax, xaxis_step=step,
                                              save_name='GLM_' + type_trans + '_' + str(yr + self.start_yr),
                                              cmap=cmap, title=type_trans,
                                              xlabel=r'Area $(km^{2})$', start_movie_yr=850,
                                              land_bg=True, do_log_cb=True, grid=True))
        plot.make_movie(imgs_all, self.movies_path, out_fname='GLM_' + out_fname + '.gif')

    def plot_compare_glm_hyde_forest(self, year, hyde_ver='3.2'):
        """
        Scatter plot comparing glm and hyde forest area estimates for year
        :param year:
        :param hyde_ver:
        :return:
        """
        logger.info('plot_compare_glm_hyde_forest')
        hndl_nc = util.open_or_die(self.path_data)

        glm_cntr_area = []
        hyde_cntr_area = []

        # List of country codes
        codes_cntr = self.map_ccodes

        # Create hyde object
        obj = process_HYDE.HYDE(constants.HYDE32_OUT_PATH, ver=hyde_ver)

        # For glm region, extract data from states.nc for 'year'
        hndl_glm = util.open_or_die(self.path_data)
        # For hyde region, extract data from obj.hyde_othr_path for 'other'
        hndl_hyde = util.open_or_die(obj.HYDE_othr_path)

        # Get glm forest area
        glm_forest_area = util.get_nc_var3d(hndl_nc, var=SubLU.primf.name, year=year) * self.carea + \
                          util.get_nc_var3d(hndl_nc, var=SubLU.secdf.name, year=year) * self.carea

        # Get hyde forest area (multiply by self.fnf to remove non-forest)
        hndl_hyde = util.open_or_die(obj.HYDE_othr_path)
        hyde_forest_area = util.get_nc_var3d(hndl_hyde, obj.lus.get(obj.HYDE_othr_path)[0], year) * self.carea * \
                           self.fnf

        # For each country extract LU_state data for 'year'
        for idx, reg in enumerate(np.unique(codes_cntr[codes_cntr > 0.0])):
            glm_cntr_area.append(np.ma.sum(glm_forest_area[codes_cntr[:] == reg][:]))
            hyde_cntr_area.append(np.ma.sum(hyde_forest_area[codes_cntr[:] == reg][:]))

        # Output to csv: country id, glm_cntr, area, hyde_cntr_area
        df_glm_hyde = pandas.DataFrame({'Country ID': np.unique(codes_cntr[codes_cntr > 0.0]).tolist(),
                                        'Country name': self.names_cntr,
                                        'GLM': glm_cntr_area,
                                        'HYDE': hyde_cntr_area})
        df_glm_hyde.to_csv(self.out_path + os.sep + 'compare_glm_hyde_forest.csv')

        # Get regression line and compute-rsquared
        import statsmodels.api as sm
        m, b = np.polyfit(np.array(hyde_cntr_area), np.array(glm_cntr_area), 1)
        results = sm.OLS(np.array(glm_cntr_area), sm.add_constant(np.array(hyde_cntr_area)), missing='drop').fit()

        # Plot scatter plot between hyde (x-axis) and glm (y-axis) forested area
        fig, ax = plt.subplots()

        # Same axis-length (min, max) for both axis
        axis_max = max(np.array(hyde_cntr_area).max(), np.array(glm_cntr_area).max())
        plt.xlim(0.0, axis_max)
        plt.ylim(0.0, axis_max)

        # Draw diagonal 1:1 line
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls='--', c='.3', alpha=0.5)
        # Draw regression line
        plt.plot(np.array(hyde_cntr_area), m * np.array(hyde_cntr_area) + b, '-', color='red', alpha=0.5)

        plt.scatter(hyde_cntr_area, glm_cntr_area, c=self.cols[0], lw=0,
                    label=r'$r^{2}$ ' + '{:0.2f}'.format(results.rsquared))

        # Create nice-looking grid for ease of visualization and plot x and y axis labels
        ax.grid(which='major', alpha=0.5, linestyle='--')
        ax.set_title(str(year + self.start_yr))
        ax.set_xlabel(r'$HYDE\ forest\ area\ (km^{2})$')
        ax.set_ylabel(r'$GLM\ forest\ area\ (km^{2})$')

        # Create a legend with transparent box around it
        leg = plt.legend(loc='upper right', fancybox=None)
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_alpha(0.5)

        # Final layout adjustment and output
        plt.tight_layout()
        plt.savefig(self.out_path + os.sep + 'scatter_hyde_glm_' + str(year) + '.png', dpi=constants.DPI)
        plt.close('all')

        # Plot map of GLM forest area
        map_glm = util.get_nc_var3d(hndl_nc, var=SubLU.primf.name, year=year) + \
                  util.get_nc_var3d(hndl_nc, var=SubLU.secdf.name, year=year)

        cmap = palettable.colorbrewer.sequential.PuRd_9.mpl_colormap
        cmap = plot.truncate_colormap(cmap, 0.1, 1.0)

        plot.plot_arr_to_map(map_glm, self.lon, self.lat, out_path=self.out_path + os.sep + 'map_GLM.png',
                             var_name='GLM', plot_type='sequential',
                             annotate_date=True, yr=int(year + self.start_yr), xlabel='Fraction of grid cell area',
                             title='GLM forest (fraction of grid cell area)',
                             cmap=cmap,
                             any_time_data=False, land_bg=True, grid=True)

        # Plot HYDE map
        map_hyde = util.get_nc_var3d(hndl_hyde, obj.lus.get(obj.HYDE_othr_path)[0], year) * self.fnf
        plot.plot_arr_to_map(map_hyde, self.lon, self.lat, out_path=self.out_path + os.sep + 'map_HYDE.png',
                             var_name='HYDE', plot_type='sequential',
                             annotate_date=True, yr=int(year + self.start_yr), xlabel='Fraction of grid cell area',
                             title='HYDE forest (fraction of grid cell area)',
                             cmap=cmap,
                             any_time_data=False, land_bg=True, grid=True)

    def map_wood_harvest(self, year):
        """

        :param year:
        :return:
        """
        logger.info('Plotting map of wood harvest for given year')
        assert not isinstance(year, list)

        hndl_trans = util.open_or_die(self.path_nc_trans)
        cmap = palettable.colorbrewer.sequential.PuRd_9.mpl_colormap
        cmap = plot.truncate_colormap(cmap, 0.1, 1.0)

        # Wood-harvest map
        arr = np.zeros([len(self.lat), len(self.lon)])
        for count, (name, lname) in enumerate(self.whrv.iteritems(), 1):
            logger.info('wood-harvest: ' + name)
            arr = arr + hndl_trans.variables[name][int(year), :, :]

        # Plot whrv
        plot.plot_arr_to_map(arr, self.lon, self.lat, out_path=self.out_path + os.sep + 'mgt_whrv.png',
                             var_name='whrv', plot_type='sequential', annotate_date=True,
                             yr=int(year + self.start_yr),
                             xlabel='Fraction of grid cell area',
                             title='Wood harvest',
                             cmap=cmap,
                             any_time_data=False, land_bg=True, grid=True)

        plot.output_kml(arr, self.lon, self.lat, path_out=self.out_path,
                        xmin=np.nanmin(arr), xmax=np.nanmax(arr), step=(np.nanmax(arr) - np.nanmin(arr))/10.,
                        cmap=cmap,
                        fname_out='mgt_whrv', name_legend='mgt_whrv ' + str(year), label='Fraction of grid cell area')

        # Wood-biomass map
        arr = np.zeros([len(self.lat), len(self.lon)])
        for count, (name, lname) in enumerate(self.wbio.iteritems(), 1):
            logger.info('wood-biomass: ' + name)
            arr = arr + (hndl_trans.variables[name][int(year), :, :] * constants.KG_TO_MG)

        # Plot wbio
        plot.plot_arr_to_map(arr, self.lon, self.lat, out_path=self.out_path + os.sep + 'mgt_wbio.png',
                             var_name='wbio', plot_type='sequential', annotate_date=True,
                             xaxis_min=0.0, xaxis_max=np.nanmax(arr), xaxis_step=np.nanmax(arr)/10.0,
                             yr=int(year + self.start_yr),
                             xlabel='Wood harvest biomass (Mg C)',
                             title='Wood harvest',
                             cmap=cmap,
                             any_time_data=False, land_bg=True, grid=True)

        plot.output_kml(arr, self.lon, self.lat, path_out=self.out_path,
                        xmin=np.nanmin(arr), xmax=np.nanmax(arr), step=(np.nanmax(arr) - np.nanmin(arr)) / 10.,
                        cmap=cmap,
                        fname_out='mgt_wbio', name_legend='mgt_wbio ' + str(year), label='Wood harvest biomass (Mg C)')

    def plot_wood_harvest_ts(self, years):
        """

        :param years:
        :return:
        """
        logger.info('plot_wood_harvest_ts')

        hndl_trans = util.open_or_die(self.path_nc_trans)

        fig, ax = plt.subplots()
        ts_wh = []
        arr = np.zeros([len(self.lat), len(self.lon)])
        # wood harvest
        for yr in years:
            for count, (name, lname) in enumerate(self.whrv.iteritems(), 1):
                arr = arr + hndl_trans.variables[name][int(yr), :, :]

            val = np.ma.sum(arr * self.carea)

            ts_wh.extend([val])
        plot.plot_np_ts(ax, ts_wh, years + self.start_yr, self.out_path + os.sep + 'ts_wh_area.png',
                        title='Wood harvest area', leg_name='Wood harvest area', ylabel=r'$Area\ (km^{2})$',
                        col=self.cols[0])
        plt.close(fig)

        fig, ax = plt.subplots()
        ts_wh = []
        arr = np.zeros([len(self.lat), len(self.lon)])
        # wood harvest biomass
        for yr in years:
            for count, (name, lname) in enumerate(self.wbio.iteritems(), 1):
                arr = arr + hndl_trans.variables[name][int(yr), :, :]

            val = np.ma.sum(arr) * constants.KG_TO_PG

            ts_wh.extend([val])

        plot.plot_np_ts(ax, ts_wh, years + self.start_yr, self.out_path + os.sep + 'ts_wh_biomass.png',
                        title='Wood harvest biomass', leg_name='Wood harvest biomass', ylabel='Pg C', col=self.cols[0])
        plt.close(fig)

    def get_poulter_age(self):
        """

        :return:
        """
        # Description from Ben Poulter
        # On the forest age dataset, I developed this as part of a geocarbon project. The dataset is based on regional
        # forest age datasets from forest inventory, which is then downscaled to a gridded product using MODIS land
        # cover. So when you reaggregate back to the inventory region, you get back the inventory age distributions.
        # For the tropics, an age-biomass curve was used to estimate age at 1 km resolution.
        sum_ages = []
        hndl_nc = util.open_or_die(constants.age_poulter)

        age_classes = hndl_nc.variables['Class'][:]

        df_age = pandas.DataFrame(columns=('Age-class', 'Area'))

        for idx, age_class in enumerate(age_classes):
            pft_ages = hndl_nc.variables['age'][idx]
            # 1. Add across the 4 PFT classes
            # 2. Convert from 0.5 to 0.25 degree
            # 3. Multiply by cell area
            sum_ages = (util.upscale_np_arr(pft_ages.sum(axis=0), block_size=2) * self.carea).sum()
            df_age.loc[len(df_age)] = [str(idx*10 + 1) + '-' + str(idx*10 + 10), sum_ages]

        weights = df_age['Area'].values[:-1]
        values = df_age.index.values[:-1]
        np.average(values, weights=weights)

    def output_glm_csvs(self, path_nc, type_region, lup_region):
        """

        :param path_nc:
        :param type_region:
        :param lup_region:
        :return:
        """
        logger.info('output_glm_csvs')
        hndl_nc = util.open_or_die(self.path_data)

        # Output csv's for all states and transitions
        dir_csvs = self.out_path + os.sep + 'csvs'
        util.make_dir_if_missing(dir_csvs)

        list_ts = map(int, util.get_nc_var1d(hndl_nc, self.tme_name).tolist())

        for name_var, varin in hndl_nc.variables.iteritems():
            df = pandas.DataFrame(columns=('code', 'name', 'variable') + tuple(list_ts))

            # Ignore: time, lat, lon, lat_bounds, lon_bounds
            if name_var in self.ignore_nc_vars:
                continue

            for id_cont, name_region in lup_region.iteritems():
                if name_region == 'Antartica':
                    continue

                name_cols = [id_cont, name_region, name_var]
                name_cols.extend(self.glm_by_region(path_nc, name_var, type_region, id_cont))
                df.loc[len(df)] = name_cols

            df.to_csv(dir_csvs + os.sep + type_region + '_' + name_var + '.csv')

        hndl_nc.close()


def do_diagnostics_table(region=None, fname='', map_region=None):
    """

    :param region: None for Global diagnostics
    :param fname:
    :param map_region:
    :return:
    """
    logger.info('####################')
    logger.info('do_diagnostics_table')
    logger.info('####################')

    df_list = []
    df_ts_list = []

    for exp in constants.glm_experiments:
        logger.info(exp)
        obj = glm(name_exp=exp,
                  path_input=constants.path_glm_input + os.sep + exp,
                  time_period='past',
                  out_path=constants.path_glm_output + os.sep + exp)

        df = obj.create_diagnostics_table(name_dataset=obj.name_exp, region=region, type_region=map_region,
                                          name_region=fname)
        df_list.append(df)

        # Get 2D mask for region/country
        if region is not None:
            arr_region = obj.get_bool_map_region(obj.map_ccodes, id_region=region)
        else:
            arr_region = None

        # Output time-series of LU's
        for var in LU:
            df_ts = obj.get_ts_as_df(obj.path_data, freq_yr=100, var=var, subset_arr=arr_region)
            df_ts_list.append(df_ts)

        # Output time-series of SubLU's
        for var in SubLU:
            df_ts = obj.get_ts_as_df(obj.path_data, freq_yr=100, var=var, subset_arr=arr_region)
            df_ts_list.append(df_ts)

        # Output LUs as csv
        frame_ts = pandas.concat(df_ts_list, axis=1)
        frame_ts.to_csv(obj.out_path + os.sep + fname + 'ts_AllLUs.csv')

    # Concatenate to one big data frame
    frame = pandas.concat(df_list, axis=1)
    # Remove duplicate columns
    dups = util.duplicate_columns(frame)
    frame = frame.drop(dups, axis=1)

    # Reorder columns so that Hurtt 2011, Hurtt 2006, Units are at the very end
    frame = frame[[c for c in frame if c not in constants.ending_diag_cols] +
                  [c for c in constants.ending_diag_cols if c in frame]]
    frame.to_excel(constants.path_glm_output + os.sep + 'diagnostic_table_' +
                   datetime.datetime.now().strftime("%B_%d_%Y") + '.xlsx', float_format='%11.2f')


def loop_region_glm_csvs():
    """
    Loop across 'continent' or 'country'
    :return:
    """
    for exp in constants.glm_experiments:
        logger.info(exp)
        obj = glm(name_exp=exp,
                  path_input=constants.path_glm_input + os.sep + exp,
                  time_period='past',
                  out_path=constants.path_glm_output + os.sep + exp)

        # Output csvs
        obj.output_glm_csvs(obj.path_data, 'continent', obj.lup_cont_codes)
        obj.output_glm_csvs(obj.path_data, 'country', obj.lup_cntr_codes)


def do_diagnostic_plots_mgt():
    """

    :return:
    """
    for exp in constants.glm_experiments:
        logger.info('do_diagnostic_plots_mgt: ' + exp)
        obj = glm(name_exp=exp,
                  path_input=constants.path_glm_input + os.sep + exp,
                  time_period='past',
                  out_path=constants.path_glm_output + os.sep + exp + os.sep + 'management')

        # shifting cultivation comparison between andreas and butler
        obj.plot_compare_sc(constants.NC_ANDREAS, constants.ASC_BUTLER,
                            years=range(850 - obj.start_yr, 2015 - obj.start_yr))

        # Make maps of glm management (N/irrigation/tillage)
        obj.maps_glm_crop_mgt(2015 - obj.start_yr)

        # Plot maps of management: biofuels/wood/harv
        obj.maps_glm_mgt(2015 - obj.start_yr)

        # Plot wood-harvest maps
        obj.map_wood_harvest(year=2014 - obj.start_yr)

        # Plot management data: N and Irrigation
        obj.plot_glm_N_diagnostics(do_global=True)
        obj.plot_glm_Irr_diagnostics(do_global=True)
        obj.plot_glm_N_diagnostics(do_global=False)
        obj.plot_glm_Irr_diagnostics(do_global=False)

        # Plot wood-harvest time-series
        obj.plot_wood_harvest_ts(years=range(850 - obj.start_yr, 2015 - obj.start_yr))

        # Compare GLM ve HYDE forest map/scatter plot
        # TODO: No point comparing the two since HYDE and GLM forest areas should be the same
        # obj.plot_compare_glm_hyde_forest(year=2010 - obj.start_yr)

        # Plot management: biofuels/wood/harv
        obj.plot_ts_mgt()


def do_diagnostic_plots_transitions():
    """

    :return:
    """
    for exp in constants.glm_experiments:
        logger.info('do_diagnostic_plots_transitions: ' + exp)
        obj = glm(name_exp=exp,
                  path_input=constants.path_glm_input + os.sep + exp,
                  time_period='past',
                  out_path=constants.path_glm_output + os.sep + exp + os.sep + 'transitions')

        # heat-map of transition matrix
        obj.plot_hmp_transitions(years=[2014 - obj.start_yr], do_cfts=False)
        obj.plot_hmp_transitions(years=range(1941 - obj.start_yr, 1961 - obj.start_yr), do_cfts=False)
        obj.plot_hmp_transitions(years=range(1951 - obj.start_yr, 1961 - obj.start_yr), do_cfts=False)
        obj.plot_hmp_transitions(years=range(1961 - obj.start_yr, 1971 - obj.start_yr), do_cfts=False)
        obj.plot_hmp_transitions(years=range(1961 - obj.start_yr, 2013 - obj.start_yr), do_cfts=False)
        obj.plot_hmp_transitions(years=range(850 - obj.start_yr, 2015 - obj.start_yr))

        # Plot transitions maps (net/gross)
        obj.plot_transitions_map(do_gross=True, years=range(2000 - obj.start_yr, 2015 - obj.start_yr))
        obj.plot_transitions_map(do_gross=True, years=range(850 - obj.start_yr, 2015 - obj.start_yr))
        obj.plot_transitions_map(do_gross=True, years=range(1960 - obj.start_yr, 2015 - obj.start_yr))
        obj.plot_transitions_map(do_gross=False, years=range(2000 - obj.start_yr, 2015 - obj.start_yr))
        obj.plot_transitions_map(do_gross=False, years=range(1960 - obj.start_yr, 2015 - obj.start_yr))
        obj.plot_transitions_map(do_gross=False, years=range(850 - obj.start_yr, 2015 - obj.start_yr))


def do_diagnostic_plots_states():
    """

    :return:
    """
    for exp in constants.glm_experiments:
        logger.info('do_diagnostic_plots_states: ' + exp)
        obj = glm(name_exp=exp,
                  path_input=constants.path_glm_input + os.sep + exp,
                  time_period='past',
                  out_path=constants.path_glm_output + os.sep + exp + os.sep + 'states')

        # Plot by region
        obj.plot_regions(region='SC', num_regions=2, vert_yr=[1961])
        obj.plot_regions(region='continent', num_regions=len(obj.lup_cont_codes), vert_yr=[1961])
        obj.plot_regions(region='country', num_regions=10, vert_yr=[1961])

        # Make movie of glm land-uses for all years
        obj.movie_glm_LU(start_yr=1850)

        # Plotting hovmoller plots all glm land-uses
        # TODO: Does not work now, because standard_name has been changed to netCDF compliant stuff :(
        # obj.plot_glm_hovmoller()

        # Plot natural vegetation (primf + primn) in biodiversity hotspots
        obj.plot_natural_veg_in_hotspots(year=850)
        obj.plot_natural_veg_in_hotspots(year=1850)
        obj.plot_natural_veg_in_hotspots(year=2005)

        # Time-series
        obj.plot_glm_time_series(lu=LU.pastr)  # glm time-series of pasture classes
        obj.plot_glm_time_series(lu=LU.crop)  # glm time-series of crop classes
        obj.plot_glm_time_series(lu=LU.all)  # glm time-series of land-use classes
        obj.plot_glm_time_series(lu=LU.scnd)  # glm time-series of scnd classes
        obj.plot_glm_time_series(lu=LU.vrgn)  # glm time-series of vrgn classes

        obj.plot_glm_scnd_diagnostics()

        # All land-uses in a stacked plot
        obj.plot_glm_stacked()

        # 2011 diagnostics
        obj.Hurtt_2011_diagnostics()

        # Difference maps
        obj.plot_glm_annual_diffs()  # Time-series of difference in land-use in consecutive years
        if obj.do_alternate:
            obj.movie_glm_diff_maps()  # Movie from maps showing spatial differences between glm versions


def do_diagnostics_plots():
    """

    :return:
    """
    logger.info('####################')
    logger.info('do_diagnostics_plots')
    logger.info('####################')

    # Diagnostic plots: management, states, transition
    do_diagnostic_plots_mgt()
    do_diagnostic_plots_states()
    do_diagnostic_plots_transitions()

    if constants.do_email:
        util.send_email(to=[constants.email_list],
                        subject='glm diagnostic table attached',
                        contents=['This diagnostic table has been emailed automatically by a script.\n',
                                  'It includes the foll. experiments: ', ' '.join(constants.glm_experiments),
                                  'regards, \n Script', constants.path_glm_output + os.sep + 'diagnostic_table.xlsx'])


def run():
    # Create output csvs
    loop_region_glm_csvs()

    # Create diagnostic table for each continent
    for idx, continent in constants.dict_conts.iteritems():
        # Ignore Antartica
        if continent == 'Antartica':
            continue

        do_diagnostics_table(region=idx, fname=continent + '_', map_region='continent')

    # Create diagnostic table for countries
    dict_cntr = pandas.read_csv('../countries.csv').set_index('ID')['Name'].to_dict()
    do_diagnostics_table(region=840, fname=dict_cntr[840] + '_', map_region='country')

    # Global
    do_diagnostics_table(fname='Global_')
    do_diagnostics_plots()


if __name__ == '__main__':
    run()
