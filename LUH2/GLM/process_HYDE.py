import logging
import numpy as np
import os
import pdb
import sys
import matplotlib.pyplot as plt

import palettable

import constants
import pygeoutil.util as util
import plot

# Logging
cur_flname = os.path.splitext(os.path.basename(__file__))[0]
LOG_FILENAME = constants.log_dir + os.sep + 'Log_' + cur_flname + '.txt'
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, filemode='w',
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt="%m-%d %H:%M")  # Logging levels are DEBUG, INFO, WARNING, ERROR, and CRITICAL
# Output to screen
logger = logging.getLogger(cur_flname)
logger.addHandler(logging.StreamHandler())


def add_to_list(inp_list, start_yr=850, end_yr=1500):
    """

    :param inp_list
    :param start_yr:
    :param end_yr:
    """
    start_list = [np.nan] * (end_yr - start_yr)
    start_list.extend(inp_list)

    return start_list


class HYDE:
    """
    Produce plots from HYDE data, compare HYDE 3.1 and HYDE 3.2
    """
    def __init__(self, out_path, ver='3.2', lon_name='lon', lat_name='lat', tme_name='time', area_name='cell_area'):
        """
        Constructor
        """
        self.list_files_to_del = []  # List of file paths to delete when done

        self.ver = ver
        if self.ver == '3.2':
            self.HYDE_crop_path = constants.HYDE32_CROP_PATH
            self.HYDE_othr_path = constants.HYDE32_OTHR_PATH
            self.HYDE_urbn_path = constants.HYDE32_URBN_PATH
            self.HYDE_past_path = constants.HYDE32_PAST_PATH
            self.HYDE_graz_path = constants.HYDE32_GRAZ_PATH
            self.HYDE_rang_path = constants.HYDE32_RANG_PATH
            self.HYDE_urbn_path = constants.HYDE32_URBN_PATH

            # Land use variables
            self.lus = {self.HYDE_crop_path: ['cropland', 'cropland fraction'],
                        self.HYDE_othr_path: ['other', 'other vegetation fraction'],
                        self.HYDE_urbn_path: ['urban', 'urban fraction'],
                        self.HYDE_graz_path: ['grazing', 'grazing land fraction'],
                        self.HYDE_past_path: ['pasture', 'managed pasture fraction'],
                        self.HYDE_rang_path: ['rangeland', 'rangeland fraction']}
        elif self.ver == '3.1':
            self.HYDE_crop_path = constants.HYDE31_CROP_PATH
            self.HYDE_othr_path = constants.HYDE31_OTHR_PATH
            self.HYDE_past_path = constants.HYDE31_PAST_PATH
            self.HYDE_urbn_path = constants.HYDE31_URBN_PATH

            # Land use variables
            self.lus = {self.HYDE_crop_path: ['cropland', 'cropland'],
                        self.HYDE_othr_path: ['primary', 'primary'],
                        self.HYDE_urbn_path: ['urban', 'urban fraction'],
                        self.HYDE_past_path: ['pasture', 'pasture']}
        elif self.ver == '3.2_v03':
            self.HYDE_crop_path = constants.HYDE32_v03_CROP_PATH
            self.HYDE_othr_path = constants.HYDE32_v03_OTHR_PATH
            self.HYDE_urbn_path = constants.HYDE32_v03_URBN_PATH
            self.HYDE_past_path = constants.HYDE32_v03_PAST_PATH
            self.HYDE_graz_path = constants.HYDE32_v03_GRAZ_PATH
            self.HYDE_rang_path = constants.HYDE32_v03_RANG_PATH
            self.HYDE_urbn_path = constants.HYDE32_v03_URBN_PATH

            # Land use variables
            self.lus = {self.HYDE_crop_path: ['cropland', 'cropland fraction'],
                        self.HYDE_othr_path: ['other', 'other vegetation fraction'],
                        self.HYDE_urbn_path: ['urban', 'urban fraction'],
                        self.HYDE_graz_path: ['grazing', 'grazing land fraction'],
                        self.HYDE_past_path: ['pasture', 'pasture fraction'],
                        self.HYDE_rang_path: ['rangeland', 'rangeland fraction']}
        elif self.ver == '3.2_v1hb':
            self.HYDE_crop_path = constants.HYDE32_v1hb_CROP_PATH
            self.HYDE_othr_path = constants.HYDE32_v1hb_OTHR_PATH
            self.HYDE_urbn_path = constants.HYDE32_v1hb_URBN_PATH
            self.HYDE_past_path = constants.HYDE32_v1hb_PAST_PATH
            self.HYDE_graz_path = constants.HYDE32_v1hb_GRAZ_PATH
            self.HYDE_rang_path = constants.HYDE32_v1hb_RANG_PATH

            # Land use variables
            self.lus = {self.HYDE_crop_path: ['cropland', 'cropland fraction'],
                        self.HYDE_othr_path: ['other', 'other vegetation fraction'],
                        self.HYDE_urbn_path: ['urban', 'urban fraction'],
                        self.HYDE_graz_path: ['grazing', 'grazing land fraction'],
                        self.HYDE_past_path: ['pasture', 'pasture fraction'],
                        self.HYDE_rang_path: ['rangeland', 'rangeland fraction']}
        elif self.ver == '3.2_march':
            self.HYDE_crop_path = constants.HYDE32_march_CROP_PATH
            self.HYDE_othr_path = constants.HYDE32_march_OTHR_PATH
            self.HYDE_urbn_path = constants.HYDE32_march_URBN_PATH
            self.HYDE_past_path = constants.HYDE32_march_PAST_PATH
            self.HYDE_graz_path = constants.HYDE32_march_GRAZ_PATH
            self.HYDE_rang_path = constants.HYDE32_march_RANG_PATH

            # Land use variables
            self.lus = {self.HYDE_crop_path: ['cropland', 'cropland fraction'],
                        self.HYDE_othr_path: ['other', 'other vegetation fraction'],
                        self.HYDE_urbn_path: ['urban', 'urban fraction'],
                        self.HYDE_graz_path: ['grazing', 'grazing land fraction'],
                        self.HYDE_past_path: ['pasture', 'pasture fraction'],
                        self.HYDE_rang_path: ['rangeland', 'rangeland fraction']}

        # Lat, Lon and time dimensions
        self.lon_name = lon_name
        self.lat_name = lat_name
        self.tme_name = tme_name
        self.area_name = area_name
        self.out_path = out_path
        self.movies_path = out_path + os.sep + 'movies'

        util.make_dir_if_missing(self.out_path)
        util.make_dir_if_missing(self.movies_path)

        # Open up netCDF and get dimensions
        ds = util.open_or_die(self.HYDE_crop_path)
        self.lat = ds.variables[lat_name][:]
        self.lon = ds.variables[lon_name][:]
        self.time = ds.variables[tme_name][:]
        ds.close()

        # GLM static data
        self.path_glm_stat = constants.path_glm_stat  # Static data, contains grid cell area (carea)
        self.path_glm_carea = constants.path_glm_carea

        # Get cell area (after subtracting ice/water fraction)
        icwtr_nc = util.open_or_die(self.path_glm_stat)
        icwtr = icwtr_nc.variables[constants.ice_water_frac][:, :]
        self.carea = util.open_or_die(self.path_glm_carea)
        self.carea_wo_wtr = util.open_or_die(self.path_glm_carea) * (1.0 - icwtr)

        # Movie frames
        self.yrs = np.arange(int(min(self.time)), int(max(self.time)), constants.MOVIE_SEP)
        # Get colors for plotting
        self.cols = plot.get_colors(palette='tableau')

    def plot_HYDE_compare_versions(self, ver31='3.1', ver32='3.2', ver32alt='3.2_v1h'):
        """
        Plots to compare HYDE versions 3.1 and 3.2
        :param ver31:
        :param ver32:
        :param ver32alt:
        :return:
        """
        obj_hyde31 = HYDE(constants.HYDE31_OUT_PATH, ver=ver31)
        obj_hyde32a = HYDE(constants.HYDE32_OUT_PATH, ver=ver32alt)

        # Create a reverse look-up table to access HYDE 3.2 paths based on HYDE 3.1 variables
        rev_dict = dict((v[0], k) for k, v in self.lus.iteritems())
        rev_dict32a = dict((v[0], k) for k, v in obj_hyde32a.lus.iteritems())
        logger.info('Compare HYDE ' + str(ver31) + ', HYDE ' + str(ver32) + ', HYDE ' + str(ver32alt))
        for path, lu in obj_hyde31.lus.iteritems():
            # Create figure
            fig, ax = plt.subplots()
            time_dim = self.time

            # Variable 'primary' in HYDE 3.1 corresponds to 'other' in HYDE 3.2
            var_hyde32 = lu[0]
            if lu[0] == 'primary':
                var_hyde32 = 'other'
            elif lu[0] == 'pasture':
                var_hyde32 = 'grazing'

            # Plot time-series of HYDE 3.1
            hyde31_arr_sum = util.sum_netcdf(path, lu[0], do_area_wt=True, arr_area=self.carea)
            hyde31_arr_sum = add_to_list(hyde31_arr_sum)
            plot.plot_multiple_ts(ax, hyde31_arr_sum, time_dim, self.out_path + os.sep + 'compare_' + var_hyde32 +
                                  '_HYDE_versions.png', leg_name='HYDE ' + str(ver31) + ' ' + var_hyde32.title(),
                                  ylabel=r'$Area  (km^{2})$', col=self.cols[0], pos='first')

            # For grazing lands, plot rangeland and pasture contribution for HYDE 3.2
            if lu[0] == 'pasture':
                # Plot rangelands HYDE 3.2
                hyde32_arr_sum = util.sum_netcdf(self.HYDE_rang_path, self.lus.get(self.HYDE_rang_path)[0],
                                                 do_area_wt=True, arr_area=self.carea)
                plot.plot_multiple_ts(ax, hyde32_arr_sum, time_dim, self.out_path + os.sep + 'compare_' + var_hyde32 +
                                      '_HYDE_versions.png', leg_name='HYDE ' + str(ver32) + ' Rangelands',
                                      col=self.cols[1], linestyle='--', pos='mid')

                # Plot pasture HYDE 3.2
                hyde32_arr_sum = util.sum_netcdf(self.HYDE_past_path, self.lus.get(self.HYDE_past_path)[0],
                                                 do_area_wt=True, arr_area=self.carea)
                plot.plot_multiple_ts(ax, hyde32_arr_sum, time_dim, self.out_path + os.sep + 'compare_' + var_hyde32 +
                                      '_HYDE_versions.png', leg_name='HYDE ' + str(ver32) + ' Pasture',
                                      col=self.cols[1], linestyle='-.', pos='mid')

                # Plot rangelands HYDE 3.2a
                hyde32a_arr_sum = util.sum_netcdf(obj_hyde32a.HYDE_rang_path,
                                                  obj_hyde32a.lus.get(obj_hyde32a.HYDE_rang_path)[0],
                                                  do_area_wt=True, arr_area=self.carea)
                plot.plot_multiple_ts(ax, hyde32a_arr_sum, time_dim, self.out_path + os.sep + 'compare_' + var_hyde32 +
                                      '_HYDE_versions.png', leg_name='HYDE ' + str(ver32alt) + ' Rangelands',
                                      col=self.cols[2], linestyle='--', pos='mid')

                # Plot pasture HYDE 3.2a
                hyde32a_arr_sum = util.sum_netcdf(obj_hyde32a.HYDE_past_path,
                                                  obj_hyde32a.lus.get(obj_hyde32a.HYDE_past_path)[0],
                                                  do_area_wt=True, arr_area=self.carea)
                plot.plot_multiple_ts(ax, hyde32a_arr_sum, time_dim, self.out_path + os.sep + 'compare_' + var_hyde32 +
                                      '_HYDE_versions.png', leg_name='HYDE ' + str(ver32alt) + ' Pasture',
                                      col=self.cols[2], linestyle='-.', pos='mid')

            # Plot time-series of HYDE 3.2
            hyde32_arr_sum = util.sum_netcdf(rev_dict[var_hyde32], var_hyde32, do_area_wt=True, arr_area=self.carea)
            plot.plot_multiple_ts(ax, hyde32_arr_sum, time_dim, self.out_path + os.sep + 'compare_' + var_hyde32 +
                                  '_HYDE_versions.png', title='Comparison of area in HYDE ' + str(ver31) + ',' +
                                                              str(ver32) + ', ' + str(ver32alt) + var_hyde32,
                                  leg_name='HYDE ' + str(ver32) + ' ' + var_hyde32.title(),
                                  ylabel=r'$Area\ (km^{2})$', col=self.cols[1], pos='mid')

            # Plot time-series of HYDE 3.2a
            hyde32a_arr_sum = util.sum_netcdf(rev_dict32a[var_hyde32], var_hyde32, do_area_wt=True, arr_area=self.carea)
            plot.plot_multiple_ts(ax, hyde32a_arr_sum, time_dim, self.out_path + os.sep + 'compare_' + var_hyde32 +
                                  '_HYDE_versions.png', title='Comparison of area in HYDE ' + str(ver31) + ',' +
                                                              str(ver32) + ', ' + str(ver32alt) + ' ' + var_hyde32,
                                  leg_name='HYDE ' + str(ver32alt) + ' ' + var_hyde32.title(),
                                  ylabel=r'$Area\ (km^{2})$', col=self.cols[2], pos='last')
        plt.close(fig)

    def plot_HYDE_diff_maps(self, ver31='3.1', ver32alt='3.2'):
        """
        Maps showing spatial differences between HYDE versions
        :param ver31:
        :param ver32alt:
        :return:
        """
        obj_hyde31 = HYDE(constants.HYDE31_OUT_PATH, ver=ver31)
        obj_hyde32a = HYDE(constants.HYDE32_OUT_PATH, ver=ver32alt)

        # Create a reverse look-up table to access HYDE 3.2 paths based on HYDE 3.1 variables
        rev_dict = dict((v[0], k) for k, v in self.lus.iteritems())

        # Create difference map and movie of difference maps HYDE3.2 and HYDE3.2alt
        for path, lu in obj_hyde32a.lus.iteritems():
            imgs_diff_movie = []
            var_hyde32 = lu[0]

            out_dir = self.out_path + os.sep + 'HYDE_diff_maps'
            util.make_dir_if_missing(out_dir)

            for yr in obj_hyde32a.time.tolist()[::constants.MOVIE_SEP]:
                logger.info('Create difference map for year ' + str(int(yr)) + ' for variable ' + var_hyde32)
                out_img_path = out_dir + os.sep + 'hyde_diff_' + var_hyde32 + '_' + str(int(yr)) + '.png'

                cmap = palettable.colorbrewer.diverging.RdYlBu_11_r.mpl_colormap

                diff_nc = util.subtract_netcdf(path, rev_dict[var_hyde32], left_var=lu[0], right_var=var_hyde32,
                                               date=int(yr))
                plot.plot_arr_to_map(diff_nc, self.lon, self.lat, out_path=out_img_path,
                                     var_name='diff_' + var_hyde32 + '_' + str(int(yr)),
                                     xaxis_min=-1.0, xaxis_max=1.1, xaxis_step=0.1, annotate_date=True, yr=int(yr),
                                     xlabel='Difference in fraction of grid cell area \n HYDE 3.2 and ' + ver32alt +
                                            ': ' + var_hyde32, title='',
                                     cmap=cmap, any_time_data=False, land_bg=False, grid=True)

                imgs_diff_movie.append(out_img_path)

            plot.make_movie(imgs_diff_movie, out_dir + os.sep + 'movies', out_fname='Diff_HYDE32Alt_' + lu[0] + '.gif')

        # Create difference map and movie of difference maps HYDE3.2 and HYDE3.1
        for path, lu in obj_hyde31.lus.iteritems():
            imgs_diff_movie = []

            # Variable 'primary' in HYDE 3.1 corresponds to 'other' in HYDE 3.2
            var_hyde32 = lu[0]
            if lu[0] == 'primary':
                var_hyde32 = 'other'
            elif lu[0] == 'pasture':
                var_hyde32 = 'grazing'

            out_dir = self.out_path + os.sep + 'HYDE31_diff_maps'
            util.make_dir_if_missing(out_dir)

            for yr in obj_hyde31.time.tolist()[::constants.MOVIE_SEP]:
                logger.info('Create difference map for year ' + str(int(yr)) + ' for variable ' + var_hyde32)
                out_img_path = out_dir + os.sep + 'hyde31_diff_' + var_hyde32 + '_' + str(int(yr)) + '.png'

                cmap = palettable.colorbrewer.diverging.RdYlBu_11_r.mpl_colormap

                diff_nc = util.subtract_netcdf(rev_dict[var_hyde32], path, left_var=var_hyde32, right_var=lu[0],
                                               date=int(yr))
                plot.plot_arr_to_map(diff_nc, self.lon, self.lat, out_path=out_img_path,
                                     var_name='diff_' + var_hyde32 + '_' + str(int(yr)),
                                     xaxis_min=-1.0, xaxis_max=1.1, xaxis_step=0.1, annotate_date=True, yr=int(yr),
                                     xlabel='Difference in fraction of grid cell area \n HYDE 3.2 and HYDE 3.1: ' +
                                            var_hyde32, title='',
                                     cmap=cmap,
                                     any_time_data=False, land_bg=False, grid=True)

                imgs_diff_movie.append(out_img_path)

            plot.make_movie(imgs_diff_movie, out_dir + os.sep + 'movies', out_fname='Diff_HYDE31_' + lu[0] + '.gif')

    def plot_HYDE_annual_diff_maps(self):
        """
        Maps showing annual diffs for LU categories in HYDE
        :return:
        """
        if self.ver != '3.2' or self.ver != '3.2a' or self.ver != '3.2_v1h':
            logger.info('plot_HYDE_annual_diff_maps does not work for HYDE version other than 3.2/3.2a')
            sys.exit(0)

        out_path = self.out_path + os.sep + 'HYDE_annual_diff_maps'

        for path, lu in self.lus.iteritems():
            imgs_diff_movie = []
            var_hyde = lu[0]
            logger.info('Create difference map for variable '+var_hyde)

            # Create difference map and movie of difference maps
            for idx, yr in enumerate(self.time.tolist()[::constants.MOVIE_SEP]):
                # Break out of loop if year + constants.MOVIE_SEP exceeds max year value
                if (yr + constants.MOVIE_SEP) >= self.time.max():
                    break

                # Plot difference map for consecutive years
                # 2nd year
                arr_two = util.open_or_die(path).variables[var_hyde][int(idx * constants.MOVIE_SEP +
                                                                         constants.MOVIE_SEP), :, :]
                # 1st year
                arr_one = util.open_or_die(path).variables[var_hyde][int(idx * constants.MOVIE_SEP), :, :]
                # Difference array
                diff_arr = arr_two - arr_one

                # Output diff as map
                out_map = plot.plot_ascii_map(diff_arr, out_path, xaxis_min=-1.0, xaxis_max=1.1, xaxis_step=0.2,
                                              plot_type='diverging', append_name=str(int(yr)), var_name=var_hyde,
                                              skiprows=0, map_label=str(int(yr)))

                imgs_diff_movie.append(out_map)
                self.list_files_to_del.append(out_map)

            plot.make_movie(imgs_diff_movie, out_path + os.sep + 'movies', out_fname='Annual_Diff_HYDE_' + var_hyde +
                            '.gif')

    def plot_HYDE_maps(self):
        """
        HYDE maps of land-use classes
        :return:
        """
        cmap = palettable.colorbrewer.sequential.PuRd_9.mpl_colormap
        cmap = plot.truncate_colormap(cmap, 0.1, 1.0)
        out_path = self.out_path + os.sep + 'HYDE_land_use_maps'

        for path, lu in self.lus.iteritems():
            # Plotting map for given year
            logger.info('Plotting map for given year ' + path)

            imgs_for_movie = plot.plot_maps_ts_from_path(path, lu[0], self.lon, self.lat,
                                                         out_path=out_path,
                                                         save_name=lu[0], xlabel='Fraction of grid cell area',
                                                         title='HYDE '+lu[0],
                                                         do_jenks=False, cmap=cmap, land_bg=False, grid=True)
            plot.make_movie(imgs_for_movie, out_path + os.sep + 'movies', out_fname='HYDE_' + lu[0] + '.gif')

    def plot_HYDE_time_series(self):
        """
        HYDE time-series of land-use classes
        :return:
        """
        # Create time-series plot
        logger.info('Create time-series plot')

        # Create figure
        fig, ax = plt.subplots()

        for count, (path, lu) in enumerate(self.lus.iteritems(), 1):
            arr_sum = util.sum_netcdf(path, lu[0], do_area_wt=True, arr_area=self.carea)

            # Determine if time series being plotted is first or last
            if count == 1:
                pos = 'first'
            elif count == len(self.lus):
                pos = 'last'
            else:
                pos = 'mid'

            plot.plot_multiple_ts(ax, arr_sum, self.time, self.out_path+os.sep+'time_series_HYDE.png',
                                  title='Land-use types in HYDE',
                                  leg_name=lu[0].title(), ylabel=r'$Area\ (km^{2})$', col=self.cols[count-1], pos=pos)

        plt.close(fig)

    def plot_HYDE_hovmoller(self):
        """
        HYDE hovmoller plots of land-use classes
        :return:
        """
        # Plot hovmoller
        logger.info('Hovmoller')
        for path, lu in self.lus.iteritems():
            plot.plot_hovmoller(path, lu[1], self.out_path+os.sep+'hovmoller_'+lu[0]+'.png', title='HYDE '+lu[0],
                                cbar='Fraction of gridcell area')

    def plot_HYDE_annual_diffs(self):
        """
        :return:
        """
        for path, lu in self.lus.iteritems():
            logger.info('plot_HYDE_annual_diffs ' + path)

            # Create figure
            fig, ax = plt.subplots()

            arr_sum = util.sum_netcdf(path, lu[0], do_area_wt=True, arr_area=self.carea)
            arr_sum = [t - s for s, t in zip(arr_sum, arr_sum[1:])]
            arr_sum.insert(0, 0.0)
            plot.plot_np_ts(ax, arr_sum, self.time, self.out_path+os.sep+'Annual_diff_HYDE_'+lu[0]+'.png',
                            vert_yr=[1850, 1961], leg_name=lu[0], ylabel=r'$Annual\ difference\ in\ area\ (km^{2})$',
                            col='k')
            plt.close(fig)

    def plot_HYDE_ver_annual_diffs(self, ver31='3.1', ver32='3.2', ver32alt='3.2a'):
        """
        :return:
        """
        obj_hyde31 = HYDE(constants.HYDE31_OUT_PATH, ver=ver31)
        obj_hyde32a = HYDE(constants.HYDE32_OUT_PATH, ver=ver32alt)

        # Create a reverse look-up table to access HYDE 3.2 paths based on HYDE 3.1 variables
        rev_dict = dict((v[0], k) for k, v in self.lus.iteritems())
        rev_dict32a = dict((v[0], k) for k, v in obj_hyde32a.lus.iteritems())
        for path, lu in obj_hyde31.lus.iteritems():
            # Create figure
            fig, ax = plt.subplots()
            time_dim = self.time

            # Variable 'primary' in HYDE 3.1 corresponds to 'other' in HYDE 3.2
            var_hyde32 = lu[0]
            if lu[0] == 'primary':
                var_hyde32 = 'other'
            elif lu[0] == 'pasture':
                var_hyde32 = 'grazing'

            # HYDE 3.1
            arr_sum = util.sum_netcdf(path, lu[0], do_area_wt=True, arr_area=self.carea)
            arr_sum = [t - s for s, t in zip(arr_sum, arr_sum[1:])]
            arr_sum.insert(0, 0.0)
            arr_sum = add_to_list(arr_sum)
            plot.plot_multiple_ts(ax, arr_sum, time_dim, vert_yr=[1961], leg_name='HYDE ' + str(ver31) + ': ' +
                                                                                  lu[0].title(),
                                  out_path=self.out_path + os.sep + 'Annual_diff_HYDE_compare_' + var_hyde32 + '.png',
                                  col=self.cols[0], ylabel=r'$Annual\ difference\ in\ area\ (km^{2})$', pos='first')

            # HYDE 3.2
            hyde32_arr_sum = util.sum_netcdf(rev_dict[var_hyde32], var_hyde32, do_area_wt=True, arr_area=self.carea)
            # TODO
            hyde32_arr_sum = [t - s for s, t in zip(hyde32_arr_sum, hyde32_arr_sum[1:])]
            hyde32_arr_sum.insert(0, 0.0)
            plot.plot_multiple_ts(ax, hyde32_arr_sum, time_dim,
                                  out_path=self.out_path + os.sep + 'Annual_diff_HYDE_compare_' + var_hyde32 + '.png',
                                  vert_yr=[1961], leg_name='HYDE ' + str(ver32) + ': ' + var_hyde32.title(),
                                  col=self.cols[1], ylabel=r'$Annual\ difference\ in\ area\ (km^{2})$', pos='mid')

            # HYDE 3.2a
            hyde32a_arr_sum = util.sum_netcdf(rev_dict32a[var_hyde32], var_hyde32, do_area_wt=True, arr_area=self.carea)
            hyde32a_arr_sum = [t - s for s, t in zip(hyde32a_arr_sum, hyde32a_arr_sum[1:])]
            hyde32a_arr_sum.insert(0, 0.0)
            plot.plot_multiple_ts(ax, hyde32a_arr_sum, time_dim,
                                  out_path=self.out_path + os.sep + 'Annual_diff_HYDE_compare_' + var_hyde32 + '.png',
                                  vert_yr=[1961], leg_name='HYDE ' + str(ver32alt) + ': ' + var_hyde32.title(),
                                  col=self.cols[2], ylabel=r'$Annual\ difference\ in\ area\ (km^{2})$', pos='last')
            plt.close(fig)

if __name__ == '__main__':
    # Set plotting preferences
    plot.set_matplotlib_params()

    # HYDE 3.2
    ver32 = ['3.2']
    for ver in ver32:
        obj = HYDE(constants.HYDE32_OUT_PATH, ver=ver)
        obj.plot_HYDE_hovmoller()  # HYDE hovmoller plots of land-use classes.
        obj.plot_HYDE_time_series()  # HYDE time-series of land-use classes
        obj.plot_HYDE_diff_maps(ver32alt='3.2_march')  # Maps showing spatial differences between HYDE versions
        obj.plot_HYDE_maps()  # HYDE maps of land-use classes
        pdb.set_trace()
        obj.plot_HYDE_compare_versions(ver31='3.1', ver32=ver, ver32alt='3.2_march')  # Plots to compare HYDE versions
        obj.plot_HYDE_ver_annual_diffs(ver31='3.1', ver32=ver, ver32alt='3.2_march')  # Plot annual differences in HYDE
        obj.plot_HYDE_annual_diffs()
        obj.plot_HYDE_annual_diff_maps()
        util.delete_files(obj.list_files_to_del)

    # HYDE 3.2a
    obj = HYDE(constants.HYDE32_OUT_PATH, ver='3.2a')
    obj.plot_HYDE_time_series()  # HYDE time-series of land-use classes
    obj.plot_HYDE_diff_maps()    # Maps showing spatial differences between HYDE versions
    obj.plot_HYDE_maps()         # HYDE maps of land-use classes
    obj.plot_HYDE_hovmoller()    # HYDE hovmoller plots of land-use classes
    obj.plot_HYDE_annual_diffs()
    obj.plot_HYDE_annual_diff_maps()
    util.delete_files(obj.list_files_to_del)

    # HYDE 3.1
    obj31 = HYDE(constants.HYDE31_OUT_PATH, ver='3.1')
    obj31.plot_HYDE_annual_diffs()
    obj31.plot_HYDE_maps()         # HYDE maps of land-use classes
    obj31.plot_HYDE_time_series()  # HYDE time-series of land-use classes
    obj31.plot_HYDE_hovmoller()    # HYDE hovmoller plots of land-use classes
    util.delete_files(obj31.list_files_to_del)
