import logging
import os
import pdb
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pygeoutil.util as util
import GLM.constants as constants_glm
import constants
import plots

reload(sys)
sys.setdefaultencoding('utf-8')
pd.options.mode.chained_assignment = None  # default='warn'

# Logging
cur_flname = os.path.splitext(os.path.basename(__file__))[0]
LOG_FILENAME = constants.log_dir + os.sep + 'Log_' + cur_flname + '.txt'
util.make_dir_if_missing(constants.log_dir)
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, filemode='w',
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt="%m-%d %H:%M")  # Logging levels are DEBUG, INFO, WARNING, ERROR, and CRITICAL
# Output to screen
logger = logging.getLogger(cur_flname)
logger.addHandler(logging.StreamHandler(sys.stdout))


class CropStats:
    """
    1. read_raw_FAO_data:
            1. Read in data on raw FAO crop acreages globally.
            2. Delete redundant data, data from continents/regions
            3. Replace NaNs by 0
    2. read_crop_lup: Read lookup table of crops
    3. plot_top_crops_by_area: Plot top crops based on how much global ag area they occupy
    4. plot_top_countries_by_crops: Plot top countries based on how much global ag area they occupy
    5. merge_FAO_CFT: Merge FAO data (raw or processed) with our crop functional type definitions
    6. belgium_luxembourg: FAO data separates Belgium and Luxembourg starting in 2000, but before that it treats them as
        a single country called Belgium-Luxembourg. Here, we compute separate crop area values for Belgium and Luxembourg.
        We do this by computing the average fraction of crop area in each country (based on data from 2000 onwards), and
        applying this fraction to the previous years data.
    7. merge_countries: Merge country data
    8. FAO_ID_concordance: Merge with csv file containing concordance for FAO id's
    9. extend_FAO_time: Extend FAO dataframe containing crop averages in time
    10. output_cft_perc_to_nc: Output a dataframe containing percentage of CFT for each country for each year into a netCDF
    11. output_cft_perc_to_csv: Output a dataframe containing percentage of CFT for each country (FAO) into csv
    12. process_crop_stats:
            1. Divide Belgium_Luxembourg values into Belgium and Luxembourg separately
            2. Unite several countries into USSR, Yugoslavia, Ethiopia, China and Indonesia respectively
            3. Create a dataframe extending from past to present (super set of 1961-present FAO period)
    """
    def __init__(self):
        self.fao_file = util.open_or_die(constants.RAW_FAO)

        # Initialize names of columns
        self.country_code = 'Country Code'  # Numeric values from 1 - ~5817, gives a unique code for country and region
        self.FAO_code = 'Country_FAO'
        self.ISO_code = 'ISO'  # ISO code for each country, not the same as country_code. ISO_code is used in HYDE

        self.crop_name = 'Item'  # Name of crop e.g. Wheat
        self.crop_id = 'Item Code'  # Id of crop e.g. 1, 2 etc.
        self.cft_id = 'functional crop id'  # crop functional type id e.g. 1
        self.cft_type = 'functional crop type'  # crop functional type e.g. C3Annual

        # Names of columns in past and future
        self.cur_cols = ['Y' + str(x) for x in range(constants.FAO_START_YR, constants.FAO_END_YR + 1)]  # 850 -> 1960
        self.past_cols = ['Y' + str(x) for x in range(constants.GLM_STRT_YR, constants.FAO_START_YR)]  # 850 -> 1960
        if constants.FAO_END_YR < constants.GLM_END_YR:
            self.futr_cols = ['Y' + str(x) for x in range(constants.FAO_END_YR + 1, constants.GLM_END_YR + 1)]  # 2014 -> 2015
        else:
            self.futr_cols = []

        self.all_cols = self.past_cols + self.cur_cols + self.futr_cols
        self.FAO_perc_all_df = pd.DataFrame()  # Dataframe containing FAO data for entire time-period
        self.FAO_mfd_df = pd.DataFrame()  # Dataframe containing CFT percentage data for each country for the year 2000

        # area_df: Area of CFT for each country in FAO era
        # gcrop: Percentage of ag area per CFT
        # gcnt: Percentage of ag area per country
        # perc_df: Percentage of ag area for each CFT by country in FAO era
        self.area_df = pd.DataFrame()
        self.gcrop = pd.DataFrame()
        self.gcnt = pd.DataFrame()
        self.perc_df = pd.DataFrame()

        # Path to csv of crop rotations
        self.csv_rotations = constants.csv_rotations

        self.dict_cont = {0: 'Antartica', 1: 'North_America', 2: 'South_America', 3: 'Europe', 4: 'Asia', 5: 'Africa',
                          6: 'Australia'}
        self.CCODES = 'country code'
        self.CONT_CODES = 'continent code'

        # continent and country code files
        self.ccodes_file = constants_glm.ccodes_file
        self.contcodes_file = constants_glm.contcodes_file

    def read_raw_FAO_data(self):
        """
        1. Read in data on raw FAO crop acreages globally.
        2. Delete redundant data, data from continents/regions
        3. Replace NaNs by 0
        :return: dataframe
        """
        logger.info('read_raw_FAO_data')
        df = self.fao_file.parse(constants.RAW_FAO_SHT)

        # Drop rows of type Y1961F...Y2013F. They are redundant. Actual data is in rows
        # with names like Y1961...Y2013
        drop_rows = ['Y' + str(x) + 'F' for x in range(constants.FAO_START_YR, constants.FAO_END_YR + 1)]
        for row in drop_rows:
            df.drop(row, axis=1, inplace=True)

        # Keep only countries, drop data from regions
        # Regions are groups of countries or continents
        df.drop(df[df[self.country_code] >= constants.FAO_REGION_CODE].index, inplace=True)
        df.fillna(0, inplace=True)

        return df

    def read_crop_lup(self, fname='', sht_name=''):
        """
        Read lookup table of crops
        :return:
        """
        logger.info('read_crop_lup')

        crp_file = util.open_or_die(fname)

        return crp_file.parse(sht_name)

    def plot_top_crops_by_area(self, grp_crop, col_name='', xlabel=''):
        """
        Plot top crops based on how much global ag area they occupy
        :param grp_crop:
        :param col_name:
        :param xlabel:
        :return:
        """
        logger.info('plot_top_crops_by_area')
        cols = plots.get_colors()
        ax = grp_crop[col_name][:constants.PLOT_CROPS].plot(kind='barh', color=cols[0], linewidth=0)

        # Remove spines from top and right of plot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Remove y-axis label
        ax.set_ylabel('')
        ax.set_xlabel(xlabel)

        # Ensure that the axis ticks only show up on the bottom and left of the plot
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        # Final layout adjustment and output
        plt.tight_layout()
        plt.savefig(constants.out_dir + 'crops_by_area.png', dpi=constants.DPI)
        plt.close()

    def plot_top_countries_by_crops(self, grp_cnt, col_name='', xlabel=''):
        """
        Plot top countries based on how much global ag area they occupy
        :param grp_cnt:
        :param col_name:
        :param xlabel:
        :return:
        """
        logger.info('plot_top_countries_by_crops')
        cols = plots.get_colors()

        # Number of countries to plot is given by constants.PLOT_CNTRS
        ax = grp_cnt[col_name][:constants.PLOT_CNTRS].plot(kind='barh', color=cols[0], linewidth=0)

        # Remove spines from top and right of plot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Remove y-axis label
        ax.set_ylabel('')
        ax.set_xlabel(xlabel)

        # Ensure that the axis ticks only show up on the bottom and left of the plot
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        # Final layout adjustment and output
        plt.tight_layout()
        plt.savefig(constants.out_dir + 'cntrs_by_area.png', dpi=constants.DPI)
        plt.close()

    def plot_stacked_country_cft(self, df, arr_legend, path_out, fname, ncols=2, xlabel='', ylabel='', title=''):
        """

        :param df:
        :param arr_legend:
        :param path_out:
        :param fname:
        :param xlabel:
        :param ylabel:
        :param title:
        :return:
        """
        df = (df
              .reset_index()
              .fillna(0.0))

        ax = (df
              .plot
              .bar(stacked=True, colormap=plots.get_colors(palette='tableau', cmap=True), linewidth=0, use_index=False))

        # Legend
        leg = ax.legend(fancybox=None, ncol=ncols, prop={'size': 6})
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_alpha(0.5)

        # Create nice-looking grid for ease of visualization
        ax.grid(which='minor', alpha=0.2, linestyle='--')
        ax.grid(which='major', alpha=0.5, linestyle='--')

        plt.xticks(df.index, df[self.FAO_code])
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)

        plt.tight_layout()
        plt.savefig(path_out + os.sep + fname, dpi=constants.DPI)
        plt.close()

    def get_stacked_df_FAO_by_CFT(self, processed_fao_df):
        """
        Output stacked plot showing for top 10 countries by cropland area (averaged from 1961 - present)
        Each stack includes crop functional type area by country
        :param processed_fao_df:
        :return:
        """
        logger.info('get_stacked_df_FAO_by_CFT')

        # Change from FAO indices to FAO ID's used in wood harvesting
        processed_fao_df = self.FAO_ID_concordance(processed_fao_df, mfd_yr=2000)

        # Read a lookup table that translates from crop type e.g. maize to crop functional type e.g. C4 Annual
        crop_df = self.read_crop_lup(fname=constants.CROP_LUP, sht_name=constants.CROP_LUP_SHT)

        # Merge processed FAO data and crop lookup table
        fao_df = pd.merge(processed_fao_df, crop_df, on=[self.crop_id, self.crop_id])

        # Get top 10 countries, plot top 5 crops for each country
        top10 = fao_df.groupby(self.FAO_code, sort=False).mean_area.sum().nlargest(10).index
        df_top_crop_stacked = fao_df[fao_df[self.FAO_code].isin(top10)]

        df_top_crop_stacked['country_total'] = df_top_crop_stacked.groupby([self.FAO_code]).mean_area.transform(sum)
        df_top_crop_stacked = (df_top_crop_stacked[[self.FAO_code, self.crop_name, 'country_total', 'mean_area']]
                               .sort_values([self.FAO_code, 'mean_area'], ascending=False)
                               .groupby(self.FAO_code)
                               .head(5)
                               .sort_values(['country_total', 'mean_area'], ascending=False)
                               .reset_index()
                               .drop(['country_total', 'index'], axis=1))

        unique_cfts = df_top_crop_stacked[self.crop_name].unique()
        df_top_crop_stacked = df_top_crop_stacked.pivot(index=self.FAO_code, columns=self.crop_name)

        # Plot stacked plot for top 10 countries by cropland area (averaged from 1961 - present)
        self.plot_stacked_country_cft(df_top_crop_stacked,
                                      arr_legend=unique_cfts,
                                      path_out=constants.out_dir,
                                      fname='stacked_crop_and_country.png',
                                      ncols=3,
                                      xlabel='',
                                      title=r'$Cropland\ area\ by\ functional\ type\ and\ country$',
                                      ylabel=r'$Area\ (km^{2})$')

        # Create a dataframe in descending order of sum of mean_area for all CFTs by country
        df_stacked = (fao_df[[self.FAO_code, self.cft_type, 'mean_area']]
                      .groupby([self.FAO_code, self.cft_type])
                      .mean()
                      .fillna(0)
                      .reset_index())

        # Make stacked plot for top 10 countries by cropland area (averaged from 1961 - present)
        df_top_stacked = (df_stacked
                          .loc[df_stacked[self.FAO_code]
                          .isin(df_stacked.groupby(self.FAO_code).sum().nlargest(10, 'mean_area').index)])

        unique_cfts = df_top_stacked[self.cft_type].unique()
        df_top_stacked = df_top_stacked.pivot(index=self.FAO_code, columns=self.cft_type)

        # Plot stacked plot for top 10 countries by cropland area (averaged from 1961 - present)
        self.plot_stacked_country_cft(df_top_stacked,
                                      arr_legend=unique_cfts,
                                      path_out=constants.out_dir,
                                      fname='stacked_crop_by_cft_and_country.png',
                                      xlabel='',
                                      title=r'$Cropland\ area\ by\ functional\ type\ and\ country$',
                                      ylabel=r'$Area\ (km^{2})$')

        return df_stacked

    def merge_FAO_CFT(self, processed_fao_df):
        """
        Merge FAO data (raw or processed) with our crop functional type definitions
        :param processed_fao_df: FAO data (raw or processed)
        :return:Combines our crop functional type definitions with FAO data
        """
        logger.info('merge_FAO_CFT')
        # Change from FAO indices to FAO ID's used in wood harvesting
        processed_fao_df = self.FAO_ID_concordance(processed_fao_df, mfd_yr=2000)

        # Read a lookup table that translates from crop type e.g. maize to crop functional type e.g. C4 Annual
        crop_df = self.read_crop_lup(fname=constants.CROP_LUP, sht_name=constants.CROP_LUP_SHT)

        # Merge processed FAO data and crop lookup table
        fao_df = pd.merge(processed_fao_df, crop_df, on=[self.crop_id, self.crop_id])

        # Select subset df with only Item, Country code and area sum
        # Item refers to crop name
        # mean_area is sum of all area from constants.FAO_START_YR to constants.FAO_END_YR for a given crop in a country
        sub_df = fao_df[[self.FAO_code, self.crop_name, 'mean_area']]

        # Compute a dataframe with data on what percentage of global crop area is occupied by each CROP
        grp_crop = sub_df.groupby(self.crop_name).sum().sort_values(by='mean_area', ascending=False)
        # Percent of all crop area
        grp_crop['pct'] = sub_df.groupby(self.crop_name).sum()*100.0/sub_df.groupby(self.crop_name).sum().sum()

        # Compute a dataframe with data on what percentage of global crop area is occupied by each COUNTRY
        grp_cnt = sub_df.groupby(self.FAO_code).sum().sort_values(by='mean_area', ascending=False)
        # Percent of all country area
        grp_cnt['pct'] = sub_df.groupby(self.FAO_code).sum()*100.0/sub_df.groupby(self.FAO_code).sum().sum()

        # Compute a dataframe subset by country and crop
        cols = ['Y' + str(x) for x in range(constants.FAO_START_YR, constants.FAO_END_YR + 1)]
        cols.extend([self.FAO_code, self.ISO_code, self.cft_id, self.cft_type])
        out_df = fao_df[cols]

        per_df = pd.DataFrame()
        for yr in ['Y' + str(x) for x in range(constants.FAO_START_YR, constants.FAO_END_YR+1)]:
            grp_df = out_df.groupby([self.FAO_code, self.ISO_code, self.cft_id, self.cft_type]).agg({yr: 'sum'})
            grp_df.fillna(0.0, inplace=True)
            pct_df = grp_df.groupby(level=0).apply(lambda x: 100*x/float(x.sum()))

            per_df = pd.concat([per_df, pct_df], axis=1, join='inner')
        per_df.reset_index(inplace=True)

        # process FAO data so that the crop types are aggregated to our crop functional types
        yrs = ['Y' + str(x) for x in range(constants.FAO_START_YR, constants.FAO_END_YR + 1, 1)]
        proc_fao_df = fao_df.groupby([self.ISO_code, self.FAO_code, self.cft_type])[yrs].sum()
        proc_fao_df.reset_index(inplace=True)

        return proc_fao_df, grp_crop, grp_cnt, per_df

    def belgium_luxembourg(self):
        """
        FAO data separates Belgium and Luxembourg starting in 2000, but before that it treats them as a single country
        called Belgium-Luxembourg. Here, we compute separate crop area values for Belgium and Luxembourg. We do this by
        computing the average fraction of crop area in each country (based on data from 2000 onwards), and applying this
        fraction to the previous years data.
        :return:
        """
        logger.info('belgium_luxembourg')
        code_bel = 255  # FAO country code for Belgium
        code_lux = 256  # FAO country code for Luxembourg
        code_blx = 15  # FAO country code for Belgium_Luxembourg

        fao_df = self.read_raw_FAO_data()
        belgium = fao_df[fao_df[self.country_code] == code_bel]
        luxembg = fao_df[fao_df[self.country_code] == code_lux]
        bel_lux = fao_df[fao_df[self.country_code] == code_blx]

        # Keep the year and Item columns. Item gives the crop code, which we use later to loop over.
        belgium = belgium.filter(regex=r'^Y\d{4}$|^Item$')
        luxembg = luxembg.filter(regex=r'^Y\d{4}$|^Item$')
        bel_lux = bel_lux.filter(regex=r'^Y\d{4}$|^Item$')

        # Replace all 0 values with Nan, makes it easier to replace them later
        fao_df = fao_df[fao_df != 0]

        # Loop over all crops
        for idx, crop in enumerate(belgium[self.crop_name]):
            # Extract row for a specific crop
            vals_belgium = belgium[belgium[self.crop_name] == crop].values
            vals_luxembg = luxembg[luxembg[self.crop_name] == crop].values
            vals_bel_lux = bel_lux[bel_lux[self.crop_name] == crop]
            vals_bel_lux = vals_bel_lux.drop(self.crop_name, axis=1)

            # Compute ratio based on non-zero values. frac is the fraction of cropland area going to Belgium. Therefore
            # (1 - frac) is the fraction of cropland area going to Luxembourg.
            if len(vals_belgium) == 0:
                frac = 0
            elif len(vals_luxembg) == 0:
                frac = 1
            else:
                sum_belgium = vals_belgium[0][1:].astype(float).sum()
                sum_luxembg = vals_luxembg[0][1:].astype(float).sum()

                frac = sum_belgium / (sum_belgium +  sum_luxembg)

            # Compute values to be used to fill Belgium and Luxembourg rows
            if len(vals_bel_lux) > 0:
                fill_bel = pd.Series((vals_bel_lux.values * frac).squeeze())
                fill_lux = pd.Series((vals_bel_lux.values * (1.0 - frac)).squeeze())
            else:
                # For some crops, we do not have Belgium-Luxembourg data e.g. Aubergines, for such crops, just use
                # the individual country data else fill with 0's
                if len(vals_belgium) > 0:
                    fill_bel = pd.Series(vals_belgium[0][1:].astype(float))
                else:
                    fill_bel = pd.Series(np.zeros(constants.FAO_END_YR - constants.FAO_START_YR + 1))

                if len(vals_luxembg) > 0:
                    fill_lux = pd.Series(vals_luxembg[0][1:].astype(float))
                else:
                    fill_lux = pd.Series(np.zeros(constants.FAO_END_YR - constants.FAO_START_YR + 1))

            # Add a new index to each time series for Belgium and Luxembourg
            yrs = ['Y' + str(x) for x in range(constants.FAO_START_YR, constants.FAO_END_YR + 1, 1)]
            fill_bel.index = yrs
            fill_lux.index = yrs

            # Replace the value for Belgium for specified crop
            fao_df[(fao_df[self.crop_name] == crop) & (fao_df[self.country_code] == code_bel)] = \
                fao_df[(fao_df[self.crop_name] == crop) & (fao_df[self.country_code] == code_bel)].fillna(fill_bel)
            # Replace the value for Luxembourg for specified crop
            fao_df[(fao_df[self.crop_name] == crop) & (fao_df[self.country_code] == code_lux)] = \
                fao_df[(fao_df[self.crop_name] == crop) & (fao_df[self.country_code] == code_lux)].fillna(fill_lux)

        # Drop Belgium_Luxembourg i.e country code 15
        fao_df = fao_df[fao_df[self.country_code] != code_blx]

        return fao_df

    def merge_countries(self, df, replace_cnt=-1, new_iso=-1, cnt_list=[]):
        """
        Merge country data
        :param df: dataframe
        :param replace_cnt: ID of country which will be changed to new_iso
        :param new_iso: New ISO(ID)
        :param cnt_list: Countries that will be merged
        :return: dataframe containing merged countries
        """
        logger.info('merge_countries')
        # For each crop, add all the rows from the cnt_list list and replace existing row in replace_cnt
        all_df = df[df[self.country_code].isin(cnt_list)]

        for idx, crop in enumerate(all_df[self.crop_name]):
            s = all_df[(all_df[self.crop_name] == crop)].filter(regex=r'^Y\d{4}$').sum()

            df.loc[(df[self.country_code] == replace_cnt) & (df[self.crop_name] == crop), s.index] = \
                pd.DataFrame(columns=s.index, data=s.values.reshape(1, len(s.index))).values

        # Get list of countries which will be dropped
        cnt_list.remove(replace_cnt)

        # Drop countries which have been aggregated and are no longer needed
        for cntry in cnt_list:
            df = df[df[self.country_code] != cntry]

        # Rename replace_cnt as new_iso
        if new_iso > 0:
            df.loc[df[self.country_code] == replace_cnt, self.country_code] = new_iso

        return df

    def FAO_ID_concordance(self, cond_df, mfd_yr=2000):
        """
        Merge with csv file containing concordance for FAO id's
        :param cond_df:
        :param mfd_yr:
        :return:
        """
        logger.info('FAO_ID_concordance')
        fao_id = pd.read_csv(constants.FAO_CONCOR)
        cond_df = pd.merge(cond_df, fao_id, how='outer', left_on=self.country_code, right_on='ID')
        # Drop rows which have NA's in ISO column
        cond_df.dropna(subset=[self.ISO_code], inplace=True)

        # Replace NA's by 0 in valid rows and compute sum of area in each row
        # valid rows have names like Y1961...Y2013
        vld_rows = ['Y' + str(x) for x in range(constants.FAO_START_YR, constants.FAO_END_YR + 1)]

        if mfd_yr == -1:
            # Compute sum of each row i.e. get sum of crop areas total across time for each country
            cond_df['mean_area'] = cond_df[vld_rows].mean(axis=1)
        else:
            cond_df['mean_area'] = cond_df['Y'+str(mfd_yr)]
        return cond_df

    def extend_FAO_time(self, per_df):
        """
        Extend FAO dataframe containing crop averages in time
        i.e. use FAO averages from 1961-201x to fill in values for remaining years
        :param per_df: Dataframe containing percentage of CFTs in each year for each country
        :return: dataframe, no side-effect
        """
        logger.info('extend_FAO_time')
        # Compute average of first CFT_FRAC_YR years worth of FAO data and use it to fill information from GLM_STRT_YR
        # to start of FAO era -> Y1961 - Y1965
        cols = ['Y' + str(x) for x in range(constants.FAO_START_YR, constants.FAO_START_YR + constants.CFT_FRAC_YR)]

        # TODO HACK alert
        # Virgin Islands (U.S.) do not have 100% crops across the FAO period (1961-2013). As a result, the average of
        # all CFTs for Virgin Islands (U.S.) sums up to less than 0. Manually changing it to 100.
        # Should get smarter about detecting and fixing such anomalies. Perhaps fill in global data instead
        per_df.ix[per_df[self.FAO_code] == 'Virgin Isl. (US.)', cols] = 100.0

        # TODO HACK alert
        # Czechoslovakia has 0 ha in croplands for all of the CFT types from 1993 onwards. Messes up netCDF, so fixing
        # this by assigning values from the year 1992 to all years after that
        czech_yr = 1992
        czech_cols = ['Y' + str(x) for x in range(czech_yr + 1, constants.FAO_END_YR + 1)] # Y1993 - Y2013
        per_df.ix[(per_df[self.FAO_code] == 'Czechoslovakia'), czech_cols] = \
            per_df.ix[(per_df[self.FAO_code] == 'Czechoslovakia'), 'Y' + str(czech_yr)]

        cols.extend([self.FAO_code])
        past_avg_df = per_df[cols]
        past_avg_df['mean_CFT'] = past_avg_df.mean(axis=1) # Average entire column

        # Compute average of last CFT_FRAC_YR years worth of FAO data and use it to fill information from END_YR to
        # GLM_END_YR
        cols = ['Y' + str(x) for x in range(constants.FAO_END_YR - constants.CFT_FRAC_YR, constants.FAO_END_YR + 1)] # Y2008 - Y2013
        cols.extend([self.FAO_code])
        futr_avg_df = per_df[cols]
        futr_avg_df['mean_CFT'] = futr_avg_df.mean(axis=1) # Average entire column

        # Extend dataframe in past and future
        all_df = pd.DataFrame()
        for col in [self.FAO_code, self.ISO_code, self.cft_id, self.cft_type]:
            if col in per_df.columns:
                all_df[col] = per_df.loc[:, col]

        # Join past columns
        for col in self.past_cols:
            all_df[col] = past_avg_df['mean_CFT']

        # Join current columns
        for col in self.cur_cols:
            all_df[col] = per_df[col]

        # Join future columns if any
        for col in self.futr_cols:
            all_df[col] = futr_avg_df['mean_CFT']

        return all_df

    def constant_FAO_time(self, per_df):
        """
        Extend FAO dataframe containing crop averages in time
        i.e. use FAO averages from 1961-201x to fill in values for all year
        CFT fraction of any country will stay constant in time
        :param per_df: Dataframe containing percentage of CFTs in each year for each country
        :return: dataframe, no side-effect
        """
        logger.info('constant_FAO_time')
        # Compute average of first CFT_FRAC_YR years worth of FAO data and use it to fill information from GLM_STRT_YR
        # to GLM_END_YR
        cols = ['Y' + str(x) for x in range(constants.FAO_START_YR, constants.FAO_END_YR + 1)] # Y1961 - Y2013

        # TODO HACK alert
        # Virgin Islands (U.S.) do not have 100% crops across the FAO period (1961-2013). As a result, the average of
        # all CFTs for Virgin Islands (U.S.) sums up to less than 0. Manually changing it to 100.
        # Should get smarter about detecting and fixing such anomalies. Perhaps fill in global data instead
        per_df.ix[per_df[self.FAO_code] == 'Virgin Isl. (US.)', cols] = 100.0

        # TODO HACK alert
        # Czechoslovakia has 0 ha in croplands for all of the CFT types from 1993 onwards. Messes up netCDF, so fixing
        # this by assigning values from the year 1992 to all years after that
        czech_yr = 1992
        czech_cols = ['Y' + str(x) for x in range(czech_yr + 1, constants.FAO_END_YR + 1)] # Y1993 - Y2013
        per_df.ix[(per_df[self.FAO_code] == 'Czechoslovakia'), czech_cols] = \
            per_df.ix[(per_df[self.FAO_code] == 'Czechoslovakia'), 'Y' + str(czech_yr)]

        cols.extend([self.FAO_code])
        past_avg_df = per_df[cols]
        past_avg_df['mean_CFT'] = past_avg_df.mean(axis=1)  # Average entire column

        # Extend dataframe in past and future
        all_df = pd.DataFrame()
        for col in [self.FAO_code, self.ISO_code, self.cft_id, self.cft_type]:
            if col in per_df.columns:
                all_df[col] = per_df.loc[:, col]

        for col in self.all_cols:
            all_df[col] = past_avg_df['mean_CFT']

        return all_df

    def output_cft_frac_to_nc(self, df, nc_name):
        """
        Output a dataframe containing fraction of CFT for each country for each year into a netCDF
        :param df:
        :return:
        """
        logger.info('output_cft_frac_to_nc')
        # Get list of ALL country codes
        fao_id = pd.read_csv(constants.FAO_CONCOR)
        all_cntrs = fao_id[self.ISO_code].unique()

        # Create a lookup dictionary between crop ids (1,2 etc.) and crop names (wheat etc.)
        crp_ids = df[self.cft_id].unique().tolist()
        crp_names = df[self.cft_type].unique().tolist()
        dict_crps = dict(zip(crp_ids, crp_names))

        # Compute global average of all CFT area percentages (one average value for each CFT)
        cols = self.all_cols[:]
        cols.extend([self.cft_id])
        global_cft_avg = df[cols].groupby(self.cft_id).sum()*100.0/df[cols].groupby(self.cft_id).sum().sum()
        global_cft_avg = global_cft_avg.mean(axis=1)

        # Read in HYDE dataset to get lat, lon info
        ds = util.open_or_die(constants.hyde_dir)
        tme = ds.variables['time'][:]

        onc = util.open_or_die(constants.out_dir + nc_name, 'w')

        # Create dimensions
        onc.createDimension('time', np.shape(tme)[0])
        onc.createDimension('country_code', len(all_cntrs))

        # Create variables
        time = onc.createVariable('time', 'i4', ('time',))
        cntrs = onc.createVariable('country_code', 'i4', ('country_code',))

        # Assign time
        time[:] = tme

        # Metadata
        cntrs.units = ''
        cntrs.standard_name = 'FAO country codes'

        # Assign data to countries
        cntrs[:] = all_cntrs

        all = onc.createVariable('sum', 'f4', ('time', 'country_code', ), fill_value=np.nan)
        all[:, :] = np.zeros((np.shape(tme)[0], len(all_cntrs)))

        # Assign data for crop functional types
        for key, val in dict_crps.iteritems():
            cft = onc.createVariable(val, 'f4', ('time', 'country_code', ), fill_value=np.nan)
            cft.units = 'fraction'

            # Iterate over all countries over all years
            for idc, i in enumerate(all_cntrs):
                # Check if country is present in dataframe
                cntr_present = i in df[self.ISO_code].values

                # If country is present, then fill CFT values
                if cntr_present:
                    # Get data corresponding to country code 'i' and crop id 'val' and for all years (all_cols)
                    vals = df[(df[self.ISO_code] == i) & (df[self.cft_type] == val)][self.all_cols].values

                    # If CFT data is missing, then vals will be an empty array.
                    # In that case, fill with 0.0
                    if len(vals) == 0:
                        vals = np.zeros((1, len(tme)))
                else: # country data not present, fill with global average
                    vals = np.repeat(global_cft_avg[global_cft_avg.index == key].values, len(tme))

                if constants.TEST_CFT:
                    # Assign to each of the 5 CFTs, a value of 1 / 5.0 or 20%
                    vals = np.empty(len(tme))
                    vals.fill(20.0)
                    cft[:, idc] = vals.T/100.0
                    all[:, idc] = all[:, idc] + vals.T/100.0
                else:
                    cft[:, idc] = vals.T/100.0  # Convert from percentage to fraction
                    all[:, idc] = all[:, idc] + cft[:, idc]

        onc.close()

    def output_constant_cft_frac_to_nc(self, df, nc_name):
        """
        Create a netCDF with constant CFT fraction values for each country across time
        :return:
        """
        logger.info('output_constant_cft_frac_to_nc')
        # Get list of ALL country codes
        fao_id = pd.read_csv(constants.FAO_CONCOR)
        all_cntrs = fao_id[self.ISO_code].unique()

        # Create a lookup dictionary between crop ids (1,2 etc.) and crop names (wheat etc.)
        crp_ids = df[self.cft_id].unique().tolist()
        crp_names = df[self.cft_type].unique().tolist()
        dict_crps = dict(zip(crp_ids, crp_names))

        # Compute global average of all CFT area percentages (one average value for each CFT)
        cols = self.all_cols[:]
        cols.extend([self.cft_id])
        global_cft_avg = df[cols].groupby(self.cft_id).sum()*100.0/df[cols].groupby(self.cft_id).sum().sum()
        global_cft_avg = global_cft_avg.mean(axis = 1)

        # Read in HYDE dataset to get lat, lon info
        ds  = util.open_or_die(constants.hyde_dir)
        tme = ds.variables['time'][:]

        onc = util.open_or_die(constants.out_dir + nc_name, 'w')

        # Create dimensions
        onc.createDimension('time', np.shape(tme)[0])
        onc.createDimension('country_code', len(all_cntrs))

        # Create variables
        time = onc.createVariable('time', 'i4', ('time',))
        cntrs = onc.createVariable('country_code', 'i4', ('country_code',))

        # Assign time
        time[:] = tme

        # Metadata
        cntrs.units = ''
        cntrs.standard_name = 'FAO country codes'

        # Assign data to countries
        cntrs[:] = all_cntrs

        all = onc.createVariable('sum', 'f4', ('time', 'country_code', ), fill_value=np.nan)
        all[:, :] = np.zeros((np.shape(tme)[0], len(all_cntrs)))

        # Assign data for crop functional types
        for key, val in dict_crps.iteritems():
            cft = onc.createVariable(val, 'f4', ('time', 'country_code', ), fill_value=np.nan)
            cft.units = 'fraction'

            # Iterate over all countries over all years
            for idc, i in enumerate(all_cntrs):
                # Check if country is present in dataframe
                cntr_present = i in df[self.ISO_code].values

                # If country is present, then fill CFT values
                if cntr_present:
                    # Get data corresponding to country code 'i' and crop id 'val' and for all years (all_cols)
                    vals = df[(df[self.ISO_code] == i) & (df[self.cft_type] == val)][self.all_cols].values

                    # If CFT data is missing, then vals will be an empty array.
                    # In that case, fill with 0.0
                    if len(vals) == 0:
                        vals = np.zeros((1, len(tme)))
                else:  # country data not present, fill with global average
                    vals = np.repeat(global_cft_avg[global_cft_avg.index == key].values, len(tme))

                cft[:, idc] = vals.T/100.0  # Convert from percentage to fraction
                all[:, idc] = all[:, idc] + cft[:, idc]

        onc.close()

    def create_rotations_nc(self, df):
        """
        :param df: Pandas dataframe
            Dataframe containing cropland area for each country x functional crop type combination
            Country_FAO    functional crop type     mean_area
            Albania             C3annual         6.687115e+03
            Albania          C3perennial         4.139867e+03
            Albania             C4annual         5.300000e+04
            Albania             N-fixing         4.460714e+03
            Algeria             C3annual         5.371344e+04
        :return:
        """
        logger.info('create_rotations_nc')

        df_rotations = util.open_or_die(self.csv_rotations, csv_header=0)

        # Read in country and continent file
        # Create dataframe combining country and continent files
        ccodes = pd.read_csv(self.ccodes_file, header=None)
        ccodes.columns = [self.CCODES]
        contcodes = pd.read_csv(self.contcodes_file, header=None)
        contcodes.columns = [self.CONT_CODES]
        lup_codes = pd.concat([ccodes, contcodes], axis=1)

        # Merge dataframe
        df_merge = pd.merge(df_rotations, lup_codes, on=self.CCODES)

        out_nc = constants_glm.path_glm_output + os.sep + 'national_crop_rotation_data_850_2015_new.nc'
        nc_data = util.open_or_die(out_nc, perm='w', format='NETCDF4_CLASSIC')
        nc_data.description = ''

        # dimensions
        tme = np.arange(constants.GLM_STRT_YR, constants.GLM_END_YR + 1)

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

        c4ann_to_c3nfx = nc_data.createVariable('c4ann_to_c3nfx', 'f4', ('time', 'country',))
        c4ann_to_c3ann = nc_data.createVariable('c4ann_to_c3ann', 'f4', ('time', 'country',))
        c3ann_to_c3nfx = nc_data.createVariable('c3ann_to_c3nfx', 'f4', ('time', 'country',))

        c4ann_to_c3nfx.units = 'fraction of crop type area undergoing crop rotation'
        c4ann_to_c3nfx.long_name = 'Crop rotations: C4 Annual, C3 N-Fixing'

        c4ann_to_c3ann.units = 'fraction of crop type area undergoing crop rotation'
        c4ann_to_c3ann.long_name = 'Crop rotations: C4 Annual, C3 Annual'

        c3ann_to_c3nfx.units = 'fraction of crop type area undergoing crop rotation'
        c3ann_to_c3nfx.long_name = 'Crop rotations: C3 Annual, C3 N-Fixing'

        # Loop over all countries
        for index, row in lup_codes.iterrows():
            # print index, row[self.CCODES], row[self.CONT_CODES]

            # Find row containing country in df_merge
            row_country = df_merge[df_merge[self.CCODES] == row[self.CCODES]]
            if len(row_country):
                c4ann_to_c3nfx[:, index] = row_country['c4ann_to_c3nfx'].values[0]
                c4ann_to_c3ann[:, index] = row_country['c4ann_to_c3ann'].values[0]
                c3ann_to_c3nfx[:, index] = row_country['c3ann_to_c3nfx'].values[0]
            else:
                # TODO Find the average crop rotation rate for the continent

                c4ann_to_c3nfx[:, index] = 0.03  # 0.53
                c4ann_to_c3ann[:, index] = 0.01
                c3ann_to_c3nfx[:, index] = 0.02

        nc_data.close()

    def output_cft_area_to_df(self, df, yr=2000):
        """
        Output FAO ag area for each CFT by country as a csv file
        :param df:
        :param yr: Which year to use from FAO?
        :return:
        """
        logger.info('output_cft_area_to_df')
        list_df = []

        # Get list of ALL country codes
        fao_id = pd.read_csv(constants.FAO_CONCOR)  # column names: Country_FAO   ID  ISO
        all_cntrs = fao_id[self.ISO_code].unique()  # .unique() is redundant here, but harmless

        # Get crop functional type names (C3Annual etc.)
        cft_names = df[self.cft_type].unique().tolist()

        # Create output csv of the form:
        # ISO  Country_FAO Area_FAO FAO_c4annual FAO_c4perren FAO_c3perren FAO_ntfixing FAO_c3annual
        # 4    Afghanistan    ...
        for idc, i in enumerate(all_cntrs):
            try:
                # Get name of country
                cnt_name = df[df[self.ISO_code] == i][self.FAO_code].unique()[0]
                # Get sum of area of cropland in country based on FAO data
                cnt_area = df[df[self.ISO_code] == i]['Y' + str(yr)].values.sum()
            except:
                # For countries that are missing in FAO data, fill areas as 0.0
                cnt_name = fao_id[fao_id[self.ISO_code] == i][self.FAO_code].iloc[0]
                cnt_area = 0.0

            # Add country ISO code, name and area to dictionary
            dict_all = {'ISO': int(i), 'Country_FAO': cnt_name, 'Area_FAO': cnt_area}

            # Get area of individual CFTs for each country
            for key in cft_names:
                try:
                    area_sum = df[(df[self.ISO_code] == i) & (df[self.cft_type] == key)]['Y' + str(yr)].values.sum()
                except:
                    # For countries that are missing in FAO data, fill areas as 0.0
                    area_sum = 0.0

                dict_cft = {'FAO_'+key.lower(): area_sum}
                dict_all.update(dict_cft)

            # Add CFT area to dictionary
            list_df.append(dict_all)

        return pd.DataFrame(list_df)

    def process_crop_stats(self):
        """
        1. Divide Belgium_Luxembourg values into Belgium and Luxembourg separately
        2. Unite several countries into USSR, Yugoslavia, Ethiopia, China and Indonesia respectively
        3. Create a dataframe extending from past to present (super set of 1961-present FAO period)
        :return: Nothing, outputs dataframes
        """
        logger.info('process_crop_stats')

        # Divide Belgium_Luxembourg values into Belgium and Luxembourg separately
        blx_fao_df = self.belgium_luxembourg()

        # Unite several countries into USSR
        ussr_vals = [1, 52, 57, 63, 73, 108, 113, 119, 126, 146, 185, 208, 213, 230, 228, 235]
        df = self.merge_countries(blx_fao_df, replace_cnt=228, new_iso=228, cnt_list=ussr_vals)

        # Unite several countries into Yugoslavia
        yugo_vals = [80, 98, 154, 186, 198, 248, 272, 273]
        df = self.merge_countries(df, replace_cnt=248, new_iso=248, cnt_list=yugo_vals)

        etop_vals = [62, 178, 238]
        df = self.merge_countries(df, replace_cnt=62, new_iso=62, cnt_list=etop_vals)

        chna_vals = [41, 96]
        df = self.merge_countries(df, replace_cnt=41, new_iso=41, cnt_list=chna_vals)

        insa_vals = [101, 176]
        df = self.merge_countries(df, replace_cnt=101, new_iso=101, cnt_list=insa_vals)

        # Merge FAO data on crop area distribution over all countries globally with
        # GLM's lookup table from crop names to crop functional type classifications
        # Fill all missing values.
        # area_df: Area of CFT for each country in FAO era
        # gcrop: Percentage of ag area per CFT
        # gcnt: Percentage of ag area per country
        # perc_df: Percentage of ag area for each CFT by country in FAO era
        self.area_df, self.gcrop, self.gcnt, self.perc_df = self.merge_FAO_CFT(df)

        # Output stacked plot showing for top 10 countries by cropland area (averaged from 1961 - present)
        # Each stack includes crop functional type area by country
        df_stacked = self.get_stacked_df_FAO_by_CFT(df)
        # create crop rotations netCDF file
        self.create_rotations_nc(df_stacked)

        # Fill in missing values using values from that row only
        self.area_df = self.area_df.fillna(axis=1, method='backfill')
        self.perc_df = self.perc_df.fillna(axis=1, method='backfill')

        # Fill in values that are still missing with 0.0
        self.area_df = self.area_df.fillna(0.0)
        self.perc_df = self.perc_df.fillna(0.0)

        # 12/01/2015: Create a dataframe containing percentages of crop functional types for all countries from start of
        # paleo period (currently 850 AD) to end of current time-period (currently 2015)
        self.all_df = self.extend_FAO_time(self.area_df)

        # Extend from FAO era to entire time-period
        self.FAO_perc_all_df = self.extend_FAO_time(self.perc_df)

        # Get CFT percentage data for each country for the year 2000 i.e. year for which monfreda map was made
        self.FAO_mfd_df = self.output_cft_area_to_df(self.FAO_perc_all_df, yr=2000)

    def output_crop_stats(self):
        """
        Output class members to csv or nc or plot them
        :return:
        """
        logger.info('output_crop_stats')

        self.output_cft_frac_to_nc(self.extend_FAO_time(self.FAO_perc_all_df), nc_name='FAO_CFT_fraction.nc')
        self.output_constant_cft_frac_to_nc(self.constant_FAO_time(self.perc_df), 'FAO_CFT_constant_fraction.nc')
        csv_df = self.output_cft_area_to_df(self.all_df, yr=2000)

        # Plot
        plots.set_matplotlib_params()
        self.plot_top_crops_by_area(self.gcrop, col_name='pct', xlabel='% of global ag area by crop')
        self.plot_top_countries_by_crops(self.gcnt, col_name='pct', xlabel='% of global ag area by country')

        # Output to csv
        self.area_df.to_csv(constants.out_dir + os.sep + 'FAO_CFT_areas_subset.csv')  # Only FAO era data
        csv_df.to_csv(constants.out_dir + os.sep + 'FAO_CFT_areas_all.csv')  # From paleo period (850 A.D.) to now (2015)
        self.gcnt.to_csv(constants.out_dir + os.sep + 'country_ag.csv')
        self.all_df.to_csv(constants.out_dir + os.sep + 'all_crops.csv')
        self.FAO_mfd_df.to_csv(constants.out_dir + os.sep + 'FAO_monfreda_perc_CFT.csv')
        self.FAO_perc_all_df.to_csv(constants.out_dir + os.sep + 'FAO_perc_all_df.csv')

if __name__ == '__main__':
    obj = CropStats()

    obj.process_crop_stats()
    obj.output_crop_stats()
