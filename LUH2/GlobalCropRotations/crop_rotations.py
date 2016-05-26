import itertools
import logging
import os
import re
import sys
import pdb

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

import pygeoutil.util as util
import constants
import crop_stats
import plots


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


class CropRotations:
    def __init__(self):
        self.name_country_col = 'Country_FAO'
        self.cft_type = 'functional crop type'

    @staticmethod
    def get_list_decades(lyrs):
        """
        Convert a list of years to lists of list of years
        FROM: [1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977]
        TO: [[1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969],
             [1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977]]
        :param lyrs:
        :return:
        """
        return np.array([list(g) for k, g in itertools.groupby(lyrs, lambda i: i // 10)])

    @staticmethod
    def get_list_yrs(df, already_processed=False):
        """
        From an input dataframe, get the list of years. The years could be in the form of columns labeled
        Y1961... or in the form of 1961....
        :param df:
        :param already_processed:
        :return:
        """
        if already_processed:
            vals = df.columns[df.columns.astype(str).str.contains(r'\d{4}$')].values
        else:
            # Create a list of columns of the form Y1961...remove the 'Y' and return list of integers
            years = df.filter(regex=r'^Y\d{4}$').columns.values
            vals = [y[1:] for y in years]

        return map(int, vals)

    def select_data_by_country(self, df, country, name_column):
        """
        Select data for a country/region by country code or name
        :param df:
        :param country:
        :param name_column:
        :return:
        """
        df_country = df[df[name_column] == country]

        return df_country

    def rename_years(self, col_name):
        """
        If col_name is of the form Y1961 then return 1961,
        If col_name is like 1961 then return 1961
        :param col_name:
        :return:
        """
        if re.match(r'^Y\d{4}', col_name):
            return int(col_name[1:])
        else:
            return col_name

    def per_CFT_by_decade(self, df, cnt_name, already_processed=False):
        """
        Aggregate years to decades and compute fraction
        of each crop functional type in that decade
        :param df:
        :param cnt_name:
        :param already_processed:
        :return:
        """
        dec_df = pd.DataFrame()

        # Get list of years in FAO data
        list_yrs = CropRotations.get_list_yrs(df, already_processed)

        if not already_processed:
            # renaming columns for years so that they do not start with a 'Y'
            print self.rename_years
            df.rename(columns=self.rename_years, inplace=True)

        # Separate years into decades
        yrs_dec = CropRotations.get_list_decades(list_yrs)

        # Select data by country
        out_df = self.select_data_by_country(df, cnt_name, name_column=self.name_country_col)

        for dec in yrs_dec:
            dec_name = str(util.round_closest(dec[0])) + 's'

            total_ar = np.sum(out_df.ix[:, dec].values)
            dec_df[dec_name] = out_df.ix[:, dec].sum(axis=1)/total_ar * 100

        # Join the decadal dataframe with country and crop functional type name columns
        dec_df = pd.concat([out_df[[self.name_country_col, self.cft_type]], dec_df], axis=1, join='inner')

        return dec_df

    def per_CFT_annual(self, df, cnt_name, already_processed=False):
        """
        Convert a dataframe containing cropland areas by CFT for each country into percentage values
        :param df:
        :param cnt_name:
        :param already_processed:
        :return:
        """
        per_df = pd.DataFrame()

        # Select data by country
        out_df = self.select_data_by_country(df, cnt_name, name_column=self.name_country_col)

        # Get list of years in FAO data
        list_yrs = CropRotations.get_list_yrs(out_df, already_processed)

        for yr in list_yrs:
            grp_df = out_df.groupby([self.name_country_col, self.cft_type]).agg({yr: 'sum'})
            pct_df = grp_df.groupby(level=0).apply(lambda x: 100*x/float(x.sum()))

            per_df = pd.concat([per_df, pct_df], axis=1, join='inner')

        return per_df

    def diff_ann_decadal(self):
        pass

    def call_R(self):
        pass

    def read_processed_FAO_data(self):
        """
        Read in data on FAO crop acreages globally (already processed)
        :return:
        """
        fao_file = util.open_or_die(constants.data_dir + os.sep + constants.FAO_FILE)

        return fao_file.parse(constants.FAO_SHEET)

    def plot_cnt_decade(self, inp_fao_df, cnt, already_processed=False):
        """
        Plot percentage of cropland area occupied by each crop functional type for a country
        :param inp_fao_df:
        :param cnt:
        :param already_processed:
        :return:
        """
        out_dec_df = self.per_CFT_by_decade(inp_fao_df, cnt, already_processed)

        out_dec_df = out_dec_df.set_index(self.cft_type)
        ax = out_dec_df.drop(self.name_country_col, axis=1).T.\
            plot(kind='bar', stacked=True, color=plots.get_colors(palette='tableau'), linewidth=0)

        plots.simple_axis(ax)  # Simple axis, no axis on top and right of plot

        # Transparent legend in lower left corner
        leg = plt.legend(loc='lower left', fancybox=None)
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_alpha(0.5)

        # Set X and Y axis labels and title
        ax.set_title(cnt)
        ax.set_xlabel('')
        plt.ylim(ymax=100)
        ax.set_ylabel('Percentage of cropland area \noccupied by each crop functional type')
        fmt = '%.0f%%'  # Format you want the ticks, e.g. '40%'
        yticks = mtick.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)

        # remove ticks from X axis
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off')         # ticks along the top edge are off

        # Rotate the X axis labels to be horizontal
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=0)

        plt.tight_layout()
        plt.savefig(constants.out_dir + os.sep + cnt + '.png', bbox_inches='tight', dpi=600)
        plt.close()

    def plot_cnt_mean_decade(self, inp_fao_df, cnt, already_processed=False):
        """
        Plot mean crop functional type area in each decade
        :param inp_fao_df:
        :param cnt:
        :param already_processed:
        :return:
        """
        out_dec_df = self.per_CFT_by_decade(inp_fao_df, cnt, already_processed)

        out_dec_df = out_dec_df.set_index(self.cft_type)
        ax = out_dec_df.drop(self.name_country_col, axis=1).T.\
            plot(kind='bar', stacked=True, color=plots.get_colors(palette='tableau'), linewidth=0)

        plots.simple_axis(ax)  # Simple axis, no axis on top and right of plot

        # Transparent legend in lower left corner
        leg = plt.legend(loc='lower left', fancybox=None)
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_alpha(0.5)

        # Set X and Y axis labels and title
        ax.set_title(cnt)
        ax.set_xlabel('')
        plt.ylim(ymax=100)
        ax.set_ylabel('Mean crop functional type area in each decade')
        fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
        yticks = mtick.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)

        # remove ticks from X axis
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off')         # ticks along the top edge are off

        # Rotate the X axis labels to be horizontal
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=0)

        plt.tight_layout()
        plt.savefig(constants.out_dir + os.sep + 'Mean_' + cnt + '.png', bbox_inches='tight', dpi=600)
        plt.close()

    def process_rotations(self):
        cs = crop_stats.CropStats()

        # 1. Read in data on raw FAO crop acreages globally.
        # 2. Delete redundant data, data from continents/regions
        # 3. Replace NaNs by 0
        fao_df = cs.read_raw_FAO_data()

        # Merge FAO data (raw or processed) with our crop functional type definitions
        fao_df, grp_crp, grp_cnt, per_df = cs.merge_FAO_CFT(fao_df)
        already_processed = False

        list_countries = fao_df[self.name_country_col].unique()

        for country in list_countries:
            logger.info(country)

            self.plot_cnt_decade(fao_df, country, already_processed)
            if not already_processed:
                already_processed = True


if __name__ == '__main__':
    obj = CropRotations()
    obj.process_rotations()
