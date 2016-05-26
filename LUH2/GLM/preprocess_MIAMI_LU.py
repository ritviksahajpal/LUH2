import logging
import os
import pdb
import math
import numpy

import constants
import pygeoutil.util as util
import plot


def process_MIAMI_LU():
    """
    Convert MIAMI LU AGB and NPP estimates to quarter degree
    :return:
    """
    # Read in MIAMI-LU data
    mlu_data = util.open_or_die(constants.miami_lu_nc)

    # Get last year for which biomass is extracted
    lyr = mlu_data.variables['biomass'].shape[0] - 1

    # Extract agb and npp layers and convert to quarter degree format
    mlu_vba = mlu_data.variables['biomass'][lyr, :, :]
    mlu_vba = mlu_vba.repeat(2, 0).repeat(2, 1)
    mlu_npp = mlu_data.variables['aa_NPP'][lyr, :, :]
    mlu_npp = mlu_npp.repeat(2, 0).repeat(2, 1)

    # Output
    numpy.savetxt(constants.out_dir + 'miami_vba_quarter_deg.txt', mlu_vba, fmt='%1.1f')
    numpy.savetxt(constants.out_dir + 'miami_npp_quarter_deg.txt', mlu_npp, fmt='%1.1f')

    # Read in previous estimates
    prev_mlu_npp = util.open_or_die(constants.miami_npp)
    prev_mlu_vba = util.open_or_die(constants.miami_vba)

    # Compare
    diff_vba = mlu_vba - prev_mlu_vba
    diff_npp = mlu_npp - prev_mlu_npp

    # Plot
    xaxis_min, xaxis_max, xaxis_step = util.get_ascii_plot_parameters(mlu_vba)
    plot.plot_ascii_map(mlu_vba, constants.out_dir, xaxis_min, xaxis_max, xaxis_step, plot_type='sequential',
                        xlabel=r'$Biomass\ (kg\ C/m^{2})$', title='', var_name='Biomass', skiprows=0)

    xaxis_min, xaxis_max, xaxis_step = util.get_ascii_plot_parameters(prev_mlu_vba)
    plot.plot_ascii_map(prev_mlu_vba, constants.out_dir, 0.0, 27.0, 3.0, plot_type='sequential',
                        xlabel=r'$Biomass\ (kg\ C/m^{2})$', title='', var_name='prev_Biomass', skiprows=0)

    xaxis_min, xaxis_max, xaxis_step = util.get_ascii_plot_parameters(mlu_npp)
    plot.plot_ascii_map(mlu_npp, constants.out_dir, xaxis_min, xaxis_max, xaxis_step, plot_type='sequential',
                        xlabel=r'$NPP\ (kg\ C/m^{2})$', title='', var_name='NPP', skiprows=0)

    # Difference
    xaxis_min, xaxis_max, xaxis_step = util.get_ascii_plot_parameters(diff_vba)
    plot.plot_ascii_map(diff_vba, constants.out_dir, xaxis_min, xaxis_max, xaxis_step, plot_type='diverging',
                        xlabel=r'$(New - Old)\ Biomass\ (kg\ C/m^{2})$', title='', var_name='Difference_Biomass', skiprows=0)

    xaxis_min, xaxis_max, xaxis_step = util.get_ascii_plot_parameters(diff_npp)
    plot.plot_ascii_map(diff_npp, constants.out_dir, xaxis_min, xaxis_max, xaxis_step, plot_type='diverging',
                        xlabel=r'$(New - Old)\ NPP\ (kg\ C/m^{2})$', title='', var_name='Difference_NPP', skiprows=0)

if __name__ == '__main__':
    process_MIAMI_LU()
