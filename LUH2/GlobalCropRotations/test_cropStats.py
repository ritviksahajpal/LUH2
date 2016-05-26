import pdb, os, numpy
from unittest import TestCase
from crop_stats import CropStats
import GLM
import constants
import pygeoutil.util as util

class TestCropStats(TestCase):
    def test_output_cft_frac_to_nc(self):
        # TODO: Maximum grid cell value should not exceed 1
        # TODO Minimum grid cell value should not be less than 1
        # Add the fractions from all crop functional types (they should sum up to 1.0 for every year)
        obj = CropStats()
        obj.process_crop_stats()

        obj.output_cft_frac_to_nc(obj.extend_FAO_time(obj.FAO_perc_all_df), nc_name='test_FAO_CFT_fraction.nc')

        # Open up netCDF file just created
        nc = util.open_or_die(constants.out_dir + os.sep + 'test_FAO_CFT_fraction.nc')
        # Get list of CFTs
        cfts = obj.FAO_perc_all_df[obj.cft_type].unique().tolist()

        # Add up all CFT fractions
        arr = numpy.zeros(nc[cfts[0]].shape)
        for idx, key in enumerate(cfts):
            arr = arr + nc[key][:]

        # CFT fraction sum for each country should be ~= 1.0
        zeros = arr == 0.0
        without_zeros = arr[~zeros]  # Remove all 0.0s
        self.assertEqual(numpy.allclose(without_zeros, 1), True)



