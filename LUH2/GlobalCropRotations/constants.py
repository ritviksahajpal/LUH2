import os, logging, multiprocessing, pdb, ast, errno, sys
if sys.version_info.major == 3:
    from configparser import SafeConfigParser
else:
    from ConfigParser import SafeConfigParser
from pathlib2 import Path

import pygeoutil.util as util

# Parse config file
parser = SafeConfigParser()
parser.read('config_rotations.txt')

# Get directory path (3 levels up is the parent directory)
dir_prj = str(Path(__file__).parents[3])

FAO_START_YR = parser.getint('PARAMETERS', 'FAO_START_YR') # Starting year of FAO data
FAO_END_YR = parser.getint('PARAMETERS', 'FAO_END_YR')    # Ending year of FAO data
TAG = parser.get('PROJECT', 'TAG')
FAO_FILE = parser.get('PROJECT', 'fao_data')
FAO_SHEET = parser.get('PROJECT', 'fao_sheet')
PROJ_NAME = parser.get('PROJECT', 'project_name')
DO_PARALLEL = parser.getboolean('PARAMETERS', 'DO_PARALLEL')           # Use multiprocessing or not?
NUM_LATS = 180.0
NUM_LONS = 360.0
PLOT_CNTRS = 10
PLOT_CROPS = 10
DPI = 300
CFT_FRAC_YR = parser.getint('PARAMETERS', 'CFT_FRAC_YR')
GLM_STRT_YR = parser.getint('PARAMETERS', 'GLM_STRT_YR')
GLM_END_YR = parser.getint('PARAMETERS', 'GLM_END_YR')
FILL_ZEROS = parser.getboolean('PARAMETERS', 'FILL_ZEROS')
TEST_CFT = parser.getboolean('PROJECT', 'TEST_CFT')

# Directories
data_dir = dir_prj + os.sep + parser.get('PATHS', 'data_dir') + os.sep
out_dir = dir_prj + os.sep + parser.get('PATHS', 'out_dir') + os.sep + PROJ_NAME + os.sep
log_dir = out_dir + os.sep + 'Logs'
fao_dir = data_dir + os.sep + parser.get('PATHS', 'fao_dir') + os.sep
hyde_dir = data_dir + os.sep + parser.get('HYDE', 'hyde32_crop_path')

# FAO CONSTANTS
FAO_REGION_CODE = 1000 # FAO country codes maximum limit
RAW_FAO = fao_dir + os.sep + parser.get('PROJECT', 'RAW_FAO')
RAW_FAO_SHT = parser.get('PROJECT', 'RAW_FAO_SHT')
CROP_LUP = fao_dir + os.sep + parser.get('PROJECT', 'CROP_LUP')
CROP_LUP_SHT = parser.get('PROJECT', 'CROP_LUP_SHT')
FAO_CONCOR = fao_dir + os.sep + parser.get('PROJECT', 'FAO_CONCOR')

# Crop rotations csv file
csv_rotations = data_dir + os.sep + parser.get('PROJECT', 'CSV_ROTATIONS')

# Create directories
util.make_dir_if_missing(data_dir)
util.make_dir_if_missing(out_dir)
util.make_dir_if_missing(log_dir)

max_threads = multiprocessing.cpu_count() - 1
