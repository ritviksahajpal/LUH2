import ast
import os
import sys

if sys.version_info.major == 3:
    from configparser import SafeConfigParser
else:
    from ConfigParser import SafeConfigParser
from pathlib2 import Path

# Ignore matplotlib deprecation warnings
import warnings
import matplotlib.cbook
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning)  # TODO: Might be unadvisable to suppress all future warnings

# Parse config file
parser = SafeConfigParser()
parser.read('../config_IAM.txt')

# Get directory path (3 levels up is the parent directory)
dir_prj = str(Path(__file__).parents[3])

# Common values
dict_conts = {0: 'Antartica', 1: 'North_America', 2: 'South_America', 3: 'Europe', 4: 'Asia', 5: 'Africa',
              6: 'Australia'}

#####################################################################################
# Tags to be modified by user
#####################################################################################
do_email = parser.getboolean('CONTROL', 'do_email')
email_list = ast.literal_eval(parser.get('CONTROL', 'email_list'))
SHFT_MAP = parser.get('CONTROL', 'SHFT_MAP')  # Use Andreas or Butler?
MOVIE_SEP = 10
do_LUH1 = parser.getboolean('CONTROL', 'do_LUH1')
PLOT_HYDE = parser.getboolean('CONTROL', 'PLOT_HYDE')
PREPROCESS_GCAM = parser.getboolean('CONTROL', 'PREPROCESS_GCAM')
PREPROCESS_IMAG = parser.getboolean('CONTROL', 'PREPROCESS_IMAG')
CONVERT_WH = parser.getboolean('CONTROL', 'CONVERT_WH')  # Convert WH information from AEZ to country level
ending_diag_cols = ast.literal_eval(parser.get('CONTROL', 'ending_diag_cols'))
MATURITY_AGE = parser.getfloat('CONTROL', 'MATURITY_AGE')

# Directories
input_dir = dir_prj + os.sep + parser.get('GLM', 'path_input')
gcam_dir = input_dir + os.sep + parser.get('PATHS', 'gcam_dir') + os.sep
out_dir = dir_prj + os.sep + parser.get('PATHS', 'out_dir') + os.sep + parser.get('PROJECT', 'project_name') + os.sep
log_dir = out_dir + os.sep + 'Logs'
codes_dir = input_dir + os.sep + parser.get('PATHS', 'codes_dir')  # Continent and country codes

# project-specific constants
TAG = parser.get('PROJECT', 'TAG')
CROPS = ast.literal_eval(parser.get('GCAM', 'CROPS'))
PASTURE = ast.literal_eval(parser.get('GCAM', 'PASTURE'))
FOREST = ast.literal_eval(parser.get('GCAM', 'FOREST'))
URBAN = ast.literal_eval(parser.get('GCAM', 'URBAN'))
FNF_DEFN = 2.0  # Forest/Non-forest definition

# GLM
cell_area_name = 'carea'
ice_water_frac = 'icwtr'
# Control parameters for glm
do_alternate = parser.getboolean('GLM', 'do_alternate')
legend_glm = parser.get('GLM', 'legend_glm')
legend_alt_glm = parser.get('GLM', 'legend_alt_glm')
# glm static data
path_glm_stat = dir_prj + os.sep + parser.get('GLM', 'path_glm_stat')  # static data e.g. cell area
path_glm_carea = dir_prj + os.sep + parser.get('GLM', 'path_glm_carea')  # glm cell area quarter dedpigree
path_glm_vba = dir_prj + os.sep + parser.get('GLM', 'path_glm_vba')  # glm miami lu based virgin biomass
path_glm_new_vba = dir_prj + os.sep + parser.get('GLM', 'path_glm_new_vba')  # glm NEW miami lu based virgin biomass
# Path glm input/output
path_glm_input = dir_prj + os.sep + parser.get('GLM', 'path_glm_input')
glm_experiments = ast.literal_eval(parser.get('GLM', 'folder_glm'))
path_glm_output = dir_prj + os.sep + parser.get('GLM', 'path_glm_output')  # output path
# netCDF glm
path_nc_states = parser.get('GLM', 'path_nc_states')  # states data for past
path_nc_trans = parser.get('GLM', 'path_nc_trans')  # Transition data for past
path_nc_mgt = parser.get('GLM', 'path_nc_mgt')  # management data for past
# netCDF future glm
path_nc_futr_states = parser.get('GLM', 'path_nc_futr_states')  # states data for future
path_nc_futr_mgt = parser.get('GLM', 'path_nc_futr_mgt')  # management data for futue
# netCDF past + future glm
path_nc_all_states = parser.get('GLM', 'path_nc_all_states')  # states data for past + future
path_nc_all_mgt = parser.get('GLM', 'path_nc_all_mgt')  # management data for past + future
# Second set of glm outputs (for comparison with first set)
path_nc_alt_state = parser.get('GLM', 'path_nc_alt_state')  # alternate glm data states
path_nc_alt_trans = parser.get('GLM', 'path_nc_alt_trans')  # transitions for alternate glm data

# HYDE3.2_march
HYDE32_march_CROP_PATH = dir_prj + os.sep + parser.get('HYDE3.2_march', 'hyde32_march_crop_path')
HYDE32_march_OTHR_PATH = dir_prj + os.sep + parser.get('HYDE3.2_march', 'hyde32_march_othr_path')
HYDE32_march_PAST_PATH = dir_prj + os.sep + parser.get('HYDE3.2_march', 'hyde32_march_past_path')
HYDE32_march_GRAZ_PATH = dir_prj + os.sep + parser.get('HYDE3.2_march', 'hyde32_march_graz_path')
HYDE32_march_RANG_PATH = dir_prj + os.sep + parser.get('HYDE3.2_march', 'hyde32_march_rang_path')
HYDE32_march_URBN_PATH = dir_prj + os.sep + parser.get('HYDE3.2_march', 'hyde32_march_urbn_path')
HYDE32_march_OUT_PATH = dir_prj + os.sep + parser.get('HYDE3.2_march', 'out_hyde32_march_path')

# HYDE3.2
HYDE32_CROP_PATH = dir_prj + os.sep + parser.get('HYDE3.2', 'hyde32_crop_path')
HYDE32_OTHR_PATH = dir_prj + os.sep + parser.get('HYDE3.2', 'hyde32_othr_path')
HYDE32_PAST_PATH = dir_prj + os.sep + parser.get('HYDE3.2', 'hyde32_past_path')
HYDE32_GRAZ_PATH = dir_prj + os.sep + parser.get('HYDE3.2', 'hyde32_graz_path')
HYDE32_RANG_PATH = dir_prj + os.sep + parser.get('HYDE3.2', 'hyde32_rang_path')
HYDE32_URBN_PATH = dir_prj + os.sep + parser.get('HYDE3.2', 'hyde32_urbn_path')
HYDE32_OUT_PATH = dir_prj + os.sep + parser.get('HYDE3.2', 'out_hyde32_path')

# HYDE3.2v1h_beta_crop_path
HYDE32_v1hb_CROP_PATH = dir_prj + os.sep + parser.get('HYDE3.2_v1h_beta', 'hyde32_v1h_beta_crop_path')
HYDE32_v1hb_OTHR_PATH = dir_prj + os.sep + parser.get('HYDE3.2_v1h_beta', 'hyde32_v1h_beta_othr_path')
HYDE32_v1hb_PAST_PATH = dir_prj + os.sep + parser.get('HYDE3.2_v1h_beta', 'hyde32_v1h_beta_past_path')
HYDE32_v1hb_GRAZ_PATH = dir_prj + os.sep + parser.get('HYDE3.2_v1h_beta', 'hyde32_v1h_beta_graz_path')
HYDE32_v1hb_RANG_PATH = dir_prj + os.sep + parser.get('HYDE3.2_v1h_beta', 'hyde32_v1h_beta_rang_path')
HYDE32_v1hb_URBN_PATH = dir_prj + os.sep + parser.get('HYDE3.2_v1h_beta', 'hyde32_v1h_beta_urbn_path')
HYDE32_v1hb_OUT_PATH = dir_prj + os.sep + parser.get('HYDE3.2_v1h_beta', 'out_hyde32_v1h_beta_path')

# HYDE3.2 # Used in version 0.3
HYDE32_v03_CROP_PATH = dir_prj + os.sep + parser.get('HYDE3.2_v0.3', 'hyde32_v0.3_crop_path')
HYDE32_v03_OTHR_PATH = dir_prj + os.sep + parser.get('HYDE3.2_v0.3', 'hyde32_v0.3_othr_path')
HYDE32_v03_PAST_PATH = dir_prj + os.sep + parser.get('HYDE3.2_v0.3', 'hyde32_v0.3_past_path')
HYDE32_v03_GRAZ_PATH = dir_prj + os.sep + parser.get('HYDE3.2_v0.3', 'hyde32_v0.3_graz_path')
HYDE32_v03_RANG_PATH = dir_prj + os.sep + parser.get('HYDE3.2_v0.3', 'hyde32_v0.3_rang_path')
HYDE32_v03_URBN_PATH = dir_prj + os.sep + parser.get('HYDE3.2_v0.3', 'hyde32_v0.3_urbn_path')
HYDE32_v03_OUT_PATH = dir_prj + os.sep + parser.get('HYDE3.2_v0.3', 'out_hyde32_v0.3_path')

# HYDE 3.1
HYDE31_CROP_PATH = dir_prj + os.sep + parser.get('HYDE3.1', 'hyde31_crop_path')
HYDE31_OTHR_PATH = dir_prj + os.sep + parser.get('HYDE3.1', 'hyde31_othr_path')
HYDE31_PAST_PATH = dir_prj + os.sep + parser.get('HYDE3.1', 'hyde31_past_path')
HYDE31_URBN_PATH = dir_prj + os.sep + parser.get('HYDE3.1', 'hyde31_urbn_path')
HYDE31_OUT_PATH = dir_prj + os.sep + parser.get('HYDE3.1', 'out_hyde31_path')

# GCAM
GCAM_OUT = dir_prj + os.sep + parser.get('GCAM', 'GCAM_OUT')
GCAM_CROPS = dir_prj + os.sep + parser.get('GCAM', 'GCAM_CROPS')
DPI = 150  # dots per inch for saved figures
NUM_LATS = 180.0
NUM_LONS = 360.0

# FAO
FAO_CONCOR = input_dir + os.sep + parser.get('COUNTRIES', 'FAO_CONCOR')

# Country and continent codes
ccodes_file = codes_dir + os.sep + parser.get('CODES', 'ccodes_file')
contcodes_file = codes_dir + os.sep + parser.get('CODES', 'contcodes_file')

# Common data
CNTRY_CODES = input_dir + os.sep + parser.get('LUMIP_DATA', 'ccodes')  # Country codes
CELL_AREA_Q = input_dir + os.sep + parser.get('LUMIP_DATA', 'careaq')  # quarter degree cell area
CELL_AREA_H = input_dir + os.sep + parser.get('LUMIP_DATA', 'careah')  # half degree cell area
CELL_AREA_Q = input_dir + os.sep + parser.get('LUMIP_DATA', 'careao')  # one degree cell area

# Shifting cultivation
shft_by_country = parser.getboolean('SHIFTING', 'shft_by_country')
ASC_BUTLER = input_dir + os.sep + parser.get('SHIFTING', 'butler_ascii')  # Butler map (ascii)
TIF_ANDREAS = input_dir + os.sep + parser.get('SHIFTING', 'andreas_map')  # Andreas map (tiff)
NC_ANDREAS = input_dir + os.sep + parser.get('SHIFTING', 'andreas_nc')  # Andreas netCDF
ASC_ANDREAS = input_dir + os.sep + parser.get('SHIFTING', 'andreas_ascii')  # Andreas ascii
ASC_ANDREAS_2100 = input_dir + os.sep + parser.get('SHIFTING', 'andreas_ascii_2100')  # Andreas ascii

# Age
age_poulter = input_dir + os.sep + parser.get('AGE', 'age_poulter')

# Other
rs_forest = input_dir + os.sep + parser.get('OTHER', 'rs_forest')
wh_file = input_dir + os.sep + parser.get('OTHER', 'wh_file')

# Monfreda
MFD_DATA_DIR = input_dir + os.sep + parser.get('MONFREDA', 'mon_data_dir')

# MIAMI-LU
miami_lu_nc = input_dir + os.sep + parser.get('MIAMI_LU', 'miami_lu_nc')
miami_npp = input_dir + os.sep + parser.get('MIAMI_LU', 'miami_npp')  # Previous MIAMI-LU NPP estimates
miami_vba = input_dir + os.sep + parser.get('MIAMI_LU', 'miami_vba')  # Previous MIAMI-LU biomass estimates

# HOTSPOTS
file_hotspots = dir_prj + os.sep + parser.get('HOTSPOTS', 'path_hotspots')

# GCAM Wood harvest file
WOOD_HARVEST = dir_prj + os.sep + parser.get('GCAM', 'WOOD_HARVEST')
FERT_DATA = dir_prj + os.sep + parser.get('GCAM', 'FERT_DATA')
GCAM_START_YR = parser.getint('GCAM', 'GCAM_START_YR')
GCAM_END_YR = parser.getint('GCAM', 'GCAM_END_YR')
GCAM_STEP_YR = parser.getint('GCAM', 'GCAM_STEP_YR')
SKIP_GCAM_COLS = parser.getint('GCAM', 'SKIP_GCAM_COLS')
GCAM_MAPPING = dir_prj + os.sep + parser.get('GCAM', 'GCAM_MAPPING')

# CONSTANTS
FILL_VALUE = 1e20
M2_TO_KM2 = 0.000001
M2_TO_HA = 1e-4
KM2_TO_HA = 100.0
IMG_SIZE = 100.0
BILLION = 1e9
KG_TO_PG = 1e-12
KG_TO_TG = 1e-9
KG_TO_MG = 1e-6
TO_MILLION = 1e-6
AGB_TO_BIOMASS = 4.0/3.0

