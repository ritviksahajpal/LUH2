import os, logging, ast, sys
import logging.handlers

if sys.version_info.major == 3:
    from configparser import SafeConfigParser
else:
    from ConfigParser import SafeConfigParser
from pathlib2 import Path

# Parse config file
parser = SafeConfigParser()
parser.read('config_AgNPP.txt')

# Get directory path (3 levels up is the parent directory)
dir_prj = str(Path(__file__).parents[3])

TAG = parser.get('PROJECT', 'TAG')

# AgNPP
prj_name = parser.get('PROJECT', 'project_name')
base_dir = parser.get('PATHS', 'input_dir')
out_dir = parser.get('PATHS', 'out_dir') + os.sep + prj_dir + os.sep
log_dir = out_dir + os.sep + 'Logs'
inp_dir = prj_dir + os.sep + base_dir + os.sep
AgNPP_fname = parser.get('AgNPP', 'AgNPP_file')
AgNPP_file = inp_dir + os.sep + AgNPP_fname

# Create directories
import pygeoutil.util as util
util.make_dir_if_missing(base_dir)
util.make_dir_if_missing(out_dir)
util.make_dir_if_missing(log_dir)

# Logging
LOG_FILENAME   = out_dir + os.sep + 'Log_' + TAG + '.txt'
logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO,\
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',\
                    datefmt="%m-%d %H:%M") # Logging levels are DEBUG, INFO, WARNING, ERROR, and CRITICAL
# Add a rotating handler
logging.getLogger().addHandler(logging.handlers.RotatingFileHandler(LOG_FILENAME, maxBytes=50000, backupCount=5))
# Output to screen
logging.getLogger().addHandler(logging.StreamHandler())