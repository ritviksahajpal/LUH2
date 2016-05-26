from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import numpy as np
import pdb

filename = '/Users/ritvik/Documents/Projects/ED/global_aug4/global_aug4.region.nc'
pdb.set_trace()
with Dataset(filename, mode='r') as fh:
   lons = fh.variables['lon'][:]
   lats = fh.variables['lat'][:]
   biom = fh.variables['biomass'][:].squeeze()

lons_sub, lats_sub = np.meshgrid(lons[::4], lats[::4])

sst_coarse = Basemap.interp(biom, lons, lats, lons_sub, lats_sub, order=1)