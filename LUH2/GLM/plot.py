import os
import click
import logging
import palettable
import subprocess
import pdb
import math
from jenks import jenks  # pip install -e "git+https://github.com/perrygeo/jenks.git#egg=jenks"
import sys
from jenks import jenks

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as tkr
from matplotlib import rcParams, ticker
from mpl_toolkits.basemap import Basemap, maskoceans
from matplotlib.ticker import MaxNLocator, AutoLocator
from matplotlib.ticker import LogFormatter
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from simplekml import (Kml, OverlayXY, ScreenXY, Units, RotationXY, AltitudeMode, Camera)
from random import shuffle
from cycler import cycler
from tqdm import tqdm
import seaborn.apionly as sns

import constants
import pygeoutil.util as util

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


@click.group()
def plot_glm():
    pass


def set_matplotlib_params():
    """
    Set matplotlib defaults to nicer values.
    """
    # rcParams dict
    rcParams['mathtext.default'] = 'regular'
    rcParams['axes.labelsize'] = 12
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 12
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.serif'] = ['Helvetica']
    # rcParams['figure.figsize'] = 7.3, 4.2


def get_colors(palette='colorbrewer', cmap=False):
    """
    Get palettable colors, which are nicer
    :param palette
    :param cmap
    """
    if palette == 'colorbrewer':
        bmap = palettable.colorbrewer.diverging.PRGn_11.mpl_colors
        if cmap:
            bmap = palettable.colorbrewer.diverging.PRGn_11.mpl_colormap
    elif palette == 'tableau':
        bmap = palettable.tableau.Tableau_20.mpl_colors
        if cmap:
            bmap = palettable.tableau.Tableau_20.mpl_colormap
        bmap = bmap[0::2] + bmap[1::2]  # Move the even numbered colors to end of list (they are very light)
    elif palette == 'cubehelix':
        bmap = palettable.cubehelix.cubehelix2_16.mpl_colors
        if cmap:
            bmap = palettable.cubehelix.cubehelix2_16.mpl_colormap
    elif palette == 'qualitative':
        bmap = palettable.tableau.GreenOrange_12.mpl_colors
        if cmap:
            bmap = palettable.tableau.GreenOrange_12.mpl_colormap

    all_markers = ['s', 'o', '^', '*', 'v', '<', '>', 4, 5] * 100

    max_values = len(bmap) if len(bmap) < len(all_markers) else len(all_markers)
    # Make sure equal number of cycle elements for color and markers
    markers = all_markers[:max_values]
    bmap = bmap[:max_values]

    color_cycle = cycler('color', bmap)  # color cycle
    marker_cycle = cycler('marker', markers)  # marker cycle

    plt.rc('axes', prop_cycle=(color_cycle + marker_cycle))

    return bmap


def simple_axis(ax):
    """
    Remove spines from top and right
    :param ax
    """
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def gridded_axis(ax):
    """
    Plot major, minor ticks as well as a grid
    :param ax:
    :return:
    """
    # Set number of major and minor ticks
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoLocator())

    # Create nice-looking grid for ease of visualization
    ax.grid(which='minor', alpha=0.2, linestyle='--')
    ax.grid(which='major', alpha=0.5, linestyle='--')

    # For y-axis, format the numbers
    scale = 1
    ticks = tkr.FuncFormatter(lambda x, pos: '{:0,d}'.format(int(x/scale)))
    ax.yaxis.set_major_formatter(ticks)


def simple_legend(ax):
    """
    Remove previous legend if any and draw in best location, with no fancy-box and some alpha
    :param ax:
    :param num_rows:
    :param position:
    :return:
    """
    leg = ax.legend(fancybox=None, prop={'size': 10},  loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3,
                    frameon=True)
    leg.get_frame().set_linewidth(0.0)


def get_cb_range(arr=np.empty([2, 2]), xaxis_min=0.0, xaxis_max=1.1, xaxis_step=0.1, do_jenks=True):
    """
    https://github.com/perrygeo/jenks
    :param arr:
    :param xaxis_min:
    :param xaxis_max:
    :param xaxis_step:
    :param do_jenks:
    :return:
    """
    # Array can only have shape == 2
    if len(np.shape(arr)) != 2:
        sys.exit(0)

    if do_jenks:
        # Select 11 elements, discard the highest
        arr = np.array(jenks(np.unique(np.round_(arr, decimals=1)).data, 11))[:-1]

        # return only the unique elements, sometimes jenks selects duplicate elements
        return np.unique(arr)
    else:
        return np.arange(xaxis_min, xaxis_max, xaxis_step)


def truncate_colormap(cmap, minval=0.01, maxval=1.0, n=100):
    """
    http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    :param cmap:
    :param minval:
    :param maxval:
    :param n:
    :return:
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def make_kml(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, figs, colorbar=None, **kw):
    """
    https://ocefpaf.github.io/python4oceanographers/blog/2014/03/10/gearth/
    :param llcrnrlon:
    :param llcrnrlat:
    :param urcrnrlon:
    :param urcrnrlat:
    :param figs:
    :param colorbar
    """
    """TODO: LatLon bbox, list of figs, optional colorbar figure,
    and several simplekml kw..."""

    kml = Kml()
    altitude = kw.pop('altitude', 2e7)
    roll = kw.pop('roll', 0)
    tilt = kw.pop('tilt', 0)
    altitudemode = kw.pop('altitudemode', AltitudeMode.relativetoground)
    camera = Camera(latitude=np.mean([urcrnrlat, llcrnrlat]),
                    longitude=np.mean([urcrnrlon, llcrnrlon]),
                    altitude=altitude, roll=roll, tilt=tilt,
                    altitudemode=altitudemode)

    kml.document.camera = camera
    draworder = 0
    for fig in figs:  # NOTE: Overlays are limited to the same bbox.
        draworder += 1
        ground = kml.newgroundoverlay(name='GroundOverlay')
        ground.draworder = draworder
        ground.visibility = kw.pop('visibility', 1)
        ground.name = kw.pop('name', 'overlay')
        ground.color = kw.pop('color', '9effffff')
        ground.atomauthor = kw.pop('author', 'ocefpaf')
        ground.latlonbox.rotation = kw.pop('rotation', 0)
        ground.description = kw.pop('description', 'Matplotlib figure')
        ground.gxaltitudemode = kw.pop('gxaltitudemode',
                                       'clampToSeaFloor')
        ground.icon.href = fig
        ground.latlonbox.east = llcrnrlon
        ground.latlonbox.south = llcrnrlat
        ground.latlonbox.north = urcrnrlat
        ground.latlonbox.west = urcrnrlon

    if colorbar:  # Options for colorbar are hard-coded (to avoid a big mess).
        screen = kml.newscreenoverlay(name='ScreenOverlay')
        screen.icon.href = colorbar
        screen.overlayxy = OverlayXY(x=0, y=0,
                                     xunits=Units.fraction,
                                     yunits=Units.fraction)
        screen.screenxy = ScreenXY(x=0.015, y=0.075,
                                   xunits=Units.fraction,
                                   yunits=Units.fraction)
        screen.rotationXY = RotationXY(x=0.5, y=0.5,
                                       xunits=Units.fraction,
                                       yunits=Units.fraction)
        screen.size.x = 0
        screen.size.y = 0
        screen.size.xunits = Units.fraction
        screen.size.yunits = Units.fraction
        screen.visibility = 1

    kmzfile = kw.pop('kmzfile', 'overlay.kmz')
    kml.savekmz(kmzfile)


def gearth_fig(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, pixels=20480):
    """
    https://ocefpaf.github.io/python4oceanographers/blog/2014/03/10/gearth/
    Return a Matplotlib `fig` and `ax` handles for a Google-Earth Image.
    """
    aspect = np.cos(np.mean([llcrnrlat, urcrnrlat]) * np.pi/180.0)
    xsize = np.ptp([urcrnrlon, llcrnrlon]) * aspect
    ysize = np.ptp([urcrnrlat, llcrnrlat])
    aspect = ysize / xsize

    if aspect > 1.0:
        figsize = (10.0 / aspect, 10.0)
    else:
        figsize = (10.0, 10.0 * aspect)

    if False:
        plt.ioff()  # Make `True` to prevent the KML components from poping-up.
    fig = plt.figure(figsize=figsize,
                     frameon=False,
                     dpi=pixels//10)

    # KML friendly image.  If using basemap try: `fix_aspect=False`.
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(llcrnrlon, urcrnrlon)
    ax.set_ylim(llcrnrlat, urcrnrlat)
    return fig, ax


def output_kml(trans, lon, lat, path_out, xmin, xmax, step, cmap, fname_out='out', name_legend='', label='',
               do_log_cb=False):
    """

    :param trans:
    :param lon:
    :param lat:
    :param path_out:
    :param xmin:
    :param xmax:
    :param step:
    :param cmap:
    :param fname_out:
    :param name_legend:
    :param label:
    :param do_log_cb:
    :return:
    """
    # Return if xmin == xmax
    if xmin == xmax:
        return

    logging.info('output_kml' + fname_out)

    dir_output = path_out + os.sep + 'kml'
    util.make_dir_if_missing(dir_output)

    fig, ax = gearth_fig(llcrnrlon=lon.min(), llcrnrlat=lat.min(), urcrnrlon=lon.max(), urcrnrlat=lat.max())
    lons, lats = np.meshgrid(lon, lat)

    m = Basemap(projection='cyl', resolution='c')
    x, y = m(lons, lats)
    mask_data = maskoceans(lons, lats, trans)
    m.etopo()

    if do_log_cb and np.nanmin(mask_data) > 0.0:
        # manually set log levels e.g. http://matplotlib.org/examples/images_contours_and_fields/contourf_log.html
        lev_exp = np.arange(np.floor(np.log10(np.nanmin(mask_data)) - 1), np.ceil(np.log10(np.nanmax(mask_data)) + 1))
        levs = np.power(10, lev_exp)
        cs = m.contourf(x, y, mask_data, levs, norm=colors.LogNorm(), cmap=cmap)
    else:
        cs = m.contourf(x, y, mask_data, np.arange(xmin, xmax, step), cmap=cmap)
        if abs(xmax - xmin) > 10000.0:
            format = '%.1e'
        elif abs(xmax - xmin) > 100.0:
            format = '%.0f'
        elif abs(xmax - xmin) > 1.0:
            format = '%.1f'
        elif abs(xmax - xmin) > 0.1:
            format = '%.3f'
        else:
            format = '%.4f'
    ax.set_axis_off()
    fig.savefig(dir_output + os.sep + 'kml_' + fname_out + '.png', transparent=False, format='png', dpi=800)

    # Colorbar
    fig = plt.figure(figsize=(1.0, 4.0), facecolor=None, frameon=False)

    # Colorbar legend, numbers represent: [bottom_left_x_coord, bottom_left_y_coord, width, height]
    ax = fig.add_axes([0.02, 0.05, 0.2, 0.9])
    if do_log_cb and np.nanmin(mask_data) >= 0.0:
        cb = fig.colorbar(cs, cax=ax, spacing='uniform')
    else:
        cb = fig.colorbar(cs, cax=ax, format=format, spacing='uniform')
    cb.set_label(name_legend + '\n' + label, rotation=-90, color='k', labelpad=25, size=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(dir_output + os.sep + name_legend + '.png', transparent=False, format='png', dpi=125)

    make_kml(llcrnrlon=lon.min(), llcrnrlat=lat.min(), urcrnrlon=lon.max(), urcrnrlat=lat.max(),
             figs=[dir_output + os.sep + 'kml_' + fname_out + '.png'],
             colorbar=dir_output + os.sep + name_legend + '.png',
             kmzfile=dir_output + os.sep + fname_out + '.kmz', name=fname_out)

    # Delte temp files
    # os.remove(dir_output + os.sep + 'kml_' + fname_out + '.png')
    # os.remove(dir_output + os.sep + name_legend + '.png')

    plt.close('all')


def make_movie(list_images, out_path, out_fname):
    """

    :param list_images:
    :param out_path:
    :return:
    """
    util.make_dir_if_missing(out_path)

    convert_cmd = 'convert -delay 50 -loop 1 '+' '.join(list_images) + ' ' + out_path + os.sep + out_fname
    subprocess.call(convert_cmd, shell=True)


def plot_hist(hist, bin_edges, out_path, do_per=False, do_log=True, title='', xlabel='', ylabel=''):
    """
    Plot histogram using information contained in hist (frequency count) and bin_edges.
    :param hist: values of the histogram
    :param bin_edges: return the bin edges
    :param out_path: output path and file name
    :param do_per: plot each bin as percentage of total
    :param do_log: plot logarithmic y-axis
    :param title:
    :param xlabel:
    :param ylabel:
    :return:
    """
    logger.info('Plot histogram')
    sns.set_style('whitegrid')

    if do_per:
        # Compute percentage of total sum
        per_hist = hist * 100.0 / sum(hist)
        sns.barplot(bin_edges[:-1].astype(int), per_hist, color='purple', edgecolor='none')
        plt.ylim(0.0, 100.0)
    else:
        sns.barplot(bin_edges[:-1].astype(int), hist, color='purple', edgecolor='none', log=do_log)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=constants.DPI)
    plt.close()


def plot_np_ts(ax, np_arr, xvar, out_path, vert_yr=[], title='', leg_name='', xlabel='', ylabel='', col='k',
               do_log=False):
    """
    Plot single time-series. Add a vertical line for vert_yr
    :param ax
    :param np_arr:
    :param xvar:
    :param out_path:
    :param vert_yr:
    :param title:
    :param leg_name:
    :param xlabel:
    :param ylabel:
    :param col:
    :param do_log:
    :return:
    """
    logger.info('Plot time-series of numpy array')
    ax.plot(xvar, np_arr, label=leg_name, color=col, lw=1.75, markevery=int(len(xvar)/5.0), markeredgecolor='none')
    if do_log:
        ax.set_yscale('log')

    if len(vert_yr):
        # Set x-axis limit
        ax.set_xlim([min(xvar), max(xvar)])

        for yr in vert_yr:
            plt.axvline(yr, linestyle='--', color='k', lw=1.5)
            plt.annotate(str(yr), xy=(yr-5, ax.get_ylim()[0] + 0.01 * ax.get_ylim()[1]), color='k', size=8,
                         bbox=dict(edgecolor='none', fc='white', alpha=0.5))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    gridded_axis(ax)

    # Show y-axis in scientific format
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

    simple_legend(ax)
    plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=constants.DPI, bbox_inches='tight')
    plt.close()


def plot_multiple_ts(ax, np_arr, xvar, out_path='', vert_yr=[], title='', leg_name='', xlabel='', ylabel='',
                     linestyle='-', col=None, pos='first', do_log=False, fill_between=False):
    """
    Produces a single plot showing time-series of multiple variables
    :param ax:
    :param np_arr: Numpy array to plot on y-axis
    :param xvar: Time series
    :param out_path: Output path (includes output file name)
    :param vert_yr:
    :param title: Title of plot
    :param leg_name:
    :param xlabel:
    :param ylabel:
    :param linestyle:
    :param col:
    :param pos: if 'first', then set up axis, if 'last' then save image. Refers to whether current time-series is
    first or last
    :param do_log:
    :param fill_between:
    :return: Nothing, side-effect: save an image
    """
    logger.info('Plot multiple time-series')
    ax.plot(xvar, np_arr, label=leg_name, color=col, lw=1.75, linestyle=linestyle,
            markevery=int(len(xvar)/5.0), markeredgecolor='none')

    if fill_between:
        ax.fill_between(xvar, np_arr[:len(xvar)], y2=0)
    if do_log:
        ax.set_yscale('log')

    if pos == 'last':
        # Set x-axis limit
        ax.set_xlim([min(xvar), max(xvar)])

        # Annotate
        if len(vert_yr):
            for yr in vert_yr:
                ax.axvline(yr, linestyle='--', color='k', lw=1.5)
                plt.annotate(str(yr), xy=(yr-5, ax.get_ylim()[0] + 0.01 * ax.get_ylim()[1]), color='k', size=8,
                             bbox=dict(edgecolor='none', fc='white', alpha=0.5))

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        gridded_axis(ax)

        # Show y-axis in scientific format
        formatter = tkr.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)

        simple_legend(ax)
        plt.title(title.capitalize())
        plt.tight_layout()

        plt.savefig(out_path, dpi=constants.DPI, bbox_inches='tight')


def plot_hovmoller(nc_path, var, out_path, do_latitude=True, xlabel='', ylabel='', title='', cbar=''):
    """
    Ref: http://scitools.org.uk/iris/docs/v0.9.1/examples/graphics/hovmoller.html
    :param nc_path:
    :param var:
    :param out_path:
    :param do_latitude:
    :param xlabel:
    :param ylabel:
    :param title:
    :param cbar:
    :return:
    """
    logger.info('Plot hovmoller')
    # TODO: Iris install is not working on mac os x
    if os.name == 'mac' or os.name == 'posix':
        return
    import iris
    import iris.plot as iplt
    iris.FUTURE.netcdf_promote = True
    cubes = iris.load(nc_path, var)

    # Take the mean over latitude/longitude
    if do_latitude:
        cube = cubes[0].collapsed('latitude', iris.analysis.MEAN)
    else:
        cube = cubes[0].collapsed('longitude', iris.analysis.MEAN)

    # Create the plot contour with 20 levels
    iplt.contourf(cube, 20, cmap=palettable.colorbrewer.diverging.RdYlGn_9.mpl_colormap)
    if not do_latitude:
        plt.ylabel(xlabel)  # Latitude
        plt.xlabel(ylabel)  # Years
    else:
        plt.ylabel(ylabel)  # Years
        plt.xlabel(xlabel)  # Longitude
    plt.title(title)
    plt.colorbar(orientation='horizontal', extend='both', drawedges=False, spacing='proportional').set_label(cbar)

    # Stop matplotlib providing clever axes range padding and do not draw gridlines
    plt.grid(b=False)
    plt.axis('tight')
    plt.tight_layout()
    plt.savefig(out_path, dpi=constants.DPI)
    plt.close()


# @plot_glm.command()
# @click.argument('path_nc')
# @click.argument('out_path')
# @click.argument('var_name')
# @click.option('--xaxis_min', default=0.0, help='')
# @click.option('--xaxis_max', default=1.1, help='')
# @click.option('--xaxis_step', default=0.1, help='')
# @click.option('--annotate_date', help='')
# @click.option('--yr', default=0, help='')
# @click.option('--date', default=-1, help='')
# @click.option('--xlabel', default='', help='')
# @click.option('--title', default='', help='')
# @click.option('--tme_name', default='time', help='')
# @click.option('--show_plot', help='')
# @click.option('--any_time_data', default=True, help='')
# @click.option('--format', default='%.2f', help='')
# @click.option('--land_bg', help='')
# @click.option('--cmap', default=plt.cm.RdBu, help='')
# @click.option('--grid', help='')
# @click.option('--fill_mask', help='')
def plot_map_from_nc(path_nc, out_path, var_name, xaxis_min=0.0, xaxis_max=1.1, xaxis_step=0.1,
                     annotate_date=False, yr=0, date=-1, xlabel='', title='', tme_name='time', show_plot=False,
                     any_time_data=True, format='%.2f', land_bg=True, cmap=plt.cm.RdBu, grid=False, fill_mask=False):
    """
    Plot var_name variable from netCDF file

    \b
    Args:
        path_nc: Name of netCDF file including path
        out_path: Output directory path + file name
        var_name: Name of variable in netCDF file to plot on map

    Returns:
        Nothing, side-effect: save an image
    """
    logger.info('Plotting ' + var_name + ' in ' + path_nc)

    # Read netCDF file and get time dimension
    nc = util.open_or_die(path_nc, 'r', format='NETCDF4')
    lon = nc.variables['lon'][:]
    lat = nc.variables['lat'][:]

    if any_time_data:
        ts = nc.variables[tme_name][:]  # time-series
        if date == -1:  # Plot either the last year {len(ts)-1} or whatever year the user wants
            plot_yr = len(ts) - 1
        else:
            plot_yr = date - ts[0]

    # Draw empty basemap
    m = Basemap(projection='robin', resolution='c', lat_0=0, lon_0=0)
    # m.drawcoastlines()
    # m.drawcountries()

    # Find x,y of map projection grid.
    lons, lats = np.meshgrid(lon, lat)
    x, y = m(lons, lats)
    if fill_mask:
        nc_vars = np.ma.filled(nc.variables[var_name], fill_value=np.nan)
    else:
        nc_vars = np.array(nc.variables[var_name])

    # Plot
    # Get data for the last year from the netCDF file array
    if any_time_data:
        mask_data = maskoceans(lons, lats, nc_vars[int(plot_yr), :, :])
    else:
        mask_data = maskoceans(lons, lats, nc_vars[:, :])

    m.etopo()
    if land_bg:
        m.drawlsmask(land_color='white', ocean_color='none', lakes=True)  # land_color = (0, 0, 0, 0) for transparent
    else:
        m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='none', lakes=True)

    cs = m.contourf(x, y, mask_data, np.arange(xaxis_min, xaxis_max, xaxis_step), cmap=cmap)

    if annotate_date:
        plt.annotate(str(yr), xy=(0.45, 0.1), xycoords='axes fraction', size=20)

    if grid:
        # where labels intersect = [left, right, top, bottom]
        m.drawmeridians(np.arange(-180, 180, 60), labels=[0,0,1,0], labelstyle='+/-', linewidth=0.5)
        m.drawparallels([-40, 0, 40], labels=[1, 0, 0, 0], labelstyle='+/-', linewidth=0.5)

    # Add colorbar
    cb = m.colorbar(cs, "bottom", size="3%", pad='2%', extend='both', drawedges=False, spacing='proportional',
                    format=format)
    cb.set_label(xlabel)
    plt.title(title, y=1.08)

    plt.tight_layout()
    if not show_plot:
        plt.savefig(out_path, dpi=constants.DPI)
        plt.close()
    else:
        plt.show()

    nc.close()

    return out_path


def plot_maps_ts(arr_or_nc, ts, lon, lat, out_path, var_name='', xaxis_min=0.0, xaxis_max=1.1, xaxis_step=0.1,
                 save_name='fig', xlabel='', start_movie_yr=-1, title='', tme_name='time', land_bg=True, do_etopo=False,
                 do_log_cb=False, do_jenks=True, cmap=plt.cm.RdBu, grid=False):
    """

    Args:
        arr_or_nc: Input can be numpy array or netcdf path
        ts:
        lon:
        lat:
        out_path:
        var_name:
        xaxis_min:
        xaxis_max:
        xaxis_step:
        save_name:
        xlabel:
        start_movie_yr:
        title:
        tme_name:
        land_bg:
        do_etopo:
        do_log_cb: Draw logarithmic colorbar (true) or not (false). Default: False
        do_jenks:
        cmap:
        grid:

    Returns:

    """
    logger.info('Plot time-series of maps')
    if isinstance(arr_or_nc, (np.ndarray, np.generic)):
        is_nc = False
        arr = np.copy(arr_or_nc)
    elif os.path.splitext(arr_or_nc)[1] == '.nc':
        is_nc = True
    else:
        sys.exit(0)

    list_pngs = []
    base_yr = ts[0]

    # Draw empty basemap
    m = Basemap(projection='robin', resolution='c', lat_0=0, lon_0=0)
    # m.drawcoastlines()
    # m.drawcountries()

    # Find x,y of map projection grid.
    lons, lats = np.meshgrid(lon, lat)
    x, y = m(lons, lats)

    # Plot
    # Get data for the last year from the netCDF file array
    for yr in tqdm(ts[::constants.MOVIE_SEP], disable=(len(ts[::constants.MOVIE_SEP]) < 2)):
        if do_etopo:
            m.etopo()

        if len(ts) > 1 and not is_nc:
            mask_data = maskoceans(lons, lats, arr[int(yr - base_yr), :, :])
        else:
            if is_nc:
                arr = util.get_nc_var3d(arr_or_nc, var_name, int(yr - base_yr))
            mask_data = maskoceans(lons, lats, arr[:, :])

        cb_range = get_cb_range(arr, xaxis_min=xaxis_min, xaxis_max=xaxis_max, xaxis_step=xaxis_step, do_jenks=do_jenks)

        if land_bg:
            m.drawlsmask(land_color='white', ocean_color='aqua', lakes=True)  # land_color = (0, 0, 0, 0) transparent
        else:
            m.drawlsmask(land_color=(0, 0, 0, 0), ocean_color='none', lakes=True)

        if np.any(cb_range < 0.0) or not do_log_cb:
            # If any negative values exist in basemap then do not use log scale
            cs = m.contourf(x, y, mask_data, cb_range, extend='both', cmap=cmap)
        else:
            # manually set log levels e.g. http://matplotlib.org/examples/images_contours_and_fields/contourf_log.html
            lev_exp = np.arange(np.floor(np.log10(mask_data.min()) - 1), np.ceil(np.log10(mask_data.max()) + 1))
            levs = np.power(10, lev_exp)
            cs = m.contourf(x, y, mask_data, levs, norm=colors.LogNorm(), cmap=cmap)
        plt.annotate(str(int(yr)), xy=(0.45, 0.1), xycoords='axes fraction', size=20)

        if grid:
            # where labels intersect = [left, right, top, bottom]
            m.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 1, 0], labelstyle='+/-', linewidth=0.5)
            m.drawparallels([-40, 0, 40], labels=[1, 0, 0, 0], labelstyle='+/-', linewidth=0.5)

        # Add colorbar
        cb = m.colorbar(cs, "bottom", size="3%", pad='2%', extend='both', drawedges=False, spacing='uniform')

        # Add label
        cb.set_label(xlabel)
        plt.title(title, y=1.08)

        out_png_name = out_path + os.sep + save_name + '_' + str(int(yr)) + '.png'
        list_pngs.append(out_png_name)
        plt.tight_layout()
        plt.savefig(out_png_name, dpi=constants.DPI)
        plt.close()

    return list_pngs


def plot_maps_ts_from_path(path_nc, var_name, lon, lat, out_path, xaxis_min=0.0, xaxis_max=1.1, xaxis_step=0.1,
                           save_name='fig', xlabel='', start_movie_yr=-1, title='', do_jenks=True,
                           tme_name='time', land_bg=True, cmap=plt.cm.RdBu, grid=False):
    """
    Plot map for var_name variable from netCDF file
    :param path_nc: Name of netCDF file
    :param var_name: Name of variable in netCDF file to plot on map
    :param xaxis_min:
    :param xaxis_max:
    :param xaxis_step:
    :param lon: List of lon's
    :param lat: List of lat's
    :param out_path: Output directory path + file name
    :return: List of paths of images produced, side-effect: save an image(s)
    """
    logger.info('Plotting ' + var_name + ' in ' + path_nc)

    util.make_dir_if_missing(out_path)

    # Read netCDF file and get time dimension
    nc = util.open_or_die(path_nc)

    if start_movie_yr > 0:
        ts = nc.variables[tme_name][:].astype(int)  # time-series
        ts = ts[start_movie_yr - ts[0]:]
    else:
        ts = nc.variables[tme_name][:]  # time-series

    nc.close()

    return plot_maps_ts(path_nc, ts, lon, lat, out_path, var_name=var_name,
                        xaxis_min=xaxis_min, xaxis_max=xaxis_max, xaxis_step=xaxis_step,
                        save_name=save_name, xlabel=xlabel, do_jenks=do_jenks,
                        start_movie_yr=start_movie_yr, title=title, tme_name=tme_name, land_bg=land_bg, cmap=cmap,
                        grid=grid)


def plot_arr_to_map(path_arr, lon, lat, out_path, var_name='arr', xaxis_min=0.0, xaxis_max=1.1, xaxis_step=0.1,
                    plot_type='sequential', annotate_date=False, yr=0, date=-1, xlabel='', title='', tme_name='time',
                    any_time_data=True, format='%.2f', land_bg=True, cmap=plt.cm.RdBu, grid=False, fill_mask=False):
    """
    Plot var_name variable from netCDF file
    :param path_arr: array (2D)
    :param lon: List of lon's
    :param lat: List of lat's
    :param out_path: Output directory path + file name
    :param var_name: Name of variable in netCDF file to plot on map
    :param xaxis_min:
    :param xaxis_max:
    :param xaxis_step:
    :param plot_type:
    :param annotate_date:
    :param yr:
    :param date:
    :param xlabel:
    :param title:
    :param tme_name:
    :param any_time_data: Is there any time dimension?
    :param format:
    :param land_bg:
    :param cmap:
    :param grid:
    :param fill_mask:
    :return: Nothing, side-effect: save an image
    """
    logger.info('Plotting ' + xlabel)

    # Bail if xaxis_min == xaxis_max
    if xaxis_min == xaxis_max:
        return

    # Output diff to netCDF file
    out_nc_path = os.path.split(out_path)[0] + os.sep + 'tmp.nc'

    util.convert_arr_to_nc(path_arr, var_name, lat, lon, out_nc_path)

    # Convert netCDF file to map
    plot_map_from_nc(out_nc_path, out_path, var_name,
                     xaxis_min=xaxis_min, xaxis_max=xaxis_max, xaxis_step=xaxis_step,
                     annotate_date=annotate_date, yr=int(yr),
                     xlabel=xlabel,
                     date=date, title=title, tme_name=tme_name, show_plot=False, any_time_data=any_time_data,
                     format=format, land_bg=land_bg, cmap=cmap, grid=grid, fill_mask=fill_mask)

    # Remove temporary netcdf file
    os.remove(out_nc_path)


def plot_ascii_map(asc, out_path, xaxis_min=0.0, xaxis_max=1.1, xaxis_step=0.1, plot_type='sequential', map_label='',
                   append_name='', xlabel='', title='', var_name='data', skiprows=0, num_lats=constants.NUM_LATS,
                   num_lons=constants.NUM_LONS):
    """

    :param asc:
    :param out_path:
    :param xaxis_min:
    :param xaxis_max:
    :param xaxis_step:
    :param plot_type:
    :param map_label:
    :param append_name:
    :param xlabel:
    :param title:
    :param var_name:
    :param skiprows:
    :param num_lats:
    :param num_lons:
    :return:
    """
    logger.info('Plot ascii file as map')
    out_nc = util.convert_ascii_nc(asc, out_path + os.sep + 'file_' + append_name + '.nc', skiprows=skiprows,
                                   num_lats=num_lats, num_lons=num_lons, var_name=var_name, desc='netCDF')

    nc_file = util.open_or_die(out_nc)
    nc_file.close()

    path = os.path.dirname(out_path)
    map_path = path + os.sep + var_name + '_' + append_name + '.png'

    plot_map_from_nc(out_nc, map_path, var_name, xaxis_min, xaxis_max, xaxis_step, plot_type, annotate_date=True,
                     yr=map_label, date=-1, xlabel=xlabel, title=title, any_time_data=False, land_bg=False,
                     cmap=plt.cm.RdBu, grid=True, fill_mask=True)

    os.remove(out_nc)
    return map_path


def plot_LUstate_top_regions(df, xlabel='', ylabel='', title='', out_path='', fname='', vert_yr=[]):
    """

    :param df:
    :param vert_yr:
    :param xlabel:
    :param ylabel:
    :param title:
    :param out_path:
    :param fname:
    :return:
    """
    fig, ax = plt.subplots()

    xvar = df.index
    num_columns = len(df.columns.values)
    idx = 0

    for name_col, col in df.iteritems():
        pos = 'first' if idx == 0 else 'last' if idx == (num_columns - 1) else 'mid'

        plot_multiple_ts(ax, col.values, xvar, out_path + os.sep + fname, title=title, leg_name=name_col, xlabel=xlabel,
                         ylabel=ylabel, vert_yr=vert_yr, pos=pos)
        idx += 1

    plt.close(fig)


def plot_activity_matrix(df, cmap, normalized=False, annotate=True, out_path='', title=''):
    """
    Plot activity matrix showing area of land transitioning between land-use types
    :param df:
    :param cmap:
    :param normalized:
    :param annotate:
    :param out_path:
    :param title:
    :return:
    """
    logger.info('Plot activity matrix')
    sns.set(font_scale=0.8)

    formatter = tkr.ScalarFormatter(useMathText=True)
    # normalized scale is from 0 - 100, does not need scientific scale
    if not normalized:
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))

    df = df * 100.0 if normalized else df * 1.0
    vmin = math.ceil(np.nanmin(df))
    vmax = math.ceil(np.nanmax(df))  # maximum value on colorbar
    ax = sns.heatmap(df, cbar_kws={'format': formatter}, cmap=cmap,
                     linewidths=.5, linecolor='lightgray', annot=annotate, fmt='.2g', annot_kws={'size': 6}, vmin=vmin,
                     vmax=vmax)
    # for annotation of heat map cells, use: annot=True, fmt='g', annot_kws={'size': 6}
    # ax.invert_yaxis()
    ax.set_ylabel('FROM')
    ax.set_xlabel('TO')

    ax.set_title(title)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=0)
    locs, labels = plt.yticks()
    plt.setp(labels, rotation=0)

    plt.savefig(out_path, dpi=constants.DPI)
    plt.close()

    # revert matplotlib params
    sns.reset_orig()
    set_matplotlib_params()
    get_colors(palette='tableau')

if __name__ == '__main__':
    plot_glm()
