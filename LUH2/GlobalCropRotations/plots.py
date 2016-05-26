import matplotlib, constants
import palettable
from matplotlib import rcParams
from matplotlib import pyplot as plt

def set_matplotlib_params():
    """
    Set matplotlib defaults to nicer values
    """
    # rcParams dict
    rcParams['mathtext.default'] = 'regular'
    rcParams['axes.labelsize'] = 12
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 12
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.serif'] = ['Times New Roman']
    # rcParams['figure.figsize'] = 7.3, 4.2

def get_colors(palette='colorbrewer', cmap=False):
    """
    Get palettable colors, which are nicer
    """
    if palette == 'colorbrewer':
        bmap = palettable.colorbrewer.diverging.PRGn_11.mpl_colors
        if cmap:
            bmap = palettable.colorbrewer.diverging.PRGn_11.mpl_colormap
    elif palette == 'tableau':
        bmap = palettable.tableau.Tableau_20.mpl_colors
        if cmap:
            bmap = palettable.tableau.Tableau_20.mpl_colormap
    elif palette == 'cubehelix':
        bmap = palettable.cubehelix.cubehelix2_16.mpl_colors
        if cmap:
            bmap = palettable.cubehelix.cubehelix2_16.mpl_colormap

    return bmap


def simple_axis(ax):
    """
    Remove spines from top and right, set max value of y-axis
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def simple_legend():
    leg = plt.legend(loc='upper left', fancybox=None)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_alpha(0.5)


