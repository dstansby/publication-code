# axisformat.py
#
#   Methods for automatically formatting axes
#
#   David Stansby 2015
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as pltdates
import matplotlib.ticker as pltticker
import matplotlib.transforms as mtransforms
import matplotlib.lines as mlines


def axline(ax, slope, intercept, **kwargs):
    if "transform" in kwargs:
        raise ValueError("'transform' is not allowed as a kwarg; "
                         "axline generates its own transform.")

    xtrans = mtransforms.BboxTransformTo(ax.viewLim)
    viewLimT = mtransforms.TransformedBbox(
        ax.viewLim,
        mtransforms.Affine2D().rotate_deg(90).scale(-1, 1))
    ytrans = (mtransforms.BboxTransformTo(viewLimT) +
              mtransforms.Affine2D().scale(slope).translate(0, intercept))
    trans = mtransforms.blended_transform_factory(xtrans, ytrans)

    l = mlines.Line2D([0, 1], [0, 1],
                      transform=trans + ax.transData,
                      **kwargs)
    ax.add_line(l)
    return l


# Alternates axes from L to R
# Takes a list of axes as input
def tidysubplots(axs, labels=None):
    naxs = len(axs)
    for i, ax in enumerate(axs):
        if labels is not None:
            ax.set_ylabel(labels[i])
        # Move y axis to right of figure
        if i % 2 == 1:
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
        # Turn of x axis
        if i != naxs - 1:
            ax.xaxis.set_visible(False)
        else:
            ax.tick_params(axis='x', pad=10)


# Removes x and y labels from an axis
def removeaxislabels(axis):
    axis.axes.get_xaxis().set_visible(False)
    axis.axes.get_yaxis().set_visible(False)


# Removes all items from axis
def clearaxis(axis):
    axis.patch.set_alpha(0.0)
    axis.set_frame_on(False)
    removeaxislabels(axis)


# Draws multiple horizontal lines from a list of y coordinates
def axhlines(ys, **kwargs):
    for y in ys:
        plt.axhline(y, **kwargs)


# Draws multiple vertical lines for a list of x coordiantes
def axvlines(xs, **kwargs):
    for x in xs:
        plt.axvline(x, **kwargs)


# Automatically set x limits
def xlims(axis, data):
    xmax = np.max(data)
    xmin = np.min(data)

    axis.set_xlim(right=xmax, left=xmin)


# Automatically stretch y limits to scale times given data limits
def ylims(axis, data, scale=1.05, verbose=0):
    data = np.float64(data)
    ymax = np.nanmax(data)
    if ymax > 0:
        ymax *= scale
    elif ymax < 0:
        ymax /= scale

    ymin = np.nanmin(data)
    if ymin > 0:
        ymin /= scale
    elif ymin < 0:
        ymin *= scale

    axis.set_ylim(top=ymax, bottom=ymin)
    if verbose:
        print('Top = ', ymax, ', Bottom = ', ymin)


def removemillisecs(axis, **datelocatorargs):
    locator = pltdates.AutoDateLocator(**datelocatorargs)
    formatter = pltdates.AutoDateFormatter(locator)
    formatter.scaled[1 / (24. * 60.)] = pltticker.FuncFormatter(sixtruncate)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)


# Remove pico seconds from a time axis
# datelocatorargs are passed to AutoDateLocator
def removepicosecs(axis, **datelocatorargs):
    locator = pltdates.AutoDateLocator(**datelocatorargs)
    formatter = pltdates.AutoDateFormatter(locator)
    formatter.scaled[1 / (24. * 60.)] = pltticker.FuncFormatter(threetruncate)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)


def threetruncate(x, pos=None):
    return truncate(x, pos, 3)


def sixtruncate(x, pos=None):
    return truncate(x, pos, 7)


def truncate(x, pos, n):
    x = pltdates.num2date(x)
    fmt = '%H:%M:%S.%f'
    label = x.strftime(fmt)
    label = label[:-n]
    return label


def gradient1_line(ax, data):
    '''
    Draw a gradient = 1 line going through the origin
    '''
    dmin = np.nanmin(data)
    dmax = np.nanmax(data)
    ax.plot([dmin, dmax], [dmin, dmax], color='k')
