# Copyright (c) 2011 Alun Morgan, Michael Abbott, Diamond Light Source Ltd.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#
# Contact:
#      Diamond Light Source Ltd,
#      Diamond House,
#      Chilton,
#      Didcot,
#      Oxfordshire,
#      OX11 0DE
#      alun.morgan@diamond.ac.uk, michael.abbott@diamond.ac.uk

'''Support for fitting a 2D Guassian to an image.'''
from pkg_resources import require
require("numpy")
require("fit_lib == 1.3")

import math
import numpy
import types
import time

from fit_lib import static
import levmar



# ------------------------------------------------------------------------------
# Ellipse coordinate conversion functions.

# The two functions here convert between two canonical representations of an
# ellipse centred at the origin:
#   1. Major axis, minor axis, angle of major axis
#   2. Coefficents of polynomial A x^2 + B y^2 + C x y
# It is easier and more stable to fit to a Gaussian in form (2), but we want the
# answer in form (1), hence these conversion functions.

def convert_abc(A, B, C):
    '''Converts ellipse parameters A, B, C for an ellipse of the form
           2      2
        A x  + B y  + C x y
    to major axis, minor axis squared and tilt angle.'''

    eps_skew = 1e-9
    eps_round= 1e-9
    if abs(C) < eps_skew * abs(A - B):
        # Skew is very small i.e. beam is horizontal or vertical
        t = 0.5 * C / (A - B)
    elif abs(C) < eps_round and abs(A - B) < eps_round:
        # Round horizontal beam.  In this case t becomes unconstrained, so force
        # it to zero.
        t = 0
    else:
        surd = math.sqrt((A - B)**2 + C**2)
        if A < B:
            # Take sign(surd) = sign(A - B) so that |t| <= 1, so
            #   -pi/4 <= theta <= pi/4
            surd = - surd
        t = (-A + B + surd) / C

    theta = math.atan(t)
    sigma_x = 1. / math.sqrt(2*A + C * t)
    sigma_y = 1. / math.sqrt(2*B - C * t)
    return sigma_x, sigma_y, -theta * 180. / math.pi


def convert_sigma_theta(sigma_x, sigma_y, theta):
    '''Converts from major, minor, theta form into A, B, C form of ellipse.'''
    ct = numpy.cos(-theta)
    st = numpy.sin(-theta)
    isx = 0.5 / sigma_x**2
    isy = 0.5 / sigma_y**2

    A = isx * ct**2 + isy * st**2
    B = isx * st**2 + isy * ct**2
    C = 2 * ct * st * (isx - isy)
    return A, B, C


# ------------------------------------------------------------------------------
# 1D Gaussian fit

# The 1D Gaussian model is parameterised by four parameters:
#
#   base, amplitude, x_0, A

def prefit_1D_Gaussian(data):
    assert data.ndim == 1
    min = float(data.min())
    max = float(data.max())
    x = numpy.arange(len(data))
    data_x = data / float(data.sum())
    mean = numpy.sum(x * data_x)
    var  = numpy.sum((x - mean) * data_x)
    return numpy.array([min, max - min, mean, 0.5 / var])

def Gaussian1dValid(params):
    _, _, _, A = params
    return A > 0

def Gaussian1d(params, x):
    g_0, K, x_0, A = params
    x = x - x_0
    return g_0 + K * numpy.exp(-(A * x**2))

def Gaussian1dJacobian(params, x):
    g_0, K, x_0, A = params
    x = x - x_0
    x2 = x * x
    E = numpy.exp(-(A * x2))
    KE = K * E
    return numpy.array([
        numpy.ones(len(E)),             # dG/dg_0
        E,                              # dG/dK
        2 * A * x * KE,                 # dG/dx_0
        - x2 * KE])                     # dG/dA

def Gaussian1dRescale(params, origin=0, scaling=1):
    g_0, K, x_0, A = params
    return (g_0, K, scaling * (x_0 - origin), A / (scaling*scaling))

def WindowGaussian1d(params, window):
    _, _, x_0, A = params
    w = window * math.sqrt(0.5 / A)
    return (x_0 - w, 2*w)

def fit1dGaussian(params, x, data):
    return levmar.fit(
        Gaussian1dValid, Gaussian1d, Gaussian1dJacobian, params, data, (x,))


class Fitter1dGaussian(static.Static):
    prefit = prefit_1D_Gaussian
    fit    = fit1dGaussian
    window = WindowGaussian1d


# ------------------------------------------------------------------------------
# 2D Gaussian fit

# The 2D Gaussian model is parameterised by six values:
#
#   g_0, K, x_0, y_0, A, B, C
#
# corresponding to the calculation
#
#   G(x,y) = g_0 + K * exp(-A*(x-x_0)**2 - B*(y-y_0)**2 - C*(x-x_0)*(y-y_0))


def prefit_2D_Gaussian(image):
    '''Computes initial estimates for 2D Gaussian fit to image.  Returns array
    of parameters in the order for fitting.'''

    assert image.ndim == 2 and (numpy.array(image.shape) > 1).all(), \
        'Can only fit to rectangular image'

    # This is done by projecting the image onto X and Y (by taking means) and
    # then computing statistics from these projections.  The results are the
    # combined into an initial estimate for the 2D fit.

    # Estimate vertical range
    min = float(image.min())
    max = float(image.max())
    # Project the image onto its axes, convert these into densities
    total = float(image.sum())
    image_x = image.sum(axis = 1) / total
    image_y = image.sum(axis = 0) / total
    # Compute x and y grids with given scale and origin
    x = numpy.arange(len(image_x))
    y = numpy.arange(len(image_y))

    # Compute statistics along each axis.
    # Note that these are only good if we have a complete enough curve!
    mean_x = numpy.sum(x * image_x)
    var_x  = numpy.sum((x - mean_x)**2 * image_x)
    mean_y = numpy.sum(y * image_y)
    var_y  = numpy.sum((y - mean_y)**2 * image_y)
    # Convert to initial Gaussian fit parameters
    # [baseline, amplitude, x0, y0, sigx, sigy, theta]
    return numpy.array([
        min, max - min, mean_x, mean_y, 0.5 / var_x, 0.5 / var_y, 0.0])


def WindowGaussian2d(params, window):
    '''Returns a sensible region in which to attempt the fit.  In this case
    we return +-window*sigma around the fitted origin.'''
    _, _, mean_x, mean_y, A, B, _ = params
    win_x = window * math.sqrt(0.5 / A)
    win_y = window * math.sqrt(0.5 / B)
    return ((mean_x - win_x, mean_y - win_y), (2*win_x, 2*win_y))


def Gaussian2dValid(params):
    A, B, C = params[-3:]
    return A > 0 and B > 0 and 4 * A * B > C * C

def Gaussian2d(params, xy):
    '''Test function used to compute modelled Gaussian on the given x,y
    vectors.'''
    g_0, K, x_0, y_0, A, B, C = params
    x, y = xy
    x = x - x_0
    y = y - y_0
    exp_part = numpy.exp(-(A * x**2 + B * y**2 + C * x*y))
    mult = numpy.zeros(len(exp_part))
    for i in range(len(exp_part)):
        if abs(exp_part[i]) < 1e-200:
            mult[i] = 0.0
        else:
            mult[i] = K * exp_part[i]
    return g_0 + mult

def Gaussian2dJacobian(params, xy):
    g_0, K, x_0, y_0, A, B, C = params
    x, y = xy
    x = x - x_0
    y = y - y_0
    x2 = x * x
    y2 = y * y
    xy = x * y
    E = numpy.exp(-(A * x2 + B * y2 + C * xy))
    KE = numpy.zeros(len(E))
    for i in range(len(E)):
        if abs(E[i]) < 1e-200:
            KE[i] = 0.0
        else:
            KE[i] = K * E[i]
    return numpy.array([
        numpy.ones(len(E)),             # dG/dg_0
        E,                              # dG/dK
        (2 * A * x + C * y) * KE,       # dG/dx_0
        (2 * B * y + C * x) * KE,       # dG/dy_0
        - x2 * KE,                      # dG/dA
        - y2 * KE,                      # dG/dB
        - xy * KE])                     # dG/dC


def Gaussian2d_0(params, xy, g_0):
    '''Modified Guassian calculation with zero baseline.'''
    return Gaussian2d(numpy.concatenate(([g_0], params)), xy)

def Gaussian2dJacobian_0(params, xy, g_0):
    '''Modified Jacobian calculation with zero baseline.'''
    return Gaussian2dJacobian(numpy.concatenate(([g_0], params)), xy)[1:]


def fit2dGaussian(params, xy, data, **kargs):
    '''Given a good initial estimate and flattenned and thinned data returns the
    best 2D Gaussian fit to the dataset.'''
    return levmar.fit(
        Gaussian2dValid, Gaussian2d, Gaussian2dJacobian,
        params, data, (xy,), **kargs)

def fit2dGaussian_0(params, xy, data, **kargs):
    '''A modification of fit2dGaussian which forces the baseline to a constant
    value.  Still takes and returns the same parameter set, but allows no
    variation of g_0.'''
    g_0 = params[0]
    result, chi2 = levmar.fit(
        Gaussian2dValid, Gaussian2d_0, Gaussian2dJacobian_0,
        params[1:], data, (xy, g_0), **kargs)
    return numpy.concatenate(([g_0], result)), chi2


def Gaussian2dRescale(params, origin=0, scaling=1):
    '''Rescales the coordinates of the fit to the specified origin and scaling
    so that the new coordinate and old coordinates are related by the equation

        new_coord = scaling * (old_coord - origin)
    '''
    g_0, K, x_0, y_0, A, B, C = params
    O_x, O_y = normalise_sequence(origin, 2)
    s_x, s_y = normalise_sequence(scaling, 2)
    return (
        g_0, K, s_x * (x_0 - O_x), s_y * (y_0 - O_y),
        A / (s_x*s_x), B / (s_y*s_y), C / (s_x*s_y))


# Gather the key elements of these fitters.

class Fitter2dGaussian(static.Static):
    prefit = prefit_2D_Gaussian
    fit = fit2dGaussian
    window = WindowGaussian2d

class Fitter2dGaussian_0(static.Static):
    prefit = prefit_2D_Gaussian
    fit    = fit2dGaussian_0
    window = WindowGaussian2d

# ------------------------------------------------------------------------------
# 2D donut Gaussian fit

# The 2D Gaussian model is parameterised by six values:
#
#   g_0, K, x_0, y_0, A, r
#
#   g_0 is the baseline
#   K is the amplitude
#   x0 and y0 are the offsets from the origin
#   A is the siga scaling
#   r is radius of the ring

# corresponding to the calculation
#
#   G(x,y) = g_0 + K * exp(-A*(r-(sqrt((x-x_0) + (y-y_0))))**2)

# TEMP ..... NEEDS TO BE ADAPTED TO THE NEW FUNCTION!!
def prefit_2D_donut(image):
    '''Computes initial estimates for 2D Gaussian fit to image.  Returns array
    of parameters in the order for fitting.'''

    assert image.ndim == 2 and (numpy.array(image.shape) > 1).all(), \
        'Can only fit to rectangular image'

    # This is done by projecting the image onto X and Y (by taking means) and
    # then computing statistics from these projections.  The results are the
    # combined into an initial estimate for the 2D fit.

    # Estimate vertical range
    min = float(image.min())
    max = float(image.max())
    # Project the image onto its axes, convert these into densities
    total = float(image.sum())
    image_x = image.sum(axis = 1) / total
    image_y = image.sum(axis = 0) / total
    # Compute x and y grids with given scale and origin
    x = numpy.arange(len(image_x))
    y = numpy.arange(len(image_y))

    # Compute statistics along each axis.
    # Note that these are only good if we have a complete enough curve!
    mean_x = numpy.sum(x * image_x)
    var_x  = numpy.sum((x - mean_x)**2 * image_x)
    mean_y = numpy.sum(y * image_y)
    var_y  = numpy.sum((y - mean_y)**2 * image_y)
    # Convert to initial Gaussian fit parameters
    # [baseline, amplitude, x0, y0, sigx, sigy, theta]
    return numpy.array([
        min, max - min, mean_x, mean_y, 0.5 / var_x, 0.5 / var_y, 0.0])

# TEMP ..... NEEDS TO BE ADAPTED TO THE NEW FUNCTION!!
def Windowdonut2d(params, window):
    '''Returns a sensible region in which to attempt the fit.  In this case
    we return +-window*sigma around the fitted origin.'''
    _, _, mean_x, mean_y, A, B, _ = params
    win_x = window * math.sqrt(0.5 / A)
    win_y = window * math.sqrt(0.5 / B)
    return ((mean_x - win_x, mean_y - win_y), (2*win_x, 2*win_y))

# TEMP ..... NEEDS TO BE ADAPTED TO THE NEW FUNCTION!!
def donut2dValid(params):
    A, B, C = params[-3:]
    return A > 0 and B > 0 and 4 * A * B > C * C

def donut2d(params, xy):
    '''Test function used to compute modelled Gaussian on the given x,y
    vectors.'''
    g_0, K, x_0, y_0, A, r = params
    x, y = xy
    x = x - x_0
    y = y - y_0
    return g_0 + K * numpy.exp(-(A * (r - ((x**2 + y**2)**0.5))**2))

# TEMP ..... NEEDS TO BE ADAPTED TO THE NEW FUNCTION!!
def fit2ddonut(params, xy, data, **kargs):
    '''Given a good initial estimate and flattenned and thinned data returns the
    best 2D Gaussian fit to the dataset.'''
    return levmar.fit(
        Gaussian2dValid, Gaussian2d, Gaussian2dJacobian,
        params, data, (xy,), **kargs)

# TEMP ..... NEEDS TO BE ADAPTED TO THE NEW FUNCTION!!
def donut2dRescale(params, origin=0, scaling=1):
    '''Rescales the coordinates of the fit to the specified origin and scaling
    so that the new coordinate and old coordinates are related by the equation

        new_coord = scaling * (old_coord - origin)
    '''
    g_0, K, x_0, y_0, A, B, C = params
    O_x, O_y = normalise_sequence(origin, 2)
    s_x, s_y = normalise_sequence(scaling, 2)
    return (
        g_0, K, s_x * (x_0 - O_x), s_y * (y_0 - O_y),
        A / (s_x*s_x), B / (s_y*s_y), C / (s_x*s_y))


# Gather the key elements of these fitters.

# TEMP ..... NEEDS TO BE ADAPTED TO THE NEW FUNCTION!!
class Fitter2ddonut(static.Static):
    prefit = prefit_2D_Gaussian
    fit    = fit2dGaussian
    window = WindowGaussian2d

# ------------------------------------------------------------------------------

# Windowing functionality.

def create_grid(shape):
    '''Given a shape tuple (N_1, ..., N_M) returns a coordinate grid of shape
        (M, N_1, ..., N_M)
    with g[m, n_1, ..., n_M] = n_(m+1).  This can be used as an index into an
    array of the given shape by converting the grid to a tuple, and indeed
        a[tuple(grid(a.shape))] = a
    in general.'''
    # Some rather obscure numpy index trickery.  We could write the expression
    # below more simply as
    #   numpy.mgrid[:shape[0], :shape[1]]
    # except that the form below will work for any length of shape.  The mgrid
    # construction takes an index expression and turns it into a grid which
    # cycles over all the points in the index.
    return numpy.mgrid[tuple(numpy.s_[:n] for n in shape)]


def normalise_sequence(input, rank):
    '''Lifted from numpy.  Converts input to a list of length rank, replicating
    as necessary if it's not already a sequence.'''
    if (isinstance(input, (types.IntType, types.LongType, types.FloatType))):
        return [input] * rank
    else:
        return input


def thin_uniformly(data, factor):
    '''Reduces data array by factor, which can be a single number, or an array
    of data.ndim points.'''
    factor = normalise_sequence(factor, data.ndim)
    return data[tuple(numpy.s_[::f] for f in factor)]


def flatten_grid(grid):
    '''Given an M+1 dimensional grid of shape (M, N_1, ..., N_M) converts it
    into a two dimensional grid of shape (M, N_1*...*N_M).'''
    M = grid.shape[0]
    assert grid.ndim == M + 1, 'Malformed grid'
    return grid.reshape((M, grid.size // M))


def thin_ordered(factor, grid, data):
    '''Thins the data in order of intensity.'''
    thinning = numpy.argsort(data)[::factor]
    return (grid[:, thinning], data[thinning])

def thin_ordered_by(factor):
    return lambda grid, data: thin_ordered(factor, grid, data)


def apply_ROI(data, origin, extent):
    '''Returns data[origin:origin+extent].'''
    low = numpy.array(normalise_sequence(origin, data.ndim))
    high = low + normalise_sequence(extent, data.ndim)
    low[low < 0] = 0
    assert numpy.all(high > 0), 'An axis of the window is outside the dataset'
    return data[tuple(numpy.s_[l:h] for l, h in zip(low, high))], low


def gamma_correct(data, gamma, max_data):
    '''Gamma correction.'''
    return data * numpy.exp(gamma * (data / float(max_data) - 1))



# ------------------------------------------------------------------------------

class FitResults:
    '''Simple structure used to receive fitting results.'''

def doFit(fitter, data,
          ROI=None, window_size=None,
          thinning=None, data_thinning=None,
          gamma=None,
          extra_data=False, **kargs):
    '''General fitter.  Returns the best fit together with the mean chi2 over
    the filtered data set.  If extra_data is True then a third structure
    containing intermediate fitting results is also returned.

    Takes the following arguments:

    fitter
        This object should have three attributes, .prefit, .fit and .window:

            initial = .prefit(data)
                Returns an initial estimate of the fit to data.  The result is
                an array of values suitable for passing to the .fit and .window
                routines.

            origin, extent = .window(initial, size)
                Returns a "region of interest" computed from the initial
                parameters scaled by size, normally in standard deviations.  The
                result will be truncated to integer pixel counts.

            fit, error = .fit(initial, grid, data, **kargs)
                The detailed fitting operation is performed starting from the
                computed initial parameters.  The data will have been thinned
                and windowed by this point, and the grid defines the coordinates
                of each data point with respect to the original data set, in
                pixels.  Extra arguments are passed through to the underlying
                fit algorithm.
                    data is a vector with N points, grid is a MxN matrix where M
                is the dimensionality (.ndim) of the original data set.

    data
        This is the initial data to be fitted, and should have the correct
        number of dimensions expected by the fitter.

    ROI
        If specified this is a "Region Of Interest", consisting of a pair
        (ROI_origin, ROI_extent) where ROI_origin is the index into data of the
        first point of the region of interest and ROI_extent is the size of the
        region of interest.

    window_size
        If specified this is a window size to be passed to fitter.window() to
        automatically compute a window on the selected data.

    thinning
        After windowing the data will be thinned by an integer factor.  This can
        be a single integer, or a thinning factor for each dimension of the
        data.

    data_thinning
        Data dependent thinning can also be applied to the data.  This is done
        last, just before gamma correction, so needs to work on the data
        indexing grid.  If specified this must be a function of the form

            grid, data = data_thinning(grid, data)

        where grid and data are as described above for fitter.fit()

    gamma
        If gamma correction on the data is required this should be a pair
        (factor, max_data) where max_data is the maximum normal data value, eg
        255 for 8-bit data, and factor is the required gamma correction.

    extra_data
        If this flag is set then intermediate computations will be assigned to
        fields of an extra results structure as follows:

            .grid       Final chosen grid used for fitting
            .data       Final data set used for fitting
            .origin     Offset of selected window into original data
            .extent     Dimensions of window into original data
    '''

    # Apply Region Of Interest if specified.
    if ROI:
        origin, extent = map(numpy.array, ROI)
        data, origin = apply_ROI(data, origin, extent)
    else:
        origin = numpy.zeros(data.ndim)

    # Create a sensible initial fit.
    initial = fitter.prefit(data)
    # Window the data if required.
    if window_size is not None:
        window_origin, extent = \
            map(numpy.int_, fitter.window(initial, window_size))
        data, window_origin = apply_ROI(data, window_origin, extent)
        origin += window_origin
    extent = data.shape
    if thinning is None:
        thinning = numpy.ones(data.ndim)
    else:
        # Thin the data uniformly in all dimensions.
        thinning = numpy.array(normalise_sequence(thinning, data.ndim))
        data = thin_uniformly(data, thinning)

    # Compute the coordinate grid with the correct coordinates taking thinning
    # and window offsets into account and flatten both the grid and the data to
    # a single dimension in preparation for fitting.
    grid = flatten_grid(create_grid(data.shape))
    grid = thinning[:, None] * grid + origin[:, None]
    data = data.flatten()

    # Do data dependent thinning if requested.
    if data_thinning:
        grid, data = data_thinning(grid, data)

    # Finally perform gamma correction on the data before performing the fit.
    if gamma:
        data = gamma_correct(data, *gamma)

    # Perform the fit on the reduced data set and return the result.
    fit, chi2 = fitter.fit(initial, grid, data, **kargs)
    # If intermediate results requested make them available
    if extra_data:
        results = FitResults()
        results.grid = grid
        results.data = data
        results.origin = origin
        results.extent = extent

        return fit, chi2 / len(data), results
    else:
        return fit, chi2 / len(data)


def MakeDoFit(fitter):
    return lambda image, **kargs: doFit(fitter, image, **kargs)


doFit2dGaussian = MakeDoFit(Fitter2dGaussian)
doFit2dGaussian_0 = MakeDoFit(Fitter2dGaussian_0)

doFit1dGaussian = MakeDoFit(Fitter1dGaussian)
