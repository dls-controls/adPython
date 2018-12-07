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

'''Gauss-Newton / Levenberg Marquardt nonlinear least squares'''

from numpy import arange, inner, zeros, dot
from numpy.linalg import solve


def gauss(A, b):
    '''
    Gaussian elimination with no pivoting.
    % input: A is an n x n nonsingular matrix
    %        b is an n x 1 vector
    % output: x is the solution of Ax=b.
    % post-condition: A and b have been modified.
    '''
    n = len(A)
    if b.size != n:
        raise ValueError("Invalid argument: incompatible sizes between A & b.", b.size, n)
    for pivot_row in xrange(n-1):
        for row in xrange(pivot_row+1, n):
            if A[pivot_row][pivot_row] == 0:
                multiplier = 0
            else:
                multiplier = A[row][pivot_row]/A[pivot_row][pivot_row]
            # the only one in this column since the rest are zero
            A[row][pivot_row] = multiplier
            for col in xrange(pivot_row + 1, n):
                if A[pivot_row][col] < 1e-200:
                    aSub = 0.0
                else:
                    aSub = multiplier*A[pivot_row][col]
                A[row][col] = A[row][col] - aSub
                if A[row][col] < 1e-200:
                    A[row][col] = 0.0
            # Equation solution column
            if b[pivot_row] < 1e-200:
                bSub = 0.0
            else:
                bSub = multiplier * b[pivot_row]
            b[row] = b[row] - bSub
            if b[row] < 1e-200:
                b[row] = 0.0
    x = zeros(n)
    k = n-1
    # x[k] = b[k]/A[k, k]
    while k >= 0:
        if A[k, k] == 0 and b[k] == 0:
            x[k] = 0
        else:
            x[k] = (b[k] - dot(A[k, k+1:], x[k+1:]))/A[k, k]
        k = k-1
    return x


def diag_ix(N):
    '''Returns a value suitable for indexing the diagonal of a square NxN
    matrix.'''
    return arange(N), arange(N)


def levmar_core(fdf, a, lam=1e-3, maxiter=10):
    '''Minimal implementation of Levenberg-Marquardt fitting with fixed
    lambda and without termination testing.  Note that this implementation
    updates a in place, which may not be desireable, and computes a fixed number
    of iterations.  Use the implementation below for proper work.'''
    d = diag_ix(len(a))             # Diagonal index for updating alpha

    for s in range(maxiter):
        e, de = fdf(a)              # Compute e and its derivatives
        beta = inner(de, e)         # Compute basic gradient vector
        alpha = inner(de, de)       # Approximate Hessian from derivatives
        alpha[d] *= 1 + lam         # Scale Hessian diagonal by lambda
        a -= solve(alpha, beta)     # Step to next position
    return a


# Possible termination reasons for exit from levmar() routine.  Lowest number is
# best.
LEVMAR_STATUS_CTOL = 0      # Absolute chi squared tolerance satisfied
LEVMAR_STATUS_FTOL = 1      # Relative chi squared tolerance satisfied
LEVMAR_STATUS_ITER = 2      # Max iterations exhausted
LEVMAR_STATUS_LAMBDA = 3    # Limit on lambda reached, convergence failed.

def levmar(f, df, a,
        ftol=1e-6, ctol=1e-6, lam=1e-3, maxiter=20, max_lambda=1e8):
    '''Levenberg-Marquardt non-linear optimisation.  Takes three mandatory
    parameters and a handful of control parameters.

    f   A function taking an N dimensional vector and returning an M dimensional
        array representing the error function.  The optimisation process here
        will adjust the parameter vector to minimise the sum of squares of the
        values, chi2 = sum(f(a)).  Note that both the input and output arrays
        have a one dimensional shape.

    df  A function taking an N dimensional vector and returning an MxN
        dimensional array containing the partial derivatives of f with respect
        to its input parameters.  To be precise,

            df(a)[k,i] = derivative of f(a) with respect to parameter i in a.

    a   The initial starting point for optimisation.  This must be an N
        dimensional vector.

    The result of calling levmar(f, df, a) is a 3-tuple containing the optimal
    value for a, the corresponding chi2 value, the number of iterations (less
    one), and a status code thus:

        new_a, chi2, iter, status = levmar(f, df, a)

    The following status codes can be returned:

        LEVMAR_STATUS_CTOL = 0    Absolute chi squared tolerance satisfied
        LEVMAR_STATUS_FTOL = 1    Relative chi squared tolerance satisfied
        LEVMAR_STATUS_ITER = 2    Max iterations exhausted
        LEVMAR_STATUS_LAMBDA = 3  Limit on lambda reached, convergence failed.

    The other parameters control termination:

    ftol=1e-6
        Fractional tolerance on chi2.  Searching will terminate when the
        fractional reduction in chi2 is less than ftol.

    ctol=0
        Absolute tolerance on chi2.  Searching will terminate when chi2 is less
        than this value.  The default value for ctol has no effect.

    maxiter=20
        Maximum number of outer loops (evaluations of df()).

    lam=1e-3
    max_lambda=1e6
        Initial linearisation scaling factor and ceiling on this value.
    '''

    d = diag_ix(len(a))             # Diagonal index for updating alpha

    # Initial function and chi2.
    e = f(a)
    assert e is not None, 'Bad initial parameters'
    chi2 = (e**2).sum()

    for iter in xrange(maxiter):
        # From Jacobian matric compute alpha0, an estimate of the Hessian, and
        # beta, (half) the gradient vector.
        de = df(a)
        beta = inner(de, e)
        alpha0 = inner(de, de)

        # Now seek a lambda which actually improves the value of chi2
        while lam < max_lambda:
            # Compute alpha = alpha0 * (1 + diag(lam)) and use this to compute
            # the next value for a.
            alpha = +alpha0
            alpha[d] *= 1 + lam
            a_new = a - gauss(alpha, beta)
            # Assess the new position.
            e = f(a_new)
            if e is None:
                # Oops.  Outside the boundary.  Increasing lam should eventually
                # bring us closer to a.
                lam *= 10.
            else:
                chi2_new = (e**2).sum()
                if chi2_new > chi2:
                    # Worse.  Try again closer to a and with a more linear fit.
                    lam *= 10.
                else:
                    # Good.  We have an improvement.
                    break
        else:
            # max_lambda reached.  Give up now.
            return a, chi2, iter, LEVMAR_STATUS_LAMBDA

        a = a_new
        lam *= 0.1

        if chi2_new < ctol:
            return a, chi2_new, iter, LEVMAR_STATUS_CTOL
        elif chi2 - chi2_new < ftol * chi2_new:
            # Looks like this is good enough.  Either chi2 is small enough or
            # the fractional improvement is so small that we're good enough.
            return a, chi2_new, iter, LEVMAR_STATUS_FTOL
        else:
            chi2 = chi2_new

    # Iterations exhausted.

    return a, chi2, iter, LEVMAR_STATUS_ITER


class FitError(Exception):
    '''Exception raised in response to fitting failure.'''


def fit(valid, function, derivative, initial, target, args, **kargs):
    '''Wrapper for the fitting process.  Takes the following arguments and
    returns the best fit of the parameters to the data and the resulting chi2.

        valid(params)
            Validates parameters array, returns true iff function can safely be
            called on the given parameter set.

        function(params, args)
            Computes function at parameters and fixed args, returns an array of
            numbers which will be compared with the target value data.

        derivative(params, args)
            Computes the derivative of function with respect to each parameter,
            returns a two dimensional array with the first dimension ranging
            over the parameters.

        initial
            Initial starting point for the parameters used above.

        target
            Target array.  The parameter vector will be adjusted to minimise the
            squared difference between function(params,...) and target.

        args
            Argument passed directly to function() and derivative().

        **kargs
            Keyword arguments passed through to levmar(), used to control
            convergence and iterations.
    '''

    def error(params):
        if valid is None or valid(params):
            val = function(params, *args)
            return val - target
        else:
            return None

    def Jacobian(params):
        return derivative(params, *args)

    result, chi2, _, status = levmar(error, Jacobian, initial, **kargs)
    if status == LEVMAR_STATUS_ITER:
        raise FitError('Iterations exhausted without an adequate fit')
    elif status == LEVMAR_STATUS_LAMBDA:
        raise FitError('Lambda runaway, fit failed')
    return result, chi2