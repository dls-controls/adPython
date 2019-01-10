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
from random import random
from numpy import arange, inner, zeros, dot, absolute, multiply

from numpy.linalg import solve


def gauss(a, b):
    n = a.shape[0]
    A = zeros((n, n+1))
    A[:, :n] = a
    A[:, n] = b
    for i in range(0, n):
        # Search for maximum in this column
        maxEl = abs(A[i, i])
        maxRow = i
        for k in range(i+1, n):
            if abs(A[k, i]) > maxEl:
                maxEl = abs(A[k, i])
                maxRow = k
        # Swap maximum row with current row (column by column)
        for k in range(i, n+1):
            tmp = A[maxRow, k]
            A[maxRow, k] = A[i, k]
            A[i, k] = tmp
        # Make all rows below this one 0 in current column
        for k in range(i+1, n):
            if A[i, i] == 0.0:
                return None
            c = -A[k, i]/A[i, i]
            if abs(c) < 1e-150:
                c = 0
            for j in range(i, n+1):
                if i == j:
                    A[k, j] = 0
                else:
                    A[k, j] += c * A[i, j]
    # Solve equation Ax=b for an upper triangular matrix A
    x = zeros(n)
    for i in range(n-1, -1, -1):
        if abs(A[i, n]) > 1e-150 and abs(A[i, i] > 1e-150):
            x[i] = A[i, n]/A[i, i]
        elif abs(A[i, n]) < 1e-150 and abs(A[i, i] > 1e-150):
            x[i] = 0
        else:
            return None
        for k in range(i-1, -1, -1):
            A[k, n] -= A[k, i] * x[i]
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
    d = diag_ix(len(a))  # Diagonal index for updating alpha

    for s in range(maxiter):
        e, de = fdf(a)  # Compute e and its derivatives
        beta = inner(de, e)  # Compute basic gradient vector
        alpha = inner(de, de)  # Approximate Hessian from derivatives
        alpha[d] *= 1 + lam  # Scale Hessian diagonal by lambda
        a -= solve(alpha, beta)  # Step to next position
    return a


# Possible termination reasons for exit from levmar() routine.  Lowest number is
# best.
LEVMAR_STATUS_CTOL = 0  # Absolute chi squared tolerance satisfied
LEVMAR_STATUS_FTOL = 1  # Relative chi squared tolerance satisfied
LEVMAR_STATUS_ITER = 2  # Max iterations exhausted
LEVMAR_STATUS_LAMBDA = 3  # Limit on lambda reached, convergence failed.


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
    d = diag_ix(len(a))  # Diagonal index for updating alpha

    # Initial function and chi2.
    e = f(a)
    assert e is not None, 'Bad initial parameters'
    chi2 = (e ** 2).sum()

    for iter in xrange(maxiter):
        # From Jacobian matrix compute alpha0, an estimate of the Hessian, and
        # beta, (half) the gradient vector.
        de = df(a)
        #print("Jacobian")
        beta = inner(de, e)
        beta = multiply(absolute(beta) > 1e-200, beta)
        alpha0 = inner(de, de)
        alpha0 = multiply(absolute(alpha0) > 1e-200, alpha0)


        # Now seek a lambda which actually improves the value of chi2
        print("lambda is %s" % lam)
        while lam < max_lambda:
            # Compute alpha = alpha0 * (1 + diag(lam)) and use this to compute
            # the next value for a.
            #print("Solve")
            # Assess the new position.
            x = None
            while x is None:
                alpha = +alpha0
                alpha[d] *= 1 + lam
                x = gauss(alpha, beta)  # solve(alpha, beta)
                if x is None:
                    lam *= 10 # (1 + 0.001 * (1 - 0.5 * random()))
            a_new = a - x
            a_far = a - 100*x
            e = f(a_new)
            e_far = f(a_far)

            #print("Function")
            if e is None:
                print("oops")
                # Oops.  Outside the boundary.  Increasing lam should eventually
                # bring us closer to a.
                lam *= 10.
            else:
                chi2_new = (e ** 2).sum()
                chi2_far = (e_far ** 2).sum()
                if chi2_new > chi2 and chi2_far > chi2:
                    print("bad! %s" % chi2_new)
                    # Worse.  Try again closer to a and with a more linear fit.
                    lam *= 10.
                elif chi2_new < chi2_far:
                    # Good.  We have an improvement.
                    print("better... %s" % (chi2_new - chi2))
                    break
                else:
                    print("bigger is better!... %s" % (chi2_far - chi2))
                    break
        else:
            # max_lambda reached.  Give up now.
            return a, chi2, iter, LEVMAR_STATUS_LAMBDA

        a = a_new
        lam *= 0.1

        if chi2_new < ctol:
            print("good solution!")
            return a, chi2_new, iter, LEVMAR_STATUS_CTOL
        elif chi2 - chi2_new < ftol * chi2_new:
            # Looks like this is good enough.  Either chi2 is small enough or
            # the fractional improvement is so small that we're good enough.
            print("good solution!")
            return a, chi2_new, iter, LEVMAR_STATUS_FTOL
        else:
            chi2 = chi2_new

    # Iterations exhausted.
    print("its something...")
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
