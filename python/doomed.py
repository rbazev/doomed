#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions for numerical evaluations and simulations in the paper:
Mutational meltdown in asexual populations doomed to extinction."""

###############################################################################
# Created By  : Ricardo Azevedo, Logan Chipkin
# Last Updated: Thu Apr 30 10:47:11 2020
###############################################################################


###############################################################################
# Imports
###############################################################################


from numba import jit
from numba import njit
from numba.typed import List
import numpy as np
import numpy.random as rnd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns


###############################################################################
# Plotting settings
###############################################################################


sns.set_style("ticks")
sns.set_context('talk')


# use LaTeX for typesetting, Helvetica font
# rcParams['text.usetex'] = True
# rcParams['text.latex.preamble'] = r'\usepackage{helvet}'
# rcParams['text.latex.preamble'] = r'\usepackage[EULERGREEK]{sansmath}'
# rcParams['text.latex.preamble'] = r'\sansmath'


def set_up_axes(ax, xmin, xmax, xstep, ymin, ymax, ystep, rnd, xlabel='', ylabel='', part_label=''):
    '''
    Format plot.  Use for Figures 2-4.

    Breaks for log-transformed axes.

    Parameters
    ----------
    ax : axis
        Axis object
    xmin : float
        Minimum X axis value
    xmax : float
        Maximum X axis value
    xstep : float
        Interval between ticks in the X axis
    ymin : float
        Minimum Y axis value
    ymax : float
        Maximum Y axis value
    ystep : float
        Interval between ticks in the Y axis
    rnd : int
        Rounding
    xlabel : str, optional
        X axis label
    ylabel : str, optional
        Y axis label
    part_label : str, optional
        Plot label
    '''
    xtx = np.arange(xmin, xmax + xstep / 2, xstep).round(rnd)
    ytx = np.arange(ymin, ymax + ystep / 2, ystep).round(rnd)
    if rnd == 0:
        xtx = np.array(xtx, dtype=int)
        ytx = np.array(ytx, dtype=int)
    xrg = xtx.max() - xtx.min()
    ax.set_xlim(xtx.min(), xtx.max())
    ax.set_xticks(xtx)
    ax.set_xticklabels(xtx)
    yrg = ytx.max() - ytx.min()
    ax.set_ylim(ytx.min(), ytx.max())
    ax.set_yticks(ytx)
    ax.set_yticklabels(ytx)
    ax.text(xtx.min() - .016 * xrg, ytx.max() + .15 * yrg,
            part_label, size=24, ha='center', va='center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine(offset=10)
    sns.set_context("talk")


def set_up_axes2(ax, xmin, xmax, xstep, ymin, ymax, ystep, rnd, xlabel='', ylabel='', part_label=''):
    '''
    Modified version of the previous function for Figure 5.
    '''
    xtx = np.arange(xmin, xmax + xstep / 2, xstep).round(rnd)
    xrg = xtx.max() - xtx.min()
    ax.set_xlim(xtx.min(), xtx.max())
    ax.set_xticks(xtx)
    ax.set_xticklabels(xtx)
    ytx = np.arange(ymin, ymax + ystep / 2, ystep).round(rnd)
    yrg = ytx.max() - ytx.min()
    ax.set_ylim(ytx.min(), ytx.max())
    ax.set_yticks(ytx)
    ax.set_yticklabels(ytx)
    ax.text(xtx.min() - .016 * xrg, ytx.max() + .2 * yrg,
            part_label, size=24, ha='center', va='center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


###############################################################################
# Theory functions
###############################################################################


@njit
def w(k, s):
    '''Fitness of a genotype with k deleterious mutations of effect s.

    Parameters
    ----------
    k : int
        Number of mutations
    s : float
        Deleterious effect of a mutation

    Returns
    -------
    float
        Fitness.
    '''
    return (1 - s) ** k


@njit
def mt(t, k, j, s, u):
    '''Expected number of descendants of type k+j from an individual of type k
    after t generations.

    Parameters
    ----------
    t : int
        Number of generations
    k : int
        Number of mutations in parent
    j : int
        Additional number of mutations in offspring
    s : float
        Deleterious effect of a mutation
    u : float
        Deleterious mutation rate

    Returns
    -------
    float
        Number of individuals
    '''
    x = 0
    if 0 <= j <= t:
        x = (u ** j) * (1 - u) ** (t - j) * w(k * t + j * (j - 1) / 2, s)
        for i in range(1, j + 1):
            x *= (1 - w(t + 1 - i, s)) / (1 - w(i, s))
    return x


@njit
def Eztk(Z0, t, k, s, u):
    '''Expected number of individuals with k mutations at time t.

    Parameters
    ----------
    Z0 : typed.List
        Initial number of individuals with 0, 1, ... mutations.
    t : int
        Time (generations)
    k : int
        Number of mutations.
    s : float
        Deleterious effect of a mutation
    u : float
        Deleterious mutation rate

    Returns
    -------
    float
        Population size
    '''
    msum = 0
    l = len(Z0)
    for i in range(l):
        if k >= i:
            msum += Z0[i] * mt(t, i, k-i, s, u)
    return msum


@njit
def EZt(Z0, t, s, u):
    '''Expected composition of the population at time t.

    Parameters
    ----------
    Z0 : typed.List
        Initial number of individuals with 0, 1, ... mutations.
    t : int
        Time (generations)
    s : float
        Deleterious effect of a mutation
    u : float
        Deleterious mutation rate

    Returns
    -------
    float
        Population size
    '''
    for i in range(t):
        Z0.append(0)
    l = len(Z0)
    return [Eztk(Z0, t, k, s, u) for k in range(l)]


@njit
def ENt(Z0, t, s, u):
    '''Expected total population size at time t. 

    Parameters
    ----------
    Z0 : typed.List
        Initial number of individuals with 0, 1, ... mutations.
    t : int
        Time (generations)
    s : float
        Deleterious effect of a mutation
    u : float
        Deleterious mutation rate

    Returns
    -------
    float
        Population size
    '''
    N = 0
    l = len(Z0)
    for k in range(l):
        x = 0
        for j in range(t+1):
            x += mt(t, k, j, s, u)
        N += Z0[k] * x
    return N

    
@njit
def phi_k(x, k, s, u):
    '''Joint pgf of the number of k- and k+1-type offspring of a k-type individual.

    Parameters
    ----------
    x : typed.List
        Vector
    k : int
        Number of mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Deleterious mutation rate

    Returns
    -------
    float
        Probability
    '''
    n = len(x) - 1
    v = 1 - u
    if k < n:
        y = 1 - v ** 2 * x[k] ** 2 - 2 * u * v * x[k] * x[k + 1] - \
            u ** 2 * x[k + 1] ** 2
    elif k == n:
        y = 1 - x[n] ** 2
    return 1 - y * w(k, s) / 2


@njit
def phi1(x, s, u):
    '''
    Joint pgf of Z_1.

    Parameters
    ----------
    x : typed.List
        Vector
    s : float
        Deleterious effect of a mutation
    u : float
        Deleterious mutation rate

    Returns
    -------
    float
        Probability
    '''
    n = len(x) - 1
    phix = List()
    for k in range(n + 1):
        phix.append(phi_k(x, k, s, u))
    return phix


@njit
def phit(t, x, s, u):
    '''
    Joint pgf of Z_t. Composition of phi1() with itself for t generations.

    Parameters
    ----------
    t : int
        Number of generations
    x : typed.List
        Vector
    s : float
        Deleterious effect of a mutation
    u : float
        Deleterious mutation rate

    Returns
    -------
    float
        Probability
    '''
    for i in range(t):
        x = phi1(x, s, u)
    return x


@njit
def prob_extinct(Z0, t, n, s, u):
    """Probability that a population goes extinct by generation t.

    Parameters
    ----------
    Z0 : typed.List
        Initial number of individuals with 0, 1, ... mutations.
    t : int
        Generation number.
    n : int
        Number of types to consider.
    s : float
        Deleterious effect of a mutation
    u : float
        Deleterious mutation rate

    Returns
    -------
    float
        Probability of extinction
    """
    p = 1
    for k in range(len(Z0)):
        p *= phi_k(phit(t - 1, List([0.] * n), s, u), k, s, u) ** Z0[k]
    return p


@njit
def ET(Z0, n, s, u, tol):
    '''
    Expected extinction time of a population.

    Parameters
    ----------
    Z0 : typed.List
        Initial number of individuals with 0, 1, ... mutations.
    s : float
        Deleterious effect of a mutation
    u : float
        Deleterious mutation rate
    tol : float
        Tolerance

    Returns
    -------
    tup of floats
        Expected time, variance in time
    '''
    newsum = 0
    oldsum = -1
    t = 1
    while (newsum - oldsum) > tol:
        oldsum = newsum
        newsum += 1 - prob_extinct(Z0, t, n, s, u)
        t += 1
    return 1 + newsum


# @njit
# def phi(x, k, s, u):
#     '''
#     Probability generating function (PGF) of the number of k-type offspring
#     of a k-type individual.

#     Equation 3.

#     Parameters
#     ----------
#     x : float
#         Variable
#     k : int
#         Number of mutations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Mutation rate

#     Returns
#     -------
#     float
#         PGF
#     '''
#     y = 1 - u ** 2 - 2 * u * (1 - u) * x - (1 - u) ** 2 * x ** 2
#     return 1 - w(k, s) / 2 * y


# @njit
# def phit(t, x, k, s, u):
#     '''
#     PGF of number of k-type offspring of a k-type individual after t
#     generations.

#     Composition of PGF phi() with itself for t generations.

#     Parameters
#     ----------
#     t : int
#         Number of generations
#     x : float
#         Variable
#     k : int
#         Number of mutations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Mutation rate

#     Returns
#     -------
#     float
#         PGF.
#     '''
#     x = phi(x, k, s, u)
#     for i in range(1, t):
#         x = phi(x, k, s, u)
#     return x


# @njit
# def phisum(nk, k, s, u, tol):
#     '''
#     Sum term in Equations 6, 12, and 13.

#     Calculate to within tolerance tol.

#     Parameters
#     ----------
#     nk : int
#         Number of k-type individuals
#     k : int
#         Number of mutations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Mutation rate
#     tol : float
#         Tolerance

#     Returns
#     -------
#     float
#         Sum
#     '''
#     newsum = 0
#     oldsum = -1
#     t = 1
#     while newsum - oldsum > tol:
#         oldsum = newsum
#         newsum += 1 - phit(t, 0, k, s, u) ** nk
#         t += 1
#     return newsum


# @njit
# def t0(n0, u, tol):
#     '''
#     Expected extinction time of mutation-free class (i.e., first click of the
#     ratchet).

#     Equation 13.

#     Parameters
#     ----------
#     n0 : int
#         Number of mutation-free individuals
#     u : float
#         Mutation rate
#     tol : float
#         Tolerance

#     Returns
#     -------
#     float
#         Time
#     '''
#     return 1 + phisum(n0, 0, 0, u, tol)


# @njit
# def tauk(nk, k, s, u, tol, upper):
#     '''
#     Expected extinction time of nk type-k individuals.

#     Equation 6.

#     Parameters
#     ----------
#     nk : int
#         Number of nk individuals at time t = 0
#     k : int
#         Number of mutations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Mutation rate
#     tol : float
#         Tolerance
#     upper : bool
#         Whether to use upper bound P(tauk>0)

#     Returns
#     -------
#     float
#         Time
#     '''
#     if k == 0:
#         t = t0(nk, u, tol)
#     else:
#         t = phisum(nk, k, s, u, tol)
#         if upper:
#             t += min(1, nk)
#     return t


# @njit
# def phisum2(nk, k, s, u, tol):
#     '''
#     Sum term in variance calculation.  Calculate to within tolerance 10 * tol.

#     Parameters
#     ----------
#     nk : int
#         Number of k-type individuals
#     k : int
#         Number of mutations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Mutation rate
#     tol : float
#         Tolerance

#     Returns
#     -------
#     float
#         Sum
#     '''
#     newsum = 0
#     oldsum = -1
#     t = 1
#     while newsum - oldsum > 10 * tol:
#         oldsum = newsum
#         newsum += t * (1 - phit(t, 0, k, s, u) ** nk)
#         t += 1
#     return 2 * newsum


# @njit
# def vartauk(nk, k, s, u, tol, upper):
#     '''
#     Variance in extinction time of nk type-k individuals.

#     Equation 7.

#     Parameters
#     ----------
#     nk : int
#         Number of nk individuals at time t = 0
#     k : int
#         Number of mutations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Mutation rate
#     tol : float
#         Tolerance
#     upper : bool
#         Whether to use upper bound P(tauk>0)

#     Returns
#     -------
#     float
#         Variance
#     '''
#     t = tauk(nk, k, s, u, tol, upper)
#     return phisum2(nk, k, s, u, tol) + t - t ** 2


# @njit
# def xtk(n0, k, s, u, tol, upper, var):
#     '''
#     Expected extinction time of type-k individuals, and size of k-class at
#     extinction time of k - 1 class.

#     Parameters
#     ----------
#     n0 : int
#         Number of unmutated individuals at time t = 0
#     k : int
#         Number of mutations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Mutation rate
#     tol : float
#         Tolerance
#     upper : bool
#         Whether to use upper bound P(tauk>0)
#     var : bool
#         Whether to calculate variance in time

#     Returns
#     -------
#     tuple
#         Expected size, expected time, variance in time.
#     '''
#     x = n0
#     t = t0(n0, u, tol)
#     if var:
#         v = vartauk(n0, 0, s, u, tol, upper)
#     else:
#         v = -1
#     for i in range(1, k + 1):
#         x = n0 * gj(t, i, s, u)
#         t += tauk(x, i, s, u, tol, upper)
#         if var:
#             v += vartauk(x, i, s, u, tol, upper)
#     return x, t, v


# @njit
# def T(n0, s, u, tol, upper, var):
#     '''
#     Expected extinction time of entire population.

#     Equation 10.

#     Parameters
#     ----------
#     n0 : int
#         Number of mutation-free individuals at time t = 0
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Mutation rate
#     tol : float
#         Tolerance
#     upper : bool
#         Whether to use upper bound P(tauk>0)
#     var : bool
#         Whether to calculate variance in time

#     Returns
#     -------
#     tup of floats
#         Expected time, variance in time
#     '''
#     newt = 0
#     oldt = -1
#     k = 0
#     while newt - oldt > tol:
#         oldt = newt
#         tmp, newt, vart = xtk(n0, k, s, u, tol, upper, var)
#         k += 1
#     return newt, vart


# def melt(n0, s, u, k, plot=False):
#     """
#     Calculate rate of mutational meltdown.
#
#     Parameters
#     ----------
#     n0 : int
#         Number of mutation-free individuals at time t = 0
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Mutation rate
#     k : int
#         Maximum number of mutations to consider
#     plot : bool, optional
#         Whether to plot.
#
#     Returns
#     -------
#     tuple
#         Clicks, time between clicks, fitted values, rate of meltdown
#     """
#     y = []
#     x = range(k)
#     oldt = 0
#     for i in x:
#         n, t, v = xtk(n0, i, s, u, 1e-6, False, False)
#         y.append(t - oldt)
#         oldt = t
#     x = np.array(x)
#     y = np.array(y)
#     if plot:
#         plt.semilogy(x, y, "o--")
#     slope, intercept, r_value, p_value, std_err = stats.linregress(
#         x, np.log(y))
#     if plot:
#         plt.plot(x, np.exp(intercept + slope * x))
#     return x, y, np.exp(intercept + slope * x), slope


###############################################################################
# Simulation functions
###############################################################################


# @njit
# def ind_step(r, k, s, u):
#     '''
#     Take a time step in the branching process model for a single individual
#     with k mutations.  Return the number of offspring with k and k + 1
#     mutations.
#
#     Parameters
#     ----------
#     r : float
#         Random number
#     k : int
#         Number of mutations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Deleterious mutation rate
#
#     Returns
#     -------
#     tuple
#         Number of offspring with k and k + 1 mutations, respectively
#     '''
#     z0 = 0
#     z1 = 0
#     wk = w(k, s)
#     v = 1 - u
#     # probabilities of different outcomes
#     p1 = (wk * v ** 2) / 2
#     p2 = wk * u * v
#     p3 = (wk * u ** 2) / 2
#     # outcome 1: two unmutated offspring
#     if r <= p1:
#         z0 += 2
#     # outcome 2: one mutant offspring one unmutated offspring
#     elif p1 < r <= p1 + p2:
#         z0 += 1
#         z1 += 1
#     # outcome 3: two mutant offspring
#     elif p1 + p2 < r <= p1 + p2 + p3:
#         z1 += 2
#     # outcome 4: death (no offspring)
#     return z0, z1
#
#
# @njit
# def class_step(z, k, s, u):
#     '''
#     Take a time step in the branching process model for z individuals with k
#     mutations.  Return the total number of offspring with k and k + 1
#     mutations.
#
#     Parameters
#     ----------
#     z : int
#         Number of individuals
#     k : int
#         Number of mutations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Deleterious mutation rate
#
#     Returns
#     -------
#     tuple
#         Number of offspring with k and k + 1 mutations, respectively
#     '''
#     z0, z1 = 0, 0
#     if z > 0:
#         for r in rnd.random(z):
#             dz0, dz1 = ind_step(r, k, s, u)
#             z0 += dz0
#             z1 += dz1
#     return z0, z1


def trim(x):
    '''Delete a zero at the end of a histogram.

    Parameters
    ----------
    x : list
        Histogram.

    Returns
    -------
    list
        Histogram.
    '''
    if x[-1] == 0:
        del x[-1]
    return x


# def pop_step(zz, s, u):
#     '''
#     Take a time step in the branching process model for a population.  Return
#     a new population.
#
#     Parameters
#     ----------
#     zz : list
#         Histogram of number of individuals with k = 0, 1, 2, ... mutations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Deleterious mutation rate
#
#     Returns
#     -------
#     list
#         New histogram
#     '''
#     offspring = []
#     newzz = [0]
#     if sum(zz) == 0:
#         return zz
#     else:
#         k = 0
#         for z in zz:
#             offspring.append(class_step(z, k, s, u))
#             k += 1
#         for i in range(len(offspring)):
#             newzz[i] += offspring[i][0]
#             newzz.append(offspring[i][1])
#         newzz = trim(newzz)
#         return newzz
#
#
# def sim(z, s, u):
#     '''
#     Simulate evolution until the population goes extinct.
#
#     Parameters
#     ----------
#     z : list
#         Histogram of number of individuals with k = 0, 1, 2, ... mutations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Deleterious mutation rate
#
#     Returns
#     -------
#     tuple
#         Population sizes, extinction time
#     '''
#     t = 0
#     n = sum(z)
#     extinct = (n == 0)
#     zz = [z]
#     nn = [n]
#     k = []
#     j = range(len(z))
#     k.append(sum([i * z[i] for i in j]) / n)
#     clicks = []
#     while not extinct:
#         newz = trim(pop_step(z, s, u))
#         j = range(len(newz))
#         n = sum(newz)
#         nn.append(n)
#         t += 1
#         extinct = (n == 0)
#         if not extinct:
#             k.append(sum([i * newz[i] for i in j]) / n)
#             i = len(clicks)
#             if newz[i] == 0:
#                 clicks.append(t)
#         else:
#             k.append(np.inf)
#         z = newz
#         zz.append(z)
#     return nn, k, t, clicks, zz


def type_offspring(n, k, s, u):
    '''Take a time step in the branching process model for n individuals with k
    mutations.  Return the total number of offspring with k and k + 1
    mutations.

    Parameters
    ----------
    n : int
        Number of individuals
    k : int
        Number of deleterious mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Deleterious mutation rate

    Returns
    -------
    tuple
        Number of offspring with k and k + 1 mutations, respectively
    '''
    if n > 0:
        v = 1 - u
        # probabilities of different outcomes
        p = w(k, s) * np.array([(v ** 2) / 2, u * v, (u ** 2) / 2])
        p = np.append(p, 1 - p.sum())
        outcomes = rnd.multinomial(n, p)
        # offspring
        unmutated = 2 * outcomes[0] + outcomes[1]
        mutant = outcomes[1] + 2 * outcomes[2]
        return unmutated, mutant
    else:
        return 0, 0


def next_gen(Z, s, u):
    '''Generate the next generation.

    Parameters
    ----------
    Z : list
        Histogram of number of individuals with k = 0, 1, 2, ... mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Deleterious mutation rate

    Returns
    -------
    list
        Histogram
    '''
    ntypes = len(Z)
    next_Z = [0] * (ntypes + 1)
    for k in range(ntypes):
        if Z[k] > 0:
            offspring = type_offspring(Z[k], k, s, u)
            next_Z[k] += offspring[0]
            next_Z[k+1] += offspring[1]
    return trim(next_Z)


def to_extinction(Z, s, u):
    '''Simulate evolution until the population goes extinct.

    Parameters
    ----------
    Z : list
        Histogram of number of individuals with k = 0, 1, 2, ... mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Deleterious mutation rate

    Returns
    -------
    tuple
        Population sizes, compositions, extinction time
    '''
    t = 0
    N = sum(Z)
    Z_history = [Z]
    N_history = [N]
    while N > 0:
        t += 1
        Z = next_gen(Z, s, u)
        N = sum(Z)
        Z_history.append(Z)
        N_history.append(N)
    return N_history, Z_history, t


def multiple_extinctions(Z, n, s, u):
    '''Simulate evolution of multiple populations until they all go extinct.

    Parameters
    ----------
    Z : list
        Histogram of number of individuals with k = 0, 1, 2, ... mutations
    n : int
        Number of replicate populations
    s : float
        Deleterious effect of a mutation
    u : float
        Deleterious mutation rate

    Returns
    -------
    tuple of np.arrays
        Population sizes (replicate populations in columns), extinction times
    '''
    NN = []
    tt = []
    for i in range(n):
        N_history, Z_history, t = to_extinction(Z, s, u)
        NN.append(N_history)
        tt.append(t)
    tt = np.array(tt, dtype=int)
    NN_array = np.zeros((tt.max() + 1, n))
    for i in range(n):
        for j in range(len(NN[i])):
            NN_array[j, i] = NN[i][j]
    return NN_array, tt


# def simult(z, nreps, s, u):
#     '''
#     Simulate evolution of multiple populations until they all go extinct.
#
#     Parameters
#     ----------
#     z : list
#         Histogram of number of individuals with k = 0, 1, 2, ... mutations
#     nreps : int
#         Number of replicate populations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Deleterious mutation rate
#
#     Returns
#     -------
#     tuple of np.arrays
#         Population sizes (replicate populations in columns), extinction times
#     '''
#     nn = []
#     kk = []
#     tt = []
#     clicks = []
#     for i in range(nreps):
#         n, k, t, c, zz = sim(z, s, u)
#         nn.append(n)
#         kk.append(k)
#         tt.append(t)
#         clicks.append(c)
#     ext_times = np.array(tt, dtype=int)
#     pop_sizes = np.zeros((ext_times.max() + 1, nreps))
#     for i in range(nreps):
#         for j in range(len(nn[i])):
#             pop_sizes[j, i] = nn[i][j]
#     return pop_sizes, ext_times, clicks


# @jit
# def pop_step(h, s, u, K):
#     '''
#     Take a time step in the branching process model for a population.  Return
#     a new population.
#
#     If population size N exceeds carrying capacity K, kill N – K individuals
#     at random.  If K is infinite, there is no density dependence.
#
#     Parameters
#     ----------
#     h : list
#         Histogram of number of individuals with k = 0, 1, 2, ... mutations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Mutation rate
#     K : int
#         Carrying capacity
#
#     Returns
#     -------
#     list
#         New histogram
#     '''
#     offspring = []
#     newh = [0]
#     if sum(h) == 0:
#         return h
#     else:
#         k = 0
#         for n in h:
#             offspring += [(class_step(n, k, s, u))]
#             k += 1
#         for i in range(len(offspring)):
#             newh[i] += offspring[i][0]
#             newh += [offspring[i][1]]
#         newh = trim(newh)
#         n = sum(newh)
#         if n <= K:
#             return newh
#         else:
#             p = np.array(newh) / n
#             purged = rnd.multinomial(K, p)
#             return purged.tolist()


# @jit
# def sim(h, s, u, K):
#     '''
#     Simulate evolution until the population goes extinct.  If K is infinite,
#     there is no density dependence.
#
#     Parameters
#     ----------
#     h : list
#         Histogram of number of individuals with k = 0, 1, 2, ... mutations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Mutation rate
#     K : int
#         Carrying capacity
#
#     Returns
#     -------
#     tuple
#         Population sizes, extinction time
#     '''
#     t = 0
#     n = [sum(h)]
#     k = [0]
#     extinct = (n[t] == 0)
#     clicks = []
#     while not extinct:
#         newh = pop_step(h, s, u, K)
#         j = range(len(newh))
#         n.append(sum(newh))
#         t += 1
#         # print(t, newh)
#         extinct = (n[t] == 0)
#         if not extinct:
#             k.append(sum([i * newh[i] for i in j]) / sum(newh))
#             i = len(clicks)
#             if newh[i] == 0:
#                 clicks.append(t)
#         else:
#             k.append(np.inf)
#         h = newh
#     return n, k, t, clicks


# @jit
# def simult(h, nreps, s, u, K):
#     '''
#     Simulate evolution of multiple populations until they all go extinct.  If K
#     is infinite, there is no density dependence.
#
#     Parameters
#     ----------
#     h : list
#         Histogram of number of individuals with k = 0, 1, 2, ... mutations
#     nreps : int
#         Number of replicate populations
#     s : float
#         Deleterious effect of a mutation
#     u : float
#         Mutation rate
#     K : int
#         Carrying capacity
#
#     Returns
#     -------
#     tuple of np.arrays
#         Population sizes (replicate populations in columns), extinction times
#     '''
#     nn = []
#     kk = []
#     tt = []
#     cc = []
#     for i in range(nreps):
#         n, k, t, c = sim(h, s, u, K)
#         nn.append(n)
#         kk.append(k)
#         tt.append(t)
#         cc.append(c)
#     ext_times = np.array(tt, dtype=int)
#     pop_sizes = np.zeros((ext_times.max() + 1, nreps))
#     mutations = np.ones((ext_times.max() + 1, nreps)) / \
#         np.zeros((ext_times.max() + 1, nreps))
#     for i in range(nreps):
#         for j in range(len(nn[i])):
#             pop_sizes[j, i] = nn[i][j]
#             mutations[j, i] = kk[i][j]
#     return pop_sizes, mutations, ext_times, cc
