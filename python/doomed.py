#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions for numerical evaluations and simulations in the paper:
Muller's Ratchet in Asexual Populations Doomed to Extinction."""

# =============================================================================
# Created By  : Ricardo Azevedo, Logan Chipkin
# Last Updated: Thu Apr 30 10:47:11 2020
# =============================================================================


# =============================================================================
# Imports
# =============================================================================


from numba import jit
import numpy as np
import numpy.random as rnd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns


# =============================================================================
# Plotting settings
# =============================================================================


sns.set_style("ticks")


# use LaTeX for typesetting, Helvetica font
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [
    r'\usepackage{helvet}',
    r'\usepackage[EULERGREEK]{sansmath}',
    r'\sansmath'
]


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
    ax.text(xtx.min() - .016 * xrg, ytx.max() + .15 * yrg, part_label, size=24, ha='center', va='center')
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
    ax.text(xtx.min() - .016 * xrg, ytx.max() + .2 * yrg, part_label, size=24, ha='center', va='center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# =============================================================================
# Theory functions
# =============================================================================


@jit(nopython=True)
def w(k, s):
    '''
    Fitness of a genotype with k deleterious mutations of effect s.

    Equation 2.

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


@jit(nopython=True)
def phi(x, k, s, u):
    '''
    Probability generating function (PGF) of the number of k-type offspring
    of a k-type individual.

    Equation 3.

    Parameters
    ----------
    x : float
        Variable
    k : int
        Number of mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate

    Returns
    -------
    float
        PGF
    '''
    y = 1 - u ** 2 - 2 * u * (1 - u) * x - (1 - u) ** 2 * x ** 2
    return 1 - w(k, s) / 2 * y


@jit(nopython=True)
def phit(t, x, k, s, u):
    '''
    PGF of number of k-type offspring of a k-type individual after t
    generations.

    Composition of PGF phi() with itself for t generations.

    Parameters
    ----------
    t : int
        Number of generations
    x : float
        Variable
    k : int
        Number of mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate

    Returns
    -------
    float
        PGF.
    '''
    x = phi(x, k, s, u)
    for i in range(1, t):
        x = phi(x, k, s, u)
    return x


@jit(nopython=True)
def joint_phi_k(x, k, s, u):
    '''
    Joint PGF of the number of k- and k+1-type offspring of a k-type individual.

    Parameters
    ----------
    x : list
        Vector
    k : int
        Number of mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate

    Returns
    -------
    float
        PGF
    '''
    n = len(x) - 1
    if k < n:
        y = 1 - (1 - u) ** 2 * x[k] ** 2 - 2 * u * (1 - u) * x[k] * x[k + 1] - u ** 2 * x[k + 1] ** 2
    elif k == n:
        y = 1 - x[n] ** 2
    return 1 - w(k, s) / 2 * y


@jit(nopython=True)
def joint_phi(x, s, u):
    '''
    PGF of the number of n-type offspring of a n-type individual.

    Parameters
    ----------
    x : list
        Vector
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate

    Returns
    -------
    float
        PGF
    '''
    n = len(x) - 1
    phix = []
    for k in range(n + 1):
        phix.append(joint_phi_k(x, k, s, u))
    return phix


@jit
def joint_phit(t, x, s, u):
    '''
    Composition of PGF joint_phi() with itself for t generations.

    Parameters
    ----------
    x : list
        Vector
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate

    Returns
    -------
    float
        PGF
    '''
    for i in range(t):
        x = joint_phi(x, s, u)
    return x


@jit
def p_extinct(n, t, s, u):
    """Probability that a population of size n will be extinct in generation t.

    Parameters
    ----------
    n : int
        Initial population size.
    t : int
        Generation number.
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate

    Returns
    -------
    float
        Probability of extinction
    """
    return joint_phi_k(joint_phit(t - 1, np.zeros(t), s, u), 0, s, u) ** n


@jit(nopython=True)
def phisum(nk, k, s, u, tol):
    '''
    Sum term in Equations 6, 12, and 13.

    Calculate to within tolerance tol.

    Parameters
    ----------
    nk : int
        Number of k-type individuals
    k : int
        Number of mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate
    tol : float
        Tolerance

    Returns
    -------
    float
        Sum
    '''
    newsum = 0
    oldsum = -1
    t = 1
    while newsum - oldsum > tol:
        oldsum = newsum
        newsum += 1 - phit(t, 0, k, s, u) ** nk
        t += 1
    return newsum


@jit(nopython=True)
def t0(n0, u, tol):
    '''
    Expected extinction time of mutation-free class (i.e., first click of the
    ratchet).

    Equation 13.

    Parameters
    ----------
    n0 : int
        Number of mutation-free individuals
    u : float
        Mutation rate
    tol : float
        Tolerance

    Returns
    -------
    float
        Time
    '''
    return 1 + phisum(n0, 0, 0, u, tol)


@jit(nopython=True)
def gj(t, j, s, u):
    '''
    Expected number of descendants of type j from a mutation-free individual
    after t generations.

    Equation 8.

    Parameters
    ----------
    t : int
        Number of generations
    j : int
        Type of individual
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate

    Returns
    -------
    float
        Number of individuals
    '''
    x = 0
    if t >= j:
        x = (1 - u) ** (t - j) * u ** j * w(j * (j - 1) / 2, s)
        for i in range(1, j + 1):
            x *= (1 - w(t + 1 - i, s)) / (1 - w(i, s))
    return x


@jit(nopython=True)
def tauk(nk, k, s, u, tol, upper):
    '''
    Expected extinction time of nk type-k individuals.

    Equation 6.

    Parameters
    ----------
    nk : int
        Number of nk individuals at time t = 0
    k : int
        Number of mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate
    tol : float
        Tolerance
    upper : bool
        Whether to use upper bound P(tauk>0)

    Returns
    -------
    float
        Time
    '''
    if k == 0:
        t = t0(nk, u, tol)
    else:
        t = phisum(nk, k, s, u, tol)
        if upper:
            t += min(1, nk)
    return t


@jit(nopython=True)
def phisum2(nk, k, s, u, tol):
    '''
    Sum term in variance calculation.  Calculate to within tolerance 10 * tol.

    Parameters
    ----------
    nk : int
        Number of k-type individuals
    k : int
        Number of mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate
    tol : float
        Tolerance

    Returns
    -------
    float
        Sum
    '''
    newsum = 0
    oldsum = -1
    t = 1
    while newsum - oldsum > 10 * tol:
        oldsum = newsum
        newsum += t * (1 - phit(t, 0, k, s, u) ** nk)
        t += 1
    return 2 * newsum


@jit(nopython=True)
def vartauk(nk, k, s, u, tol, upper):
    '''
    Variance in extinction time of nk type-k individuals.

    Equation 7.

    Parameters
    ----------
    nk : int
        Number of nk individuals at time t = 0
    k : int
        Number of mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate
    tol : float
        Tolerance
    upper : bool
        Whether to use upper bound P(tauk>0)

    Returns
    -------
    float
        Variance
    '''
    t = tauk(nk, k, s, u, tol, upper)
    return phisum2(nk, k, s, u, tol) + t - t ** 2


@jit(nopython=True)
def xtk(n0, k, s, u, tol, upper, var):
    '''
    Expected extinction time of type-k individuals, and size of k-class at
    extinction time of k - 1 class.

    Parameters
    ----------
    n0 : int
        Number of unmutated individuals at time t = 0
    k : int
        Number of mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate
    tol : float
        Tolerance
    upper : bool
        Whether to use upper bound P(tauk>0)
    var : bool
        Whether to calculate variance in time

    Returns
    -------
    tuple
        Expected size, expected time, variance in time.
    '''
    x = n0
    t = t0(n0, u, tol)
    if var:
        v = vartauk(n0, 0, s, u, tol, upper)
    else:
        v = -1
    for i in range(1, k + 1):
        x = n0 * gj(t, i, s, u)
        t += tauk(x, i, s, u, tol, upper)
        if var:
            v += vartauk(x, i, s, u, tol, upper)
    return x, t, v


@jit(nopython=True)
def T(n0, s, u, tol, upper, var):
    '''
    Expected extinction time of entire population.

    Equation 10.

    Parameters
    ----------
    n0 : int
        Number of mutation-free individuals at time t = 0
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate
    tol : float
        Tolerance
    upper : bool
        Whether to use upper bound P(tauk>0)
    var : bool
        Whether to calculate variance in time

    Returns
    -------
    tup of floats
        Expected time, variance in time
    '''
    newt = 0
    oldt = -1
    k = 0
    while newt - oldt > tol:
        oldt = newt
        tmp, newt, vart = xtk(n0, k, s, u, tol, upper, var)
        k += 1
    return newt, vart


@jit(nopython=True)
def N(n0, s, u, t):
    '''
    Expected total population size at time t.

    Equation 19.

    Parameters
    ----------
    n0 : int
        Number of mutation-free individuals at time t = 0
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate
    t : int
        Time (generations)

    Returns
    -------
    float
        Size
    '''
    gjsum = 0
    for j in range(t + 1):
        gjsum += gj(t, j, s, u)
    return n0 * gjsum


def melt(n0, s, u, k, plot=False):
    """
    Calculate rate of mutational meltdown.

    Parameters
    ----------
    n0 : int
        Number of mutation-free individuals at time t = 0
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate
    k : int
        Maximum number of mutations to consider
    plot : bool, optional
        Whether to plot.

    Returns
    -------
    tuple
        Clicks, time between clicks, fitted values, rate of meltdown
    """
    y = []
    x = range(k)
    oldt = 0
    for i in x:
        n, t, v = xtk(n0, i, s, u, 1e-6, False, False)
        y.append(t - oldt)
        oldt = t
    x = np.array(x)
    y = np.array(y)
    if plot:
        plt.semilogy(x, y, "o--")
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,np.log(y))
    if plot:
        plt.plot(x, np.exp(intercept + slope * x))
    return x, y, np.exp(intercept + slope * x), slope


# =============================================================================
# Simulation functions
# =============================================================================


@jit(nopython=True)
def ind_step(r, k, s, u):
    '''
    Take a time step in the branching process model for a single individual
    with k mutations.  Return the number of offspring with k and k + 1
    mutations.

    Parameters
    ----------
    r : float
        Random number
    k : int
        Number of mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate

    Returns
    -------
    tuple
        Number of offspring with k and k + 1 mutations, respectively
    '''
    # fitness
    f = w(k, s)
    y = 0
    z = 0
    # probabilities of different outcomes
    p1 = (f * (1 - u) ** 2) / 2
    p2 = f * u * (1 - u)
    p3 = (f * u ** 2) / 2
    # outcome 1: two unmutated offspring
    if r <= p1:
        y += 2
    # outcome 2: one mutant offspring one unmutated offspring
    elif p1 < r <= p1 + p2:
        y += 1
        z += 1
    # outcome 3: two mutant offspring
    elif p1 + p2 < r <= p1 + p2 + p3:
        z += 2
    # outcome 4: death (no offspring)
    return y, z


@jit(nopython=True)
def class_step(n, k, s, u):
    '''
    Take a time step in the branching process model for n individuals with k
    mutations.  Return the total number of offspring with k and k + 1
    mutations.

    Parameters
    ----------
    n : int
        Number of individuals
    k : int
        Number of mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate

    Returns
    -------
    tuple
        Number of offspring with k and k + 1 mutations, respectively
    '''
    y, z = 0, 0
    if n > 0:
        rr = rnd.random(n)
        for r in rr:
            dy, dz = ind_step(r, k, s, u)
            y += dy
            z += dz
    return y, z


@jit(nopython=True)
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


@jit
def pop_step(h, s, u, K):
    '''
    Take a time step in the branching process model for a population.  Return
    a new population.

    If population size N exceeds carrying capacity K, kill N – K individuals
    at random.  If K is infinite, there is no density dependence.

    Parameters
    ----------
    h : list
        Histogram of number of individuals with k = 0, 1, 2, ... mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate
    K : int
        Carrying capacity

    Returns
    -------
    list
        New histogram
    '''
    offspring = []
    newh = [0]
    if sum(h) == 0:
        return h
    else:
        k = 0
        for n in h:
            offspring += [(class_step(n, k, s, u))]
            k += 1
        for i in range(len(offspring)):
            newh[i] += offspring[i][0]
            newh += [offspring[i][1]]
        newh = trim(newh)
        n = sum(newh)
        if n <= K:
            return newh
        else:
            print(newh)
            p = np.array(newh) / n
            purged = rnd.multinomial(K, p)
            return purged.tolist()


@jit
def sim(h, s, u, K):
    '''
    Simulate evolution until the population goes extinct.  If K is infinite,
    there is no density dependence.

    Parameters
    ----------
    h : list
        Histogram of number of individuals with k = 0, 1, 2, ... mutations
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate
    K : int
        Carrying capacity

    Returns
    -------
    tuple
        Population sizes, extinction time
    '''
    t = 0
    n = [sum(h)]
    extinct = (n[t] == 0)
    while not extinct:
        newh = pop_step(h, s, u, K)
        n.append(sum(newh))
        t += 1
        extinct = (n[t] == 0)
        h = newh
    return n, t


@jit
def simult(h, nreps, s, u, K):
    '''
    Simulate evolution of multiple populations until they all go extinct.  If K
    is infinite, there is no density dependence.

    Parameters
    ----------
    h : list
        Histogram of number of individuals with k = 0, 1, 2, ... mutations
    nreps : int
        Number of replicate populations
    s : float
        Deleterious effect of a mutation
    u : float
        Mutation rate
    K : int
        Carrying capacity

    Returns
    -------
    tuple of np.arrays
        Population sizes (replicate populations in columns), extinction times
    '''
    nn = []
    tt = []
    for i in range(nreps):
        n, t = sim(h, s, u, K)
        nn.append(n)
        tt.append(t)
    ext_times = np.array(tt, dtype=int)
    pop_sizes = np.zeros((ext_times.max() + 1, nreps))
    for i in range(nreps):
        for j in range(len(nn[i])):
            pop_sizes[j,i] = nn[i][j]
    return pop_sizes, ext_times

