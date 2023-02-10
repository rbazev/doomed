#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions for numerical evaluations and simulations in the paper:
Mutational meltdown in asexual populations doomed to extinction."""


###############################################################################
### CREATED BY  : Ricardo Azevedo, Logan Chipkin                            ###
### LAST UPDATED: 02/10/2023                                                ###
###############################################################################


from numba import jit
from numba import njit
from numba.typed import List
import numpy as np
import numpy.random as rnd


###############################################################################
### PLOTTING SETTINGS                                                       ###
###############################################################################


sns.set_style('ticks')
sns.set_context('talk')


def set_up_axes(ax, xmin, xmax, xstep, ymin, ymax, ystep, rnd, xlabel='', ylabel='', part_label=''):
    '''
    Format plot. 

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


###############################################################################
### THEORY                                                                  ###
###############################################################################


# Size and composition of population


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
        Initial number of individuals with 0, 1, ... mutations
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
        Initial number of individuals with 0, 1, ... mutations
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
        Initial number of individuals with 0, 1, ... mutations
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

    
# Extinction time


@njit
def Phi_k(x, k, s, u):
    '''Joint probability generating function (pgf) of the number of k- and
    k+1-type offspring of a k-type individual.

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
def Phi1(x, s, u):
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
    Phix = List()
    for k in range(n + 1):
        Phix.append(Phi_k(x, k, s, u))
    return Phix


@njit
def Phit(t, x, s, u):
    '''
    Joint pgf of Z_t. Composition of Phi1() with itself for t generations.

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
        x = Phi1(x, s, u)
    return x


@njit
def prob_extinct(Z0, t, n, s, u):
    """Probability that a population goes extinct by generation t.

    Parameters
    ----------
    Z0 : typed.List
        Initial number of individuals with 0, 1, ... mutations
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
        p *= Phi_k(Phit(t - 1, List([0.] * n), s, u), k, s, u) ** Z0[k]
    return p


@njit
def extinction_time(Z0, n, s, u, tol):
    '''
    Expected value and variance of extinction time of a population.

    Parameters
    ----------
    Z0 : typed.List
        Initial number of individuals with 0, 1, ... mutations
    n : int
        Number of types to consider.
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
    oldT = 0
    newT = 1
    oldT2 = -1
    newT2 = 0
    t = 1
    while ((newT - oldT) > tol) and ((newT2 - oldT2) > tol):
        oldT = newT
        oldT2 = newT2
        q = 1 - prob_extinct(Z0, t, n, s, u)
        newT += q
        newT2 += 2 * t * q
        t += 1
    return newT, newT2 + newT - newT * newT 


# Click time


@njit
def phi_k(x, k, s, u):
    '''
    Pgf of the number of k-type offspring of a k-type individual.

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
    v = 1 - u
    y = 1 - (v ** 2 * x ** 2 + 2 * u * v * x + u ** 2)
    return 1 - y * w(k, s) / 2


@njit
def phit(t, x, k, s, u):
    '''
    Pgf of number of k-type offspring of a k-type individual after t
    generations.

    Composition of pgf phi_k() with itself for t generations.

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
    x = phi_k(x, k, s, u)
    for i in range(1, t):
        x = phi_k(x, k, s, u)
    return x


@njit
def click_time(Z0k, k, s, u, tol):
    '''
    Expected value and variance of the extinction time of the least loaded class
    (i.e., the first click of the ratchet).

    Parameters
    ----------
    Z0 : int
        Initial number of individuals with k mutations
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
    oldT = 0
    newT = 1
    oldT2 = -1
    newT2 = 0
    t = 1
    while ((newT - oldT) > tol) and ((newT2 - oldT2) > tol):
        oldT = newT
        oldT2 = newT2
        q = 1 - phit(t, 0, k, s, u) ** Z0k
        newT += q
        newT2 += 2 * t * q
        t += 1
    return newT, newT2 + newT - newT * newT 


###############################################################################
### STOCHASTIC SIMULATIONS                                                  ###
###############################################################################


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
