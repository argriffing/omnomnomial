"""
Quickly create large multinomial transition matrices.

The implementation is in Cython for speed
and uses python numpy arrays for speed and convenience.
For compilation instructions see
http://docs.cython.org/src/reference/compilation.html
For example:
$ cython -a wrightcore.pyx
$ gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
      -I/usr/include/python2.7 -o wrightcore.so wrightcore.c
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log

np.import_array()



@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_log_fact_array(int N):
    """
    Precompute some logarithms of factorials up to N.
    This is a helper function for get_lmcs.
    @param N: max integer whose log of factorial to compute
    @return: a numpy array of length N+1
    """
    cdef np.ndarray[np.float64_t, ndim=1] log_fact = np.zeros(N+1)
    cdef double accum = 0
    for i in range(2, N+1):
        accum += log(i)
        log_fact[i] = accum
    return log_fact

@cython.boundscheck(False)
@cython.wraparound(False)
def get_lmcs(
        np.ndarray[np.int_t, ndim=2] M,
        ):
    """
    This computes the lmcs ndarray with logs of multinomial coefficients.
    It only needs to be computed once for each M.
    @param M: counts of each microstate for each macrostate
    @return: a one dimensional array of log multinomial coefficients
    """
    cdef int nstates = M.shape[0]
    cdef int k = M.shape[1]
    cdef int N = np.sum(M[0])
    cdef int i
    cdef np.ndarray[np.float64_t, ndim=1] log_fact = get_log_fact_array(N)
    cdef np.ndarray[np.float64_t, ndim=1] v = np.empty(nstates)
    for i in range(nstates):
        v[i] = log_fact[N]
        for index in range(k):
            v[i] -= log_fact[M[i, index]]
    return v

@cython.boundscheck(False)
@cython.wraparound(False)
def get_neutral_lps(
        np.ndarray[np.int_t, ndim=2] M,
        ):
    """
    The next macrostate expectation will be equal to the current macrostate.
    In other words, then counts of microstates of the next macrostate
    will be picked multinomially according to their current frequencies.
    This function is not really necessary,
    and you should be able to replace it by np.log(M / float(N))
    where N is the common row sum of M.
    Transition matrices with deterministic mutation followed by
    neutral multinomial drift may be modeled
    using an lps like np.log(np.dot(M, mu) / float(N))
    where mu is a microstate transition matrix.
    @param M: counts of each microstate for each macrostate
    @return: log of a discrete microstate distribution for each macrostate
    """
    cdef int nstates = M.shape[0]
    cdef int k = M.shape[1]
    cdef int i, j
    cdef double neg_logp
    cdef double popsize
    cdef np.ndarray[np.float64_t, ndim=2] L = np.empty((nstates, k))
    for i in range(nstates):
        popsize = 0
        for j in range(k):
            popsize += M[i, j]
        neg_logp = -log(popsize)
        for j in range(k):
            L[i, j] = neg_logp + log(M[i, j])
    return L

@cython.boundscheck(False)
@cython.wraparound(False)
def get_log_transition_matrix(
        np.ndarray[np.int_t, ndim=2] M,
        np.ndarray[np.float64_t, ndim=1] lmcs,
        np.ndarray[np.float64_t, ndim=2] lps,
        ):
    """
    Build the entrywise logarithm of a multinomial transition matrix.
    The output may have -inf but it should not have nan.
    @param M: counts of each microstate for each macrostate
    @param lmcs: log of multinomial coefficient of each macrostate
    @param lps: log probability of each microstate for each macrostate
    @return: entrywise log of a multinomial transition matrix
    """
    cdef int nstates = M.shape[0]
    cdef int k = M.shape[1]
    cdef int i, j, index
    cdef np.ndarray[np.float64_t, ndim=2] L = np.zeros((nstates, nstates))
    for i in range(nstates):
        for j in range(nstates):
            L[i, j] = lmcs[j]
            for index in range(k):
                if M[j, index]:
                    L[i, j] += M[j, index] * lps[i, index]
    return L

