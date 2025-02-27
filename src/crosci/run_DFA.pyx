import cython
from cpython cimport array
import array
import numpy as np

cdef extern from "dfa.h":
    double* dfa(double *seq, long npts, long *rs, int nr, double overlap_perc)

def run_DFA(input,sampling_frequency,overlap,window_sizes):
    input_list = input.tolist()
    window_sizes_list = window_sizes.tolist()
    cdef array.array input_arr = array.array('d',input_list)
    npts = len(input_arr)
    cdef array.array rs = array.array('l',window_sizes_list)
    nr = len(rs)

    if overlap:
        overlap_perc = 0.5
    else:
        overlap_perc = 0
    
    cdef int nrc = nr
    cdef double[:] mse = <double[:nrc]> dfa(input_arr.data.as_doubles,npts,rs.data.as_longs,nr,overlap_perc)
    mse_np = np.asarray(mse)

    return((window_sizes,mse_np))