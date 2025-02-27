import cython
from cpython cimport array
import array
import numpy as np
from math import floor

cdef extern from "fEI.h":
    double* fEI(double *seq, long npts, long boxsize, double overlap)

def run_fEI(input,sampling_frequency,window_size_sec,overlap):
    input_list = input.tolist()
    npts = len(input_list)

    cdef array.array input_arr = array.array('d',input_list)

    cdef int box_size = sampling_frequency * window_size_sec

    if overlap>0:
        inc = floor(box_size*(1-overlap))
    else:
        inc = box_size

    num_w = 0
    #I subtract 1 because I add 1 in the beginning (using 1-indexing in c)
    for i in range(0,npts-box_size,inc):
        num_w = num_w+1

    cdef int nrc = num_w * 2

    cdef double[:] mse = <double[:nrc]> fEI(input_arr.data.as_doubles,npts,box_size,overlap)

    mse_np = np.asarray(mse)

    w_normalized_fluctuations = mse_np[0:num_w]
    w_original_amp = mse_np[num_w:nrc]

    return ((w_original_amp,w_normalized_fluctuations))