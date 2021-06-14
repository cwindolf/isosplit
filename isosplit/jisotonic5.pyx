import numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def jisotonic5(double[::1] x, double[::1] weights):
    cdef Py_ssize_t N = x.shape[0]

    MSE_ = np.zeros(N, dtype=np.double)
    cdef double[::1] MSE = MSE_
    y_ = np.zeros(N, dtype=np.double)
    cdef double[::1] y = y_

    unweightedcount_ = np.zeros(N, dtype=np.int)
    cdef long[::1] unweightedcount = unweightedcount_
    count_ = np.zeros(N, dtype=np.double)
    cdef double[::1] count = count_
    sum_ = np.zeros(N, dtype=np.double)
    cdef double[::1] sum = sum_
    sumsqr_ = np.zeros(N, dtype=np.double)
    cdef double[::1] sumsqr = sumsqr_
    cdef Py_ssize_t last_index = 0

    unweightedcount[last_index] = 1
    count[last_index] = weights[0]
    sum[last_index] = x[0] * weights[0]
    sumsqr[last_index] = x[0] * x[0] * weights[0]
    MSE[0] = 0

    cdef double prevMSE
    cdef double newMSE

    for j in range(1, N):
        last_index += 1
        unweightedcount[last_index] = 1
        count[last_index] = weights[j]
        sum[last_index] = x[j] * weights[j]
        sumsqr[last_index] = x[j] * x[j] * weights[j]
        MSE[j] = MSE[j - 1]

        while True:
            if last_index <= 0:
                break
            if sum[last_index - 1] / count[last_index - 1] < sum[last_index] / count[last_index]:
                break

            prevMSE = sumsqr[last_index - 1] - sum[last_index - 1] * sum[last_index - 1] / count[last_index - 1]
            prevMSE += sumsqr[last_index] - sum[last_index] * sum[last_index] / count[last_index]
            unweightedcount[last_index - 1] += unweightedcount[last_index]
            count[last_index - 1] += count[last_index]
            sum[last_index - 1] += sum[last_index]
            sumsqr[last_index - 1] += sumsqr[last_index]
            newMSE = sumsqr[last_index - 1] - sum[last_index - 1] * sum[last_index - 1] / count[last_index - 1]
            MSE[j] += newMSE - prevMSE
            last_index -= 1

    cdef Py_ssize_t ii = 0
    for k in range(last_index + 1):
        for cc in range(unweightedcount[k]):
            y[ii + cc] = sum[k] / count[k]
        ii += unweightedcount[k]

    return y_, MSE_
