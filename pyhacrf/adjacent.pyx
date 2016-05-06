#cython: boundscheck=False, wraparound=False, initializedcheck=False
#cython: infer_types=True

import numpy as np
cimport numpy as np
from numpy import ndarray
from numpy cimport ndarray
from numpy.math cimport INFINITY as inf
cdef extern from "math.h" nogil :
    np.float64_t log (np.float64_t x)
    np.float64_t exp (np.float64_t x)

cpdef dict forward(np.ndarray[np.float64_t, ndim=3] x_dot_parameters, int S):
    """ Helper to calculate the forward weights.  """
    cdef dict alpha = {}

    cdef int I, J
    I, J = x_dot_parameters.shape[0], x_dot_parameters.shape[1]

    # Fill in the edges of the state matrices
    #
    #   0 1 2 3 
    # 0 x x x x
    # 1 x - - -
    # 2 x - - -
    # 3 x - - -
    cdef int matching =  1 * S
    cdef int deletion =  2 * S
    cdef int insertion = 3 * S
    cdef np.float64_t insert, delete, match
    cdef int i, j, s 

    for s in range(S):
        alpha[0, 0, s] = x_dot_parameters[0, 0, s]
        for i in range(1, I):
            insert = (alpha[i - 1, 0, s] +
                      x_dot_parameters[i, 0, insertion + s])
            alpha[i, 0, s] = x_dot_parameters[i, 0, s] + insert

            alpha[i - 1, 0, s, i, 0, s, insertion + s] = insert
        for j in range(1, J):
            delete = (alpha[0, j - 1, s] +
                      x_dot_parameters[0, j, deletion + s])
            alpha[0, j, s] = x_dot_parameters[0, j, s] + delete

            alpha[0, j - 1, s, 0, j, s, deletion + s] = delete
        
        # Now fill in the middle of the matrix    
        for i in range(1, I):
            for j in range(1, J):
                insert = (alpha[i - 1, j, s] +
                          x_dot_parameters[i, j, insertion + s])
                delete = (alpha[i, j - 1, s] +
                          x_dot_parameters[i, j, deletion + s])
                match = (alpha[i - 1, j - 1, s] +
                         x_dot_parameters[i, j, matching + s])
                alpha[i, j, s] = (x_dot_parameters[i, j, s] +
                                  logaddexp(insert, logaddexp(delete, match)))

                alpha[i - 1, j, s, i, j, s, insertion + s] = insert
                alpha[i, j - 1, s, i, j, s, deletion + s] = delete
                alpha[i - 1, j - 1, s, i, j, s, matching + s] = match

    return alpha

cpdef dict backward(np.ndarray[np.float64_t, ndim=3] x_dot_parameters, int S):
    """ Helper to calculate the forward weights.  """
    cdef dict beta = {}

    cdef int I, J
    I, J = x_dot_parameters.shape[0], x_dot_parameters.shape[1]

    # Fill in the edges of the state matrices
    #
    #   0 1 2 3 
    # 0 - - - x
    # 1 - - - x
    # 2 - - - x
    # 3 x x x x
    cdef int matching =  1 * S
    cdef int deletion =  2 * S
    cdef int insertion = 3 * S
    cdef np.float64_t insert, delete, match
    cdef int i, j, s
    cdef int last_row = I - 1
    cdef int last_col = J - 1

    for s in range(S):
        beta[last_row, last_col, s] = 0
        for i in range(last_row - 1, -1, -1):
            insert = (beta[i + 1, last_col, s] +
                      x_dot_parameters[i + 1, last_col, s])
            beta[i, last_col, s] = (x_dot_parameters[i + 1, last_col, insertion + s]
                                    + insert)

            beta[i, last_col, s, i + 1, last_col, s, insertion + s] = insert
        for j in range(last_col - 1, -1, -1):
            delete = (beta[last_row, j + 1, s] +
                      x_dot_parameters[last_row, j + 1, s])
            beta[last_row, j, s] = (x_dot_parameters[last_row, j + 1, deletion + s]
                                    + delete)

            beta[last_row, j, s, last_row, j + 1, s, deletion + s] = delete
        
        # Now fill in the middle of the matrix    
        for i in range(last_row - 1, -1, -1):
            for j in range(last_col - 1, -1, -1):
                insert = (beta[i + 1, j, s] +
                          x_dot_parameters[i + 1, j, s])
                delete = (beta[i, j + 1, s] +
                          x_dot_parameters[i, j + 1, s])
                match = (beta[i + 1, j + 1, s] +
                         x_dot_parameters[i + 1, j + 1, s])

                beta[i, j, s, i + 1, j, s, insertion + s] = insert
                beta[i, j, s, i, j + 1, s, deletion + s] = delete
                beta[i, j, s, i + 1, j + 1, s, matching + s] = match

                insert += x_dot_parameters[i + 1, j, insertion + s]
                delete += x_dot_parameters[i, j + 1, deletion + s]
                match += x_dot_parameters[i + 1, j + 1, matching + s]
                
                beta[i, j, s] = logaddexp(insert, logaddexp(delete, match))



    return beta



cpdef np.float64_t[:, :, ::1] forward_predict(np.float64_t[:, :, ::1] x_dot_parameters,
                                              int S) :
    cdef np.float64_t[:, :, ::1] alpha = x_dot_parameters.copy()

    cdef int I, J
    I, J = alpha.shape[0], alpha.shape[1]

    # Fill in the edges of the state matrices
    #
    #   0 1 2 3 
    # 0 x x x x
    # 1 x - - -
    # 2 x - - -
    # 3 x - - -
    cdef int matching =  1 * S
    cdef int deletion =  2 * S
    cdef int insertion = 3 * S
    cdef np.float64_t insert, delete, match
    cdef int i, j, s
    
    for s in range(S):
        alpha[0, 0, s] = x_dot_parameters[0, 0, s]
        for i in range(1, I):
            insert = (alpha[i - 1, 0, s] +
                      x_dot_parameters[i, 0, insertion + s])
            alpha[i, 0, s] = x_dot_parameters[i, 0, s] + insert
        for j in range(1, J):
            delete = (alpha[0, j - 1, s] +
                      x_dot_parameters[0, j, deletion + s])
            alpha[0, j, s] = x_dot_parameters[0, j, s] + delete
        
        # Now fill in the middle of the matrix    
        for i in range(1, I):
            for j in range(1, J):
                insert = (alpha[i - 1, j, s] +
                          x_dot_parameters[i, j, insertion + s])
                delete = (alpha[i, j - 1, s] +
                          x_dot_parameters[i, j, deletion + s])
                match = (alpha[i - 1, j - 1, s] +
                         x_dot_parameters[i, j, matching + s])
                alpha[i, j, s] = (x_dot_parameters[i, j, s] +
                                  logaddexp(insert, logaddexp(delete, match)))

    return alpha

cdef np.float64_t logaddexp(np.float64_t x, np.float64_t y) nogil:
    cdef np.float64_t tmp
    if x == y :
        return x + log(2)
    else :
        tmp = x - y
        if tmp > 0 :
            return x + log(1 + exp(-tmp))
        elif tmp <= 0 :
            return y + log(1 + exp(tmp))
        else :
            return tmp
    
