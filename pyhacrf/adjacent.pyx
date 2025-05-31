#cython: boundscheck=False, wraparound=False, initializedcheck=False
#cython: infer_types=True

import numpy as np
cimport numpy as np
from numpy import ndarray
from numpy cimport ndarray
from libc.math cimport INFINITY as inf
from libc.math cimport exp
cdef extern from "log1p.h" nogil:
    np.float64_t log1p(np.float64_t x)

cdef np.float64_t LOG_2 = 0.6931471805599453
cdef np.float64_t LOG_3 = 1.0986122886681098

cpdef np.ndarray[np.float64_t, ndim=3] forward(np.float64_t[:, :, :] x_dot_parameters, int S):
    """ Helper to calculate the forward weights.  """
    cdef np.ndarray[np.float64_t, ndim=3] alpha = np.full_like(x_dot_parameters, -np.inf)

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

            alpha[i, 0, insertion + s] = insert
        for j in range(1, J):
            delete = (alpha[0, j - 1, s] +
                      x_dot_parameters[0, j, deletion + s])
            alpha[0, j, s] = x_dot_parameters[0, j, s] + delete

            alpha[0, j, deletion + s] = delete
        
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
                                  logsumexp(insert, delete, match))

                alpha[i, j, insertion + s] = insert
                alpha[i, j, deletion + s] = delete
                alpha[i, j, matching + s] = match

    return alpha

cpdef np.ndarray[np.float64_t, ndim=3] backward(np.ndarray[np.float64_t, ndim=3] x_dot_parameters, int S):
    """ Helper to calculate the forward weights.  """
    cdef np.ndarray[np.float64_t, ndim=3] beta = np.full_like(x_dot_parameters, -np.inf)

    cdef int I, J
    I, J = beta.shape[0], beta.shape[1]

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

            beta[i + 1, last_col, insertion + s] = insert
        for j in range(last_col - 1, -1, -1):
            delete = (beta[last_row, j + 1, s] +
                      x_dot_parameters[last_row, j + 1, s])
            beta[last_row, j, s] = (x_dot_parameters[last_row, j + 1, deletion + s]
                                    + delete)

            beta[last_row, j + 1, deletion + s] = delete
        
        # Now fill in the middle of the matrix    
        for i in range(last_row - 1, -1, -1):
            for j in range(last_col - 1, -1, -1):
                insert = (beta[i + 1, j, s] +
                          x_dot_parameters[i + 1, j, s])
                delete = (beta[i, j + 1, s] +
                          x_dot_parameters[i, j + 1, s])
                match = (beta[i + 1, j + 1, s] +
                         x_dot_parameters[i + 1, j + 1, s])

                beta[i + 1, j, insertion + s] = insert
                beta[i, j + 1, deletion + s] = delete
                beta[i + 1, j + 1, matching + s] = match

                insert += x_dot_parameters[i + 1, j, insertion + s]
                delete += x_dot_parameters[i, j + 1, deletion + s]
                match += x_dot_parameters[i + 1, j + 1, matching + s]
                
                beta[i, j, s] = logsumexp(insert, delete, match)

    return beta

cpdef np.float64_t[::1] forward_predict(np.float64_t[:, :, :] x_dot_parameters,
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
        for j in range(1, J):
            for i in range(1, I):
                insert = (alpha[i - 1, j, s] +
                          x_dot_parameters[i, j, insertion + s])
                delete = (alpha[i, j - 1, s] +
                          x_dot_parameters[i, j, deletion + s])
                match = (alpha[i - 1, j - 1, s] +
                         x_dot_parameters[i, j, matching + s])
                alpha[i, j, s] = (x_dot_parameters[i, j, s] +
                                  logsumexp(insert, delete, match))


    cdef np.float64_t[::1] final_alphas = alpha[I - 1, J - 1, :S]

    cdef np.float64_t Z = -inf
    for s in range(S):
        Z = logaddexp(Z, final_alphas[s])

    for s in range(S):
        final_alphas[s] = exp(final_alphas[s] - Z)

    return final_alphas

cdef np.float64_t logaddexp(np.float64_t x, np.float64_t y) nogil:
    cdef np.float64_t tmp
    if x == y :
        return x + LOG_2
    else :
        tmp = x - y
        if tmp > 0 :
            return x + log1p(exp(-tmp))
        elif tmp <= 0 :
            return y + log1p(exp(tmp))
        else :
            return tmp

cdef np.float64_t logsumexp(np.float64_t x, np.float64_t y, np.float64_t z) nogil :
    if x == y == z:
        return x + LOG_3
    elif x > y > z or x > z > y:
        return x + log1p(exp(y - x) + exp(z - x))
    elif x > y == z:
        return x + log1p(exp(y - x) * 2)
    elif x == y > z:
        return x + LOG_2 + log1p(exp(z - x - LOG_2))
    elif x == z > y:
        return x + LOG_2 + log1p(exp(y - x - LOG_2))
    elif y > x > z or y > z > x:
        return y + log1p(exp(x - y) + exp(z - y))
    elif y > x == z:
        return y + log1p(exp(x - y) * 2)
    elif y == z > x:
        return y + LOG_2 + log1p(exp(x - y - LOG_2))
    elif z > x > y or z > y > x:
        return z + log1p(exp(x - z) + exp(y - z))
    elif z > x == y:
        return z + log1p(exp(x - z) * 2)
