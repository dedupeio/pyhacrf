from collections import defaultdict
import numpy as np
cimport numpy as np
from numpy.math cimport logaddexp

cpdef dict forward(list lattice, np.ndarray[double, ndim=3] x_dot_parameters):
    """ Helper to calculate the forward weights.  """
    cdef dict alpha = {}

    cdef int i,j,s,i0,j0,s0,i1,j1,s1,edge_parameter_index
    cdef float edge_potential
    cdef tuple node

    for node in lattice:
        if len(node) == 3:
            i, j, s = node
            if i == 0 and j == 0:
                alpha[(i, j, s)] = x_dot_parameters[i, j, s]
            else:
                alpha[(i, j, s)] += x_dot_parameters[i, j, s]
        else:
            i0, j0, s0, i1, j1, s1, edge_parameter_index = node  # Actually an edge in this case
            # Use the features at the destination of the edge.
            edge_potential = (x_dot_parameters[i1, j1, edge_parameter_index]
                              + alpha[(i0, j0, s0)])
            alpha[node] = edge_potential
            alpha[(i1, j1, s1)] = logaddexp(alpha.get((i1, j1, s1), -np.inf), 
                                            edge_potential)
    return alpha


def backward(lattice, x_dot_parameters, I, J):
    """ Helper to calculate the backward weights.  """
    beta = defaultdict(lambda: -np.inf)
    for node in reversed(lattice):
        if len(node) == 3:
            i, j, s = node
            if i == I - 1 and j == J - 1:
                beta[node] = 0.0
        else:
            i0, j0, s0, i1, j1, s1, edge_parameter_index = node  # Actually an edge in this case
            # Use the features at the destination of the edge.
            edge_potential = beta[(i1, j1, s1)] + (x_dot_parameters[i1, j1, s1])
            beta[node] = edge_potential
            beta[(i0, j0, s0)] = np.logaddexp(beta[(i0, j0, s0)],
                                              edge_potential + x_dot_parameters[i1, j1, edge_parameter_index])
    return beta