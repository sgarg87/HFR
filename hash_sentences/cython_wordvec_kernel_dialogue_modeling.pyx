from cpython.exc cimport PyErr_CheckSignals


cdef sparse_func(float k, float threshold):
    cdef float k_sparse
    k_sparse = 1-((1-k)/(1-threshold))
    if k_sparse < 0.0:
        k_sparse = 0.0
    return k_sparse


cpdef compute_kernel_wordvec(float[:] wordvec1,
                             float[:] wordvec2,
                             float cs_threshold
                           ):
    #
    cdef int wordvec_size = wordvec1.size
    cdef int i
    cdef float cs=0.0
    for i in xrange(wordvec_size):
        cs += wordvec1[i]*wordvec2[i]
    #
    cdef float kij, kij_sparse
#    kij = exp(cs-1.0)
    kij = cs
    kij_sparse = sparse_func(cs, cs_threshold)
    kij *= kij_sparse
    #
    return kij
