from cpython.exc cimport PyErr_CheckSignals


def compute_subsequence_kernel_from_word_similarities(
                                                            int num_terms_1,
                                                            int num_terms_2,
                                                            int p,
                                                            float lamb_sqr,
                                                            float lamb,
                                                            float[:,:] dps,
                                                            float[:,:] dp,
                                                            float[:] k,
                                                            int[:,:] matches,
                                                            double[:] p_weights,
                                                  ):
    #
    cdef int i, j
    assert k[1] == 0
    for i in xrange(num_terms_1):
        for j in xrange(num_terms_2):
            k[1] += lamb*dps[i,j]
    #
    cdef int l
    for l in xrange(2, p+1):
        for i in xrange(num_terms_1):
            for j in xrange(num_terms_2):
                dp[i+1, j+1] = dps[i,j] + lamb*dp[i,j+1] + lamb*dp[i+1,j] - lamb_sqr*dp[i,j]
                if matches[i,j] == 1:
                    dps[i,j] = lamb_sqr*dp[i,j]
                    k[l] = k[l] + dps[i,j]
                else:
                    assert matches[i,j] == 0
    #
    cdef double k_sum = 0.0
    assert p_weights.size == (p+1)
    assert k.size == (p+1)
    for l in xrange(p+1):
        k_sum += k[l]*p_weights[l]
    #
    return k_sum
