import numpy as np


class RandomKNearestNeighborsHash:

    def compute_hashcode_bit(self, K, subset1, subset2):

        n = K.shape[0]

        # computing hash vector
        k1 = K[:, subset1].max(1)
        k2 = K[:, subset2].max(1)
        z = np.zeros(n, dtype=np.bool)
        z[k1 <= k2] = 1

        return z
