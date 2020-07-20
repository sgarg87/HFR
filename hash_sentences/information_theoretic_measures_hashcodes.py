from . import total_correlations_estimation
import numpy as np
import time


class InformationTheoreticMeasuresHashcodes:

    def __init__(self):
        pass

    def compute_marginal_entropy(self, hash_vector):

        # start_time = time.time()
        assert len(hash_vector.shape) == 1
        pos_ratio = hash_vector.mean()
        if (pos_ratio == 0.0) or (pos_ratio == 1.0):
            entropy = 0.0
        else:
            entropy = -(pos_ratio * np.log(pos_ratio)) - ((1.0 - pos_ratio) * np.log((1.0 - pos_ratio)))
        # print 'H(x): {}'.format(time.time() - start_time)

        return entropy

    def compute_marginal_entropy_non_binary(self, x):

        # start_time = time.time()
        assert len(x.shape) == 1

        entropy = 0.0
        unique_x = np.unique(x)
        for curr_val in unique_x:
            idx = np.where(x == curr_val)[0]
            fraction = float(idx.size)/x.size
            if fraction > 0.0:
                entropy += -(fraction * np.log(fraction))
        # print 'H(x): {}'.format(time.time() - start_time)

        return entropy

    def compute_conditional_entropy_x_cond_z(self, x, z):

        # start_time = time.time()
        assert x.shape == z.shape
        Pz1 = z.mean()
        Pz0 = 1.0 - Pz1

        if Pz0 == 0.0:
            Hx_cond_z1 = self.compute_marginal_entropy(x[z == 1])
            # print 'Hx_cond_z1', Hx_cond_z1
            Hx_cond_z = Pz1 * Hx_cond_z1
        elif Pz0 == 1.0:
            Hx_cond_z0 = self.compute_marginal_entropy(x[z == 0])
            # print 'Hx_cond_z0', Hx_cond_z0
            Hx_cond_z = Pz0 * Hx_cond_z0
        else:
            Hx_cond_z0 = self.compute_marginal_entropy(x[z == 0])
            # print 'Hx_cond_z0', Hx_cond_z0
            Hx_cond_z1 = self.compute_marginal_entropy(x[z == 1])
            # print 'Hx_cond_z1', Hx_cond_z1
            Hx_cond_z = Pz0*Hx_cond_z0 + Pz1*Hx_cond_z1
        # print 'H(x|z): {}'.format(time.time() - start_time)

        return Hx_cond_z

    def compute_conditional_entropy_x_cond_z_non_binary(self, x, z):

        # start_time = time.time()
        assert x.shape == z.shape
        assert len(x.shape) == 1
        num_data = x.size

        Hx_cond_z = 0.0
        for curr_z_val in np.unique(z):
            curr_idx = np.where(z == curr_z_val)[0]
            curr_prob = float(curr_idx.size)/num_data
            if curr_prob > 0.0:
                curr_Hx_cond_z = self.compute_marginal_entropy_non_binary(x[curr_idx])
                Hx_cond_z += curr_prob * curr_Hx_cond_z

        # print 'H(x|z): {}'.format(time.time() - start_time)

        return Hx_cond_z

    def compute_conditional_entropy_z_cond_C(self, z, C):

        # start_time = time.time()

        Cu, Hz_per_C, n_per_C = self.compute_entropy_z_cond_C(z, C)

        # print 'Hz_per_C', Hz_per_C

        Hz_cond_C = Hz_per_C.dot(n_per_C)/z.size

        # print 'H(z|C): {}'.format(time.time() - start_time)

        return Hz_cond_C

    def compute_entropy_z_cond_C(self,
                                 z,
                                 C,
                                 is_z_binary=True,
                                 is_return_Cu_idx_in_original_arr=False,
                            ):

        # by default, z is assumed to be binary
        # else, z values should be 0, 1, 2

        # start_time_unique = time.time()
        Cu, Cu_idx_in_original_arr = np.unique(C, axis=0, return_inverse=True)
        # print 'Cu.shape', Cu.shape
        # print 'Uq: {}'.format(time.time() - start_time_unique)

        num_clusters = Cu.shape[0]

        Hz_per_C = np.zeros(num_clusters)
        n_per_C = np.zeros(num_clusters, dtype=np.int)

        DiC = 0.0

        for curr_idx in range(num_clusters):
            start_time_data_in_cluster = time.time()

            data_idx_fr_cluster = np.where(Cu_idx_in_original_arr == curr_idx)[0]

            DiC += time.time() - start_time_data_in_cluster

            if data_idx_fr_cluster.size != 0:

                z_sel = z[data_idx_fr_cluster]
                if is_z_binary:
                    curr_Hz_per_C = self.compute_marginal_entropy(z_sel)
                else:
                    curr_Hz_per_C = self.compute_marginal_entropy_non_binary(z_sel)
                del z_sel

                Hz_per_C[curr_idx] = curr_Hz_per_C
                del curr_Hz_per_C

                n_per_C[curr_idx] = data_idx_fr_cluster.size

        # print 'DiC: {}'.format(DiC)

        if is_return_Cu_idx_in_original_arr:
            return Cu, Hz_per_C, n_per_C, Cu_idx_in_original_arr
        else:
            return Cu, Hz_per_C, n_per_C

    def count_elements_in_clusters(self, C):

        Cu, Cu_idx_in_original_arr = np.unique(C, axis=0, return_inverse=True)
        num_clusters = Cu.shape[0]
        n_per_C = np.zeros(num_clusters, dtype=np.int)
        DiC = 0.0

        for curr_idx in range(num_clusters):
            start_time_data_in_cluster = time.time()
            data_idx_fr_cluster = np.where(Cu_idx_in_original_arr == curr_idx)[0]
            DiC += time.time() - start_time_data_in_cluster
            if data_idx_fr_cluster.size != 0:
                n_per_C[curr_idx] = data_idx_fr_cluster.size

        # print 'DiC: {}'.format(DiC)

        return Cu, n_per_C

    def compute_joint_entropy_approximation(self,
                                            hashcodes,
                                            num_hidden_var=10,
                                            max_num_iter=100,
                                        ):

        # start_time = time.time()

        # computing marginal entropies
        eps = 1e-10
        hashcodes_mean = hashcodes.mean(0)
        hashcodes_mean[np.where(hashcodes_mean == 0)] += eps
        hashcodes_mean[np.where(hashcodes_mean == 1)] -= eps

        curr_e = -(hashcodes_mean * np.log(hashcodes_mean)) - ((1 - hashcodes_mean) * np.log((1 - hashcodes_mean)))
        curr_e_sum = curr_e.sum()

        joint_entropy_approx = curr_e_sum

        layer1 = total_correlations_estimation.TotalCorrelationsEstimation(n_hidden=num_hidden_var,
                                                                           seed=0,
                                                                           max_iter=max_num_iter)
        layer1.fit(hashcodes)
        # print layer1.tcs
        tc_lb = layer1.tcs.sum()

        joint_entropy_approx -= tc_lb

        # print 'H(C): {}'.format(time.time() - start_time)

        return joint_entropy_approx

