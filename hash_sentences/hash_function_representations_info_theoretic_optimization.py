import time
import numpy as np
from . import information_theoretic_measures_hashcodes as itmh
import scipy.special as scipy_special
import itertools
import numpy.random as npr
from . import parallel_computing as pc
from . import parallel_computing_wrapper as pcw
import math
from . import random_k_nearest_neighbors_hash as rknn
from . import random_maximum_margin_hash as rmm
from . import random_neural_hash as rneural
from .divide_conquer_optimize_split import DivideConquerOptimizeSplit
from . import embed_path_tuples as ept
import pickle


file_path = 'hash_function_representations'


class HashFunctionRepresentations:

    # Implemented for R1NN currently as that is the most efficient hashing
    # when optimizing the split of a subset (artificial labels) to construct a hash function.

    def __init__(
         self,
         hash_func,
         data_operations_obj,
         is_semisupervised_else_unsupervised,
         is_alpha_sampling_weighted=False,
         max_num_hash_functions_for_sampling_from_cluster=10,
         num_cores=1,
         window_size=10,
         is_delete_old_hash=True,
         is_conditional_birth=False,
         is_opt_reg_rmm=False,
         is_opt_kernel_par=False,
         is_rnd_sample_combinations_for_opt_kernel_par=False,
         num_sample_combinations=30,
         frac_trn_data=0.1,
         is_inductive_ssh=False,
         num_bits_infer_cluster=16,
         is_convolution_neural=False,
         is_images=False,
         # labeled_score_frac=0.3,
         labeled_score_frac=1.0,
         is_joint_inductive_transductive_ssh=False,
         is_weighted_transductive=False,
         seed_val=0,
         default_reg_val_rmm=1.0,
         rmm_max_iter=100,
         is_text_display=False,
         is_infer_cluster=False,
         unlabeled_induction_sampling_ratio=1.0,
         # recently added parameter (False value as per old code)
         is_subset_bits_for_objective_scores=True,
         deletion_z=1.28,
         is_optimize_split=True,
         is_divide_conquer_else_brute_force=False,
         min_set_size_to_split=4,
         is_optimize_hash_functions_nonredundancy=True,
         nonredundancy_score_beta=1.0,
         is_random_sample_data_for_computing_objective=False,
         rnd_sample_obj_size_log_base=2.0,
         is_edge_label=False,
         is_cluster_high_weightage=True,
         cluster_sample_weights=10,
         is_zero_kernel_compute_outside_cluster=False,
         is_alpha_per_cluster_size=False,
         alpha_compute_per_cluster_size__log_scale=3.0,
         is_cluster_size_and_entropy=False,
         ref_set_applicable_for_max_num_bits_to_cluster=None,
         max_hamming_dist_bits_frac_for_neighborhood=0.1,
         is_bert=False,
         wordvec_file_path=None,
         is_self_dialog_learning=False,
         is_num_hash_bits_for_cluster_dynamic=False,
    ):

        self.is_text_display = is_text_display
        self.is_infer_cluster = is_infer_cluster
        self.unlabeled_induction_sampling_ratio = unlabeled_induction_sampling_ratio

        self.max_num_hash_functions_for_sampling_from_cluster = max_num_hash_functions_for_sampling_from_cluster
        self.data_operations_obj = data_operations_obj
        self.hash_func = hash_func
        self.is_semisupervised_else_unsupervised = is_semisupervised_else_unsupervised
        self.is_inductive_ssh = is_inductive_ssh
        self.is_joint_inductive_transductive_ssh = is_joint_inductive_transductive_ssh
        self.is_weighted_transductive = is_weighted_transductive

        self.labeled_score_frac = labeled_score_frac

        self.num_bits_infer_cluster = num_bits_infer_cluster

        self.window_size = window_size

        self.num_cores = num_cores
        self.is_alpha_sampling_weighted = is_alpha_sampling_weighted

        self.itmh_obj = itmh.InformationTheoreticMeasuresHashcodes()

        self.npr_sample_alpha_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_nbhec_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_subset_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_hash_bits_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_hash_bits_infer_cluster_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_prob_deletion_old_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_combinations_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_data_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_unlabeled_data_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_cluster_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_split_obj = npr.RandomState(seed=seed_val)
        self.npr_sample_sel_compute_obj = npr.RandomState(seed=seed_val)
        self.npr_bits_subset_size_reduction = npr.RandomState(seed=seed_val)

        self.is_delete_old_hash = is_delete_old_hash
        self.is_conditional_birth = is_conditional_birth

        self.is_opt_reg_rmm = is_opt_reg_rmm
        self.is_opt_kernel_par = is_opt_kernel_par

        self.is_rnd_sample_combinations_for_opt_kernel_par = is_rnd_sample_combinations_for_opt_kernel_par
        self.num_sampled_combinations = num_sample_combinations
        self.frac_trn_data = frac_trn_data

        self.is_rnd_sample_data_for_opt_kernel_par = True

        self.word_vec_len = 30
        self.is_convolution_neural = is_convolution_neural
        self.num_words_convolution = 2
        self.max_epochs = 300

        self.default_reg_val_rmm = default_reg_val_rmm
        self.rmm_max_iter = rmm_max_iter

        self.is_subset_bits_for_objective_scores = is_subset_bits_for_objective_scores

        self.is_images = is_images

        self.deletion_z = deletion_z

        self.is_optimize_split = is_optimize_split

        self.is_divide_conquer_else_brute_force = is_divide_conquer_else_brute_force
        self.min_set_size_to_split = min_set_size_to_split

        self.is_optimize_hash_functions_nonredundancy = is_optimize_hash_functions_nonredundancy
        self.nonredundancy_score_beta = nonredundancy_score_beta

        self.is_random_sample_data_for_computing_objective = is_random_sample_data_for_computing_objective
        self.rnd_sample_obj_size_log_base = rnd_sample_obj_size_log_base

        self.wordvec_file_path = wordvec_file_path

        self.is_cluster_high_weightage = is_cluster_high_weightage
        assert isinstance(cluster_sample_weights, int)
        self.cluster_sample_weights = cluster_sample_weights

        self.is_self_dialog_learning = is_self_dialog_learning
        if is_self_dialog_learning:
            assert not self.is_random_sample_data_for_computing_objective
            assert not self.is_opt_kernel_par
            assert not self.is_cluster_high_weightage
        self.is_num_hash_bits_for_cluster_dynamic = is_num_hash_bits_for_cluster_dynamic

        assert not self.is_conditional_birth

        self.ept_obj = ept.EmbedPathTuples(
            is_word_only=(not is_edge_label),
            word_vec_len=self.word_vec_len,
            wordvec_file_path=self.wordvec_file_path,
            is_random=True,
        )

        self.is_bert = is_bert
        if self.is_bert:
            self.bs_obj = bs.BERTEmbedSentences()

        self.is_zero_kernel_compute_outside_cluster_and_neighboring = is_zero_kernel_compute_outside_cluster
        self.is_alpha_per_cluster_size = is_alpha_per_cluster_size
        self.alpha_compute_per_cluster_size__log_scale = alpha_compute_per_cluster_size__log_scale

        self.is_cluster_size_and_entropy = is_cluster_size_and_entropy

        self.ref_set_applicable_for_max_num_bits_to_cluster = ref_set_applicable_for_max_num_bits_to_cluster
        self.max_hamming_dist_bits_frac_for_neighborhood = max_hamming_dist_bits_frac_for_neighborhood

        if self.is_images:
            # assert self.hash_func == 'RNeural'
            # self.rneural_obj = rneural_images.RandomNeuralHash()
            raise NotImplementedError
        else:
            self.rknn_obj = rknn.RandomKNearestNeighborsHash()
            self.rmm_obj = rmm.RandomMaximumMarginHash()

            if self.hash_func == 'RNeural':
                if self.is_convolution_neural:
                    self.rneural_obj = rneural.RandomNeuralHash(
                        method='lstm-cnn',
                        word_embedding_len=self.word_vec_len*self.num_words_convolution,
                    )
                else:
                    self.rneural_obj = rneural.RandomNeuralHash(
                        method='lstm',
                        word_embedding_len=self.word_vec_len,
                    )

    def compute_self_dialog_mi(self, C, z):
        n = C.shape[0]

        if len(C.shape) == 1:
            C = C.reshape(C.shape[0], 1)
            assert len(C.shape) == 2

        utterance_codes = C[:n-1, :]
        responder_codes = C[1:n, :]

        utterance_z = z[:n-1]
        responder_z = z[1:n]

        utterance_codes_z = np.hstack((
            utterance_codes,
            utterance_z.reshape(utterance_z.size, 1),
        ))
        del utterance_codes

        responder_codes_z = np.hstack((
            responder_codes,
            responder_z.reshape(responder_z.size, 1),
        ))
        del responder_codes

        Hz_utterance__cond__C_responder = self.itmh_obj.compute_conditional_entropy_z_cond_C(
            z=utterance_z, C=responder_codes_z,
        )

        Hz_responder__cond__C_utterance = self.itmh_obj.compute_conditional_entropy_z_cond_C(
            z=responder_z, C=utterance_codes_z,
        )

        # not including marginal entropy of z, accounted else where in the overall objective
        mi = -Hz_utterance__cond__C_responder - Hz_responder__cond__C_utterance

        return mi

    def compute_score_for_reference_subset_partition(self,
                                                     C, z, x, y,
                                                     data_idx_from_cluster,
                                                     sample_counts=None):

        start_time_score_compute = time.time()
        # print '...............................'

        if sample_counts is not None:
            assert data_idx_from_cluster is None

            assert sample_counts.dtype == np.int
            total_sample_counts = sample_counts.sum()
            print('total_sample_counts', total_sample_counts)

            if C is not None:
                print('................')
                print('C.shape', C.shape)
                if len(C.shape) == 1:
                    org_num_bits = 1
                else:
                    assert len(C.shape) == 2
                    org_num_bits = C.shape[1]
                C = np.repeat(C, sample_counts, axis=0)
                print('C.shape', C.shape)
                assert C.shape[0] == total_sample_counts
                if len(C.shape) == 1:
                    assert org_num_bits == 1
                else:
                    assert C.shape[1] == org_num_bits

            assert z is not None
            # print 'z.shape', z.shape
            z = np.repeat(z, sample_counts, axis=0)
            # print 'z.shape', z.shape
            assert len(z.shape) == 1
            assert z.shape[0] == total_sample_counts

            if x is not None:
                # print 'x.shape', x.shape
                x = np.repeat(x, sample_counts, axis=0)
                # print 'x.shape', x.shape
                assert len(x.shape) == 1
                assert x.shape[0] == total_sample_counts

            if y is not None:
                # print 'y.shape', y.shape
                y = np.repeat(y, sample_counts, axis=0)
                # print 'y.shape', y.shape
                assert len(y.shape) == 1
                assert y.shape[0] == total_sample_counts

            del total_sample_counts, sample_counts

        if self.is_inductive_ssh:
            if self.is_joint_inductive_transductive_ssh:
                # test data
                weight_0 = 0.33
                # training labeled data
                weight_1 = 0.33
                # training unlabeled data
                weight_2 = 0.33

                # test and training data
                weight_01 = 0.33
                # test and unlabeled data
                weight_02 = 0.33
                # training labeled and unlabeled data
                weight_12 = 0.33
            else:
                # training labeled data
                weight_1 = 0.33
                # training unlabeled data
                weight_2 = 0.33
                # training labeled and unlabeled data
                weight_12 = 0.33

            if self.is_joint_inductive_transductive_ssh:
                # test data
                idx_0 = np.where(x == 0)[0]
                # train labeled data
                idx_1 = np.where(x == 1)[0]
                # train unlabeled data
                idx_2 = np.where(x == 2)[0]

                # test and training data
                idx_01 = np.where((x == 0) | (x == 1))[0]
                # test and unlabeled data
                idx_02 = np.where((x == 0) | (x == 2))[0]
                # training labeled and unlabeled data
                idx_12 = np.where((x == 1) | (x == 2))[0]
            else:
                # train labeled data
                idx_1 = np.where(x == 1)[0]
                # train unlabeled data
                idx_2 = np.where(x == 2)[0]
                # training labeled and unlabeled data
                idx_12 = np.where((x == 1) | (x == 2))[0]
        else:
            if self.is_weighted_transductive:
                assert not self.is_self_dialog_learning

                # test data
                weight_0 = 0.8

                # training labeled data
                weight_1 = 0.2

                # test data
                idx_0 = np.where(x == 0)[0]
                # train labeled data
                idx_1 = np.where(x == 1)[0]

        if self.is_self_dialog_learning:
            self_dialog_mi = self.compute_self_dialog_mi(C=C, z=z)
            assert self_dialog_mi is not None
        else:
            self_dialog_mi = 0.0
        print('self_dialog_mi', self_dialog_mi)

        # computing H(z)
        if self.is_inductive_ssh:
            Hz = 0.0

            Hz_1 = self.itmh_obj.compute_marginal_entropy(z[idx_1])
            Hz += weight_1 * Hz_1

            Hz_2 = self.itmh_obj.compute_marginal_entropy(z[idx_2])
            Hz += weight_2 * Hz_2

            if self.is_joint_inductive_transductive_ssh:
                Hz_0 = self.itmh_obj.compute_marginal_entropy(z[idx_0])
                Hz += weight_0*Hz_0

                print('Hz_0: {}, Hz_1: {}, Hz_2: {}, Hz: {}'.format(
                    Hz_0, Hz_1, Hz_2, Hz
                ))
            else:
                print('Hz_1: {}, Hz_2: {}, Hz: {}'.format(
                    Hz_1, Hz_2, Hz
                ))
        else:
            if self.is_weighted_transductive:
                Hz = 0.0

                Hz_0 = self.itmh_obj.compute_marginal_entropy(z[idx_0])
                Hz += weight_0*Hz_0

                Hz_1 = self.itmh_obj.compute_marginal_entropy(z[idx_1])
                Hz += weight_1 * Hz_1

                print('Hz_0: {}, Hz_1: {}, Hz: {}'.format(
                    Hz_0, Hz_1, Hz))
            else:
                Hz = self.itmh_obj.compute_marginal_entropy(z)
        # print 'Hz', Hz

        # computing H(z|C)
        if (C is not None) and (self.is_optimize_hash_functions_nonredundancy):
            if len(C.shape) == 2:
                if data_idx_from_cluster is not None:
                    raise(NotImplemented, 'code deprecated, never used while changing other parts of code so many times')
                    assert not self.is_inductive_ssh
                    Hz_cond_C = self.itmh_obj.compute_conditional_entropy_z_cond_C(
                        z=z[data_idx_from_cluster],
                        C=C[data_idx_from_cluster]
                    )
                else:
                    if self.is_inductive_ssh:
                        Hz_cond_C = 0.0

                        Hz_cond_C_1 = self.itmh_obj.compute_conditional_entropy_z_cond_C(
                            z=z[idx_1],
                            C=C[idx_1, :]
                        )
                        Hz_cond_C += weight_1*Hz_cond_C_1

                        Hz_cond_C_2 = self.itmh_obj.compute_conditional_entropy_z_cond_C(
                            z=z[idx_2],
                            C=C[idx_2, :]
                        )
                        Hz_cond_C += weight_2*Hz_cond_C_2

                        if self.is_joint_inductive_transductive_ssh:
                            Hz_cond_C_0 = self.itmh_obj.compute_conditional_entropy_z_cond_C(
                                z=z[idx_0],
                                C=C[idx_0, :],
                            )
                            Hz_cond_C += weight_0*Hz_cond_C_0

                            print('Hz_cond_C_0: {}, Hz_cond_C_1: {}, Hz_cond_C_2: {}, Hz_cond_C: {}'.format(
                                Hz_cond_C_0, Hz_cond_C_1, Hz_cond_C_2, Hz_cond_C
                            ))
                        else:
                            print('Hz_cond_C_1: {}, Hz_cond_C_2: {}, Hz_cond_C: {}'.format(
                                Hz_cond_C_1, Hz_cond_C_2, Hz_cond_C
                            ))
                    else:
                        if self.is_weighted_transductive:
                            Hz_cond_C = 0.0

                            Hz_cond_C_0 = self.itmh_obj.compute_conditional_entropy_z_cond_C(
                                z=z[idx_0],
                                C=C[idx_0, :],
                            )
                            Hz_cond_C += weight_0 * Hz_cond_C_0

                            Hz_cond_C_1 = self.itmh_obj.compute_conditional_entropy_z_cond_C(
                                z=z[idx_1],
                                C=C[idx_1, :]
                            )
                            Hz_cond_C += weight_1 * Hz_cond_C_1

                            print('Hz_cond_C_0: {}, Hz_cond_C_1: {}, Hz_cond_C: {}'.format(
                                    Hz_cond_C_0, Hz_cond_C_1, Hz_cond_C
                            ))
                        else:
                            Hz_cond_C = self.itmh_obj.compute_conditional_entropy_z_cond_C(
                                z=z,
                                C=C
                            )
            elif len(C.shape) == 1:
                if self.is_inductive_ssh:
                    Hz_cond_C = 0.0

                    Hz_cond_C_1 = self.itmh_obj.compute_conditional_entropy_x_cond_z(
                        x=z[idx_1],
                        z=C[idx_1],
                    )
                    Hz_cond_C += weight_1 * Hz_cond_C_1

                    Hz_cond_C_2 = self.itmh_obj.compute_conditional_entropy_x_cond_z(
                        x=z[idx_2],
                        z=C[idx_2],
                    )
                    Hz_cond_C += weight_2 * Hz_cond_C_2

                    if self.is_joint_inductive_transductive_ssh:
                        Hz_cond_C_0 = self.itmh_obj.compute_conditional_entropy_x_cond_z(
                            x=z[idx_0],
                            z=C[idx_0],
                        )
                        Hz_cond_C += weight_0 * Hz_cond_C_0

                        print('Hz_cond_C_0: {}, Hz_cond_C_1: {}, Hz_cond_C_2: {}, Hz_cond_C: {}'.format(
                            Hz_cond_C_0, Hz_cond_C_1, Hz_cond_C_2, Hz_cond_C
                        ))
                    else:
                        print('Hz_cond_C_1: {}, Hz_cond_C_2: {}, Hz_cond_C: {}'.format(
                            Hz_cond_C_1, Hz_cond_C_2, Hz_cond_C
                        ))
                else:
                    if self.is_weighted_transductive:
                        Hz_cond_C = 0.0

                        Hz_cond_C_0 = self.itmh_obj.compute_conditional_entropy_x_cond_z(
                                x=z[idx_0],
                                z=C[idx_0],
                            )
                        Hz_cond_C += weight_0 * Hz_cond_C_0

                        Hz_cond_C_1 = self.itmh_obj.compute_conditional_entropy_x_cond_z(
                            x=z[idx_1],
                            z=C[idx_1],
                        )
                        Hz_cond_C += weight_1 * Hz_cond_C_1

                        print('Hz_cond_C_0: {}, Hz_cond_C_1: {}, Hz_cond_C: {}'.format(
                                Hz_cond_C_0, Hz_cond_C_1, Hz_cond_C
                        ))
                    else:
                        Hz_cond_C = self.itmh_obj.compute_conditional_entropy_x_cond_z(
                            x=z,
                            z=C,
                        )
            else:
                raise AssertionError
        else:
            Hz_cond_C = 0.0

        # print 'Hz_cond_C', Hz_cond_C

        # todo: compute H(x|z, C) as well
        # todo: tune parameters of RkNN (k in kNN) as well as RMM (C in SVM)

        curr_score = Hz + self.nonredundancy_score_beta*Hz_cond_C + self_dialog_mi

        if self.is_semisupervised_else_unsupervised:
            # computing H(x|z)
            if self.is_inductive_ssh:
                assert y is None, 'not implemented'
                assert np.where(x == 2)[0].size > 0

                Hx_cond_z = 0.0
                Hz_cond_x = 0.0

                # labeled training examples and unlabeled training examples
                x_12 = x[idx_12]
                z_12 = z[idx_12]
                x_12[np.where(x_12 == 2)] = 0
                Hx_cond_z_12 = self.itmh_obj.compute_conditional_entropy_x_cond_z(x=x_12, z=z_12)
                Hz_cond_x_12 = self.itmh_obj.compute_conditional_entropy_x_cond_z(x=z_12, z=x_12)
                del x_12, z_12
                Hx_cond_z += weight_12*Hx_cond_z_12
                Hz_cond_x += weight_12*Hz_cond_x_12

                if self.is_joint_inductive_transductive_ssh:
                    # this code is not correct completely, need to change values of x to binary before passing into the info-theoretic func
                    # test unlabeled and train labeled examples
                    x_01 = x[idx_01]
                    z_01 = z[idx_01]
                    Hx_cond_z_01 = self.itmh_obj.compute_conditional_entropy_x_cond_z(x=x_01, z=z_01)
                    Hz_cond_x_01 = self.itmh_obj.compute_conditional_entropy_x_cond_z(x=z_01, z=x_01)
                    del x_01, z_01
                    Hx_cond_z += weight_01*Hx_cond_z_01
                    Hz_cond_x += weight_01*Hz_cond_x_01

                    # test unlabeled and train unlabeled examples
                    x_02 = x[idx_02]
                    z_02 = z[idx_02]
                    x_02[np.where(x_02 == 2)] = 1
                    Hx_cond_z_02 = self.itmh_obj.compute_conditional_entropy_x_cond_z(x=x_02, z=z_02)
                    Hz_cond_x_02 = self.itmh_obj.compute_conditional_entropy_x_cond_z(x=z_02, z=x_02)
                    del x_02, z_02
                    Hx_cond_z += weight_02*Hx_cond_z_02
                    Hz_cond_x += weight_02*Hz_cond_x_02

                    print('Hx_cond_z_01: {}, Hx_cond_z_02 : {}, Hx_cond_z_12: {}, ' \
                          'Hz_cond_x_01: {}, Hz_cond_x_02: {}, Hz_cond_x_12: {}, ' \
                          'Hx_cond_z: {}, Hz_cond_x: {}'.format(
                            Hx_cond_z_01, Hx_cond_z_02, Hx_cond_z_12,
                            Hz_cond_x_01, Hz_cond_x_02, Hz_cond_x_12,
                            Hx_cond_z, Hz_cond_x
                    ))
                else:
                    print('Hx_cond_z_12: {}, ' \
                          'Hz_cond_x_12: {}, ' \
                          'Hx_cond_z: {}, Hz_cond_x: {}'.format(
                            Hx_cond_z_12,
                            Hz_cond_x_12,
                            Hx_cond_z, Hz_cond_x
                    ))
            else:
                # todo: see if we should do synthetic sampling to deal with imbalance of class distribution, or any other principled technique
                Hx_cond_z = self.itmh_obj.compute_conditional_entropy_x_cond_z(x=x, z=z)
                Hz_cond_x = self.itmh_obj.compute_conditional_entropy_x_cond_z(x=z, z=x)
                print('Hx_cond_z: {}, Hz_cond_x: {}'.format(Hx_cond_z, Hz_cond_x))

            curr_score += (Hx_cond_z + Hz_cond_x)

            if y is not None:
                assert not self.is_weighted_transductive, 'not implemented yet'
                labeled_idx = np.where(x == 1)[0]
                assert labeled_idx.size == y.size

                if C is not None:
                    C_labeled = C[labeled_idx]
                    z_labeled = z[labeled_idx]
                    del labeled_idx

                    if len(C_labeled.shape) == 1:
                        C_labeled = C_labeled.reshape(C_labeled.size, 1)
                    else:
                        assert len(C_labeled.shape) == 2

                    Cz_labeled = np.hstack((C_labeled, z_labeled.reshape(z_labeled.size, 1)))
                    del C_labeled
                    Hy_cond_Cz = self.itmh_obj.compute_conditional_entropy_z_cond_C(
                        z=y,
                        C=Cz_labeled,
                    )
                    del Cz_labeled
                    curr_score -= Hy_cond_Cz

                    Hy_cond_z = self.itmh_obj.compute_conditional_entropy_x_cond_z(x=y, z=z_labeled)
                    Hz_cond_y = self.itmh_obj.compute_conditional_entropy_x_cond_z(x=z_labeled, z=y)
                    del z_labeled
                    curr_score -= self.labeled_score_frac*(Hy_cond_z+Hz_cond_y)
                else:
                    z_labeled = z[labeled_idx]
                    del labeled_idx

                    Hy_cond_Cz = 0.0
                    Hy_cond_z = self.itmh_obj.compute_conditional_entropy_x_cond_z(x=y, z=z_labeled)
                    Hz_cond_y = self.itmh_obj.compute_conditional_entropy_x_cond_z(x=z_labeled, z=y)
                    del z_labeled
                    curr_score -= self.labeled_score_frac*(Hy_cond_z+Hz_cond_y)

        if self.is_semisupervised_else_unsupervised:
            if y is not None:
                print('Hz: {}, Hz_C: {},' \
                      ' Hx_z: {}, Hz_x: {},' \
                      ' Hy_Cz: {},' \
                      ' Hy_z: {}, Hz_y: {},' \
                      ' score: {}'.format(
                    Hz, Hz_cond_C,
                    Hx_cond_z, Hz_cond_x,
                    Hy_cond_Cz,
                    Hy_cond_z, Hz_cond_y,
                    curr_score
                ))
            else:
                print('Hz: {}, Hz_C: {}, Hx_z: {}, Hz_x: {},  score: {}'.format(Hz, Hz_cond_C, Hx_cond_z, Hz_cond_x, curr_score))
        else:
            print('Hz: {}, Hz_C: {}, score: {}'.format(Hz, Hz_cond_C, curr_score))

        total_time_score_compute = time.time() - start_time_score_compute

        return curr_score, total_time_score_compute

    def compare_array_wrt_array_of_arrays(self, array_for_comparison, arrays):
        assert array_for_comparison.dtype == np.int
        assert arrays.dtype == np.object
        num_arrays_for_comparison = arrays.size
        for curr_idx in range(num_arrays_for_comparison):
            if np.array_equal(arrays[curr_idx], array_for_comparison):
                return True
        return False

    def get_all_combinations(self, subset_size, superset=None):
        start_time = time.time()

        if superset is None:
            superset = np.arange(subset_size, dtype=np.int)

        alpha = int(subset_size / 2)

        num_combinations = 0
        for curr_combination_size in range(1, alpha+1):
            num_combinations += int(scipy_special.comb(subset_size, curr_combination_size))

        subsets1 = np.empty(num_combinations, dtype=np.object)
        subsets2 = np.empty(num_combinations, dtype=np.object)

        idx = -1
        none_idx = []
        for curr_combination_size in range(1, alpha+1):
            for curr_subset in itertools.combinations(superset, curr_combination_size):
                idx += 1

                curr_subset = np.array(list(curr_subset))
                curr_subset_complement = np.setdiff1d(superset, curr_subset)

                # print '..........................'
                # print 'curr_subset', curr_subset
                # print 'subsets2', subsets2

                if not self.compare_array_wrt_array_of_arrays(curr_subset, subsets2):
                    subsets1[idx] = curr_subset
                    subsets2[idx] = curr_subset_complement
                else:
                    none_idx.append(idx)
                    subsets1[idx] = None
                    subsets2[idx] = None

        if none_idx:
            none_idx = np.array(none_idx)
            all_idx = np.arange(num_combinations, dtype=np.int)
            not_none_idx = np.setdiff1d(all_idx, none_idx)
            del none_idx, all_idx
            subsets1 = subsets1[not_none_idx]
            subsets2 = subsets2[not_none_idx]
            del not_none_idx

        print('Time to compute combinations', time.time() - start_time)

        return subsets1, subsets2

    def optimize_rmm_hash_for_subsets(
            self,
            K, Kr,
            subset1, subset2,
            C, x, y,
            data_idx_from_cluster=None,
            sample_counts=None
    ):
        if self.is_opt_reg_rmm:
            reg_values = np.array([0.1, 0.3, 1.0, 3.0])

            scores_fr_reg_values = np.zeros(reg_values.size)
            z_objs = np.empty(reg_values.size, dtype=np.object)

            for curr_reg_val_idx in range(reg_values.size):

                curr_reg_val = reg_values[curr_reg_val_idx]
                z = self.rmm_obj.compute_hashcode_bit(K=K, Kr=Kr, subset1=subset1, subset2=subset2, reg=curr_reg_val)
                curr_score, _ = self.compute_score_for_reference_subset_partition(
                    C=C,
                    z=z,
                    x=x,
                    y=y,
                    data_idx_from_cluster=data_idx_from_cluster,
                    sample_counts=sample_counts,
                )

                scores_fr_reg_values[curr_reg_val_idx] = curr_score
                z_objs[curr_reg_val_idx] = z

            max_score_idx = scores_fr_reg_values.argmax()
            z = z_objs[max_score_idx]
            max_score = scores_fr_reg_values[max_score_idx]
            print('Optimal reg. for RMM: {}'.format(reg_values[max_score_idx]))
        else:
            z = self.rmm_obj.compute_hashcode_bit(
                K=K, Kr=Kr,
                subset1=subset1,
                subset2=subset2,
                reg=self.default_reg_val_rmm,
                max_iter=self.rmm_max_iter,
            )
            max_score, _ = self.compute_score_for_reference_subset_partition(
                C=C,
                z=z,
                x=x,
                y=y,
                data_idx_from_cluster=data_idx_from_cluster,
                sample_counts=sample_counts,
            )

        return z, max_score

    def optimize_rknn_hash_for_subsets(self,
                                       K,
                                       subset1,
                                       subset2,
                                       C,
                                       x,
                                       y,
                                       data_idx_from_cluster=None,
                                       sample_counts=None):

        z = self.rknn_obj.compute_hashcode_bit(K=K, subset1=subset1, subset2=subset2)

        curr_score, _ = self.compute_score_for_reference_subset_partition(
            C=C,
            z=z,
            x=x,
            y=y,
            data_idx_from_cluster=data_idx_from_cluster,
            sample_counts=sample_counts,
        )

        return z, curr_score

    def optimize_rneural_hash_for_subsets(self,
                                          path_tuples_all_embeddings,
                                          subset_path_tuples_kernel_references_embeddings,
                                          subset1,
                                          subset2,
                                          C,
                                          x,
                                          y,
                                          max_epochs,
                                          data_idx_from_cluster=None,
                                          sample_counts=None):

        print('x', x)
        if x is not None:
            print('x.shape', x.shape)

        z = self.rneural_obj.compute_hashcode_bit(
            path_tuples_all_embeddings,
            subset_path_tuples_kernel_references_embeddings,
            subset1=subset1,
            subset2=subset2,
            max_epochs=max_epochs,
        )
        print('z', z)
        print('z.shape', z.shape)

        curr_score, _ = self.compute_score_for_reference_subset_partition(
            C=C,
            z=z,
            x=x,
            y=y,
            data_idx_from_cluster=data_idx_from_cluster,
            sample_counts=sample_counts,
        )

        return z, curr_score

    def compute_scores_for_optimize_reference_subsets_wrapper(
            self,
            path_tuples_all_embeddings,
            subset_path_tuples_kernel_references_embeddings,
            C,
            x,
            y,
            K,
            Kr,
            data_idx_from_cluster,
            subsets1_arr,
            subsets2_arr,
            sample_counts=None,
    ):
        # start_time_split = time.time()

        num_combinations = subsets1_arr.size
        # print 'num_combinations', num_combinations
        assert num_combinations == subsets2_arr.size

        scores = np.zeros(num_combinations)
        # total_time_score_compute = 0.0

        for curr_combination_idx in range(num_combinations):
            # print 'curr_combination_idx', curr_combination_idx

            subset1 = subsets1_arr[curr_combination_idx]
            subset2 = subsets2_arr[curr_combination_idx]

            # computing hash vector
            if self.hash_func == 'RMM':
                z, curr_score = self.optimize_rmm_hash_for_subsets(
                      K,
                      Kr,
                      subset1,
                      subset2,
                      C,
                      x,
                      y,
                      data_idx_from_cluster=data_idx_from_cluster,
                      sample_counts=sample_counts,
                )
            elif self.hash_func == 'RkNN':
                z, curr_score = self.optimize_rknn_hash_for_subsets(
                       K,
                       subset1,
                       subset2,
                       C,
                       x,
                       y,
                       data_idx_from_cluster=data_idx_from_cluster,
                       sample_counts=sample_counts,
                )
            elif self.hash_func == 'RNeural':
                z, curr_score = self.optimize_rneural_hash_for_subsets(
                      path_tuples_all_embeddings,
                      subset_path_tuples_kernel_references_embeddings,
                      subset1,
                      subset2,
                      C,
                      x,
                      y,
                      max_epochs=self.max_epochs,
                      data_idx_from_cluster=data_idx_from_cluster,
                      sample_counts=sample_counts,
                )
            else:
                raise AssertionError

            # total_time_score_compute += curr_total_time_score_compute
            scores[curr_combination_idx] = curr_score

        # print '({})'.format(total_time_score_compute)
        # print 'Time to optimize split in wrapper', time.time() - start_time_split

        return scores

    def subselect_bits(
        self, C, max_num_bits_for_sel,
        num_bits_for_sel_high_entropy_clusters_all_hash=None,
    ):
        if C is None:
            return None, None
        else:
            assert max_num_bits_for_sel >= 1
            if len(C.shape) == 2:
                if num_bits_for_sel_high_entropy_clusters_all_hash is not None:
                    assert C.shape[1] == num_bits_for_sel_high_entropy_clusters_all_hash.size
                    # assert np.all(num_bits_for_sel_high_entropy_clusters_all_hash > 0)
                    weights = 1 / (num_bits_for_sel_high_entropy_clusters_all_hash+1)
                    weights /= weights.sum()
                else:
                    weights = None
                print('weights', weights)

                num_bits = C.shape[1]
                if num_bits > max_num_bits_for_sel:
                    rnd_subset_bits = self.npr_sample_hash_bits_obj.choice(
                        num_bits, max_num_bits_for_sel, replace=False,
                        p=weights,
                    )
                    C = C[:, rnd_subset_bits]
                else:
                    rnd_subset_bits = np.arange(num_bits, dtype=np.int)
            else:
                # C remain unchanged since only one bit
                # random selection of bits is just the original bit since only one bit to select
                rnd_subset_bits = np.array([0], dtype=np.int)

            return C, rnd_subset_bits

    def select_high_entropy_cluster(self,
            C, x, subset_size,
            is_z_binary,
            ref_sample_idx=None,
            sample_weights=None,
            is_return_cluster_data_indices=False,
            is_return_neighboring_cluster_data_indices=False
    ):

        if len(C.shape) == 1:
            num_bits = 1
        elif len(C.shape) == 2:
            num_bits = C.shape[1]
        else:
            raise AssertionError

        if ref_sample_idx is not None:
            if (self.ref_set_applicable_for_max_num_bits_to_cluster is None) \
                    or (self.ref_set_applicable_for_max_num_bits_to_cluster >= num_bits):
                C = C[ref_sample_idx]
                assert C.shape[0] == ref_sample_idx.size

                if x is not None:
                    x = x[ref_sample_idx]
                    assert x.size == ref_sample_idx.size

                if sample_weights is not None:
                    sample_weights = sample_weights[ref_sample_idx]
            else:
                ref_sample_idx = None

        num_data = C.shape[0]

        if self.is_semisupervised_else_unsupervised:
            Cu, Hz_per_C, n_per_C = self.itmh_obj.compute_entropy_z_cond_C(z=x, C=C, is_z_binary=is_z_binary)

            print('Hz_per_C', Hz_per_C)
            print('Hz_per_C.min()', Hz_per_C.min())
            print('Hz_per_C.max()', Hz_per_C.max())
            print('Hz_per_C.mean()', Hz_per_C.mean())
            print('Hz_per_C.std()', Hz_per_C.std())

            print('n_per_C', n_per_C)
            print('n_per_C.min()', n_per_C.min())
            print('n_per_C.max()', n_per_C.max())
            print('n_per_C.mean()', n_per_C.mean())
            print('n_per_C.std()', n_per_C.std())

            # print 'Hz_per_C.shape', Hz_per_C.shape
            # print 'n_per_C.shape', n_per_C.shape
            # cluster_scores = Hz_per_C*n_per_C
            # print 'cluster_scores.shape', cluster_scores.shape

            sel_clusters_idx = np.where(n_per_C >= subset_size)[0]
            # print 'sel_clusters_idx.shape', sel_clusters_idx.shape

            if self.is_cluster_size_and_entropy:
                cluster_sel_criterion = n_per_C[sel_clusters_idx]*Hz_per_C[sel_clusters_idx]
                cluster_idx = sel_clusters_idx[cluster_sel_criterion.argmax()]
                del cluster_sel_criterion
            else:
                cluster_idx = sel_clusters_idx[Hz_per_C[sel_clusters_idx].argmax()]

            print('cluster_idx', cluster_idx)
            print('Hz_per_C[cluster_idx]', Hz_per_C[cluster_idx])
            print('n_per_C[cluster_idx]', n_per_C[cluster_idx])
        else:
            Cu, n_per_C = self.itmh_obj.count_elements_in_clusters(C=C)

            max_n_per_C = n_per_C.max()
            mul_cluster_idx = np.where(n_per_C == max_n_per_C)[0]
            del max_n_per_C
            if mul_cluster_idx.size == 1:
                cluster_idx = mul_cluster_idx[0]
            else:
                cluster_idx = self.npr_sample_cluster_obj.choice(mul_cluster_idx)
            del mul_cluster_idx

            print('cluster_idx', cluster_idx)
            print('n_per_C', n_per_C)
            print('n_per_C.min()', n_per_C.min())
            print('n_per_C.max()', n_per_C.max())
            print('n_per_C.mean()', n_per_C.mean())
            print('n_per_C.std()', n_per_C.std())
            print('n_per_C[cluster_idx]', n_per_C[cluster_idx])

        # todo: make this computation faster, see how we do it in the function count_elements_in_clusters or compute_entropy_z_cond_C
        if len(C.shape) == 2:
            cluster = Cu[cluster_idx, :]
            data_idx_from_cluster = np.where((C == cluster).all(axis=1))[0]
        elif len(C.shape) == 1:
            cluster = Cu[cluster_idx]
            data_idx_from_cluster = np.where(C == cluster)[0]
        else:
            raise AssertionError

        if self.is_alpha_per_cluster_size:
            # added for changing alpha (subset size) as per the number of data points in the cluster
            subset_size = self.compute_subset_size_per_cluster_size(
                subset_size=subset_size,
                cluster_size=data_idx_from_cluster.size,
            )

        if data_idx_from_cluster.size <= subset_size:
            subset_idx = np.copy(data_idx_from_cluster)
            subset_size = data_idx_from_cluster.size
        else:
            assert subset_size >= 2
            if sample_weights is not None:
                sample_weights__data_idx_from_cluster = sample_weights[data_idx_from_cluster]
                print('sample_weights__data_idx_from_cluster', sample_weights__data_idx_from_cluster)

                sample_weights__data_idx_from_cluster =\
                    self.get_sample_weights_normalized_from_counts(sample_weights__data_idx_from_cluster, None)
                print(sample_weights__data_idx_from_cluster)
            else:
                sample_weights__data_idx_from_cluster = None

            assert (sample_weights__data_idx_from_cluster is None) \
                   or np.all(sample_weights__data_idx_from_cluster > 0.0)
            subset_idx = self.npr_sample_subset_obj.choice(
                data_idx_from_cluster,
                subset_size,
                replace=False,
                p=sample_weights__data_idx_from_cluster,
            )
            del sample_weights__data_idx_from_cluster

        if is_return_neighboring_cluster_data_indices:
            # neighbor clusters including the cluster
            neighboring_clusters = self.find_neighboring_clusters_including_the_cluster(Cu=Cu, cluster=cluster)
            print('neighboring_clusters.shape', neighboring_clusters.shape)

            if neighboring_clusters.shape[0] == Cu.shape[0]:
                data_idx_from_neighbor_clusters = np.arange(num_data, dtype=np.int)
            else:
                assert len(neighboring_clusters.shape) == 2
                data_idx_from_neighbor_clusters = np.array([], dtype=np.int)
                for curr_neighbor_cluster in neighboring_clusters:
                    data_idx_from_neighbor = np.where((C == curr_neighbor_cluster).all(axis=1))[0]
                    data_idx_from_neighbor_clusters =\
                        np.concatenate((data_idx_from_neighbor_clusters, data_idx_from_neighbor))

        if ref_sample_idx is not None:
            subset_idx = ref_sample_idx[subset_idx]
            data_idx_from_cluster = ref_sample_idx[data_idx_from_cluster]
            if is_return_neighboring_cluster_data_indices:
                data_idx_from_neighbor_clusters = ref_sample_idx[data_idx_from_neighbor_clusters]

        if is_return_neighboring_cluster_data_indices:
            if is_return_cluster_data_indices:
                return subset_idx, subset_size, cluster, data_idx_from_cluster, neighboring_clusters, data_idx_from_neighbor_clusters
            else:
                return subset_idx, subset_size, cluster, neighboring_clusters, data_idx_from_neighbor_clusters
        else:
            if is_return_cluster_data_indices:
                return subset_idx, subset_size, cluster, data_idx_from_cluster
            else:
                return subset_idx, subset_size, cluster

    def infer_labels_for_unlabeled_examples_in_clusters(self, C_org, x, max_num_bits):

        # print(x.tolist()
        num_labeled = np.where(x == 1)[0].size
        print('num_labeled', num_labeled)

        num_unlabeled = np.where(x == 2)[0].size
        print('num_unlabeled', num_unlabeled)

        assert len(C_org.shape) == 2
        assert np.where((x == 1) | (x == 2))[0].size == x.size
        num_bits = C_org.shape[1]
        assert num_bits > max_num_bits

        x_binary = np.copy(x)
        x_binary[np.where(x_binary == 2)] = 0

        # num_trials = (num_bits/max_num_bits)*10
        num_trials = 10

        for curr_trial_idx in range(num_trials):
            # print('curr_trial_idx', curr_trial_idx

            rnd_subset_bits = self.npr_sample_hash_bits_infer_cluster_obj.choice(num_bits, max_num_bits, replace=False)
            C = C_org[:, rnd_subset_bits]

            Cu, Hx_per_C, n_per_C, Cu_idx_in_original_arr = self.itmh_obj.compute_entropy_z_cond_C(
                z=x_binary,
                C=C,
                is_z_binary=True,
                is_return_Cu_idx_in_original_arr=True,
            )

            print('Hx_per_C.max()', Hx_per_C.max())

            num_clusters = Cu.shape[0]
            # print('num_clusters', num_clusters)

            for cluster_idx in range(num_clusters):
                # print('cluster_idx', cluster_idx
                data_idx_fr_cluster = np.where(Cu_idx_in_original_arr == cluster_idx)[0]
                # Hx = Hx_per_C[cluster_idx]

                ratio_num_labeled_in_cluster = x_binary[data_idx_fr_cluster].mean()
                # print('ratio_num_labeled_in_cluster', ratio_num_labeled_in_cluster

                if ratio_num_labeled_in_cluster > 0.75:
                        # print('Hx', Hx
                        print('data_idx_fr_cluster', data_idx_fr_cluster)
                        print('x[data_idx_fr_cluster]', x[data_idx_fr_cluster])
                        x[data_idx_fr_cluster] = 1
                        print('x[data_idx_fr_cluster]', x[data_idx_fr_cluster])

        num_labeled_new = np.where(x == 1)[0].size
        print('num_labeled_new', num_labeled_new)

        num_inferred_labels = num_labeled_new - num_labeled
        print('num_inferred_labels', num_inferred_labels)

        return x

    def compute_subset_size_per_cluster_size(self, subset_size, cluster_size):
        subset_size = max(subset_size, int(math.log(cluster_size, self.alpha_compute_per_cluster_size__log_scale)**2))
        return subset_size

    def find_neighboring_clusters_including_the_cluster(self, Cu, cluster):

        if len(Cu.shape) == 1:
            neighbor_clusters = Cu
        else:
            assert len(Cu.shape) == 2
            num_clusters = Cu.shape[0]
            hamming_distance_arr = np.zeros(num_clusters)

            for curr_idx in range(num_clusters):
                curr_cluster = Cu[curr_idx, :]
                hamming_distance_arr[curr_idx] = np.count_nonzero(np.bitwise_xor(curr_cluster, cluster))

            # including the original cluster
            # neighbor_hamming_distance_rad = int(math.ceil(hamming_distance_arr.max()/25))
            # neighbor_hamming_distance_rad = min(int(math.ceil(hamming_distance_arr.max()/10.0)), 5)
            neighbor_hamming_distance_max = hamming_distance_arr.max()
            print('neighbor_hamming_distance_max', neighbor_hamming_distance_max)
            neighbor_hamming_distance_rad =\
                min(int(round(neighbor_hamming_distance_max*self.max_hamming_dist_bits_frac_for_neighborhood)), 10)
            print('neighbor_hamming_distance_rad', neighbor_hamming_distance_rad)
            neighbor_clusters_idx = np.where(hamming_distance_arr <= neighbor_hamming_distance_rad)[0]
            neighbor_clusters = Cu[neighbor_clusters_idx, :]

        return neighbor_clusters

    def evaluate_hash_func_scores(self,
                                  path_tuples_all_embedding,
                                  curr_subset_path_tuples_kernel_references_embedding,
                                  K, Kr,
                                  C, x, y,
                                  num_combinations,
                                  data_idx_from_clusters,
                                  subsets1_arr, subsets2_arr,
                                  subset_size,
                                  sample_counts=None):

        assert subsets1_arr.size == num_combinations
        assert subsets2_arr.size == num_combinations

        num_cores = min(num_combinations, self.num_cores)
        start_time_opt_split = time.time()

        if self.is_random_sample_data_for_computing_objective and (C is not None):
            assert data_idx_from_clusters is None

            if len(C.shape) == 1:
                num_all_samples = C.size
            elif len(C.shape) == 2:
                num_all_samples = C.shape[0]
            else:
                raise AssertionError
            print('num_all_samples', num_all_samples)

            if len(C.shape) == 1:
                num_bits = 1
            else:
                num_bits = C.shape[1]

            assert (K is None) or (K.shape[0] == num_all_samples)
            assert (x is None) or (x.size == num_all_samples)
            assert (y is None) or (y.size == num_all_samples)

            num_samples_sel = int(num_all_samples/max(math.log(num_bits, self.rnd_sample_obj_size_log_base), 1.0))
            print('num_samples_sel', num_samples_sel)

            rnd_sel_idx = self.npr_sample_sel_compute_obj.choice(num_all_samples, num_samples_sel)
            del num_samples_sel

            print(rnd_sel_idx.shape)
            print('rnd_sel_idx', rnd_sel_idx)

            C = C[rnd_sel_idx]
            print(C.shape)

            if x is not None:
                x = x[rnd_sel_idx]
                print(x.shape)

            if y is not None:
                y = y[rnd_sel_idx]
                print(y.shape)

            if K is not None:
                K = K[rnd_sel_idx]
                print(K.shape)

            if sample_counts is not None:
                sample_counts = sample_counts[rnd_sel_idx]
                print(sample_counts.shape)

            del rnd_sel_idx

        if num_cores == 1:
            scores = self.compute_scores_for_optimize_reference_subsets_wrapper(
                path_tuples_all_embedding,
                curr_subset_path_tuples_kernel_references_embedding,
                C,
                x,
                y,
                K,
                Kr,
                data_idx_from_clusters,
                subsets1_arr,
                subsets2_arr,
                sample_counts=sample_counts,
            )
        else:
            scores = np.zeros(num_combinations)

            idx_range_parallel = pc.uniform_distribute_tasks_across_cores(num_combinations, num_cores)
            args_tuples_map = {}
            for currCore in range(num_cores):
                args_tuples_map[currCore] = (
                    path_tuples_all_embedding,
                    curr_subset_path_tuples_kernel_references_embedding,
                    C,
                    x,
                    y,
                    K,
                    Kr,
                    data_idx_from_clusters,
                    subsets1_arr[idx_range_parallel[currCore]],
                    subsets2_arr[idx_range_parallel[currCore]],
                    sample_counts,
                )

            pcw_obj = pcw.ParallelComputingWrapper(num_cores=num_cores)
            results_map = pcw_obj.process_method_parallel(
                method=self.compute_scores_for_optimize_reference_subsets_wrapper,
                args_tuples_map=args_tuples_map,
            )

            for curr_core in range(num_cores):
                curr_result = results_map[curr_core]
                scores[idx_range_parallel[curr_core]] = curr_result

        print(time.time() - start_time_opt_split)

        print('subset_size', subset_size)
        print('num_combinations', num_combinations)

        mean_score = scores.mean()
        min_score = scores.min()
        max_score = scores.max()
        print('scores', scores)
        print('scores.min()', min_score)
        print('scores.max()', max_score)
        print('scores.mean()', mean_score)
        print('scores.std()', scores.std())

        return scores

    def compute_kernel_for_hash_func(
        self,
        path_tuples,
        curr_subset_path_tuples_kernel_references,
        path_tuples_all_embedding,
        curr_subset_path_tuples_kernel_references_embedding,
        is_zero_kernel_compute_outside_cluster=False,
        data_idx_from_neighbor_clusters_and_the_cluster=None,
        references_idx_in_all=None,
    ):

        if self.is_bert:
            assert path_tuples_all_embedding is not None
            assert curr_subset_path_tuples_kernel_references_embedding is not None
            path_tuples = path_tuples_all_embedding
            curr_subset_path_tuples_kernel_references = curr_subset_path_tuples_kernel_references_embedding
            del path_tuples_all_embedding, curr_subset_path_tuples_kernel_references_embedding

        if self.hash_func == 'RMM':
            if is_zero_kernel_compute_outside_cluster \
                    and (data_idx_from_neighbor_clusters_and_the_cluster is not None) \
                    and (data_idx_from_neighbor_clusters_and_the_cluster.size < path_tuples.size):

                path_tuples_sel = path_tuples[data_idx_from_neighbor_clusters_and_the_cluster]
                path_tuples_all_sel = np.concatenate((path_tuples_sel, curr_subset_path_tuples_kernel_references))
                print('path_tuples_all_sel.size', path_tuples_all_sel.size)

                Kall_sel = self.data_operations_obj.__compute_kernel_matrix_parallel__(
                    path_tuples_all_sel,
                    curr_subset_path_tuples_kernel_references,
                    is_sparse=True,
                    is_normalize=self.data_operations_obj.is_normalize_kernel_fr_hashcodes,
                )
                del path_tuples_all_sel

                Kall_sel = Kall_sel.toarray()
                Kr = Kall_sel[path_tuples_sel.size:, :]
                # print('Kr', Kr.tolist()
                print('Kr.mean()', Kr.mean())

                K = np.zeros((path_tuples.size, Kall_sel.shape[1]), dtype=Kall_sel.dtype)
                K_sel = Kall_sel[:path_tuples_sel.size]
                del Kall_sel
                K[data_idx_from_neighbor_clusters_and_the_cluster] = K_sel
                del K_sel
            else:
                path_tuples_all = np.concatenate((path_tuples, curr_subset_path_tuples_kernel_references))
                Kall = self.data_operations_obj.__compute_kernel_matrix_parallel__(
                    path_tuples_all,
                    curr_subset_path_tuples_kernel_references,
                    is_sparse=True,
                    is_normalize=self.data_operations_obj.is_normalize_kernel_fr_hashcodes,
                )
                Kall = Kall.toarray()
                del path_tuples_all

                K = Kall[:path_tuples.size, :]
                print('K.mean()', K.mean())
                print('K.mean(0)', K.mean(0))
                print('K.std(0)', K.std(0))

                Kr = Kall[path_tuples.size:, :]
                # print('Kr', Kr.tolist()
                print('Kr.mean()', Kr.mean())
        else:
            assert self.hash_func == 'RkNN'
            if self.is_zero_kernel_compute_outside_cluster_and_neighboring \
                    and (data_idx_from_neighbor_clusters_and_the_cluster is not None) \
                    and (data_idx_from_neighbor_clusters_and_the_cluster.size < path_tuples.size):

                path_tuples_sel = path_tuples[data_idx_from_neighbor_clusters_and_the_cluster]

                K_sel = self.data_operations_obj.__compute_kernel_matrix_parallel__(
                    path_tuples_sel,
                    curr_subset_path_tuples_kernel_references,
                    is_sparse=True,
                    is_normalize=self.data_operations_obj.is_normalize_kernel_fr_hashcodes,
                )
                del path_tuples_sel
                K_sel = K_sel.toarray()

                K = np.zeros((path_tuples.size, K_sel.shape[1]), dtype=K_sel.dtype)
                K[data_idx_from_neighbor_clusters_and_the_cluster, :] = K_sel
                del K_sel
            else:
                K = self.data_operations_obj.__compute_kernel_matrix_parallel__(
                    path_tuples,
                    curr_subset_path_tuples_kernel_references,
                    is_sparse=True,
                    is_normalize=self.data_operations_obj.is_normalize_kernel_fr_hashcodes,
                )
                K = K.toarray()

            # if references_idx_in_all is not None:
            #     print('Setting self similarity to be zero.')
            #     K[references_idx_in_all, np.arange(references_idx_in_all.size, dtype=np.int)] = 0.0

            print('K.mean()', K.mean())
            print('K.mean(0)', K.mean(0))
            print('K.std(0)', K.std(0))
            Kr = None

        return K, Kr

    def sample_from_train_set(self, path_tuples_all, x, C, sample_counts=None):
        assert self.is_rnd_sample_data_for_opt_kernel_par

        num_data = path_tuples_all.size
        if C is not None:
            assert num_data == C.shape[0]
        if x is not None:
            assert num_data == x.size

        if x is not None:
            all_idx = np.arange(num_data, dtype=np.int)
            train_idx = np.where(x == 1)[0]
            test_idx = np.setdiff1d(all_idx, train_idx)

            if train_idx.size > test_idx.size:
                num_samples = int(self.frac_trn_data*train_idx.size)
                train_data_sample_idx = npr.choice(train_idx, num_samples)

                if num_samples < test_idx.size:
                    test_data_sample_idx = npr.choice(test_idx, num_samples)
                else:
                    test_data_sample_idx = test_idx
            else:
                num_samples = int(self.frac_trn_data*test_idx.size)
                test_data_sample_idx = npr.choice(test_idx, num_samples)

                if num_samples < train_idx.size:
                    train_data_sample_idx = npr.choice(train_idx, num_samples)
                else:
                    train_data_sample_idx = train_idx

            all_sample_idx = np.concatenate((train_data_sample_idx, test_data_sample_idx))
            train_data_sample_idx = None
            test_data_sample_idx = None
        else:
            num_samples = int(self.frac_trn_data*num_data)
            all_sample_idx = npr.choice(num_data, num_samples)

        path_tuples_all_sampled = path_tuples_all[all_sample_idx]

        if C is not None:
            C_sampled = C[all_sample_idx]
        else:
            C_sampled = None

        if x is not None:
            x_sampled = x[all_sample_idx]
        else:
            x_sampled = None

        if sample_counts is not None:
            sample_counts_sampled = sample_counts[all_sample_idx]
        else:
            sample_counts_sampled = None

        all_sample_idx = None

        if sample_counts is None:
            return path_tuples_all_sampled, x_sampled, C_sampled
        else:
            return path_tuples_all_sampled, x_sampled, C_sampled, sample_counts_sampled

    def optimize_kernel_hash_func(
          self,
          path_tuples_all,
          curr_subset_path_tuples_kernel_references,
          path_tuples_all_embedding,
          curr_subset_path_tuples_kernel_references_embedding,
          C,
          x,
          y,
          num_combinations,
          data_idx_from_clusters,
          subsets1_arr,
          subsets2_arr,
          subset_size,
          sample_counts=None,
          is_zero_kernel_compute_outside_cluster=False,
          data_idx_from_neighbor_clusters_and_the_cluster=None,
          references_idx_in_all=None,
    ):

        path_tuples_all_embedding = None
        curr_subset_path_tuples_kernel_references_embedding = None

        if self.is_opt_kernel_par:
            assert y is None, 'not implemented'
            y_sampled = None

            assert self.is_rnd_sample_data_for_opt_kernel_par
            path_tuples_all_sampled, x_sampled, C_sampled, sample_counts_sampled = self.sample_from_train_set(
                path_tuples_all, x, C, sample_counts=sample_counts,
            )

            if self.is_rnd_sample_combinations_for_opt_kernel_par and (num_combinations > self.num_sampled_combinations):
                print('Random sampling combinations ...')
                assert num_combinations == subsets1_arr.size
                assert num_combinations == subsets2_arr.size
                comb_idx = self.npr_sample_combinations_obj.choice(num_combinations, self.num_sampled_combinations, replace=False)
                subsets1_arr_sampled = subsets1_arr[comb_idx]
                subsets2_arr_sampled = subsets2_arr[comb_idx]
                num_sampled_combinations = self.num_sampled_combinations
            else:
                subsets1_arr_sampled = subsets1_arr
                subsets2_arr_sampled = subsets2_arr
                num_sampled_combinations = num_combinations

            optimization_str_list = []

            max_score_config = None
            max_score = -1.0e100

            for curr_sparse_kernel_threshold in [0.0, 0.1, 0.2, 0.3, 0.4]:
                self.data_operations_obj.__sparse_kernel_threshold__ = curr_sparse_kernel_threshold
                self.data_operations_obj.__kernel_buffer_node_edge_tuple__ = {}
                self.data_operations_obj.__kernel_buffer_node_tuple__ = {}

                for curr_lamda in [0.8, 0.9, 0.99]:
                    self.data_operations_obj.__lamb__ = curr_lamda

                    curr_config = {'lamb': curr_lamda, 'sparse_kernel_threshold': curr_sparse_kernel_threshold}

                    assert path_tuples_all_sampled is not None
                    K, Kr = self.compute_kernel_for_hash_func(
                        path_tuples_all_sampled,
                        curr_subset_path_tuples_kernel_references,
                        path_tuples_all_embedding,
                        curr_subset_path_tuples_kernel_references_embedding,
                        is_zero_kernel_compute_outside_cluster=is_zero_kernel_compute_outside_cluster,
                        data_idx_from_neighbor_clusters_and_the_cluster=data_idx_from_neighbor_clusters_and_the_cluster,
                    )

                    scores, _ = self.evaluate_hash_func_scores(
                        path_tuples_all_embedding,
                        curr_subset_path_tuples_kernel_references_embedding,
                        K,
                        Kr,
                        C_sampled,
                        x_sampled,
                        y_sampled,
                        num_sampled_combinations,
                        data_idx_from_clusters,
                        subsets1_arr_sampled,
                        subsets2_arr_sampled,
                        subset_size,
                        sample_counts=sample_counts_sampled,
                    )

                    curr_max_score = scores.max()
                    curr_str = 'lamb: {}, sparse_kernel_threshold: {}, max_score: {}'.format(
                        curr_lamda,
                        curr_sparse_kernel_threshold,
                        curr_max_score,
                    )
                    optimization_str_list.append(curr_str)
                    curr_str = None

                    if curr_max_score > max_score:
                        max_score = curr_max_score
                        max_score_config = curr_config

            self.data_operations_obj.__sparse_kernel_threshold__ = max_score_config['sparse_kernel_threshold']
            self.data_operations_obj.__lamb__ = max_score_config['lamb']
            self.data_operations_obj.__kernel_buffer_node_edge_tuple__ = {}
            self.data_operations_obj.__kernel_buffer_node_tuple__ = {}

            optimized_str = 'Optimized, lamb: {}, sparse_kernel_threshold: {}, max_score: {}'.format(
                max_score_config['lamb'],
                max_score_config['sparse_kernel_threshold'],
                max_score,
            )
            print('\n'.join(optimization_str_list)+'\n'+optimized_str)

        K, Kr = self.compute_kernel_for_hash_func(
            path_tuples_all,
            curr_subset_path_tuples_kernel_references,
            path_tuples_all_embedding,
            curr_subset_path_tuples_kernel_references_embedding,
            is_zero_kernel_compute_outside_cluster=is_zero_kernel_compute_outside_cluster,
            data_idx_from_neighbor_clusters_and_the_cluster=data_idx_from_neighbor_clusters_and_the_cluster,
            references_idx_in_all=references_idx_in_all,
        )

        scores = self.evaluate_hash_func_scores(
            path_tuples_all_embedding,
            curr_subset_path_tuples_kernel_references_embedding,
            K,
            Kr,
            C,
            x,
            y,
            num_combinations,
            data_idx_from_clusters,
            subsets1_arr,
            subsets2_arr,
            subset_size,
            sample_counts=sample_counts,
        )

        return scores, K, Kr

    def format_path_tuple_as_text(self, curr_element_in_ref_set):
        if 'path_tuple' in curr_element_in_ref_set:
            curr_element_in_ref_set = curr_element_in_ref_set['path_tuple']

        if isinstance(curr_element_in_ref_set, list):
            curr_text = ' '.join([curr_pos_tuple[0] for curr_pos_tuple in curr_element_in_ref_set if curr_pos_tuple is not None])
        else:
            curr_text = curr_element_in_ref_set

        curr_text = '\'{}\''.format(curr_text)
        return curr_text

    def get_sample_weights_normalized_from_counts(self, path_tuples_all_counts, path_tuples_all_vocab_counts, idx=None):
        if path_tuples_all_vocab_counts is not None:
            assert path_tuples_all_counts.dtype == np.int
            assert path_tuples_all_vocab_counts.dtype == np.int
            assert np.all(path_tuples_all_vocab_counts >= 1)
            print('debug ...')
            print(path_tuples_all_counts.mean())
            path_tuples_all_counts = path_tuples_all_counts*path_tuples_all_vocab_counts
            print(path_tuples_all_counts.mean())

        if path_tuples_all_counts is not None:
            if idx is not None:
                sample_weights = path_tuples_all_counts[idx]
                epsilon = 1.0e-30
                if sample_weights.dtype != np.float:
                    assert sample_weights.dtype == np.int
                    sample_weights = ((sample_weights.astype(np.float)+epsilon) / float(sample_weights.sum()))
                else:
                    sample_weights = (sample_weights+epsilon) / sample_weights.sum()
            else:
                if path_tuples_all_counts.dtype != np.float:
                    assert path_tuples_all_counts.dtype == np.int
                    sample_weights = path_tuples_all_counts.astype(np.float) / float(path_tuples_all_counts.sum())
                else:
                    sample_weights = path_tuples_all_counts / path_tuples_all_counts.sum()
        else:
            sample_weights = None

        return sample_weights

    def embed_path_tuples(self, path_tuples_all, is_bert=False):
        path_tuples_all_embedding = self.ept_obj.embed_path_tuples_neural(
            path_tuples_all, is_bert=is_bert,
        )
        print('path_tuples_all_embedding.shape', path_tuples_all_embedding.shape)
        assert len(path_tuples_all_embedding.shape) == 3
        assert path_tuples_all_embedding.shape[0] == path_tuples_all.size
        assert path_tuples_all_embedding.shape[2] == self.word_vec_len

        if self.is_convolution_neural:
            assert self.num_words_convolution == 2, 'not implemented as requires padding zeros'
            assert (path_tuples_all_embedding.shape[1] % 2) == 0
            new_path_len_for_embeddings = int(path_tuples_all_embedding.shape[1] / self.num_words_convolution)
            path_tuples_all_embedding = path_tuples_all_embedding.reshape(
                path_tuples_all_embedding.shape[0],
                new_path_len_for_embeddings,
                self.word_vec_len * self.num_words_convolution,
            )
            del new_path_len_for_embeddings

        return path_tuples_all_embedding

    def embed_sentences_bert(self, arr_text_sent_1, arr_text_sent_2):
        n1 = arr_text_sent_1.size
        n2 = arr_text_sent_2.size

        arr_text_sent = np.concatenate((arr_text_sent_1, arr_text_sent_2))
        arr_text_sent_bert = self.bs_obj.compute_bert_embeddings(sentences=arr_text_sent, is_avg=False)
        arr_text_sent_bert_1 = arr_text_sent_bert[:n1]
        assert arr_text_sent_bert_1.size == n1
        arr_text_sent_bert_2 = arr_text_sent_bert[n1:]
        assert arr_text_sent_bert_2.size == n2

        return arr_text_sent_bert_1, arr_text_sent_bert_2

    def optimize_reference_subsets_greedy_stochastic(
            self,
            path_tuples_all,
            x,
            num_hash_functions,
            alpha=None,
            class_labels=None,
            unlabeled_path_tuples_arr=None,
            ref_sample_idx=None,
            is_return_all_unlabeled=False,
            path_tuples_all_counts_org=None,
            path_tuples_all_vocab_count=None,
            return_neighbor_clusters=False,
    ):
        # todo: to reduce kernel compute cost, we could cache the kernel computations from each greedy step as some of those would be reused/
        # todo: if normalizing kernel, caching self kernel similarity is definitely the way to go, and easy & clean to implement
        # todo: if returning the reference set and the subsets, modify structures considering the imbalanced split of subsets.

        if path_tuples_all_counts_org is not None:
            assert path_tuples_all_counts_org.dtype == np.int
            assert path_tuples_all_counts_org.size == path_tuples_all.size
            print('path_tuples_all_counts_org', path_tuples_all_counts_org)
            assert np.all(path_tuples_all_counts_org >= 1)

            if path_tuples_all_vocab_count is not None:
                assert path_tuples_all_vocab_count.dtype == np.int
                assert path_tuples_all_vocab_count.size == path_tuples_all.size
                print('path_tuples_all_vocab_count', path_tuples_all_vocab_count)
                assert np.all(path_tuples_all_vocab_count >= 1)
        else:
            assert path_tuples_all_vocab_count is None

        if self.hash_func in ['RMM', 'RkNN']:
            org_lamda = self.data_operations_obj.__lamb__
            org_sparse_kernel_threshold = self.data_operations_obj.__sparse_kernel_threshold__

        if self.is_semisupervised_else_unsupervised:
            assert x is not None
        else:
            assert x is None

        if class_labels is not None:
            assert x is not None
            assert np.where(x == 1)[0].size == class_labels.size

        print('alpha', alpha)

        if alpha is not None:
            if isinstance(alpha, np.ndarray):
                subset_sizes = alpha
            else:
                subset_sizes = np.array([alpha])
        else:
            if self.hash_func == 'RNeural':
                # PubMed
                subset_sizes = np.array([6])
            else:
                # subset_sizes = np.array([6])

                # PubMed
                # subset_sizes = np.array([6])
                # subset_sizes = np.array([4, 8, 10])
                # subset_sizes = np.array([4, 6])

                # trying larger size after implementing divide and conquer version
                subset_sizes = np.array([4, 8, 16, 32, 64])

                # subset_sizes = np.array([4, 8, 12])
                # PPI
                # subset_sizes = np.array([6])

        print('subset_sizes', subset_sizes)

        if self.max_num_hash_functions_for_sampling_from_cluster is not None:
            if isinstance(self.max_num_hash_functions_for_sampling_from_cluster, np.ndarray):
                choices_num_bits_for_high_entropy_sel = self.max_num_hash_functions_for_sampling_from_cluster
            else:
                choices_num_bits_for_high_entropy_sel = [self.max_num_hash_functions_for_sampling_from_cluster]
        else:
            choices_num_bits_for_high_entropy_sel = [10]

        print('choices_num_bits_for_high_entropy_sel', choices_num_bits_for_high_entropy_sel)

        num_subset_sizes = subset_sizes.size

        scores_opt_hash_func = np.zeros(num_hash_functions)

        # subset_size__mean_score_map = {}
        # subset_size__min_score_map = {}
        subset_size__max_score_map = {}
        subset_size__count_map = {}

        max_subset_size = self.compute_subset_size_per_cluster_size(
            subset_size=subset_sizes.max(),
            cluster_size=path_tuples_all.size
        )
        print('max_subset_size', max_subset_size)
        for subset_size in range(1, max_subset_size+1):
            # subset_size__mean_score_map[subset_size] = 0.0
            # subset_size__min_score_map[subset_size] = 0.0
            subset_size__max_score_map[subset_size] = 0.0
            subset_size__count_map[subset_size] = 0
        del max_subset_size

        if self.is_optimize_split and (not self.is_divide_conquer_else_brute_force):
            start_time_combinations = time.time()
            subsets1_map = {}
            subsets2_map = {}

            for subset_size in range(1, subset_sizes.max()+1):
                subsets1_arr, subsets2_arr = self.get_all_combinations(subset_size=subset_size)
                subsets1_map[subset_size] = subsets1_arr
                subsets2_map[subset_size] = subsets2_arr

            print('Time to get all combinations for partitions', time.time() - start_time_combinations)

        # set1_points = -1 * np.ones(shape=(num_hash_functions, alpha), dtype=np.int64)
        # set2_points = -1 * np.ones(shape=(num_hash_functions, alpha), dtype=np.int64)
        # path_tuples_kernel_references = np.array([])

        C = None

        if self.is_inductive_ssh:
            assert not self.is_images
            assert unlabeled_path_tuples_arr is not None

            if self.unlabeled_induction_sampling_ratio < 1.0:
                num_samples_unlabeled = int(self.unlabeled_induction_sampling_ratio*unlabeled_path_tuples_arr.size)
                sampled_unlabeled_idx = self.npr_sample_unlabeled_data_obj.choice(
                    unlabeled_path_tuples_arr.size,
                    num_samples_unlabeled
                )
                del num_samples_unlabeled
                unlabeled_path_tuples_arr = unlabeled_path_tuples_arr[sampled_unlabeled_idx]
                del sampled_unlabeled_idx

            x = np.concatenate((x, 2*np.ones(unlabeled_path_tuples_arr.size, dtype=np.int)))
            path_tuples_all = np.concatenate((path_tuples_all, unlabeled_path_tuples_arr))
            del unlabeled_path_tuples_arr

            # x changes during optimization
            # x_po is pre-optimization
            x_po = np.copy(x)

        if self.is_images:
            num_data = path_tuples_all.shape[0]
        else:
            num_data = path_tuples_all.size

        if self.hash_func == 'RNeural':
            if self.is_images:
                path_tuples_all_embedding = path_tuples_all
            else:
                path_tuples_all_embedding = self.embed_path_tuples(path_tuples_all)
        else:
            if self.is_bert:
                path_tuples_all_embedding = self.bs_obj.compute_bert_embeddings(sentences=path_tuples_all, is_avg=False)
            else:
                path_tuples_all_embedding = None

        curr_set_idx = -1
        num_hash_computations = 0

        print('ref_sample_idx', ref_sample_idx)

        # to save on memory, commenting the ones which are not necessary at present
        # todo: if uncomment the following, make sure to change the order of elemnets when order of functions change in the end of each iteration
        path_tuples_kernel_references_objs = np.empty(num_hash_functions, dtype=np.object)
        opt_subset1_objs = np.empty(num_hash_functions, dtype=np.object)
        opt_subset2_objs = np.empty(num_hash_functions, dtype=np.object)
        # data_idx_from_cluster_objs = np.empty(num_hash_functions, dtype=np.object)

        rnd_subsel_bits_for_clustering_objs = np.empty(num_hash_functions, dtype=np.object)
        sel_cluster_for_sampling_objs = np.empty(num_hash_functions, dtype=np.object)
        sel_neighboring_clusters_objs = np.empty(num_hash_functions, dtype=np.object)
        data_idx_from_neighbor_clusters_and_the_cluster_objs = np.empty(num_hash_functions, dtype=np.object)

        # if self.is_num_hash_bits_for_cluster_dynamic:
        num_bits_for_sel_high_entropy_clusters_all_hash = np.zeros(num_hash_functions, dtype=np.int)

        while curr_set_idx < (num_hash_functions-1):
            start_time_hash_opt_greedy = time.time()

            num_hash_computations += 1
            curr_set_idx += 1

            print('......................................')
            print('curr_set_idx', curr_set_idx)

            self.data_operations_obj.time_load_wordvecs = 0.0
            print('self.data_operations_obj.time_load_wordvecs', self.data_operations_obj.time_load_wordvecs)

            if (not self.is_alpha_sampling_weighted) or (curr_set_idx < 15):
                p = None
            else:
                p = np.zeros(num_subset_sizes)
                for curr_idx in range(num_subset_sizes):
                    p[curr_idx] = subset_size__max_score_map[subset_sizes[curr_idx]]/float(subset_size__count_map[subset_sizes[curr_idx]])
                p /= p.sum()
            print('p', p)

            num_bits_for_sel_high_entropy_cluster = self.npr_sample_nbhec_obj.choice(
                choices_num_bits_for_high_entropy_sel
            )

            if self.is_num_hash_bits_for_cluster_dynamic:
                num_bits_for_sel_high_entropy_cluster = min(num_bits_for_sel_high_entropy_cluster, curr_set_idx)

                if num_bits_for_sel_high_entropy_cluster != 0:
                    # multiple hash functions of same granularity
                    # random_val_for_bits_subset_size_reduction = self.npr_bits_subset_size_reduction.rand()
                    # if random_val_for_bits_subset_size_reduction < 0.5:
                    num_bits_for_sel_high_entropy_cluster = int(
                        math.sqrt(num_bits_for_sel_high_entropy_cluster)*
                        math.ceil(math.log(num_bits_for_sel_high_entropy_cluster, 10.0))
                    )
                    # elif random_val_for_bits_subset_size_reduction < 0.75:
                    #     num_bits_for_sel_high_entropy_cluster = int(math.log(num_bits_for_sel_high_entropy_cluster, 2))
                    # del random_val_for_bits_subset_size_reduction

                if (num_bits_for_sel_high_entropy_cluster == 0) and (C is not None):
                    num_bits_for_sel_high_entropy_cluster = 1

            num_bits_for_sel_high_entropy_clusters_all_hash[curr_set_idx] = num_bits_for_sel_high_entropy_cluster

            print('num_bits_for_sel_high_entropy_cluster', num_bits_for_sel_high_entropy_cluster)

            subset_size = self.npr_sample_alpha_obj.choice(subset_sizes, p=p)

            C_subsel_bits, rnd_subsel_bits_for_clustering = self.subselect_bits(
                C, num_bits_for_sel_high_entropy_cluster,
                num_bits_for_sel_high_entropy_clusters_all_hash[:curr_set_idx],
            )
            rnd_subsel_bits_for_clustering_objs[curr_set_idx] = rnd_subsel_bits_for_clustering
            print('rnd_subsel_bits_for_clustering', rnd_subsel_bits_for_clustering)
            del rnd_subsel_bits_for_clustering

            if C_subsel_bits is not None:
                print('C_subsel_bits.shape', C_subsel_bits.shape)
            # print('alpha: {}'.format(subset_size))
            if (C_subsel_bits is not None) and (len(C_subsel_bits.shape) == 2) and (C_subsel_bits.shape[1] > 1):
                subset_size = min(int(subset_size / math.log(C_subsel_bits.shape[1], 4.0)), subset_size)
            print('alpha: {}'.format(subset_size))

            cluster = None
            # todo: select subset from a selected cluster
            if (curr_set_idx == 0) or (num_bits_for_sel_high_entropy_cluster == 0):
                if ref_sample_idx is None:
                    print(num_data, subset_size)
                    curr_subset12 = self.npr_sample_subset_obj.choice(
                        num_data,
                        subset_size,
                        replace=False,
                        p=self.get_sample_weights_normalized_from_counts(
                            path_tuples_all_counts_org, path_tuples_all_vocab_count)
                    )
                else:
                    curr_subset12 = self.npr_sample_subset_obj.choice(
                        ref_sample_idx,
                        subset_size,
                        replace=False,
                        p=self.get_sample_weights_normalized_from_counts(
                            path_tuples_all_counts_org, path_tuples_all_vocab_count, idx=ref_sample_idx)
                    )

                data_idx_from_cluster = None
                neighboring_clusters = None
                data_idx_from_neighbor_clusters_and_the_cluster = None
            else:
                print('self.is_inductive_ssh', self.is_inductive_ssh)

                if self.is_inductive_ssh:
                    if self.is_infer_cluster and (curr_set_idx > int(self.num_bits_infer_cluster*1.2)):
                        # training labeled and unlabeled data
                        idx_12 = np.where((x_po == 1) | (x_po == 2))[0]

                        # using the original labels to decide the boundaries,
                        # so as not to propagate errors on the inference
                        x_12 = self.infer_labels_for_unlabeled_examples_in_clusters(
                            C_org=C[idx_12],
                            x=x_po[idx_12],
                            max_num_bits=self.num_bits_infer_cluster,
                        )
                        idx_12_changed = idx_12[np.where((x_po[idx_12] == 2) & (x_12 == 1))[0]]
                        x[idx_12_changed] = 1
                        del idx_12, x_12, idx_12_changed

                    print('No. of unlabeled examples inferred', np.where(x == 1)[0].size - np.where(x_po == 1)[0].size)

                    if not self.is_joint_inductive_transductive_ssh:
                        assert ref_sample_idx is None, 'not implemented'
                        # test and training data
                        idx_12 = np.where((x == 1) | (x == 2))[0]
                        curr_subset12, subset_size, cluster, data_idx_from_cluster, \
                        neighboring_clusters, data_idx_from_neighbor_clusters_and_the_cluster = self.select_high_entropy_cluster(
                                C_subsel_bits[idx_12],
                                x[idx_12],
                                subset_size=subset_size,
                                is_z_binary=False,
                                sample_weights=self.get_sample_weights_normalized_from_counts(
                                    path_tuples_all_counts_org, path_tuples_all_vocab_count, idx=idx_12
                                ),
                                is_return_cluster_data_indices=True,
                                is_return_neighboring_cluster_data_indices=True
                        )
                        curr_subset12 = idx_12[curr_subset12]
                        del idx_12
                    else:
                        # test and training data
                        curr_subset12, subset_size, cluster, data_idx_from_cluster, \
                        neighboring_clusters, data_idx_from_neighbor_clusters_and_the_cluster = self.select_high_entropy_cluster(
                                C_subsel_bits,
                                x,
                                subset_size=subset_size,
                                is_z_binary=False,
                                ref_sample_idx=ref_sample_idx,
                                sample_weights=self.get_sample_weights_normalized_from_counts(
                                    path_tuples_all_counts_org, path_tuples_all_vocab_count),
                                is_return_cluster_data_indices=True,
                                is_return_neighboring_cluster_data_indices=True
                        )
                else:
                    curr_subset12, subset_size, cluster, data_idx_from_cluster,\
                    neighboring_clusters, data_idx_from_neighbor_clusters_and_the_cluster\
                        = self.select_high_entropy_cluster(
                            C_subsel_bits,
                            x,
                            subset_size=subset_size,
                            is_z_binary=True,
                            ref_sample_idx=ref_sample_idx,
                            sample_weights=self.get_sample_weights_normalized_from_counts(
                                path_tuples_all_counts_org, path_tuples_all_vocab_count),
                            is_return_cluster_data_indices=True,
                            is_return_neighboring_cluster_data_indices=True
                    )

            print('data_idx_from_cluster', data_idx_from_cluster)

            print('data_idx_from_neighbor_clusters_and_the_cluster', data_idx_from_neighbor_clusters_and_the_cluster)

            # data_idx_from_cluster_objs[curr_set_idx] = data_idx_from_cluster

            sel_cluster_for_sampling_objs[curr_set_idx] = cluster
            del cluster
            sel_neighboring_clusters_objs[curr_set_idx] = neighboring_clusters
            data_idx_from_neighbor_clusters_and_the_cluster_objs[curr_set_idx] =\
                data_idx_from_neighbor_clusters_and_the_cluster

            subset_size = curr_subset12.size
            print('curr_subset12', subset_size)
            if data_idx_from_neighbor_clusters_and_the_cluster is not None:
                print('data_idx_from_neighbor_clusters_and_the_cluster.size', data_idx_from_neighbor_clusters_and_the_cluster.size)
            else:
                print('data_idx_from_neighbor_clusters_and_the_cluster', data_idx_from_neighbor_clusters_and_the_cluster)

            if path_tuples_all_counts_org is not None:
                if self.is_cluster_high_weightage and (data_idx_from_neighbor_clusters_and_the_cluster is not None):
                    path_tuples_all_counts = np.zeros(path_tuples_all_counts_org.size, dtype=np.int)
                    path_tuples_all_counts[data_idx_from_neighbor_clusters_and_the_cluster] =\
                        path_tuples_all_counts_org[data_idx_from_neighbor_clusters_and_the_cluster]
                else:
                    path_tuples_all_counts = np.copy(path_tuples_all_counts_org)
            else:
                path_tuples_all_counts = None

            if self.is_optimize_split and (not self.is_divide_conquer_else_brute_force):
                subsets1_arr = subsets1_map[subset_size]
                subsets2_arr = subsets2_map[subset_size]
                num_combinations = subsets1_arr.size
                print('num_combinations', num_combinations)

            curr_subset_path_tuples_kernel_references = path_tuples_all[curr_subset12]

            if (self.hash_func == 'RNeural') or self.is_bert:
                curr_subset_path_tuples_kernel_references_embedding = path_tuples_all_embedding[curr_subset12]
            else:
                curr_subset_path_tuples_kernel_references_embedding = None

            if path_tuples_all_counts is not None:
                curr_subset_path_tuples_kernel_references_weights = path_tuples_all_counts[curr_subset12]
            else:
                curr_subset_path_tuples_kernel_references_weights = None

            if path_tuples_all_vocab_count is not None:
                curr_subset_path_tuples_all_vocab_count = path_tuples_all_vocab_count[curr_subset12]
            else:
                curr_subset_path_tuples_all_vocab_count = None

            if not self.is_images:
                print('******************************************************')
                if self.is_inductive_ssh:
                    print(x_po[curr_subset12])

                if x is not None:
                    print(x[curr_subset12])

                for curr_element_in_ref_set_idx, curr_element_in_ref_set in enumerate(curr_subset_path_tuples_kernel_references):

                    if curr_subset_path_tuples_kernel_references_weights is not None:
                        curr_element_in_ref_set_weight = curr_subset_path_tuples_kernel_references_weights[curr_element_in_ref_set_idx]
                    else:
                        curr_element_in_ref_set_weight = ''

                    if curr_subset_path_tuples_all_vocab_count is not None:
                        curr_element_vocab_count = curr_subset_path_tuples_all_vocab_count[curr_element_in_ref_set_idx]
                    else:
                        curr_element_vocab_count = ''

                    if self.is_text_display:
                        print(curr_element_in_ref_set['text'], curr_element_in_ref_set_weight, curr_element_vocab_count)
                    else:
                        print(self.format_path_tuple_as_text(curr_element_in_ref_set), curr_element_in_ref_set_weight, curr_element_vocab_count)
                        # print(curr_element_in_ref_set['path_tuple']

            if self.is_optimize_split:
                if self.is_divide_conquer_else_brute_force:
                    hash_func_org = self.hash_func
                    if self.hash_func not in ['RMM', 'RkNN']:
                        assert self.hash_func == 'RNeural'
                        self.hash_func = 'RkNN'

                    dcos_obj = DivideConquerOptimizeSplit(
                        subset_size,
                        hash_func=self.hash_func,
                        min_set_size_to_split=self.min_set_size_to_split,
                    )

                    opt_subset1, opt_subset2, max_score, K, Kr = dcos_obj.optimize(
                        path_tuples_all,
                        curr_subset_path_tuples_kernel_references,
                        path_tuples_all_embedding,
                        curr_subset_path_tuples_kernel_references_embedding,
                        C_subsel_bits,
                        x,
                        class_labels,
                        None,
                        subset_size,
                        info_theoretic_opt_obj=self,
                        sample_counts=path_tuples_all_counts,
                        is_zero_kernel_compute_outside_cluster=self.is_zero_kernel_compute_outside_cluster_and_neighboring,
                        data_idx_from_neighbor_clusters_and_the_cluster=data_idx_from_neighbor_clusters_and_the_cluster,
                        references_idx_in_all=curr_subset12,
                    )
                    del dcos_obj
                    self.hash_func = hash_func_org
                    del hash_func_org
                else:
                    if self.hash_func in ['RMM', 'RkNN']:
                        assert self.is_subset_bits_for_objective_scores
                        scores, K, Kr = self.optimize_kernel_hash_func(
                              path_tuples_all,
                              curr_subset_path_tuples_kernel_references,
                              path_tuples_all_embedding,
                              curr_subset_path_tuples_kernel_references_embedding,
                              C_subsel_bits,
                              x,
                              class_labels,
                              num_combinations,
                              None,
                              subsets1_arr,
                              subsets2_arr,
                              subset_size,
                              sample_counts=path_tuples_all_counts,
                              is_zero_kernel_compute_outside_cluster=self.is_zero_kernel_compute_outside_cluster_and_neighboring,
                              data_idx_from_neighbor_clusters_and_the_cluster=data_idx_from_neighbor_clusters_and_the_cluster,
                              references_idx_in_all=curr_subset12,
                        )
                    else:
                        assert self.hash_func == 'RNeural'
                        K = None
                        Kr = None
                        assert self.is_subset_bits_for_objective_scores
                        scores = self.evaluate_hash_func_scores(
                               path_tuples_all_embedding,
                               curr_subset_path_tuples_kernel_references_embedding,
                               K,
                               Kr,
                               C_subsel_bits,
                               x,
                               class_labels,
                               num_combinations,
                               None,
                               subsets1_arr,
                               subsets2_arr,
                               subset_size,
                               sample_counts=path_tuples_all_counts,
                        )

                    assert scores.size == subsets1_arr.size
                    assert scores.size == subsets2_arr.size
                    max_score = scores.max()

                    opt_idx = scores.argmax()
                    print('opt_idx', opt_idx)

                    opt_subset1 = np.copy(subsets1_arr[opt_idx])
                    opt_subset2 = np.copy(subsets2_arr[opt_idx])

                # subset_size__mean_score_map[subset_size] += mean_score
                # subset_size__min_score_map[subset_size] += min_score
                subset_size__max_score_map[subset_size] += max_score
                subset_size__count_map[subset_size] += 1

                # print('subset_size__mean_score_map', subset_size__mean_score_map
                # print('subset_size__min_score_map', subset_size__min_score_map
                print('subset_size__max_score_map', subset_size__max_score_map)
                print('subset_size__count_map', subset_size__count_map)

                scores_opt_hash_func[curr_set_idx] = max_score
                print('scores_opt_hash_func', scores_opt_hash_func[:curr_set_idx+1])
            else:
                if self.hash_func == 'RNeural':
                    K = None
                    Kr = None
                else:
                    K, Kr = self.compute_kernel_for_hash_func(
                        path_tuples_all,
                        curr_subset_path_tuples_kernel_references,
                        path_tuples_all_embedding,
                        curr_subset_path_tuples_kernel_references_embedding,
                        is_zero_kernel_compute_outside_cluster=self.is_zero_kernel_compute_outside_cluster_and_neighboring,
                        data_idx_from_neighbor_clusters_and_the_cluster=data_idx_from_neighbor_clusters_and_the_cluster,
                    )

                # opt_idx = self.npr_sample_split_obj.choice(num_combinations)
                opt_subset1 = self.npr_sample_split_obj.choice(subset_size, int(subset_size/2), replace=False)
                opt_subset2 = np.setdiff1d(np.arange(subset_size, dtype=np.int), opt_subset1)

            if K is not None:
                print('.....................................')
                print('K[:, opt_subset1].mean()', K[:, opt_subset1].mean())
                print('K[:, opt_subset2].mean()', K[:, opt_subset2].mean())
                print('K.mean()', K.mean())

                if Kr is not None:
                    print('..............................')
                    print('Kr[opt_subset1, opt_subset2].mean()', Kr[:, opt_subset2][opt_subset1, :].mean())
                    print('Kr.mean()', Kr.mean())
                    print('Kr[opt_subset1, opt_subset1].mean()', Kr[:, opt_subset1][opt_subset1, :].mean())
                    print('Kr[opt_subset2, opt_subset2].mean()', Kr[:, opt_subset2][opt_subset2, :].mean())

            print('.......................................')
            print(opt_subset1.size)
            print('opt_subset1', opt_subset1)

            path_tuples_opt_subset1 = curr_subset_path_tuples_kernel_references[opt_subset1]
            if curr_subset_path_tuples_kernel_references_weights is not None:
                path_tuples_opt_subset1_weights = curr_subset_path_tuples_kernel_references_weights[opt_subset1]
                for curr_element_in_ref_set_idx, curr_element_in_ref_set in enumerate(path_tuples_opt_subset1):
                    print(self.format_path_tuple_as_text(curr_element_in_ref_set), path_tuples_opt_subset1_weights[curr_element_in_ref_set_idx])
                del curr_element_in_ref_set_idx, curr_element_in_ref_set, path_tuples_opt_subset1_weights
            else:
                for curr_element_in_ref_set in path_tuples_opt_subset1:
                    print(self.format_path_tuple_as_text(curr_element_in_ref_set))
                del curr_element_in_ref_set
            del path_tuples_opt_subset1

            print('.......................................')
            print(opt_subset2.size)
            print('opt_subset2', opt_subset2)

            path_tuples_opt_subset2 = curr_subset_path_tuples_kernel_references[opt_subset2]
            if curr_subset_path_tuples_kernel_references_weights is not None:
                path_tuples_opt_subset2_weights = curr_subset_path_tuples_kernel_references_weights[opt_subset2]
                for curr_element_in_ref_set_idx, curr_element_in_ref_set in enumerate(path_tuples_opt_subset2):
                    print(self.format_path_tuple_as_text(curr_element_in_ref_set), path_tuples_opt_subset2_weights[curr_element_in_ref_set_idx])
                del curr_element_in_ref_set_idx, curr_element_in_ref_set, path_tuples_opt_subset2_weights
            else:
                for curr_element_in_ref_set in path_tuples_opt_subset2:
                    print(self.format_path_tuple_as_text(curr_element_in_ref_set))
                del curr_element_in_ref_set
            del path_tuples_opt_subset2

            path_tuples_kernel_references_objs[curr_set_idx] = curr_subset_path_tuples_kernel_references
            opt_subset1_objs[curr_set_idx] = opt_subset1
            opt_subset2_objs[curr_set_idx] = opt_subset2

            # computing hash vector
            if self.hash_func == 'RMM':
                assert self.is_subset_bits_for_objective_scores
                c, score_recomputed_for_opt_subsets = self.optimize_rmm_hash_for_subsets(
                    K,
                    Kr,
                    opt_subset1,
                    opt_subset2,
                    C_subsel_bits,
                    x,
                    class_labels,
                    data_idx_from_cluster=None,
                    sample_counts=path_tuples_all_counts,
                )
            elif self.hash_func == 'RkNN':
                assert self.is_subset_bits_for_objective_scores
                c, score_recomputed_for_opt_subsets = self.optimize_rknn_hash_for_subsets(
                    K,
                    opt_subset1,
                    opt_subset2,
                    C_subsel_bits,
                    x,
                    class_labels,
                    data_idx_from_cluster=None,
                    sample_counts=path_tuples_all_counts,
                )
            elif self.hash_func == 'RNeural':
                # c = Z[:, opt_idx].flatten()
                # assert c.size == x.size
                # score_recomputed_for_opt_subsets = scores[opt_idx]

                assert self.is_subset_bits_for_objective_scores
                # rather than relearning network, which is almost double compute time for the case of neural models,
                # we just use the ones from optimization
                c, score_recomputed_for_opt_subsets = self.optimize_rneural_hash_for_subsets(
                    path_tuples_all_embedding,
                    curr_subset_path_tuples_kernel_references_embedding,
                    opt_subset1,
                    opt_subset2,
                    C_subsel_bits,
                    x,
                    class_labels,
                    max_epochs=self.max_epochs,
                    data_idx_from_cluster=None,
                    sample_counts=path_tuples_all_counts,
                )
            else:
                raise AssertionError

            print('curr_set_idx', curr_set_idx)
            print(c.shape)

            print('score_recomputed_for_opt_subsets', score_recomputed_for_opt_subsets)
            scores_opt_hash_func[curr_set_idx] = score_recomputed_for_opt_subsets

            # opt_subset1 += path_tuples_kernel_references.size
            # opt_subset2 += path_tuples_kernel_references.size
            # set1_points[curr_set_idx, :] = opt_subset1
            # set2_points[curr_set_idx, :] = opt_subset2
            # path_tuples_kernel_references = np.concatenate((
            #                                     path_tuples_kernel_references,
            #                                     curr_subset_path_tuples_kernel_references
            #                                 ))

            print('scores_opt_hash_func', scores_opt_hash_func[:curr_set_idx+1])

            if curr_set_idx == 0:
                assert C is None
                C = c
            else:
                if curr_set_idx == 1:
                    C = C.reshape(C.size, 1)

                c = c.reshape(c.size, 1)
                C = np.hstack((C, c))
                print('C.shape', C.shape)

            # Commented on April 6, 2020. Deletion code may be in conflict with changes in code above, best to rewrite it carefully.
            # I think, the flag was False anyways.
            #
            # deletion_window_size = 25
            # if self.is_delete_old_hash and ((num_hash_computations % 4) == 3):
            #     # (curr_set_idx >= 3):
            #     print('Death of neurons (hash functions) ...')
            #     rnd_sample_uniform = self.npr_sample_prob_deletion_old_obj.random_sample()
            #     print('rnd_sample_uniform', rnd_sample_uniform)
            #
            #     if rnd_sample_uniform < 0.999999:
            #         # delete old and new hash functions
            #
            #         if (curr_set_idx+1) > deletion_window_size:
            #             curr_scores_opt_hash_func = scores_opt_hash_func[curr_set_idx + 1-deletion_window_size:curr_set_idx + 1]
            #         else:
            #             curr_scores_opt_hash_func = scores_opt_hash_func[:curr_set_idx+1]
            #
            #         deletion_threshold_score_opt_hash_func = curr_scores_opt_hash_func.mean() - self.deletion_z*curr_scores_opt_hash_func.std()
            #         print('deletion_threshold_score_opt_hash_func', deletion_threshold_score_opt_hash_func)
            #
            #         not_deleted_idx = np.where(scores_opt_hash_func[:curr_set_idx+1] > deletion_threshold_score_opt_hash_func)[0]
            #         print('not_deleted_idx', not_deleted_idx)
            #
            #         num_deletions = C.shape[1] - not_deleted_idx.size
            #         print('num_deletions', num_deletions)
            #
            #         if num_deletions > 0:
            #             print('deleting ...')
            #             C = C[:, not_deleted_idx]
            #             curr_set_idx -= num_deletions
            #             scores_opt_hash_func_old = np.copy(scores_opt_hash_func)
            #             scores_opt_hash_func[:curr_set_idx+1] = scores_opt_hash_func_old[not_deleted_idx]
            #             del scores_opt_hash_func_old
            #             scores_opt_hash_func[curr_set_idx+1:] = 0.0
            #             print('after deletion: scores_opt_hash_func', scores_opt_hash_func)

            print('Time to optimize greedy hash function', time.time() - start_time_hash_opt_greedy)

            print('self.data_operations_obj.time_load_wordvecs', self.data_operations_obj.time_load_wordvecs)

            if (len(C.shape) == 2) and ((C.shape[1] % 10) == 0):
                np.save('C', C)
                np.save('x', x)
                if self.is_inductive_ssh:
                    np.save('x_po', x_po)

        # assert np.all(set1_points >= 0)
        # assert np.all(set2_points >= 0)

        if self.hash_func in ['RMM', 'RkNN']:
            self.data_operations_obj.__lamb__ = org_lamda
            self.data_operations_obj.__sparse_kernel_threshold__ = org_sparse_kernel_threshold

        self.path_tuples_kernel_references_objs = path_tuples_kernel_references_objs
        self.opt_subset1_objs = opt_subset1_objs
        self.opt_subset2_objs = opt_subset2_objs

        # self.rnd_subsel_bits_for_clustering_objs = rnd_subsel_bits_for_clustering_objs
        # self.sel_cluster_for_sampling_objs = sel_cluster_for_sampling_objs
        # self.data_idx_from_cluster_objs = data_idx_from_cluster_objs
        # self.data_idx_from_neighbor_clusters_and_the_cluster_objs = data_idx_from_neighbor_clusters_and_the_cluster_objs
        # self.sel_neighboring_clusters_objs = sel_neighboring_clusters_objs
        # self.sel_cluster_for_sampling_objs = sel_cluster_for_sampling_objs
        # self.rnd_subsel_bits_for_clustering_objs = rnd_subsel_bits_for_clustering_objs

        # self.__dump__()

        if self.is_inductive_ssh:
            if is_return_all_unlabeled:
                idx_01 = np.where((x_po == 0) | (x_po == 1))[0]
                C_train_test = C[idx_01, :]
                del idx_01

                assert x.size == x_po.size
                idx_2 = np.where(x_po == 2)[0]
                C_unlabeled = C[idx_2, :]
                del idx_2

                if return_neighbor_clusters:
                    return C_train_test, C_unlabeled, data_idx_from_neighbor_clusters_and_the_cluster_objs, sel_cluster_for_sampling_objs, rnd_subsel_bits_for_clustering_objs, sel_neighboring_clusters_objs
                else:
                    return C_train_test, C_unlabeled
            else:
                idx_01 = np.where((x_po == 0) | (x_po == 1))[0]
                C_train_test = C[idx_01, :]
                del idx_01

                assert x.size == x_po.size
                idx_2_inferred = np.where((x == 1) & (x_po == 2))[0]
                C_unlabeled_inferred = C[idx_2_inferred, :]
                del idx_2_inferred

                if return_neighbor_clusters:
                    return C_train_test, C_unlabeled_inferred, data_idx_from_neighbor_clusters_and_the_cluster_objs, sel_cluster_for_sampling_objs, rnd_subsel_bits_for_clustering_objs, sel_neighboring_clusters_objs
                else:
                    return C_train_test, C_unlabeled_inferred
        else:
            if return_neighbor_clusters:
                return C, data_idx_from_neighbor_clusters_and_the_cluster_objs, sel_cluster_for_sampling_objs, rnd_subsel_bits_for_clustering_objs, sel_neighboring_clusters_objs
            else:
                return C

    def compute_hashcodes(self, path_tuples_all):
        # model is assumed to be trained

        assert self.hash_func in ['RkNN', 'RMM']
        assert not self.is_opt_reg_rmm
        assert not self.is_zero_kernel_compute_outside_cluster_and_neighboring

        path_tuples_kernel_references_objs = self.path_tuples_kernel_references_objs
        assert path_tuples_kernel_references_objs is not None
        num_hash_functions = path_tuples_kernel_references_objs.size

        opt_subset1_objs = self.opt_subset1_objs
        assert opt_subset1_objs is not None
        assert opt_subset1_objs.size == num_hash_functions

        opt_subset2_objs = self.opt_subset2_objs
        assert opt_subset2_objs is not None
        assert opt_subset2_objs.size == num_hash_functions

        C = None

        for curr_set_idx in range(num_hash_functions):
            start_time_hash_opt_greedy = time.time()

            print('......................................')
            print('curr_set_idx', curr_set_idx)

            curr_subset_path_tuples_kernel_references = path_tuples_kernel_references_objs[curr_set_idx]
            opt_subset1 = opt_subset1_objs[curr_set_idx]
            opt_subset2 = opt_subset2_objs[curr_set_idx]

            K, Kr = self.compute_kernel_for_hash_func(
                path_tuples_all,
                curr_subset_path_tuples_kernel_references,
                path_tuples_all_embedding=None,
                curr_subset_path_tuples_kernel_references_embedding=None,
                is_zero_kernel_compute_outside_cluster=False,
                data_idx_from_neighbor_clusters_and_the_cluster=None,
            )

            # computing hash vector
            if self.hash_func == 'RMM':
                c = self.rmm_obj.compute_hashcode_bit(
                    K=K, Kr=Kr,
                    subset1=opt_subset1,
                    subset2=opt_subset2,
                    reg=self.default_reg_val_rmm,
                    max_iter=self.rmm_max_iter,
                )
            elif self.hash_func == 'RkNN':
                c = self.rknn_obj.compute_hashcode_bit(
                    K=K,
                    subset1=opt_subset1,
                    subset2=opt_subset2,
                )
            else:
                raise AssertionError

            print('curr_set_idx', curr_set_idx)
            print(c.shape)

            if curr_set_idx == 0:
                assert C is None
                C = c
            else:
                if curr_set_idx == 1:
                    C = C.reshape(C.size, 1)

                c = c.reshape(c.size, 1)
                C = np.hstack((C, c))
                print('C.shape', C.shape)

            print('Time to compute hash function', time.time() - start_time_hash_opt_greedy)

            if (len(C.shape) == 2) and ((C.shape[1] % 10) == 0):
                np.save('C', C)

        return C

    def __dump__(self):

        print('dumping the object ....')
        start_time = time.time()

        # self.data_operations_obj.__dump__()
        self.data_operations_obj = None

        np.save('path_tuples_kernel_references_objs', self.path_tuples_kernel_references_objs)
        self.path_tuples_kernel_references_objs = None

        np.save('opt_subset1_objs', self.opt_subset1_objs)
        self.opt_subset1_objs = None

        np.save('opt_subset2_objs', self.opt_subset2_objs)
        self.opt_subset2_objs = None

        # np.save('rnd_subsel_bits_for_clustering_objs', self.rnd_subsel_bits_for_clustering_objs)
        # self.rnd_subsel_bits_for_clustering_objs = None
        #
        # np.save('sel_cluster_for_sampling_objs', self.sel_cluster_for_sampling_objs)
        # self.sel_cluster_for_sampling_objs = None

        # np.save('data_idx_from_cluster_objs', self.data_idx_from_cluster_objs)
        # self.data_idx_from_cluster_objs = None

        # np.save('sel_neighboring_clusters_objs', self.sel_neighboring_clusters_objs)
        # self.sel_neighboring_clusters_objs = None

        # np.save('data_idx_from_neighbor_clusters_and_the_cluster_objs',
        #         self.data_idx_from_neighbor_clusters_and_the_cluster_objs)
        # self.data_idx_from_neighbor_clusters_and_the_cluster_objs = None

        with open(file_path+'.pickle', 'wb') as h:
            pickle.dump(self, h)

        print('time to dump was ', time.time()-start_time)


def load():
    with open(file_path+'.pickle', 'wb') as h:
        obj = pickle.load(h)

    obj.path_tuples_kernel_references_objs = np.load('./path_tuples_kernel_references_objs.npy')
    obj.opt_subset1_objs = np.load('./opt_subset1_objs.npy')
    obj.opt_subset2_objs = np.load('./opt_subset2_objs.npy')
    obj.data_idx_from_neighbor_clusters_and_the_cluster_objs = np.load('./data_idx_from_neighbor_clusters_and_the_cluster_objs.npy')

    return obj
