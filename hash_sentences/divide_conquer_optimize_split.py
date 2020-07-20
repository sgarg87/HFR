import numpy as np
import time
import math


class DivideConquerOptimizeSplit:
    # this is relevant for optimizing splits of large sets

    def __init__(self,
                 alpha,
                 hash_func,
                 min_set_size_to_split=4):

        # assert (alpha%min_set_size_to_split) == 0
        assert hash_func in ['RMM', 'RkNN']
        self.alpha = alpha
        self.hash_func = hash_func
        self.min_set_size_to_split = min_set_size_to_split

    def random_divide(self, set):
        # assert (set.size%self.min_set_size_to_split) == 0
        num_divisions = int(math.ceil(float(set.size)/self.min_set_size_to_split))
        print('num_divisions', num_divisions)

        divisions = [[] for d1 in range(num_divisions)]
        for curr_division in range(num_divisions):
            divisions[curr_division] = np.arange(curr_division, set.size, num_divisions, dtype=np.int)
        print('divisions', divisions)

        return divisions

    def elements_in_division(self, division, splits):

        elements = np.array([], dtype=np.int)

        for curr_idx in division:
            assert isinstance(splits[curr_idx], np.ndarray)
            elements = np.concatenate((elements, splits[curr_idx]))

        return elements

    def expand_combinations_per_elements_in_splits(self,
                                                   subsets_arr,
                                                   splits):

        assert subsets_arr.dtype == np.object

        for curr_idx in range(subsets_arr.size):
            curr_subset = subsets_arr[curr_idx]
            elements_in_subset = self.elements_in_division(curr_subset, splits=splits)
            subsets_arr[curr_idx] = elements_in_subset

        return subsets_arr

    def map_combinations_via_dict(self, subsets_arr, dict_element_in_division_to_index):

        assert subsets_arr.dtype == np.object

        new_subsets_arr = -10000*np.ones(subsets_arr.shape, dtype=np.object)

        for curr_idx in range(subsets_arr.size):
            curr_subset = subsets_arr[curr_idx]
            curr_subset = self.map_elements_via_dict(
                elements=curr_subset,
                dict_element_in_division_to_index=dict_element_in_division_to_index,
            )
            new_subsets_arr[curr_idx] = curr_subset

        return new_subsets_arr

    def map_elements_via_dict(self, elements, dict_element_in_division_to_index):

        assert elements.dtype == np.int
        new_elements = -1000*np.ones(elements.shape, dtype=elements.dtype)

        for curr_idx in range(elements.size):
            curr_element = elements[curr_idx]
            new_elements[curr_idx] = dict_element_in_division_to_index[curr_element]

        return new_elements

    def inv_array_indices(self, elements_in_division):

        if elements_in_division.size > 100:
            print('warning: this implementation may be slow for large arrays')
            # raise AssertionError, 'this implementation may be slow for large arrays'

        dict_element_in_division_to_index = {}
        for curr_idx in range(elements_in_division.size):
            curr_element = elements_in_division[curr_idx]
            dict_element_in_division_to_index[curr_element] = curr_idx

        return dict_element_in_division_to_index

    def get_combinations(self,
                         curr_division,
                         splits,
                         info_theoretic_opt_obj,
    ):

        subsets1_arr, subsets2_arr = info_theoretic_opt_obj.get_all_combinations(
            subset_size=curr_division.size, superset=curr_division
        )

        subsets1_arr = self.expand_combinations_per_elements_in_splits(
            subsets1_arr,
            splits,
        )
        subsets2_arr = self.expand_combinations_per_elements_in_splits(
            subsets2_arr,
            splits,
        )

        return subsets1_arr, subsets2_arr

    def optimize(self,
                 path_tuples_all,
                 curr_subset_path_tuples_kernel_references,
                 path_tuples_all_embedding,
                 curr_subset_path_tuples_kernel_references_embedding,
                 C,
                 x,
                 y,
                 data_idx_from_clusters,
                 subset_size,
                 info_theoretic_opt_obj,
                 sample_counts=None,
                 is_zero_kernel_compute_outside_cluster=False,
                 data_idx_from_neighbor_clusters_and_the_cluster=None,
                 references_idx_in_all=None,
    ):
        assert subset_size == curr_subset_path_tuples_kernel_references.size

        K, Kr = info_theoretic_opt_obj.compute_kernel_for_hash_func(
            path_tuples_all,
            curr_subset_path_tuples_kernel_references,
            path_tuples_all_embedding,
            curr_subset_path_tuples_kernel_references_embedding,
            is_zero_kernel_compute_outside_cluster=is_zero_kernel_compute_outside_cluster,
            data_idx_from_neighbor_clusters_and_the_cluster=data_idx_from_neighbor_clusters_and_the_cluster,
            references_idx_in_all=references_idx_in_all,
        )

        splits = np.arange(subset_size, dtype=np.int).reshape((subset_size, 1))

        while len(splits) > 2:

            print('*****************************************')
            print('splits', splits)

            num_splits = len(splits)

            set = np.arange(num_splits, dtype=np.int)
            print('set', set)

            divisions = self.random_divide(set)
            print('divisions', divisions)

            new_splits = []

            for curr_division in divisions:

                print('.............................')
                print('curr_division', curr_division)

                # --------------------------------------------------------------------------------
                start_time_preprocessing = time.time()

                elements_in_division = self.elements_in_division(curr_division, splits=splits)
                print('elements_in_division', elements_in_division)

                subsets1_arr, subsets2_arr = self.get_combinations(
                    curr_division,
                    splits,
                    info_theoretic_opt_obj,
                )
                num_combinations = subsets1_arr.size
                assert num_combinations == subsets2_arr.size
                print('subsets1_arr', subsets1_arr)
                print('subsets2_arr', subsets2_arr)

                # taking appropriate projections of columns from the kernel matrices and mapping the combinations accordingly
                curr_K = K[:, elements_in_division]
                if Kr is not None:
                    curr_Kr = Kr[elements_in_division, :]
                    curr_Kr = curr_Kr[:, elements_in_division]
                else:
                    curr_Kr = None
                dict_element_in_division_to_index = self.inv_array_indices(elements_in_division)
                subsets1_arr_adjusted_per_kernel_matrix = self.map_combinations_via_dict(
                    subsets_arr=subsets1_arr,
                    dict_element_in_division_to_index=dict_element_in_division_to_index,
                )
                subsets2_arr_adjusted_per_kernel_matrix = self.map_combinations_via_dict(
                    subsets_arr=subsets2_arr,
                    dict_element_in_division_to_index=dict_element_in_division_to_index,
                )

                print(time.time() - start_time_preprocessing)
                # -----------------------------------------------------------------------------------------------------

                scores = info_theoretic_opt_obj.evaluate_hash_func_scores(
                    None,
                    None,
                    curr_K,
                    curr_Kr,
                    C,
                    x,
                    y,
                    num_combinations,
                    data_idx_from_clusters,
                    subsets1_arr_adjusted_per_kernel_matrix,
                    subsets2_arr_adjusted_per_kernel_matrix,
                    elements_in_division.size,
                    sample_counts=sample_counts,
                )

                opt_idx = scores.argmax()
                print('opt_idx', opt_idx)

                max_score = scores[opt_idx]

                opt_subset1 = subsets1_arr[opt_idx]
                print('opt_subset1', opt_subset1)
                new_splits.append(np.copy(opt_subset1))

                opt_subset2 = subsets2_arr[opt_idx]
                print('opt_subset2', opt_subset2)
                new_splits.append(np.copy(opt_subset2))

            new_splits = np.array(new_splits)
            splits = new_splits
            del new_splits

        print('splits', splits)
        assert isinstance(splits, np.ndarray)
        print('splits.shape', splits.shape)
        assert splits.shape[0] == 2
        assert (splits[0].size + splits[1].size) == subset_size

        return splits[0], splits[1], max_score, K, Kr
