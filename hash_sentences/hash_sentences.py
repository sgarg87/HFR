import math
import numpy as np
import nltk
from . import text_kernel as tk
from . import expanding_contractions as ec
from . import hash_function_representations_info_theoretic_optimization as hf_ito


file_path = 'hash_sentences_model'


class HashSentences:

    def __init__(
            self,
            hash_func,
            num_cores=1,
            alpha=32,
            alpha_compute_per_cluster_size__log_scale=4.0,
            is_bert_embedding=False,
            is_zero_kernel_compute_outside_cluster=False,
            max_hamming_dist_bits_frac_for_neighborhood=0.4,
            wordvec_file_path=None
    ):
        self.hash_func = hash_func
        self.num_cores = num_cores
        self.data_operations_obj = None
        self.min_freq_ngram = 2
        self.is_zero_kernel_compute_outside_cluster = is_zero_kernel_compute_outside_cluster
        self.alpha = alpha
        self.alpha_compute_per_cluster_size__log_scale = alpha_compute_per_cluster_size__log_scale
        self.is_bert_embedding = is_bert_embedding
        self.max_hamming_dist_bits_frac_for_neighborhood = max_hamming_dist_bits_frac_for_neighborhood
        assert wordvec_file_path is not None
        self.wordvec_file_path = wordvec_file_path

    def init_data_operations_obj(self, is_edge_label, is_custom_p_weights=True):

        if self.data_operations_obj is None:

            lamb = 0.95
            # lamb = 0.9
            cs = 0.65
            # cs = 0.0
            # cs = 0.5
            # cs = 0.75
            # cs = 0.80

            if is_custom_p_weights:
                # p_weights = [0.01, 0.1, 1.0]
                # p_weights = [0.01, 0.1, 1.0, 10.0]

                p_weights = [0.01, 0.03, 0.1, 0.3, 1.0]

                # p_weights = [0.01, 0.01, 0.1, 0.1, 1.0, 10.0]
                # p_weights = [0.01, 0.01, 0.1, 0.1, 1.0, 1.0, 10.0]

                # p_weights = [0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 1.0, 3.0, 10.0]

                p = len(p_weights)-1
            else:
                p = 4
                is_uniform_weights = True
                if is_uniform_weights:
                    p_weights = self.__compute_uniform_weight_subsequence_length__(p=p)
                else:
                    p_weights = self.__compute_weight_subsequences_length__(
                        p=p,
                        lamb=lamb,
                        weight_power_factor=2,
                    )
                print('p_weights', p_weights)

            is_normalize_kernel_fr_hashcodes = True
            if self.is_bert_embedding:
                self.data_operations_obj = btk.TextKernel(
                    num_cores=self.num_cores,
                    lamb=lamb,
                    cs=cs,
                    p=p,
                    is_normalize_kernel_fr_hashcodes=is_normalize_kernel_fr_hashcodes,
                    p_weights=p_weights,
                )
            else:
                self.data_operations_obj = tk.TextKernel(
                    num_cores=self.num_cores,
                    is_edge_label_sim=is_edge_label,
                    lamb=lamb,
                    cs=cs,
                    p=p,
                    is_normalize_kernel_fr_hashcodes=is_normalize_kernel_fr_hashcodes,
                    p_weights=p_weights,
                    is_wordvec_large=False,
                    wordvec_file_path=self.wordvec_file_path,
                )
            del is_normalize_kernel_fr_hashcodes

    def __compute_weight_subsequences_length__(self, p, lamb, weight_power_factor=1):

        # Even if there is no gap, long sub-sequence matches get less weight due to lambda decay.
        # That is not appropriate. So, computing weights as per value of p.
        # self.__p_weights__ = np.array([self.__lamb__**x for x in range(self.__p__+1)])
        # even more stronger weights on

        p_weights = np.array([(lamb**(x*weight_power_factor)) for x in range(p+1)])
        p_weights = 1/p_weights
        p_weights /= p_weights.sum()
        return p_weights

    def __compute_uniform_weight_subsequence_length__(self, p):
        p_weights = np.ones(p + 1)
        p_weights /= (p_weights.sum() + 1e-100)
        return p_weights

    def init_hash_func_obj(
        self,
        seed_val=0,
        max_num_hash_functions_for_sampling_from_cluster=None,
        is_edge_label=False,
        is_self_dialog_learning=False,
    ):
        self.init_data_operations_obj(
            is_edge_label=is_edge_label
        )

        is_divide_conquer_else_brute_force = True
        is_optimize_split = True

        hash_func_obj = hf_ito.HashFunctionRepresentations(
            hash_func=self.hash_func,
            data_operations_obj=self.data_operations_obj,
            is_semisupervised_else_unsupervised=False,
            is_alpha_sampling_weighted=False,
            num_cores=self.num_cores,
            is_opt_reg_rmm=False,
            is_opt_kernel_par=False,
            is_rnd_sample_combinations_for_opt_kernel_par=True,
            frac_trn_data=None,
            is_inductive_ssh=False,
            max_num_hash_functions_for_sampling_from_cluster=max_num_hash_functions_for_sampling_from_cluster,
            is_convolution_neural=False,
            is_images=False,
            labeled_score_frac=None,
            is_joint_inductive_transductive_ssh=None,
            is_weighted_transductive=None,
            seed_val=seed_val,
            # default_reg_val_rmm=0.3,
            default_reg_val_rmm=1.0,
            rmm_max_iter=100,
            is_conditional_birth=False,
            is_delete_old_hash=False,
            # high value suitable if not computing redundancy scores since marginal entropies are concentrated
            # deletion_z=4.0,
            deletion_z=2.0,
            # deletion_z=1.25,
            is_optimize_split=is_optimize_split,
            is_divide_conquer_else_brute_force=is_divide_conquer_else_brute_force,
            min_set_size_to_split=4,
            is_optimize_hash_functions_nonredundancy=True,
            nonredundancy_score_beta=100.0,
            is_random_sample_data_for_computing_objective=False,
            is_edge_label=is_edge_label,
            rnd_sample_obj_size_log_base=10.0,
            is_cluster_high_weightage=True,
            cluster_sample_weights=10,
            is_zero_kernel_compute_outside_cluster=self.is_zero_kernel_compute_outside_cluster,
            is_alpha_per_cluster_size=True,
            # alpha_compute_per_cluster_size__log_scale=3.0,
            alpha_compute_per_cluster_size__log_scale=self.alpha_compute_per_cluster_size__log_scale,
            # alpha_compute_per_cluster_size__log_scale=5.0,
            # alpha_compute_per_cluster_size__log_scale=8.0,
            is_cluster_size_and_entropy=False,
            ref_set_applicable_for_max_num_bits_to_cluster=None,
            max_hamming_dist_bits_frac_for_neighborhood=self.max_hamming_dist_bits_frac_for_neighborhood,
            is_bert=self.is_bert_embedding,
            wordvec_file_path=self.wordvec_file_path,
            is_self_dialog_learning=is_self_dialog_learning,
            is_num_hash_bits_for_cluster_dynamic=False,
        )
        del is_divide_conquer_else_brute_force, is_optimize_split

        return hash_func_obj

    def process_text(self, text):

        new_text = ''

        for x in text:
            if (x == ' ') or (x.isalpha()):
                pass
            else:
                x = ' '
            new_text += x

        while True:
            if '   ' in new_text:
                new_text = new_text.replace('   ', ' ')
            elif '  ' in new_text:
                new_text = new_text.replace('  ', ' ')
            else:
                break

        new_text = new_text.strip()
        new_text = new_text.lower()

        return new_text

    def preprocess_text_sent(self, sent, is_print=False):
        sent = ec.expand_contractions(sent)
        if is_print:
            print(sent)

        # curr_sent = ''.join(e for e in curr_sent if (e.isalpha() or e == ' '))
        sent = self.process_text(sent)
        if is_print:
            print(sent)

        return sent

    def preprocess_and_tokenize_sent(self, sent, is_print=True, is_pos_tag=True):
        sent = self.preprocess_text_sent(sent=sent, is_print=is_print)

        if (sent is None) or (not sent):
            return None, 0
        else:
            token_sent = nltk.word_tokenize(sent)

            if is_pos_tag:
                token_sent = nltk.pos_tag(token_sent)
                # print token_sent
            assert token_sent is not None

            sent_len = len(token_sent)

            return token_sent, sent_len

    def preprocess_and_tokenize_sentences(self, sentences, is_return_sentences=False):

        num_sentenes = sentences.size
        tokenized_sentences = np.empty(num_sentenes, dtype=np.object)
        len_sentences = np.zeros(num_sentenes, dtype=np.int)

        # non_null_idx = []

        for curr_idx in range(num_sentenes):
            print('.........................')
            curr_sent = sentences[curr_idx]
            print(curr_sent)

            assert curr_sent is not None
            curr_token_sent, curr_sent_len = self.preprocess_and_tokenize_sent(
                sent=curr_sent,
            )

            if curr_token_sent is None:
                curr_token_sent, curr_sent_len = self.preprocess_and_tokenize_sent(
                    sent='Null',
                )
                assert curr_token_sent is not None

            # if (curr_token_sent is not None) and (curr_sent_len > 0):
            #     non_null_idx.append(curr_idx)

            len_sentences[curr_idx] = curr_sent_len
            tokenized_sentences[curr_idx] = curr_token_sent

        # non_null_idx = np.array(non_null_idx)
        # len_sentences = len_sentences[non_null_idx]
        # tokenized_sentences = tokenized_sentences[non_null_idx]
        # if is_return_sentences:
        #     sentences = sentences[non_null_idx]

        if is_return_sentences:
            return tokenized_sentences, len_sentences, sentences
        else:
            return tokenized_sentences, len_sentences

    def hashcodes_for_sentences(
            self,
            sentences,
            seed_val=0,
            num_hash_bits=10,
            unique_sentences_counts=None,
            unique_sentences_vocab_count=None,
            is_self_dialog_learning=False,
    ):
        # max_num_hash_functions_for_sampling_from_cluster = np.array([16, 32, 64, 128])
        max_num_hash_functions_for_sampling_from_cluster = np.array(
            [
                # int(math.log(num_hash_bits, 2)),
                # int(math.sqrt(num_hash_bits)),
                # int(math.log(num_hash_bits, 3)**2),
                num_hash_bits,
            ]
        )
        # max_num_hash_functions_for_sampling_from_cluster = np.array([2, 4, 8, 16, 32, 64, 128])

        hash_func_obj = self.init_hash_func_obj(
            seed_val=seed_val,
            max_num_hash_functions_for_sampling_from_cluster=max_num_hash_functions_for_sampling_from_cluster,
            is_self_dialog_learning=is_self_dialog_learning,
        )

        alpha = np.array([self.alpha])
        hashcodes, data_idx_from_neighbor_clusters_and_the_cluster_objs, clusters, rnd_subsel_bits_for_clustering_objs, neighboring_clusters = \
            hash_func_obj.optimize_reference_subsets_greedy_stochastic(
                sentences,
                None,
                num_hash_bits,
                alpha,
                class_labels=None,
                unlabeled_path_tuples_arr=None,
                ref_sample_idx=None,
                path_tuples_all_counts_org=unique_sentences_counts,
                path_tuples_all_vocab_count=unique_sentences_vocab_count,
                return_neighbor_clusters=True,
            )
        assert hashcodes.shape[1] == num_hash_bits
        assert data_idx_from_neighbor_clusters_and_the_cluster_objs.size == num_hash_bits
        assert clusters.size == num_hash_bits
        assert neighboring_clusters.size == num_hash_bits
        assert rnd_subsel_bits_for_clustering_objs.size == num_hash_bits

        # if self.is_zero_kernel_compute_outside_cluster:
        #     hashcodes_non_binary = np.copy(hashcodes).astype(np.int)
        #     all_idx = np.arange(hashcodes.shape[0], dtype=np.int)
        #     for curr_bit_idx in range(num_hash_bits):
        #         curr_bit__data_idx_from_neighbor_clusters = data_idx_from_neighbor_clusters_and_the_cluster_objs[curr_bit_idx]
        #
        #         if curr_bit__data_idx_from_neighbor_clusters is not None:
        #             complement__curr_bit_data_idx_from_neighbor_clusters = np.setdiff1d(
        #                 all_idx,
        #                 curr_bit__data_idx_from_neighbor_clusters,
        #             )
        #             del curr_bit__data_idx_from_neighbor_clusters
        #             hashcodes_non_binary[complement__curr_bit_data_idx_from_neighbor_clusters, curr_bit_idx] = -1
        #
        #     hashcodes = hashcodes_non_binary
        #     del hashcodes_non_binary

        return hashcodes, neighboring_clusters, clusters, rnd_subsel_bits_for_clustering_objs

    def remove_pos_tags(self, ngrams):
        num_ngrams = ngrams.size

        print('..............................')
        for curr_idx in range(num_ngrams):
            curr_ngram = ngrams[curr_idx]

            print(curr_ngram)
            for curr_token_idx in range(len(curr_ngram)):
                curr_token = curr_ngram[curr_token_idx]
                curr_token = list(curr_token)
                curr_token[1] = 'X'
                curr_token = tuple(curr_token)
                curr_ngram[curr_token_idx] = curr_token
                print(curr_token)

            print(curr_ngram)

        return ngrams

    def compute_vocab_counts_map_from_sentences(self, tokenized_sentences):
        if not self.is_bert_embedding:
            tokenized_sentences_str = self.encode_ngram_objs_strings(tokenized_sentences)
            del tokenized_sentences
        else:
            tokenized_sentences_str = tokenized_sentences

        sent_vocab_count_map = {}
        for curr_sent in tokenized_sentences_str:
            tokens_in_sent = curr_sent.split()
            del curr_sent

            for curr_token in tokens_in_sent:
                if curr_token not in sent_vocab_count_map:
                    sent_vocab_count_map[curr_token] = 0
                sent_vocab_count_map[curr_token] += 1
        return sent_vocab_count_map

    def encode_ngram_objs_strings(self, ngram_objs):
        ngram_strings = np.empty(ngram_objs.size, dtype=np.object)

        for curr_ngram_idx, curr_ngram_obj in enumerate(ngram_objs):
            print(curr_ngram_obj)

            curr_ngram_str = self.encode_ngram_as_string(curr_ngram_obj)
            ngram_strings[curr_ngram_idx] = curr_ngram_str
            del curr_ngram_str

        return ngram_strings

    def encode_ngram_as_string(self, curr_ngram_obj):
        if isinstance(curr_ngram_obj, dict):
            curr_ngram_obj = curr_ngram_obj['path_tuple']
        else:
            assert isinstance(curr_ngram_obj, list)

        curr_ngram_str = ''
        for curr_token in curr_ngram_obj:
            curr_ngram_str += ' ' + curr_token[0]
        curr_ngram_str = curr_ngram_str.strip()

        return curr_ngram_str

    def compute_sentence_vocab_log_count_mean(
            self,
            sentences_str, vocab_count_map,
    ):
        sentences_vocab_log_count_mean = np.zeros(sentences_str.size, dtype=np.int)
        for curr_sent_idx, curr_sent_str in enumerate(sentences_str):
            # print '................................'
            # print curr_sent_str
            tokens_in_sentence = curr_sent_str.split()
            num_tokens = len(tokens_in_sentence)
            curr_sentence__sum_of_vocab_log_count = 0.0
            for curr_token in tokens_in_sentence:
                curr_count = vocab_count_map[curr_token]
                curr_log_of_count = max(math.log(curr_count, 2.0), 1.0)
                del curr_count
                # print '{}, {}'.format(curr_token, curr_log_of_count)
                curr_sentence__sum_of_vocab_log_count += curr_log_of_count
            curr_sentence__sum_of_vocab_log_count = int(round(curr_sentence__sum_of_vocab_log_count/float(num_tokens)))
            assert curr_sentence__sum_of_vocab_log_count >= 1
            sentences_vocab_log_count_mean[curr_sent_idx] = curr_sentence__sum_of_vocab_log_count
        assert np.all(sentences_vocab_log_count_mean >= 1)
        return sentences_vocab_log_count_mean

    def main(self, sentences, is_remove_pos_tags=False, num_hash_bits=100, is_self_dialog_learning=False, seed_val=0):

        print('Number of sentences for hashing', sentences.size)

        if self.is_bert_embedding:
            return self.main_bert(
                sentences=sentences, num_hash_bits=num_hash_bits,
            )

        assert sentences is not None
        tokenized_sentences, _ = self.preprocess_and_tokenize_sentences(sentences=sentences)
        assert tokenized_sentences.size == sentences.size
        sentences = tokenized_sentences
        del tokenized_sentences

        vocab_count_map = self.compute_vocab_counts_map_from_sentences(
            tokenized_sentences=sentences,
        )

        if is_remove_pos_tags:
            print('+'*40)
            sentences = self.remove_pos_tags(sentences)

        if is_self_dialog_learning:
            sentences_str = self.encode_ngram_objs_strings(sentences)
            assert sentences_str.size == sentences.size

            sentences_vocab_log_count_mean = self.compute_sentence_vocab_log_count_mean(
                sentences_str=sentences_str,
                vocab_count_map=vocab_count_map,
            )
            del vocab_count_map
            print('sentences_vocab_log_count_mean', sentences_vocab_log_count_mean)
            assert sentences_vocab_log_count_mean.size == sentences_str.size

            sentences_counts = np.ones(sentences.size, dtype=np.int)
            assert is_self_dialog_learning
            sentences_hashcodes, neighboring_clusters, clusters, rnd_subsel_bits_for_clustering_objs =\
                self.hashcodes_for_sentences(
                    sentences,
                    num_hash_bits=num_hash_bits,
                    unique_sentences_counts=sentences_counts,
                    unique_sentences_vocab_count=sentences_vocab_log_count_mean,
                    is_self_dialog_learning=True,
                    seed_val=seed_val,
                )
            print('sentences_hashcodes.shape', sentences_hashcodes.shape)
            assert sentences_hashcodes.shape[0] == sentences.size
            assert sentences_hashcodes.shape[1] == num_hash_bits
        else:
            unique_sentences, unique_sentences_inv_indices, unique_sentences_counts = np.unique(
                sentences, return_inverse=True, return_counts=True,
            )
            assert unique_sentences_inv_indices.size == sentences.size
            assert unique_sentences.size == unique_sentences_counts.size

            print('unique_sentences.size', unique_sentences.shape)
            print('unique_sentences_inv_indices', unique_sentences_inv_indices.shape)
            # print 'unique_sentences_inv_indices', unique_sentences_inv_indices

            unique_sentences_str = self.encode_ngram_objs_strings(unique_sentences)
            assert unique_sentences_str.size == unique_sentences.size

            unique_sentences_vocab_log_count_mean = self.compute_sentence_vocab_log_count_mean(
                sentences_str=unique_sentences_str,
                vocab_count_map=vocab_count_map,
            )
            print('unique_sentences_vocab_log_count_mean', unique_sentences_vocab_log_count_mean)
            assert unique_sentences_vocab_log_count_mean.size == unique_sentences_str.size
            del unique_sentences_str

            unique_sentences_hashcodes, neighboring_clusters, clusters, rnd_subsel_bits_for_clustering_objs =\
                self.hashcodes_for_sentences(
                    unique_sentences,
                    num_hash_bits=num_hash_bits,
                    unique_sentences_counts=unique_sentences_counts,
                    unique_sentences_vocab_count=unique_sentences_vocab_log_count_mean,
                    seed_val=seed_val,
                )
            del unique_sentences_vocab_log_count_mean, unique_sentences_counts
            print('unique_sentences_hashcodes.shape', unique_sentences_hashcodes.shape)
            assert unique_sentences_hashcodes.shape[0] == unique_sentences.size
            assert unique_sentences_hashcodes.shape[1] == num_hash_bits
            del unique_sentences

            np.save('unique_sentences_hashcodes', unique_sentences_hashcodes)

            sentences_hashcodes = unique_sentences_hashcodes[unique_sentences_inv_indices, :]
            del unique_sentences_hashcodes, unique_sentences_inv_indices

        assert sentences_hashcodes.shape[0] == sentences.size
        assert sentences_hashcodes.shape[1] == num_hash_bits
        # np.save('sentences_hashcodes', sentences_hashcodes)

        return sentences_hashcodes, neighboring_clusters, clusters, rnd_subsel_bits_for_clustering_objs

    def main_bert(self, sentences, num_hash_bits=100, is_self_dialog_learning=False, seed_val=0):
        assert sentences is not None
        assert not is_self_dialog_learning, 'not implemented, need to make sure not to use unique sentences but original order, refactor code'

        vocab_count_map = self.compute_vocab_counts_map_from_sentences(
            tokenized_sentences=sentences,
        )

        unique_sentences, unique_sentences_inv_indices, unique_sentences_counts = np.unique(
            sentences, return_inverse=True, return_counts=True,
        )
        assert unique_sentences_inv_indices.size == sentences.size
        assert unique_sentences.size == unique_sentences_counts.size

        print('unique_sentences.size', unique_sentences.shape)
        print('unique_sentences_inv_indices', unique_sentences_inv_indices.shape)
        # print 'unique_sentences_inv_indices', unique_sentences_inv_indices

        unique_sentences_vocab_log_count_mean = self.compute_sentence_vocab_log_count_mean(
            sentences_str=unique_sentences,
            vocab_count_map=vocab_count_map,
        )
        print('unique_sentences_vocab_log_count_mean', unique_sentences_vocab_log_count_mean)
        assert unique_sentences_vocab_log_count_mean.size == unique_sentences.size

        unique_sentences_hashcodes, neighboring_clusters, clusters = self.hashcodes_for_sentences(
            unique_sentences,
            num_hash_bits=num_hash_bits,
            unique_sentences_counts=unique_sentences_counts,
            unique_sentences_vocab_count=unique_sentences_vocab_log_count_mean,
            is_self_dialog_learning=is_self_dialog_learning,
            seed_val=seed_val,
        )
        print('unique_sentences_hashcodes.shape', unique_sentences_hashcodes.shape)
        assert unique_sentences_hashcodes.shape[0] == unique_sentences.size
        assert unique_sentences_hashcodes.shape[1] == num_hash_bits
        np.save('unique_sentences_hashcodes', unique_sentences_hashcodes)

        sentences_hashcodes = unique_sentences_hashcodes[unique_sentences_inv_indices, :]
        assert sentences_hashcodes.shape[0] == sentences.size
        assert sentences_hashcodes.shape[1] == num_hash_bits
        np.save('sentences_hashcodes', sentences_hashcodes)

        return sentences_hashcodes

    def format_path_tuple_as_text(
            self,
            curr_element_in_ref_set
    ):
        curr_text = ' '.join([curr_pos_tuple[0] for curr_pos_tuple in curr_element_in_ref_set])
        curr_text = '\'{}\''.format(curr_text)
        return curr_text
