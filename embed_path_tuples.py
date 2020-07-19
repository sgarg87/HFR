import numpy as np
import numpy.random as npr
from . import word_embeddings as we
import math


class EmbedPathTuples:

    def __init__(
            self,
            is_random=False,
            word_vec_len=100,
            max_response_length_fr_embedding=12000,
            num_words_per_lstm_step=10,
            is_word_only=True,
            wordvec_file_path=None,
    ):
        self.is_random = is_random
        self.word_vec_len = word_vec_len

        if not self.is_random:
            assert self.word_vec_len in [100, 300]

            if self.word_vec_len == 100:
                is_wordvec_large = False
            elif self.word_vec_len == 300:
                is_wordvec_large = True
            else:
                raise AssertionError

            self.word_embedding_obj = we.WordEmbeddings(
                is_wordvec_large=is_wordvec_large,
                wordvec_file_path=wordvec_file_path,
            )

        self.max_response_length_fr_embedding = max_response_length_fr_embedding
        self.num_words_per_lstm_step = num_words_per_lstm_step
        self.is_word_only = is_word_only

    def __get_word_vector__(self, label):
        assert label is not None

        if not self.is_random:
            word_vector = self.word_embedding_obj.__get_word_vector__(word_label=label)
            return word_vector
        else:
            if (not hasattr(self, '__randomvec_buffer__')) or (self.__randomvec_buffer__ is None):
                print('initializing random vectors buffer')
                self.__randomvec_buffer__ = {}

            if label not in self.__randomvec_buffer__:
                random_i_vector = npr.random(self.word_vec_len)
                self.__randomvec_buffer__[label] = random_i_vector
            else:
                random_i_vector = self.__randomvec_buffer__[label]

            return random_i_vector

    def embed_path_tuples_neural(
            self, path_tuples, is_var_length=False, is_max_len_frm_data=False,
            is_multiword_as_single=False, is_bert=False
    ):
        assert not is_bert, 'not implemented'

        if self.is_word_only:
            word_len = self.word_vec_len
        else:
            word_len = self.word_vec_len * 2

        max_path_len = 1
        num_path_tuples = path_tuples.size

        for curr_path_idx in range(num_path_tuples):
            curr_path_tuple = path_tuples[curr_path_idx]
            if 'path_tuple' in curr_path_tuple:
                curr_path_tuple = curr_path_tuple['path_tuple']

            curr_path_len = len(curr_path_tuple)
            # print curr_path_len

            max_path_len = max(max_path_len, curr_path_len)

        print('max_path_len', max_path_len)

        if is_var_length:
            embeddings = []
        else:
            if is_max_len_frm_data:
                embeddings = np.zeros(
                    (num_path_tuples, max_path_len, word_len)
                )
            else:
                embeddings = np.zeros(
                    (num_path_tuples, self.max_response_length_fr_embedding, word_len)
                )

        for curr_path_idx in range(num_path_tuples):
            curr_path_tuple = path_tuples[curr_path_idx]
            if 'path_tuple' in curr_path_tuple:
                curr_path_tuple = curr_path_tuple['path_tuple']
            # print 'curr_path_tuple', curr_path_tuple

            curr_path_len = len(curr_path_tuple)
            # print curr_path_len

            if is_var_length:
                curr_embedding = np.zeros((curr_path_len, word_len))

            # if not is_var_length:
            #     assert curr_path_len <= self.max_response_length_fr_embedding

            curr_label_idx = -1

            for curr_tuple in curr_path_tuple:

                curr_label_idx += 1

                # print curr_tuple

                if not self.is_word_only:
                    curr_label_vec1 = self.__get_word_vector__(curr_tuple[0])
                    curr_label_vec2 = self.__get_word_vector__(curr_tuple[1])
                    curr_label_vec = np.concatenate((curr_label_vec1, curr_label_vec2))
                    curr_label_vec1 = None
                    curr_label_vec2 = None
                else:
                    curr_label_vec = self.__get_word_vector__(curr_tuple[0])

                assert curr_label_vec is not None

                if is_var_length:
                    curr_embedding[curr_label_idx, :] = curr_label_vec
                else:
                    embeddings[curr_path_idx, curr_label_idx, :] = curr_label_vec

            if is_var_length:
                embeddings.append(curr_embedding)

        if is_var_length:
            embeddings = np.array(embeddings)

        if is_multiword_as_single:
            # print 'embeddings.shape', embeddings.shape

            assert not is_var_length, 'not implemented for variable length'
            sequence_length = embeddings.shape[1]

            # print 'sequence_length', sequence_length

            new_sequence_length = int(math.ceil(sequence_length/float(self.num_words_per_lstm_step))*self.num_words_per_lstm_step)
            # new_sequence_length = int(round(sequence_length+4, -1))
            # new_sequence_length = int(round(sequence_length+((self.num_words_per_lstm_step/2)-1), -1))

            # print 'new_sequence_length', new_sequence_length

            embeddings_new = np.zeros((embeddings.shape[0], new_sequence_length, embeddings.shape[2]))
            embeddings_new[:, :sequence_length, :] = embeddings
            embeddings_new = embeddings_new.reshape((
                                embeddings.shape[0],
                                int(new_sequence_length/self.num_words_per_lstm_step),
                                embeddings.shape[2]*self.num_words_per_lstm_step
                            ))
            embeddings = embeddings_new
            embeddings_new = None

            # print 'embeddings.shape', embeddings.shape

        return embeddings
