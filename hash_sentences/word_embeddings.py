import numpy as np
import time
import math
import numpy.random as npr
from . import nonascii_text_processing as ntp


class WordEmbeddings:

    def __init__(self, is_wordvec_large=False, wordvec_file_path=None):
        self.__is_debug__ = False
        self.__is_coarse_debug__ = False

        # kernel computation related settings
        self.__wordvec_buffer__ = {}

        # for parallel computing with a high number of cores, buffer slows down kernel matrix computations so. so one can switch use of different buffer used.
        # make sure to use this flag whenever implementing any buffer or a uncommenting an old one.
        self.__is_buffer__ = False
        self.__word_vectors_map__ = None

        self.__is_coarse_debug__ = False

        self.__is_buffer__ = True
        self.__is_wordvec_kernel_false__ = True

        if not self.__is_wordvec_kernel_false__:
            self.glove_wordvec_map = self.load_glove_model()

        if wordvec_file_path is not None:
            self.gloveFile = wordvec_file_path
            if not is_wordvec_large:
                self.wordvec_length = 100
            else:
                self.wordvec_length = 300
        else:
            if not is_wordvec_large:
                self.gloveFile = './glove.6B.100d.txt'
                self.wordvec_length = 100
            else:
                self.gloveFile = './glove.840B.300d.txt'
                self.wordvec_length = 300

        self.__reset_buffers__(is_wordvec_buffer_reset=True)
        self.load_glove_model()
        self.__reset_buffers__()

    def __reset_buffers__(self, is_wordvec_buffer_reset=False):
        self.__randomvec_buffer__ = {}
        if is_wordvec_buffer_reset:
            self.__wordvec_buffer__ = {}

    def __get_random_vector__(self, label):
        assert label is not None

        if (not hasattr(self, '__randomvec_buffer__')) or (self.__randomvec_buffer__ is None):
            print('initializing random vectors buffer')
            self.__randomvec_buffer__ = {}

        if label not in self.__randomvec_buffer__:
            print('warning: random vector')
            random_i_vector = npr.random(self.random_word_vec_len)
            self.__randomvec_buffer__[label] = random_i_vector
        else:
            random_i_vector = self.__randomvec_buffer__[label]

        return random_i_vector

    def load_glove_model(self):
        gloveFile = self.gloveFile

        start_time = time.time()
        print("Loading Glove Model ...")
        f = open(gloveFile, 'r')
        model = {}

        for line in f:
            try:
                splitLine = line.split()
                word = splitLine[0]
                # print('word', word)
                embedding = [float(val) for val in splitLine[1:]]
                embedding = np.array(embedding)
                # print('embedding', embedding)
                model[word] = embedding
            except Exception as e:
                print(e)

        print("Done.", len(model), " words loaded!")
        print('Time to load was ', time.time() - start_time)
        start_time = None

        self.glove_wordvec_map = model

    def preprocess_word_fr_cs(self, curr_word):
        curr_word = curr_word.lower()
        curr_word = curr_word.strip('.')
        curr_word = curr_word.strip()
        curr_word = curr_word.strip('\'')
        curr_word = ntp.remove_non_ascii(curr_word)
        return curr_word

    def __search_object_for_word_from_map__(self,
                                            word_str,
                                            objects_map,
                                            is_simple_search=False):

        if word_str in objects_map:
            curr_object = objects_map[word_str]
            return curr_object

        if not is_simple_search:
            word_str_lower = word_str.lower()
            if word_str_lower in objects_map:
                curr_object = objects_map[word_str_lower]
                return curr_object
            word_str_lower = None

            word_str_upper = word_str.upper()
            if word_str_upper in objects_map:
                curr_object = objects_map[word_str_upper]
                return curr_object
            word_str_upper = None

            word_str_formatted = word_str.strip('.')
            word_str_formatted = word_str_formatted.strip()
            word_str_formatted = word_str_formatted.strip('\'')
            word_str_formatted = ntp.remove_non_ascii(word_str_formatted)
            word_str_formatted = word_str_formatted
            if word_str_formatted in objects_map:
                curr_object = objects_map[word_str_formatted]
                return curr_object
            word_str_formatted = None

            # # this operation is very expensive, must avoid it, so commented  (can take fraction of second for one single call)
            # word_str_spelled = autocorrect.spell(word_str)
            # if word_str_spelled in objects_map:
            #     curr_object = objects_map[word_str_spelled]
            #     return curr_object
            # word_str_spelled = None

        return None

    def get_wordvector_wrapper(self, word_str):
        assert self.glove_wordvec_map is not None
        return self.__search_object_for_word_from_map__(word_str=word_str, objects_map=self.glove_wordvec_map)

    def __get_word_vector__(self, word_label, is_rnd_init=True):
        # start_time = time.time()

        assert word_label is not None

        if (not self.__is_buffer__) or (word_label not in self.__wordvec_buffer__):

            word_i_vector = self.get_wordvector_wrapper(word_label)

            if is_rnd_init and (word_i_vector is None):
                # random initialization
                if (not hasattr(self, 'npr_state_wordvec_random')) or (self.npr_state_wordvec_random is None):
                    self.npr_state_wordvec_random = npr.RandomState(seed=0)

                word_i_vector = self.npr_state_wordvec_random.randn(self.wordvec_length)
                assert word_i_vector.size == self.wordvec_length

                is_rnd_word_vec_initialized = True

            if word_i_vector is not None:
                l2_norm = word_i_vector.dot(word_i_vector)
                l2_norm = math.sqrt(l2_norm)
                # print l2_norm
                word_i_vector /= l2_norm
                l2_norm = None
                word_i_vector = word_i_vector.astype(np.float32)

            self.__wordvec_buffer__[word_label] = word_i_vector
        else:
            word_i_vector = self.__wordvec_buffer__[word_label]

        return word_i_vector
