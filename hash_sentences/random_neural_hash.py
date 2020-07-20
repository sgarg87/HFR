import numpy as np
import gc
from . import rnn_hash_model as rhm
import time


class RandomNeuralHash:

    def __init__(self,
                 method='lstm',
                 # method = 'lstm-cnn',
                 word_embedding_len=100,
    ):
        # todo: define different configuration files for different projects
        self.method = method
        self.word_embedding_len = word_embedding_len

        self.max_sent_len = None
        self.batch_size = 32
        self.nn_units = [8, 8]
        # self.nn_units = [32, 32]
        # self.nn_units = [8, 16, 8]
        # self.nn_units = [8, 16, 32, 16, 8]
        self.dense_units = []
        # self.dense_units = [8]

        # self.dropout = 0.0
        # self.recurrent_dropout = 0.0
        self.dropout = 0.5
        self.recurrent_dropout = 0.5

        # learn_algo='sg'
        self.learn_algo = 'adam'
        self.is_regularize = False
        # self.is_regularize = True
        self.regularize_const = 1e-4

        # self.learning_rate = 1e-2
        # self.learning_rate = 3e-4
        # self.learning_rate = 3e-3
        self.learning_rate = 1e-3

    def compute_hashcode_bit(self,
                            path_tuples_embedding,
                            path_tuples_references_embedding,
                            subset1,
                            subset2,
                            max_epochs=30
                    ):

        superset_size = path_tuples_references_embedding.shape[0]

        labels = np.zeros(superset_size, dtype=np.int)
        labels[subset1] = -1
        labels[subset2] = 1
        print('labels', labels)

        neural_clf_obj = rhm.RNNHashModel(
            max_sent_len=self.max_sent_len,
            word_emb_dim=self.word_embedding_len,
            batch_size=self.batch_size,
            max_epochs=max_epochs,
            n_units=self.nn_units,
            dense_units=self.dense_units,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            learn_algo=self.learn_algo,
            is_verbose=True,
            is_regularize=self.is_regularize,
            regularize_const=self.regularize_const,
            method=self.method,
            learning_rate=self.learning_rate,
        )

        start_time = time.time()
        neural_clf_obj.train_rnn(path_tuples_references_embedding, labels)
        print('Trained RNN in {} seconds.'.format(time.time() - start_time))

        start_time = time.time()
        z_prob = neural_clf_obj.test_rnn(path_tuples_embedding)
        print('Inferred with RNN in {} seconds.'.format(time.time() - start_time))

        del neural_clf_obj
        gc.collect()

        z = np.zeros(z_prob.size, dtype=bool)
        z[np.where(z_prob > 0.5)] = 1

        return z
