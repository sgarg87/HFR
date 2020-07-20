''' Trains RNN given a set of sentences and corresponding labels
'''
import numpy as np

# import os
# os.environ['KERAS_BACKEND'] = 'theano'

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Bidirectional, RNN, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
import keras.regularizers as Reg
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
# import keras.utils as keras_util

is_self_attention = False
if is_self_attention:
    from keras_self_attention import SeqSelfAttention


class RNNHashModel:
    def __init__(self,
                 max_sent_len=None,
                 word_emb_dim=100,
                 method=None,
                 learning_rate=1e-3,
                 n_units=None,
                 dense_units=[],
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 batch_size=1,
                 max_epochs=1000,
                 learn_algo='adam',
                 filters=64,
                 kernel_size=5,
                 pool_size=4,
                 is_verbose=True,
                 is_regularize=False,
                 regularize_const=1e-4,
                 is_cross_entropy_loss=False,
                 is_self_activation=False,
    ):
        ''' Constructs an LSTM or GRU model

        Args:
            max_sent_len: length of the longest sentence
                          (each sentence should be padded to this length)
            word_emb_dim: dimension of embedding of each word in a sentence
            method: neural unit - currently supports LSTM and GRU
            n_units: array containing dimensions of each recurrent layer
            dense_units: array containing dimensions of each dense layer
            bias_reg: bias regularization
            input_reg: input regularization
            recurr_reg: recurrent regularization
            batch_size: batch size in training and prediction
            max_epochs: maximum number of training epochs (Early stopping is used)
        '''

        if n_units is None:
            n_units = [10, 10]
        if dense_units is None:
            dense_units = []

        if is_regularize:
            input_reg = Reg.l1_l2(l1=regularize_const, l2=regularize_const)
            recurr_reg = Reg.l1_l2(l1=regularize_const, l2=regularize_const)
        else:
            input_reg = None
            recurr_reg = None

        self.max_sent_len = max_sent_len
        self.word_emb_dim = word_emb_dim
        self.method = method
        self.learning_rate = learning_rate
        self.n_units = n_units
        self.dense_units = dense_units
        self.input_reg = input_reg
        self.recurr_reg = recurr_reg
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.learn_algo = learn_algo
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.is_verbose = is_verbose
        self.is_cross_entropy_loss = is_cross_entropy_loss
        self.is_self_activation = is_self_activation

        # if self.is_cross_entropy_loss:
        #     self.label_dim = 2
        # else:
        self.label_dim = 1

        self.model = self.__construct_rnn_model__()

    def __construct_rnn_model__(self):
        model = Sequential()
        n_rnn_layers = len(self.n_units)

        if n_rnn_layers > 1:
            is_return_sequences = True
        else:
            is_return_sequences = False

        if self.is_self_activation:
            is_return_sequences_last_recurrent_layer = True
        else:
            is_return_sequences_last_recurrent_layer = False
        print(is_return_sequences_last_recurrent_layer)

        assert self.method is not None

        if self.method == 'rnn':
            print(self.max_sent_len)
            model.add(RNN(self.n_units[0],
                           input_shape=(self.max_sent_len, self.word_emb_dim),
                           return_sequences=is_return_sequences,
                           kernel_regularizer=self.input_reg,
                           recurrent_regularizer=self.recurr_reg,
                           dropout=self.dropout,
                           recurrent_dropout=self.recurrent_dropout,
                        )
                      )

            for l_idx, n_unit in enumerate(self.n_units[1:]):
                if l_idx < (n_rnn_layers - 2):
                    model.add(RNN(n_unit,
                                   return_sequences=True,
                                   kernel_regularizer=self.input_reg,
                                   recurrent_regularizer=self.recurr_reg,
                                   dropout=self.dropout,
                                   recurrent_dropout=self.recurrent_dropout,
                            )
                    )
                else:
                    model.add(RNN(
                        n_unit,
                        return_sequences=is_return_sequences_last_recurrent_layer,
                        kernel_regularizer=self.input_reg,
                        recurrent_regularizer=self.recurr_reg,
                        dropout=self.dropout,
                        recurrent_dropout=self.recurrent_dropout,
                    ))
        elif self.method == 'lstm':
            print(self.max_sent_len)
            model.add(LSTM(self.n_units[0],
                           input_shape=(self.max_sent_len, self.word_emb_dim),
                           return_sequences=is_return_sequences,
                           kernel_regularizer=self.input_reg,
                           recurrent_regularizer=self.recurr_reg,
                           dropout=self.dropout,
                           recurrent_dropout=self.recurrent_dropout,
                        )
                      )
            for l_idx, n_unit in enumerate(self.n_units[1:]):
                if l_idx < n_rnn_layers - 2:
                    model.add(LSTM(n_unit,
                                   return_sequences=True,
                                   kernel_regularizer=self.input_reg,
                                   recurrent_regularizer=self.recurr_reg,
                                   dropout=self.dropout,
                                   recurrent_dropout=self.recurrent_dropout,
                                )
                            )
                else:
                    model.add(LSTM(
                                n_unit,
                                return_sequences=is_return_sequences_last_recurrent_layer,
                                kernel_regularizer=self.input_reg,
                                recurrent_regularizer=self.recurr_reg,
                                dropout=self.dropout,
                                recurrent_dropout=self.recurrent_dropout,
                    ))
        elif self.method == 'gru':
            model.add(GRU(self.n_units[0],
                          input_shape=(self.max_sent_len, self.word_emb_dim),
                          return_sequences=is_return_sequences,
                          kernel_regularizer=self.input_reg,
                          recurrent_regularizer=self.recurr_reg
                        )
                    )
            for l_idx, n_unit in enumerate(self.n_units[1:]):
                if l_idx < n_rnn_layers - 2:
                    model.add(GRU(n_unit,
                                  return_sequences=True,
                                  kernel_regularizer=self.input_reg,
                                  recurrent_regularizer=self.recurr_reg
                                )
                              )
                else:
                    model.add(GRU(n_unit,
                                  return_sequences=is_return_sequences_last_recurrent_layer,
                                  kernel_regularizer=self.input_reg,
                                  recurrent_regularizer=self.recurr_reg
                                  )
                              )
        elif self.method == 'bi-lstm':
                    print(self.max_sent_len)
                    model.add(
                        Bidirectional(
                            LSTM(
                                self.n_units[0],
                                return_sequences=(is_return_sequences or is_return_sequences_last_recurrent_layer),
                                kernel_regularizer=self.input_reg,
                                recurrent_regularizer=self.recurr_reg,
                                dropout=self.dropout,
                                recurrent_dropout=self.recurrent_dropout,
                            ),
                            input_shape=(self.max_sent_len, self.word_emb_dim),
                        )
                    )
                    for l_idx, n_unit in enumerate(self.n_units[1:]):
                        if l_idx < (n_rnn_layers - 2):
                            model.add(Bidirectional(LSTM(n_unit,
                                                         return_sequences=True,
                                                         kernel_regularizer=self.input_reg,
                                                         recurrent_regularizer=self.recurr_reg,
                                                         dropout=self.dropout,
                                                         recurrent_dropout=self.recurrent_dropout,
                                                         ))
                                      )
                        else:
                            model.add(Bidirectional(LSTM(n_unit,
                                                         return_sequences=is_return_sequences_last_recurrent_layer,
                                                         kernel_regularizer=self.input_reg,
                                                         recurrent_regularizer=self.recurr_reg,
                                                         dropout=self.dropout,
                                                         recurrent_dropout=self.recurrent_dropout,
                            )))
        elif self.method == 'bi-gru':
            model.add(Bidirectional(
                        GRU(self.n_units[0],
                          return_sequences=is_return_sequences,
                          kernel_regularizer=self.input_reg,
                          recurrent_regularizer=self.recurr_reg
                        ),
                        input_shape=(self.max_sent_len, self.word_emb_dim),
                    ))
            for l_idx, n_unit in enumerate(self.n_units[1:]):
                if l_idx < (n_rnn_layers - 2):
                    model.add(Bidirectional(GRU(n_unit,
                                  return_sequences=True,
                                  kernel_regularizer=self.input_reg,
                                  recurrent_regularizer=self.recurr_reg
                                ))
                              )
                else:
                    model.add(Bidirectional(GRU(
                        n_unit,
                        return_sequences=is_return_sequences_last_recurrent_layer,
                        kernel_regularizer=self.input_reg,
                        recurrent_regularizer=self.recurr_reg
                    )))
        elif self.method == 'lstm-cnn':
            print(self.max_sent_len)
            model.add(Conv1D(self.filters,
                      self.kernel_size,
                      input_shape=(self.max_sent_len, self.word_emb_dim),
                      padding='valid',
                      activation='relu',
                      strides=1
                )
            )
            model.add(MaxPooling1D(pool_size=self.pool_size))
            model.add(LSTM(self.n_units[0],
                           return_sequences=is_return_sequences,
                           kernel_regularizer=self.input_reg,
                           recurrent_regularizer=self.recurr_reg,
                           dropout=self.dropout,
                           recurrent_dropout=self.recurrent_dropout,
                        )
                      )
            for l_idx, n_unit in enumerate(self.n_units[1:]):
                if l_idx < n_rnn_layers - 2:
                    model.add(LSTM(n_unit,
                                   return_sequences=True,
                                   kernel_regularizer=self.input_reg,
                                   recurrent_regularizer=self.recurr_reg,
                                   dropout=self.dropout,
                                   recurrent_dropout=self.recurrent_dropout,
                                )
                            )
                else:
                    model.add(LSTM(n_unit,
                                   return_sequences=is_return_sequences_last_recurrent_layer,
                                   kernel_regularizer=self.input_reg,
                                   recurrent_regularizer=self.recurr_reg,
                                   dropout=self.dropout,
                                   recurrent_dropout=self.recurrent_dropout,
                                )
                            )
        elif self.method == 'gru-cnn':
            print(self.max_sent_len)
            model.add(Conv1D(self.filters,
                      self.kernel_size,
                      input_shape=(self.max_sent_len, self.word_emb_dim),
                      padding='valid',
                      activation='relu',
                      strides=1
                )
            )
            model.add(MaxPooling1D(pool_size=self.pool_size))
            model.add(GRU(self.n_units[0],
                           return_sequences=is_return_sequences,
                           kernel_regularizer=self.input_reg,
                           recurrent_regularizer=self.recurr_reg,
                           dropout=self.dropout,
                           recurrent_dropout=self.recurrent_dropout,
                        )
                      )
            for l_idx, n_unit in enumerate(self.n_units[1:]):
                if l_idx < n_rnn_layers - 2:
                    model.add(GRU(n_unit,
                                   return_sequences=True,
                                   kernel_regularizer=self.input_reg,
                                   recurrent_regularizer=self.recurr_reg,
                                   dropout=self.dropout,
                                   recurrent_dropout=self.recurrent_dropout,
                                )
                            )
                else:
                    model.add(GRU(n_unit,
                                  return_sequences=is_return_sequences_last_recurrent_layer,
                                  kernel_regularizer=self.input_reg,
                                  recurrent_regularizer=self.recurr_reg,
                                  dropout=self.dropout,
                                  recurrent_dropout=self.recurrent_dropout,
                    ))
        elif self.method == 'cnn':
            print(self.max_sent_len)
            model.add(Conv1D(self.filters,
                      self.kernel_size,
                      input_shape=(self.max_sent_len, self.word_emb_dim),
                      padding='valid',
                      activation='relu',
                      strides=1
                )
            )
            model.add(GlobalMaxPooling1D())

        if self.is_self_activation:
            model.add(SeqSelfAttention(attention_activation='sigmoid'))
            model.add(Flatten())

        for dense_n_unit in self.dense_units:
            model.add(Dense(dense_n_unit, activation='relu'))

        model.add(Dense(self.label_dim, activation='sigmoid'))

        if self.learn_algo == 'sg':
            optimizer = SGD(
                        lr=self.learning_rate,
                        momentum=0.0,
                        decay=0.0,
                        nesterov=False
                      )
        elif self.learn_algo == 'adam':
            optimizer = Adam(
                lr=self.learning_rate,
                # beta_1=0.9,
                # beta_2=0.999,
                # epsilon=1e-08,
                amsgrad=True,
            )
        else:
            raise NotImplementedError

        if self.is_cross_entropy_loss:
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['binary_accuracy'],
            )
        else:
            model.compile(
                loss='mean_squared_error',
                optimizer=optimizer,
            )

        return model

    def train_rnn(self, trainX, trainY, class_weights_map=None):

        # trainX: training set of sentences
        #         (numpy array of shape [num_tr_sent, max_sent_len, word_em_dim])
        # trainY: training labels (1-D array)
        if self.is_cross_entropy_loss:
            assert np.setdiff1d(np.unique(trainY), np.array([0, 1])).size == 0
        else:
            assert np.setdiff1d(np.unique(trainY), np.array([-1, 1])).size == 0

        trainY = trainY.reshape((trainY.size, 1))
        print(trainY)

        # if self.is_cross_entropy_loss:
        #     trainY = keras_util.to_categorical(trainY)

        # early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        early_stop = EarlyStopping(
            monitor='loss', verbose=1, patience=30,
        )

        if self.is_verbose:
            print('Fitting data to the model')
            verbose = 2
        else:
            verbose = 0

        batch_size = min(self.batch_size, trainX.shape[0])
        print('batch_size', batch_size)

        self.model.fit(
            trainX, trainY,
            epochs=self.max_epochs,
            batch_size=batch_size,
            # validation_split=0.5,
            # validation_split=0.2,
            # validation_data=(trainX, trainY),
            callbacks=[early_stop],
            verbose=verbose,
            class_weight=class_weights_map,
        )

        if self.is_verbose:
            print('Fitted data to the model')

    def test_rnn(self, testX):
        # prediction = self.model.predict(testX, batch_size=self.batch_size, verbose=0)
        prediction = self.model.predict(testX, batch_size=1024, verbose=0)
        print(prediction.shape)
        print(prediction)

        # if self.is_cross_entropy_loss:
        #     prediction = prediction[:, 1]
        #     print(prediction.shape)
        #     print(prediction)

        prediction = prediction.reshape(prediction.size)
        return prediction


if __name__ == '__main__':
    ''' Example usage
        Generates num_sent sentences in n_bins buckets
        Note that for sentences with varying number of words, padding can be performed:
            from keras.preprocessing import sequence
            sequence.pad_sequences(x_train, maxlen=maxlen)
    '''
    num_sent = 1000
    n_bins = 30
    max_sent_len = 100
    emb_dim = 100
    reg = 1e-03
    n_per_bin = num_sent // n_bins
    trainX = np.zeros((num_sent, max_sent_len, emb_dim))
    trainY = np.zeros((num_sent, 1))
    testX = np.zeros((num_sent, max_sent_len, emb_dim))
    testY = np.zeros((num_sent, 1))

    # Generate training examples
    for k in range(n_bins):
        trainX[k * n_per_bin:(k + 1) * n_per_bin, :, :] = \
            np.ones((n_per_bin, max_sent_len, emb_dim)) / (k + 1) + \
            np.random.randn(n_per_bin, max_sent_len, emb_dim) / 1000

        # trainY[k * n_per_bin:(k + 1) * n_per_bin, 0] = 1.0 / (k + 1)

        if (k % 2) == 1:
            trainY[k * n_per_bin:(k + 1) * n_per_bin, 0] = 1

    # Generate test examples
    for k in range(n_bins):
        testX[k * n_per_bin:(k + 1) * n_per_bin, :, :] = \
            np.ones((n_per_bin, max_sent_len, emb_dim)) / (k + 1) + \
            np.random.randn(n_per_bin, max_sent_len, emb_dim) / 1000

        # testY[k * n_per_bin:(k + 1) * n_per_bin, 0] = 1.0 / (k + 1)

        if (k % 2) == 1:
            testY[k * n_per_bin:(k + 1) * n_per_bin, 0] = 1

    print('testY', testY.flatten())

    rnn_model = RNNHashModel(method='cnn')

    rnn_model.train_rnn(trainX,
                        trainY)

    pred = rnn_model.test_rnn(testX)
    print('pred', pred.flatten())

    # err = mse(testY, pred)[0]
    # err_b = mse(testY, np.zeros((num_sent, 1)))[0]
    # print('MSE test error: %f, baseline error: %f' % (err, err_b))
