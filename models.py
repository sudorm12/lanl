from keras.models import Sequential, Model
from keras.layers import CuDNNLSTM, LSTM, Dense, Dropout, Input, Reshape, concatenate, Permute
from keras.regularizers import l2
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


class DenseNN:
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, l2_reg=0, 
                 epochs=5, batch_size=256, dropout=0, verbose=1):
        self._epochs = epochs
        self._batch_size = batch_size
        self._verbose = verbose
        self._model = Sequential()

        for i in range(num_layers):
            self._model.add(Dense(units=hidden_dim,
                                  activation='relu',
                                  kernel_regularizer=l2(l2_reg),
                                  input_dim=input_dim,
                                  name='dense_{}'.format(i)))
            self._model.add(Dropout(dropout, name='dropout_{}'.format(i)))

        self._model.add(Dense(units=1, activation='sigmoid', name='dense_final'))

        self._model.compile(loss='binary_crossentropy',
                            optimizer='Adam',
                            metrics=['accuracy'])

    def fit(self, data_train, target_train, validation_data=None):
        self._model.fit(data_train, target_train,
                        epochs=self._epochs,
                        batch_size=self._batch_size,
                        validation_data=validation_data,
                        verbose=self._verbose)

    def predict(self, data):
        return self._model.predict(data)


class GBC:
    def __init__(self, input_shape=None, n_estimators=10, max_depth=3, verbose=0, min_samples_split=2, learning_rate=0.1):
        self._model = GradientBoostingClassifier(n_estimators=n_estimators,
                                                 max_depth=max_depth,
                                                 verbose=verbose,
                                                 min_samples_split=min_samples_split,
                                                 learning_rate=learning_rate)

    def fit(self, data_train, target_train, validation_data=None):
        self._model.fit(data_train, target_train)

    def predict(self, data):
        return self._model.predict_proba(data)[:, 1]


class ABC:
    def __init__(self, input_shape=None, n_estimators=10, learning_rate=1):
        self._model = AdaBoostClassifier(n_estimators=n_estimators,
                                         learning_rate=learning_rate)

    def fit(self, data_train, target_train, validation_data=None):
        self._model.fit(data_train, target_train)

    def predict(self, data):
        return self._model.predict_proba(data)[:, 1]


class DTC:
    def __init__(self, input_shape=None, class_weight='balanced', min_samples_split=2):
        self._model = DecisionTreeClassifier(class_weight=class_weight, min_samples_split=min_samples_split)

    def fit(self, data_train, target_train, validation_data=None):
        self._model.fit(data_train, target_train)

    def predict(self, data):
        return self._model.predict_proba(data)[:, 1]


class MultiLSTMWithMetadata:
    def __init__(self, input_shapes,
                 sequence_dense_layers=1, meta_dense_layers=1, comb_dense_layers=1,
                 sequence_dense_width=32, meta_dense_width=32, comb_dense_width=32,
                 sequence_l2_reg=0, meta_l2_reg=0, comb_l2_reg=0,
                 sequence_dropout=0, meta_dropout=0, comb_dropout=0,
                 lstm_units=8, lstm_l2_reg=0, lstm_gpu=False,
                 epochs=5, batch_size=256):

        """
        input_shapes: a list of tuples indicating shape of each input, meta first as
        shape (meta_features,) and then sequence shapes as (sequence_lengths, sequence_features)
        """
        self._num_seq_inputs = len(input_shapes) - 1
        self._num_epochs = epochs
        self._batch_size = batch_size

        lstm_inputs = []
        lstm_outputs = []
        lstm_forward = []

        # build up the lstm network for each time series input
        for i, sequence_shape in enumerate(input_shapes[1:]):
            # lstm input and reshape from flat to sequence length x features shape
            lstm_input = Input(shape=(sequence_shape[0] * sequence_shape[1],), 
                               name='lstm_input_{}'.format(i))
            lstm_inputs.append(lstm_input)

            reshaped_input = Reshape(sequence_shape, 
                                     name='reshaped_input_{}'.format(i))(lstm_input)

            # lstm layer selected based on gpu acceleration
            if lstm_gpu:
                lstm = CuDNNLSTM(lstm_units, kernel_regularizer=l2(lstm_l2_reg),
                                 name='lstm_{}'.format(i))(reshaped_input)
            else:
                lstm = LSTM(lstm_units, kernel_regularizer=l2(lstm_l2_reg),
                            name='lstm_{}'.format(i))(reshaped_input)

            # add dense layers to output of lstm
            for j in range(sequence_dense_layers):
                lstm = Dense(sequence_dense_width, activation='relu', kernel_regularizer=l2(sequence_l2_reg),
                             name='seq_dense_{}_{}'.format(i, j))(lstm)
                lstm = Dropout(sequence_dropout,
                               name='lstm_dropout_{}'.format(i))(lstm)
            lstm_forward.append(lstm)
            lstm_outputs.append(Dense(1, activation='sigmoid', name='lstm_output_{}'.format(i))(lstm))

        # meta data input and dense layers
        meta_input = Input(shape=input_shapes[0], name='meta_input')
        meta_dense = meta_input
        for i in range(meta_dense_layers):
            meta_dense = Dense(meta_dense_width, activation='relu', kernel_regularizer=l2(meta_l2_reg),
                               name='meta_dense_{}'.format(i))(meta_dense)
            meta_dense = Dropout(meta_dropout,
                                 name='meta_dropout_{}'.format(i))(meta_dense)

        # combine metadata and sequence outputs and add dense layers
        x = concatenate([*lstm_forward, meta_dense], name='concatenate')
        for i in range(comb_dense_layers):
            x = Dense(comb_dense_width, activation='relu', kernel_regularizer=l2(comb_l2_reg),
                      name='combined_dense_{}'.format(i))(x)
            x = Dropout(rate=comb_dropout,
                        name='combined_dropout_{}'.format(i))(x)
        main_output = Dense(1, activation='sigmoid', name='main_output')(x)

        # initialize and compile keras model
        self._model = Model(inputs=[meta_input, *lstm_inputs], outputs=[main_output, *lstm_outputs])
        self._model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            loss_weights=[1.] + [0.2] * self._num_seq_inputs,
                            metrics=['accuracy'])

    def fit(self, data_train, target_train, validation_data=None, verbose=2):
        num_outputs = self._num_seq_inputs + 1
        if validation_data is not None:
            validation_data = (validation_data[0], [validation_data[1]] * num_outputs)
        history = self._model.fit(data_train, [target_train] * num_outputs,
                                  validation_data=validation_data,
                                  epochs=self._num_epochs, batch_size=self._batch_size, verbose=verbose)
        return history

    def predict(self, data):
        return self._model.predict(data)[0]

    def model_summary(self):
        return self._model.summary()


class MultiLSTM:
    def __init__(self, input_shapes, lstm_units, dense_units):
        # TODO: add support for l2 regularization
        # TODO: add support for dropout layers
        # TODO: add support for GPU accelerated LSTM layers
        # TODO: add support for alternate optimizers
        # TODO: add batch size and epochs as parameters

        # check if dense units is int and convert to single-item list if so
        if isinstance(dense_units, int):
            dense_units = [dense_units]
            # TODO: allow int to represent multiple layers at the same width

        if len(input_shapes) != len(lstm_units):
            raise ValueError('input shapes and units iterables must have same number of elements')

        inputs = []
        lstms = []
        for i, input_shape in enumerate(input_shapes):
            lstm_input = Input(shape=input_shape, name='lstm_input_{}'.format(i))
            lstm_permute = Permute(dims=(2, 1), name='lstm_permute_{}'.format(i))(lstm_input)
            lstm = LSTM(units=lstm_units[i], name='lstm_{}'.format(i))(lstm_permute)
            inputs.append(lstm_input)
            lstms.append(lstm)

        combined = concatenate(lstms, name='combined')

        for i, units in enumerate(dense_units):
            combined = Dense(units, name='combined_dense_{}'.format(i))(combined)

        output = Dense(1, name='output')(combined)

        self._model = Model(inputs=inputs, outputs=output)
        self._model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    def fit(self, data_train, target_train, validation_data=None, verbose=2):
        history = self._model.fit(data_train, target_train, validation_data=validation_data, 
                                  verbose=verbose, epochs=25, batch_size=64)
        return history

    def predict(self, data):
        return self._model.predict(data)

    def summary(self):
        return self._model.summary()
