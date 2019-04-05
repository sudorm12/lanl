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
    def __init__(self, input_shapes, lstm_units, dense_units,
                 lstm_l2=0, lstm_dropout=0, lstm_gpu=False, lstm_backwards=False,
                 dense_l2=0, dense_dropout=0.2,
                 optimizer='adam'):
        """
        Create a keras network with multiple inputs and a single output for time-series or otherwise sequenced data.
        Each sequence input will be connected to an LSTM layer, and the LSTM layers will all be combined and
        connected to one or more dense layers.
        :param input_shapes: a list of tuples, each tuple being two dimensional with order (timesteps, features)
        :param lstm_units: list or int, the number of units for each lstm
        :param dense_units: list or int, the number of units for each dense layer
        :param lstm_l2: list or real, the level of l2 kernel regularization to apply to each lstm layer
        :param lstm_dropout: list or real, the level of dropout to apply to each lstm layer
        :param lstm_gpu: boolean, True to use the CuDNN optimized LSTM layer
        :param dense_l2: list or real, the level of l2 regularization to apply to each dense layer
        :param dense_dropout: list or real, the level of dropout to apply to each dense layer
        :param optimizer: the type of optimizer to use when compiling the keras model
        """
        # find number of lstm inputs
        self._n_inputs = len(input_shapes)

        # determine if LSTM networks will be GPU accelerated
        if lstm_gpu:
            LSTMLayer = CuDNNLSTM
        else:
            LSTMLayer = LSTM

        # these lists will end up containing the lstm input and output layers
        inputs = []
        lstms = []
        outputs = []

        # iterate over the provided input shapes and build up the lstm networks layer by layer
        for i, input_shape in enumerate(input_shapes):
            # create the input layer
            lstm_input = Input(shape=input_shape, name='lstm_input_{}'.format(i))

            # permute input layer dimensions from (timesteps, features) to (features, timesteps)
            lstm_permute = Permute(dims=(2, 1), name='lstm_permute_{}'.format(i))(lstm_input)

            # determine the number of units the lstm should have
            try:
                units = lstm_units[i]
            except TypeError:
                units = int(lstm_units)
            except IndexError:
                units = lstm_units[-1]
                print('Mismatch between input shape and lstm units, using last lstm unit value for input {}'.format(i))

            # determine L2 regularization
            try:
                l2_reg = l2(lstm_l2[i])
            except TypeError:
                l2_reg = l2(lstm_l2)
            except IndexError:
                l2_reg = l2(lstm_l2[-1])
                print('Mismatch between input shape and lstm l2 regularization values, using last lstm l2 value')

            # determine dropout
            try:
                dropout = lstm_dropout[i]
            except TypeError:
                dropout = lstm_dropout
            except IndexError:
                dropout = lstm_dropout[-1]
                print('Mismatch between input shape and lstm dropout values, using last lstm dropout value')

            # create the lstm layer
            lstm = LSTMLayer(
                units=units,
                kernel_regularizer=l2_reg,
                # dropout=dropout,
                go_backwards=lstm_backwards,
                name='lstm_{}'.format(i)
            )(lstm_permute)

            # add an output to help each lstm training
            lstm_output = Dense(1, activation='linear', name='lstm_{}_output'.format(i))(lstm)

            # add lstm inputs and outputs to their respective lists
            inputs.append(lstm_input)
            lstms.append(lstm)
            outputs.append(lstm_output)

        # combine lstm outputs together
        combined = concatenate(lstms, name='combined')

        # add dense layers to the output of the lstms
        try:
            for i, units in enumerate(dense_units):
                # determine L2 regularization
                try:
                    l2_reg = l2(dense_l2[i])
                except TypeError:
                    l2_reg = l2(dense_l2)
                except IndexError:
                    l2_reg = l2(dense_l2[-1])
                    print('Mismatch between input shape and dense l2 regularization values, using last dense l2 value')

                # determine dropout
                try:
                    dropout = dense_dropout[i]
                except TypeError:
                    dropout = dense_dropout
                except IndexError:
                    dropout = dense_dropout[-1]
                    print('Mismatch between input shape and dense dropout values, using last dense dropout value')

                # create a dense layer with dropout
                combined = Dense(
                    units,
                    kernel_regularizer=l2_reg,
                    name='combined_dense_{}'.format(i)
                )(combined)
                combined = Dropout(
                    dropout,
                    name='dropout_{}'.format(i)
                )(combined)

        # if dense_units appears to not be iterable
        except TypeError:
            # determine l2 regularization
            try:
                l2_reg = l2(dense_l2[0])
            except TypeError:
                l2_reg = l2(dense_l2)

            # determine dropout
            try:
                dropout = dense_dropout[0]
            except TypeError:
                dropout = int(dense_dropout)

            # create a single dense layer with dropout
            combined = Dense(
                int(dense_units),
                kernel_regularizer=l2_reg,
                activation='relu',
                name='combined_dense'
            )(combined)
            combined = Dropout(
                dropout,
                name='dropout'
            )(combined)

        # add an output layer
        output = Dense(1, activation='linear', name='output')(combined)

        # initialize the model and compile
        self._model = Model(inputs=inputs, outputs=[output, *outputs])
        self._model.compile(optimizer=optimizer, loss_weights=[1.] + [0.2] * self._n_inputs, loss='mse')

    def fit(self, data_train, target_train, validation_data=None, verbose=2, batch_size=64, epochs=5, callbacks=None):
        if validation_data is not None:
            validation_data = (validation_data[0], [validation_data[1]] * (self._n_inputs + 1))

        history = self._model.fit(data_train, [target_train] * (self._n_inputs + 1), 
                                  validation_data=validation_data, 
                                  verbose=verbose, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        return history

    def predict(self, data):
        return self._model.predict(data)

    def summary(self):
        return self._model.summary()
