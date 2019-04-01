import logging
import re
import ast
from datetime import datetime
import pandas as pd
from prepare_data import LANLDataLoader
from models import MultiLSTM
from grid_search import ts_reg_grid_search


def test_basic_operation():
    train_args = {
        'n_samples': 40,
    }
    val_args = {
        'n_samples': 10
    }
    loader_args = {
        'st_size': 3000,
        'overlap_size': 500,
        'fft_f_cutoff': 500
    }

    loader = LANLDataLoader()
    train_data, train_target, val_data, val_target = loader.load_train_val(
        train_args=train_args, val_args=val_args, loader_args=loader_args)
    input_shapes = loader.get_input_shape()
    logging.debug(input_shapes)

    multi_lstm = MultiLSTM(input_shapes, [8, 32], 64)
    history = multi_lstm.fit(train_data, train_target,
                             validation_data=(val_data, val_target))

    test_data, test_labels = loader.load_test_data()
    predictions = multi_lstm.predict([data[:10, :].compute() for data in test_data])

    regex = re.compile('.csv')
    results_file_name = 'data/results/test_predictions_{:%Y%m%d_%H%M%S}.csv'.format(datetime.now())

    pd.DataFrame({
        'seg_id': [re.sub(regex, '', label) for label in test_labels[:10]],
        'time_to_failure': predictions.squeeze()
    }).to_csv(results_file_name)


def perform_grid_search():
    # test over parameter space given by a text file
    hp_file = 'lstm_grid_params.txt'

    loader_args = {
        'st_size': 1500,
        'overlap_size': 500,
        'fft_f_cutoff': 500
    }
    train_args = {
        'n_samples': 5000,
    }
    val_args = {
        'n_samples': 500
    }

    model_args = {
        'lstm_gpu': True
    }
    fit_args = {
        'epochs': 40,
        'batch_size': 256
    }

    ts_reg_grid_search(MultiLSTM, LANLDataLoader, hp_file, 
                       loader_args, model_args, train_args, val_args, fit_args, draws=10)


def predict():
    hp_file = 'lstm_grid_params.txt'

    # load hyperparameters from file
    with open(hp_file, 'r') as f:
        model_args = ast.literal_eval(f.read())

    # train on input data and provide prediction on test data
    loader_args = {
        'st_size': 10000,
        'overlap_size': 5000,
        'fft_f_cutoff': 500
    }

    train_args = {
        'n_samples': 100,
    }

    fit_args = {
        'epochs': 10,
        'batch_size': 64
    }

    loader = LANLDataLoader()
    train_data, train_target, test_data, test_labels = loader.load_train_val(
        train_args=train_args, loader_args=loader_args)

    model = MultiLSTM(loader.get_input_shape(), **model_args)
    model.fit(train_data, train_target, **fit_args)

    predictions = model.predict(test_data)

    regex = re.compile('.csv')
    pd.DataFrame({
        'seg_id': [re.sub(regex, '', label) for label in test_labels],
        'time_to_failure': predictions.squeeze()
    })


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    perform_grid_search()
