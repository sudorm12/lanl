import logging
from prepare_data import LANLDataLoader
from models import MultiLSTM
from grid_search import ts_reg_grid_search


def test_basic_operation():
    train_args = {
        'n_samples': 100,
    }
    val_args = {
        'n_samples': 40
    }

    loader = LANLDataLoader()
    train_data, train_target, val_data, val_target = loader.load_train_val(train_args=train_args, val_args=val_args)
    input_shapes = loader.get_input_shape()
    logging.debug(input_shapes)

    multi_lstm = MultiLSTM(input_shapes, [8, 32], 64)
    history = multi_lstm.fit([data.compute() for data in train_data], train_target,
                             validation_data=([data.compute() for data in val_data], val_target))
    return history


def perform_grid_search():
    # test over parameter space given by a text file
    hp_file = 'lstm_grid_params.txt'

    loader_args = {
        'st_size': 10000,
        'overlap_size': 5000,
        'fft_f_cutoff': 500
    }
    train_args = {
        'n_samples': 100,
    }
    val_args = {
        'n_samples': 40
    }

    model_args = {
        'lstm_gpu': False
    }
    fit_args = {
        'epochs': 10,
        'batch_size': 64
    }

    ts_reg_grid_search(MultiLSTM, LANLDataLoader, hp_file, loader_args, model_args, train_args, val_args, fit_args)


def predict():
    # train on input data and provide prediction on test data
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    perform_grid_search()
