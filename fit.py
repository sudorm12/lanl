import logging
from prepare_data import LANLDataLoader
from models import MultiLSTM

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    train_args = {
        'n_samples': 200,
    }
    val_args = {
        'n_samples': 40
    }

    loader = LANLDataLoader()
    train_data, train_target, val_data, val_target = loader.load_train_val(train_args=train_args, val_args=val_args)
    input_shapes = loader.get_input_shape()
    logging.debug(input_shapes)

    multi_lstm = MultiLSTM(input_shapes, [8, 32], 64)
    multi_lstm.fit([data.compute() for data in train_data], train_target,
                   validation_data=([data.compute() for data in val_data], val_target))
