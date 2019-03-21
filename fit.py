import logging
from prepare_data import LANLDataLoader

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    loader = LANLDataLoader()
    data, train = loader.load_train_data(n_samples=200)
    logging.debug(loader.get_input_shape())
