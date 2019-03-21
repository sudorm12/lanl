import logging
import numpy as np
import dask
import dask.dataframe as dd
from dask.array import fft
from loader import DataLoader


class LANLDataLoader(DataLoader):
    def __init__(self):
        super().__init__()

        # train file name and length
        self._train_file = 'data/train.csv'
        self._train_file_length = 629145479
        self._sample_length = 150000
        self._t_step = 2.62e-07

        # load training data into dask array
        logging.debug('Loading training data...')
        data_dask = dd.read_csv("data/train.csv", dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64},
                                blocksize=25e6)
        array_dask = data_dask.to_dask_array(lengths=True)
        self._data_array = dask.array.rechunk(array_dask, chunks=(self._sample_length, 2))
        logging.debug('Done')

        # prepare variables set during data loading
        self._input_shape = None
        self._fft_quartiles = None
        self._stat_quartiles = None

    def get_index(self):
        return np.arange(self._train_file_length)

    def get_input_shape(self):
        return self._input_shape

    def load_test_data(self, **loader_args):
        # TODO: add functionality for loading test data
        raise NotImplementedError

    def load_train_data(self, split_index=None, fit_transform=True, n_samples=200,
                        split_length=50000000, st_size=10000, overlap_size=5000, fft_f_cutoff=500):
        # get random samples from time series data and perform transformations on those samples
        logging.debug('Preparing {} random samples...'.format(n_samples))
        ts_sample, sample_index = self.get_random_samples(self._data_array[:, 0], n_samples, self._sample_length)
        logging.debug('Calculating rolling statistics...')
        ts_resample = self.overlapping_resample(ts_sample, st_size, overlap_size)
        sample_statistics = self.get_summary_statistics(ts_resample)
        logging.debug('Performing DFT...')
        stdft_samples = np.abs(self.filtered_fft(ts_resample, self._t_step, fft_f_cutoff))

        # scale data
        logging.debug('Scaling data...')
        if fit_transform:
            self._stat_quartiles = dask.array.percentile(sample_statistics.flatten(), q=[20, 50, 80]).compute()
            self._fft_quartiles = np.percentile(stdft_samples, q=[20, 50, 80])
        if self._stat_quartiles is not None and self._fft_quartiles is not None:
            scaled_sample_statistics = (sample_statistics - self._stat_quartiles[1]) / (self._stat_quartiles[2] - self._stat_quartiles[0])
            scaled_stdft_samples = (stdft_samples - self._fft_quartiles[1]) / (self._fft_quartiles[2] - self._fft_quartiles[0])
        else:
            raise ValueError('fit_transform set as False but no scaler fit exists')

        # set the input shape for dynamically creating neural network models later
        logging.debug('Calculating input shapes...')
        self._input_shape = [
            tuple(scaled_sample_statistics.shape[1:]),
            tuple(scaled_stdft_samples[0].compute().shape)
        ]

        data = [scaled_sample_statistics, scaled_stdft_samples]
        target = self._data_array[sample_index + self._sample_length, 1]

        logging.debug('Done')
        return data, target

    @staticmethod
    def get_random_samples(x, n, l):
        sample_index_start = np.random.choice(len(x) - l, size=n)
        sample_index = [np.arange(i, i + l) for i in sample_index_start]
        samples = x[np.concatenate(sample_index)].reshape((n, l))
        return samples, sample_index_start

    @staticmethod
    def overlapping_resample(x, sample_size, overlap_size):
        resample_index = np.concatenate([np.arange(i, i + sample_size) for i in
                                         np.arange(0, x.shape[1] - sample_size + 1, sample_size - overlap_size)])
        resample = x[:, resample_index].reshape((x.shape[0], -1, sample_size))
        return resample

    @staticmethod
    def get_summary_statistics(x):
        summary_stats = dask.array.concatenate([
            x.mean(axis=-1, keepdims=True),
            x.std(axis=-1, keepdims=True),
            x.max(axis=-1, keepdims=True),
            x.min(axis=-1, keepdims=True)
        ], axis=2)
        return summary_stats

    @staticmethod
    def filtered_fft(x, t_step, f_cutoff):
        x_freq = fft.fft(x)
        f = fft.fftfreq(x.shape[-1]) / t_step / 1000  # in kHz
        x_freq_filtered = x_freq[:, :, np.abs(f) < f_cutoff]
        return x_freq_filtered
