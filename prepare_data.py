import logging
from os import listdir
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

    def load_test_data(self, st_size=10000, overlap_size=5000, fft_f_cutoff=500):
        logging.debug('Preparing test data')

        target_labels = sorted(listdir('data/test'))

        test_dd = dd.read_csv('data/test/*.csv')
        test_darray = test_dd['acoustic_data'].to_dask_array(lengths=True).reshape(-1, 150000)

        logging.debug('Calculating rolling statistics...')
        ts_resample = self.overlapping_resample(test_darray, st_size, overlap_size)
        sample_statistics = self.get_summary_statistics(ts_resample)
        logging.debug('Performing DFT...')
        stdft_samples = np.abs(self.filtered_fft(ts_resample, self._t_step, fft_f_cutoff))

        # scale data
        logging.debug('Scaling data...')
        if self._stat_quartiles is not None and self._fft_quartiles is not None:
            scaled_sample_statistics = (sample_statistics - self._stat_quartiles[1]) / (self._stat_quartiles[2] - self._stat_quartiles[0])
            scaled_stdft_samples = (stdft_samples - self._fft_quartiles[1]) / (self._fft_quartiles[2] - self._fft_quartiles[0])
        else:
            raise ValueError('fit_transform set as False but no scaler fit exists')

        logging.debug('Performing delayed computations...')
        data = [scaled_sample_statistics.compute(), scaled_stdft_samples.compute()]

        logging.debug('Done')
        return data, target_labels

    def load_train_data(self, fit_transform=True, n_samples=200,
                        fft_resample_params={'sample_size': 10000, 'overlap_size': 5000},
                        stat_resample_params=[{'sample_size': 10000, 'overlap_size': 5000}],
                        fft_f_cutoff=500):
        """
        Load data from continuous file provided as training data for earthquake prediction. Selects
        n_samples random subsets of length as defined in the LANLDataLoader constructor. Returns a
        two-item tuple (data, target). Two data matrices provided are a rolling statistical summary
        and a short-time discrete fourier transform (ST-DFT) analysis.
        :param fit_transform: True to scale based on data selected in this draw, False to scale based
        on stored scaling from a previous draw
        :param n_samples: number of random subsets to draw from the input data
        :param fft_resample_params: arguments for overlapping_resample of data for FFT
        :param stat_resample_params: list of dicts of arguments for overlapping_resample of data 
        for summary statistics
        :param fft_f_cutoff: max frequency (in kHz) of returned fft based on timestep defined in
        LANLDataLoader constructor
        :return: two-item tuple (data, target), data being a two-item list returing rolling statistical
        calculations and ST-DFT as a dask array of shape (samples, calculations/frequencies, timesteps) and
        target being a dask array of shape (samples, time_to_failure)
        """
        # get random samples from time series data and perform transformations on those samples
        logging.debug('Preparing {} random samples...'.format(n_samples))
        ts_sample, sample_index = self.get_random_samples(self._data_array[:, 0], n_samples, self._sample_length)

        # resample and calculate fft
        logging.debug('Calculating DFT...')
        fft_resample = self.overlapping_resample(ts_sample, **fft_resample_params)
        stdft_samples = np.abs(self.filtered_fft(fft_resample, self._t_step, fft_f_cutoff)).compute()

        # find scaling parameters for fft output
        logging.debug('Scaling DFT...')
        if fit_transform:
            self._fft_quartiles = np.percentile(stdft_samples, q=[20, 50, 80])

        # resample and calculate bin statistics
        logging.debug('Calculating rolling statistics...')

        stat_resample = [
            self.overlapping_resample(ts_sample, **params).compute() for params in stat_resample_params
        ]
        sample_statistics = [
            self.get_summary_statistics(stats) for stats in stat_resample
        ]

        # find scaling parameters for statistics output
        logging.debug('Scaling statistics...')
        if fit_transform:
            self._stat_quartiles = np.zeros((len(sample_statistics), 3, sample_statistics[0].shape[2]))
            for i in range(len(sample_statistics)):
                for j in range(sample_statistics[i].shape[2]):
                    self._stat_quartiles[i, :, j] = np.percentile(sample_statistics[i][:, :, j], q=[20, 50, 80])

        # apply scaling
        if self._stat_quartiles is not None and self._fft_quartiles is not None:
            scaled_sample_statistics = []
            for i in range(len(sample_statistics)):
                logging.debug(self._stat_quartiles)
                sample_stat_center = self._stat_quartiles[i, 1, :]
                sample_stat_scale = self._stat_quartiles[i, 2, :] - self._stat_quartiles[i, 0, :]
                scaled_sample_statistics.append((sample_statistics[i] - sample_stat_center) / sample_stat_scale)

            stdft_stat_center = np.atleast_3d(self._fft_quartiles[1])
            stdft_stat_scale = np.atleast_3d(self._fft_quartiles[2] - self._fft_quartiles[0])
            scaled_stdft_samples = (stdft_samples - stdft_stat_center) / stdft_stat_scale
        else:
            raise ValueError('fit_transform set as False but no scaler fit exists')

        # set the input shape for dynamically creating neural network models later
        logging.debug('Calculating input shapes...')
        self._input_shape = [
            *[tuple(scaled_stats[0].shape) for scaled_stats in scaled_sample_statistics],
            tuple(scaled_stdft_samples[0].shape)
        ]
        logging.debug(self._input_shape)

        logging.debug('Performing delayed computations...')
        data = [*scaled_sample_statistics, scaled_stdft_samples]
        target = self._data_array[sample_index + self._sample_length, 1].compute()

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
        """
        :param x: the data to resample
        :param sample_size: number of samples for each rolling block
        :param overlap_size: overlap size between each rolling block
        :return: array with resampled data
        """
        resample_index = np.concatenate([np.arange(i, i + sample_size) for i in
                                         np.arange(x.shape[1] - sample_size, -1, -(sample_size - overlap_size))])
        resample = x[:, resample_index].reshape((x.shape[0], -1, sample_size))
        return resample

    @staticmethod
    def get_summary_statistics(x):
        summary_stats = np.concatenate([
            x.mean(axis=-1, keepdims=True),
            x.std(axis=-1, keepdims=True),
            x.max(axis=-1, keepdims=True),
            x.min(axis=-1, keepdims=True),
            np.diff(x).max(axis=-1, keepdims=True),
            *[np.percentile(x, q=qt, axis=-1, keepdims=True) for qt in [1, 5, 10, 25, 50, 75, 90, 95, 99]]
        ], axis=2)
        return summary_stats

    @staticmethod
    def filtered_fft(x, t_step, f_cutoff):
        x_freq = fft.fft(x)
        f = fft.fftfreq(x.shape[-1]) / t_step / 1000  # in kHz
        x_freq_filtered = x_freq[:, :, np.abs(f) < f_cutoff]
        return x_freq_filtered
