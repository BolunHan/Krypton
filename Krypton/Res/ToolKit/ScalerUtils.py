import json
import os
from typing import Union, Dict, Optional

import numpy as np
import pandas as pd

from . import GLOBAL_LOGGER

LOGGER = GLOBAL_LOGGER.getChild(os.path.basename(os.path.splitext(__file__)[0]))
__all__ = ['PandasScaler', 'ArrayScaler']


class PandasScaler(object):
    def __init__(self, with_mean: bool = False, by_serial: bool = False, fill_method: Optional[str] = None):
        """
        scaler for pandas DataFrame and Series
        :param with_mean: scaler sample mean to zero
        :param by_serial: if sample is a DataFrame, scale it by series or not
        :param fill_method: the method use to fill NaN and Inf
        """
        self.with_mean = with_mean
        self.by_serial = by_serial

        self.mean: Optional[Dict[str, float]] = None
        self.variance: Optional[Dict[str, float]] = None
        self._sample_mean: Optional[float] = None
        self._sample_variance: Optional[float] = None
        self._fitted: bool = False
        self.fill_method = fill_method

    # noinspection DuplicatedCode
    def fit(self, samples: Union[pd.DataFrame, pd.Series]):
        """
        fit the scaler
        :param samples: can be DataFrame or Series, by_serial only apply to DataFrames
        :return:
        """
        if self.by_serial and not isinstance(samples, pd.DataFrame):
            LOGGER.warning('by_serial disabled! Scaling by series requires a pandas DataFrame.')
            self.by_serial = False

        mean = {}
        variance = {}

        if self.by_serial:

            reshaped_sample = samples.values.reshape(-1)
            reshaped_sample = reshaped_sample[~np.isnan(reshaped_sample)]
            reshaped_sample = reshaped_sample[~np.isinf(reshaped_sample)]

            mu = np.mean(reshaped_sample) if self.with_mean else 0.0
            sigma = np.std(reshaped_sample)

            self._sample_mean = mu
            self._sample_variance = sigma

            for serial_name in samples.columns:
                serial: pd.Series = samples[serial_name]

                reshaped_sample = serial.values.reshape(-1)
                reshaped_sample = reshaped_sample[~np.isnan(reshaped_sample)]
                reshaped_sample = reshaped_sample[~np.isinf(reshaped_sample)]

                mu = np.mean(reshaped_sample) if self.with_mean else 0.0
                sigma = np.std(reshaped_sample)

                assert not (np.isnan(mu) or np.isnan(sigma)), 'Fitting scaler failed!'

                mean[serial_name] = mu
                variance[serial_name] = sigma
        else:
            reshaped_sample = samples.values.reshape(-1)
            reshaped_sample = reshaped_sample[~np.isnan(reshaped_sample)]
            reshaped_sample = reshaped_sample[~np.isinf(reshaped_sample)]

            mu = np.mean(reshaped_sample) if self.with_mean else 0.0
            sigma = np.std(reshaped_sample)

            self._sample_mean = mu
            self._sample_variance = sigma

            assert not (np.isnan(mu) or np.isnan(sigma)), 'Fitting scaler failed!'
            if isinstance(samples, pd.Series):
                mean[samples.name] = mu
                variance[samples.name] = sigma
            else:
                for serial_name in samples.columns:
                    mean[serial_name] = mu
                    variance[serial_name] = sigma

        self.mean = mean
        self.variance = variance
        self._fitted = True

    def transform(self, samples: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        transform DataFrame or Series with fitted mean and variance
        :param samples:
        :return:
        """
        assert self._fitted, 'Scaler not initialized'

        if isinstance(samples, pd.Series):
            # noinspection PyTypeChecker
            mean = self.mean.get(samples.name)
            # noinspection PyTypeChecker
            variance = self.variance.get(samples.name)

            if mean is None:

                if self.by_serial:
                    raise Exception('Serial name not match! Stored names {}, Transforming name {}.'.format(list(self.mean.keys()), samples.name))
                else:
                    if len(self.mean.keys()) < 10:
                        LOGGER.warning('Serial name not match! Stored names {}, Transforming name {}. Ignoring error...'.format(list(self.mean.keys()), samples.name))
                    else:
                        LOGGER.warning('Serial name not match! Stored {} names, Transforming name {}. Ignoring error...'.format(len(self.mean.keys()), samples.name))

                    mean = self._sample_mean
                    variance = self._sample_variance

            if self.fill_method == 'mean':
                samples_transformed = (samples.replace([np.inf, -np.inf], np.nan).add(-mean, fill_value=mean)) / variance
            elif self.fill_method is None:
                samples_transformed = (samples.replace([np.inf, -np.inf], np.nan).add(-mean, fill_value=None)) / variance
            else:
                LOGGER.warning('Invalid fill method! Using mean as fill method...')
                samples_transformed = (samples.replace([np.inf, -np.inf], np.nan).add(-mean, fill_value=mean)) / variance
        elif isinstance(samples, pd.DataFrame):
            samples_transformed = pd.DataFrame()
            for serial_name in samples.columns:
                mean = self.mean.get(serial_name)
                variance = self.variance.get(serial_name)

                if mean is None:

                    if self.by_serial:
                        raise Exception('Serial name not match! Stored names {}, Transforming name {}.'.format(list(self.mean.keys()), serial_name))
                    else:
                        if len(self.mean.keys()) < 10:
                            LOGGER.warning('Serial name not match! Stored names {}, Transforming name {}. Ignoring error...'.format(list(self.mean.keys()), serial_name))
                        else:
                            LOGGER.warning('Serial name not match! Stored {} names, Transforming name {}. Ignoring error...'.format(len(self.mean.keys()), serial_name))

                        mean = self._sample_mean
                        variance = self._sample_variance

                if self.fill_method == 'mean':
                    samples_transformed[serial_name] = (samples[serial_name].replace([np.inf, -np.inf], np.nan).add(-mean, fill_value=mean)) / variance
                elif self.fill_method is None:
                    samples_transformed[serial_name] = (samples[serial_name].replace([np.inf, -np.inf], np.nan).add(-mean, fill_value=None)) / variance
                else:
                    LOGGER.warning('Invalid fill method! Using mean as fill method...')
                    samples_transformed[serial_name] = (samples[serial_name].replace([np.inf, -np.inf], np.nan).add(-mean, fill_value=mean)) / variance
        else:
            raise Exception('Samples must be a pandas DataFrame or a Series')

        return samples_transformed

    def fit_transform(self, samples: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        Fit than transform the given data
        :param samples: the given data
        :return:
        """
        self.fit(samples)
        return self.transform(samples)


class ArrayScaler(object):
    """
    scale data with function D = D * alpha + beta
    """

    def __init__(self, with_beta: bool = True, by_serial: bool = True, min_max=False, positive_only=False):
        self.with_beta = with_beta
        self.by_serial = by_serial
        self.min_max = min_max
        self.positive_only = positive_only

        self.mean: Optional[np.ndarray] = None
        self.variance: Optional[np.ndarray] = None

    def fit(self, sample: np.ndarray):
        if self.by_serial:
            self.mean = np.mean(sample, axis=0)
            if self.min_max:
                if self.positive_only:
                    self.mean = np.min(sample, axis=0)
                    self.variance = np.max(sample, axis=0) - np.min(sample, axis=0)
                else:
                    self.variance = (np.max(sample, axis=0) - np.min(sample, axis=0)) / 2
            else:
                self.variance = np.std(sample, axis=0)
        else:
            self.mean = np.mean(sample)
            if self.min_max:
                if self.positive_only:
                    self.mean = np.min(sample)
                    self.variance = np.max(sample) - np.min(sample)
                else:
                    self.variance = (np.max(sample) - np.min(sample)) / 2
            else:
                self.variance = np.std(sample)

    def transform(self, sample: np.ndarray):
        if self.with_beta:
            sample = sample.astype(float) - self.mean

        sample = sample.astype(float) / self.variance

        return sample

    def reverse_transform(self, sample: np.ndarray):
        if self.with_beta:
            sample = sample.astype(float) + self.mean

        sample = sample.astype(float) * self.variance

        return sample

    def fit_transform(self, sample: np.ndarray):
        self.fit(sample)
        return self.transform(sample)

    def to_json(self):
        data_dict = {
            'with_beta': self.with_beta,
            'by_serial': self.by_serial,
            'min_max': self.min_max,
            'positive_only': self.positive_only
        }

        if self.mean is not None:
            data_dict['mean'] = self.mean.tolist()

        if self.variance is not None:
            data_dict['variance'] = self.variance.tolist()

        return json.dumps(data_dict)

    @classmethod
    def from_json(cls, json_message: Union[str, bytes, bytearray, dict]) -> 'ArrayScaler':
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)
        with_beta = json_dict['with_beta']
        by_serial = json_dict['by_serial']
        min_max = json_dict['by_serial']
        self = cls(with_beta=with_beta, by_serial=by_serial, min_max=min_max)

        if 'mean' in json_dict:
            self.mean = np.array(json_dict['mean'])

        if 'variance' in json_dict:
            self.variance = np.array(json_dict['variance'])

        return self

    def __reduce__(self):
        return self.__class__.from_json, (self.to_json(),)
