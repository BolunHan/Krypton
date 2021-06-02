import datetime
import os
from collections import defaultdict
from enum import Enum
from typing import List, Union, Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import GLOBAL_LOGGER
from .Collections import AttrDict
from .MarketUtils import MarketData

LOGGER = GLOBAL_LOGGER.getChild(os.path.basename(os.path.splitext(__file__)[0]))
__all__ = ['WaveDecoder']


class WaveDecoder(object):
    """
    this module implements a decoder based on Elliott Wave Principle https://en.wikipedia.org/wiki/Elliott_wave_principle
    convert raw price movement data into a vectorized, multi-level wave
    """

    class WaveFlag(Enum):
        Undetermined = 0
        Up = U = 1
        Down = D = 2
        Consolidate = C = 3
        Fluctuate = F = 4

    def __init__(
            self,
            max_level: int,
            detail_level: int = 0,
            historical_data: Union[List[MarketData], pd.Series] = None,
            key: Union[str, Callable] = 'market_price',
    ):
        if isinstance(key, str) or isinstance(key, Callable):
            self.key = key
        else:
            raise ValueError(f'Invalid key {self.key}!')

        self.max_level = max_level
        self.detail_level = detail_level  # {0: U/D, 1: U/D/F, 2: U/D/F/C}

        self._trend = defaultdict(dict)
        self._trend_df = None
        self._volatility_df = None
        self.local_extreme = {}

        if isinstance(historical_data, list):
            for market_data in historical_data:
                self.input(market_data=market_data)
        elif isinstance(historical_data, pd.Series):
            for market_time in historical_data.index:
                if AttrDict.valid_attr(historical_data.name, nested=True):
                    self.key = historical_data.name
                else:
                    self.key = 'market_price'
                self.input(market_data=AttrDict({'market_time': market_time, self.key: historical_data.at[market_time]}))

    def input(self, market_data):
        next_level_input = self._init_decode(market_data)

        for level in range(1, self.max_level):

            if next_level_input is None:
                break
            if self.detail_level == 0:
                next_level_input = self._decode(level=level, market_data=next_level_input)
            else:
                next_level_input = self._deep_decode(level=level, market_data=next_level_input)

        self._trend_df = None

    def _init_decode(self, market_data: MarketData):
        market_time = market_data.market_time
        last_data_point, last_trend = self.local_extreme.get('init', [None, self.WaveFlag.Undetermined])
        last_update = list(self._trend['MarketPrice'].keys())[-1] if self._trend['MarketPrice'].keys() else None

        if last_update is not None and market_time <= last_update:
            LOGGER.warning(f'MarketData obsolete! {market_data}')
            return

        if isinstance(self.key, str):
            market_price: float = getattr(market_data, self.key)
        elif isinstance(self.key, Callable):
            market_price: float = self.key(market_data)
        else:
            raise ValueError(f'Invalid key {self.key}')

        self._trend['MarketPrice'][market_time] = market_price

        new_data_point = [market_time, market_price]

        if last_data_point is not None:
            if last_data_point[1] > new_data_point[1]:
                new_trend = self.WaveFlag.Down
            elif last_data_point[1] < new_data_point[1]:
                new_trend = self.WaveFlag.Up
            else:
                new_trend = last_trend
        else:
            new_trend = last_trend

        self.local_extreme['init'] = [new_data_point, new_trend]

        if last_trend != new_trend:
            self._trend[0][last_data_point[0]] = new_trend
            return last_data_point
        else:
            return None

    def _deep_decode(self, level: int, market_data):
        next_level_input = None
        dp_5 = [market_data[0], market_data[1], self.WaveFlag.Undetermined]
        dp_0, dp_1, dp_2, dp_3, dp_4 = self.local_extreme.get(level, [None, None, None, None, None])

        if dp_3 is not None and (dp_3[1] < dp_4[1] < dp_5[1] or dp_3[1] > dp_4[1] > dp_5[1]):
            extremes = [dp_0, dp_1, dp_2, dp_3, dp_5]
            for lv in range(level, self.max_level):
                self._trend.get(lv, {}).pop(dp_4[0], None)
        else:
            extremes = [dp_1, dp_2, dp_3, dp_4, dp_5]

        if extremes[0] is not None:
            extreme_ob, extreme_0, extreme_1, extreme_2, extreme_3 = extremes
            if extreme_0[1] < extreme_2[1] and extreme_1[1] < extreme_3[1]:
                new_trend = self.WaveFlag.Up
                # a up trend must start with a local minimal extreme
                if extreme_0[1] > extreme_1[1]:
                    extreme_index = 1
                else:
                    extreme_index = 0

                for i in range(len(extremes)):
                    x = extremes[i]
                    if i == extreme_index:
                        last_trend = x[2]
                        if last_trend not in [self.WaveFlag.Undetermined, new_trend]:
                            next_level_input = [x[0], x[1], new_trend]
                        x[2] = new_trend
                    elif i >= extreme_index:
                        x[2] = new_trend
            elif extreme_0[1] > extreme_2[1] and extreme_1[1] > extreme_3[1]:
                new_trend = self.WaveFlag.Down

                # a up trend must start with a local minimal extreme
                if extreme_0[1] < extreme_1[1]:
                    extreme_index = 1
                else:
                    extreme_index = 0

                for i in range(len(extremes)):
                    x = extremes[i]
                    if i == extreme_index:
                        last_trend = x[2]
                        if last_trend not in [self.WaveFlag.Undetermined, new_trend]:
                            next_level_input = [x[0], x[1], new_trend]
                        x[2] = new_trend
                    elif i >= extreme_index:
                        x[2] = new_trend
            else:
                if self.detail_level == 2:
                    if extreme_0[1] < extreme_2[1] and extreme_1[1] > extreme_3[1]:
                        if extreme_2[1] < extreme_3[1]:
                            new_trend = self.WaveFlag.Consolidate
                        else:
                            new_trend = self.WaveFlag.Fluctuate
                    elif extreme_0[1] > extreme_2[1] and extreme_1[1] < extreme_3[1]:
                        if extreme_2[1] < extreme_3[1]:
                            new_trend = self.WaveFlag.Fluctuate
                        else:
                            new_trend = self.WaveFlag.Consolidate
                    else:
                        new_trend = extreme_2[2]
                elif self.detail_level == 1:
                    new_trend = self.WaveFlag.Fluctuate
                else:
                    new_trend = self.WaveFlag.Undetermined

                last_trend = extreme_2[2]
                extreme_px = [x[1] for x in extremes]
                if last_trend == self.WaveFlag.Up:
                    extreme_index = extreme_px.index(max(extreme_px))
                elif last_trend == self.WaveFlag.Down:
                    extreme_index = extreme_px.index(min(extreme_px))
                elif last_trend == new_trend:
                    extreme_index = 4
                else:
                    extreme_index = 3

                for i in range(len(extremes)):
                    x = extremes[i]
                    if i == extreme_index:
                        last_trend = x[2]
                        if last_trend not in [self.WaveFlag.Undetermined, new_trend]:
                            next_level_input = [x[0], x[1], new_trend]
                        x[2] = new_trend
                    elif i >= extreme_index:
                        x[2] = new_trend

        self.local_extreme[level] = extremes

        if next_level_input is not None:
            self._trend[level][next_level_input[0]] = next_level_input[2]
            return [next_level_input[0], next_level_input[1]]
        else:
            return None

    def _decode(self, level: int, market_data):
        next_level_input = None
        dp_3 = [market_data[0], market_data[1], self.WaveFlag.Undetermined]
        dp_0, dp_1, dp_2 = self.local_extreme.get(level, [None, None, None])

        if dp_0 is not None and (dp_1[1] < dp_2[1] < dp_3[1] or dp_1[1] > dp_2[1] > dp_3[1]):
            extremes = [dp_0, dp_1, dp_3]
            for lv in range(level, self.max_level):
                self._trend.get(lv, {}).pop(dp_2[0], None)
        else:
            extremes = [dp_1, dp_2, dp_3]

        if extremes[0] is not None:
            extreme_0, extreme_1, extreme_2 = extremes
            if extreme_0[1] < extreme_2[1]:
                new_trend = self.WaveFlag.Up
                # a up trend must start with a local minimal extreme
                if extreme_0[1] > extreme_1[1]:
                    extreme_index = 1
                else:
                    extreme_index = 0
            elif extreme_0[1] > extreme_2[1]:
                new_trend = self.WaveFlag.Down
                # a down trend must start with a local maximal extreme
                if extreme_0[1] < extreme_1[1]:
                    extreme_index = 1
                else:
                    extreme_index = 0
            else:
                new_trend = extreme_2[2] = extreme_1[2]
                extreme_index = -1

            for i in range(len(extremes)):
                x = extremes[i]
                if i == extreme_index:
                    last_trend = x[2]
                    if last_trend not in [self.WaveFlag.Undetermined, new_trend]:
                        next_level_input = [x[0], x[1], new_trend]
                    x[2] = new_trend
                elif i >= extreme_index:
                    x[2] = new_trend

        self.local_extreme[level] = extremes

        if next_level_input is not None:
            self._trend[level][next_level_input[0]] = next_level_input[2]
            return [next_level_input[0], next_level_input[1]]
        else:
            return None

    def wave(self, level: int, start: Optional[int] = None, end: Optional[int] = None):
        trend = self.trend.ffill().iloc[start: end]
        price = trend['MarketPrice'].to_list()
        market_time = trend.index.to_list()
        wave_flags = trend[level].to_list()

        last_flag = wave_flags[0]
        vector_start = (market_time[0], price[0])

        vector_list = []
        for i in range(1, len(wave_flags)):
            flag = wave_flags[i]

            if isinstance(last_flag, self.WaveFlag) and isinstance(flag, self.WaveFlag) and last_flag != flag:
                vector_end = (market_time[i], price[i])
                vector_list.append([vector_start, vector_end])
                vector_start = vector_end

            last_flag = flag

        return vector_list

    def next_extreme(self, level, market_time):
        if level in self.trend.columns:
            end_market_time = self.trend.loc[self.trend.index > market_time][level].first_valid_index()

            if end_market_time is None:
                return None
            else:
                end_price = self.trend.at[end_market_time, 'MarketPrice']
                return end_market_time, end_price
        else:
            return None

    def previous_extreme(self, level, market_time):
        if level in self.trend.columns:
            end_market_time = self.trend.loc[self.trend.index <= market_time][level].last_valid_index()

            if end_market_time is None:
                return None
            else:
                end_price = self.trend.at[end_market_time, 'MarketPrice']
                return end_market_time, end_price
        else:
            return None

    def plot(self):
        fig = go.Figure()
        fig.add_scatter(x=self.trend['MarketPrice'].index, y=self.trend['MarketPrice'], mode='lines', name='MarketPrice')

        for level in range(len(self.trend.columns) - 1):
            waves = self.wave(level=level)
            if waves:
                x = []
                y = []
                for vector in waves:
                    x.append(vector[0][0])
                    y.append(vector[0][1])

                x.append(waves[-1][1][0])
                y.append(waves[-1][1][1])

                fig.add_scatter(x=x, y=y, mode='lines', name=f'level_{level}')

        return fig

    def _volatility(self, interval: int = 60, multiplier=245) -> pd.DataFrame:
        vol = defaultdict(dict)

        for level in range(self.max_level):
            data = {
                'last_extreme': None,  # [index_int, market_price]
                'extremes': [],  # List[[index_int, average_return]]
                'active': False
            }
            for i in range(len(self.trend.index)):
                market_time = self.trend.index[i]
                market_price = self.trend.at[market_time, 'MarketPrice']
                extreme = self._trend.get(level, {}).get(market_time)

                # initialize
                if not data['extremes'] or extreme is not None:
                    data['extremes'].append([i, market_price])
                    data['active'] = None
                # append extremes data
                elif extreme is not None:
                    data['extremes'].append([i, market_price])
                    data['active'] = None
                else:
                    data['active'] = [i, market_price]

                # delete obsolete data
                if (data['active'] is not None and len(data['extremes']) > 3) or (len(data['extremes']) > 4):
                    second_extreme = data['extremes'][1]

                    if second_extreme[0] < i - interval:
                        data['extremes'].pop(0)

                    # calculate volatility
                    rt_list = []
                    wt_list = []
                    last_extreme: Optional[list] = None

                    for extreme in data['extremes']:
                        if last_extreme is not None:
                            rt = extreme[1] / last_extreme[1] - 1
                            trend_length = extreme[0] - last_extreme[0]
                            average_rt = rt / trend_length
                            weight = extreme[0] - max(i - interval, last_extreme[0])

                            if weight < 0:
                                weight = 0

                            rt_list.append(average_rt)
                            wt_list.append(weight)

                        last_extreme = extreme

                    if data['active'] is not None:
                        rt = data['active'][1] / last_extreme[1] - 1
                        trend_length = data['active'][0] - last_extreme[0]
                        average_rt = rt / trend_length
                        weight = data['active'][0] - max(i - interval, last_extreme[0])

                        if weight < 0:
                            weight = 0

                        rt_list.append(average_rt)
                        wt_list.append(weight)

                    vol[level][market_time] = np.sqrt(np.cov(rt_list, aweights=wt_list)) * multiplier

        vol_df = pd.DataFrame(vol)
        vol_df.insert(loc=0, column='Raw', value=self.trend['MarketPrice'].pct_change().rolling(window=interval).std() * multiplier)

        return vol_df

    @property
    def trend(self):
        if self._trend_df is None:
            self._trend_df = pd.DataFrame(self._trend)

        return self._trend_df

    @property
    def volatility(self):
        if self._volatility_df is None:
            self._volatility_df = self._volatility()

        return self._volatility_df


class WaveAugmentation(object):
    @classmethod
    def synthesis_next_extreme(cls, level: int, market_time: datetime.datetime, decoder: WaveDecoder):
        next_extreme = decoder.next_extreme(level=level, market_time=market_time)

        if next_extreme is None:
            return None

        sigma = decoder.volatility.at[market_time, max(level - 2, 0)]
        length = len(decoder.trend.loc[decoder.trend.index < next_extreme[0]].loc[decoder.trend.index > market_time])
        noise = sigma * np.sqrt(length) * np.random.normal()

        return next_extreme[0], next_extreme[1] * (1 + noise)

    @classmethod
    def synthesis_price(cls, price_series: pd.Series, level: int = 4, threshold: float = 1, scale: float = 1, wavelet='haar', mode='smooth', plot: bool = False):
        import pywt
        # price -> return
        return_series = price_series.pct_change().fillna(0)

        # decomposed data
        dwt = pywt.wavedec(data=return_series, wavelet=wavelet, level=level, mode=mode)
        adjusted_threshold = threshold * np.nanstd(return_series)
        low_pass_dwt = [dwt[0]] + [pywt.threshold(i, value=adjusted_threshold, mode="soft") for i in dwt[1:]]

        # low-pass-filtered
        low_pass_reconstructed = pywt.waverec(coeffs=low_pass_dwt, wavelet=wavelet, mode=mode)
        low_pass_return = pd.Series(index=price_series.index, data=low_pass_reconstructed)
        low_pass_price = low_pass_return.add(1).cumprod() * price_series.iat[0]

        # high-pass-filtered
        high_pass_return = return_series - low_pass_return
        high_pass_price = (high_pass_return * price_series.iat[0]).cumsum()

        # synthetic-price
        synthetic_high_pass_return = high_pass_return + np.random.normal(size=len(high_pass_return), scale=high_pass_return.rolling(60).std().bfill()) * scale
        synthetic_high_pass_price = (synthetic_high_pass_return * price_series.iat[0]).cumsum()
        synthetic_price = low_pass_price + synthetic_high_pass_price

        if plot:
            fig = go.Figure()
            fig.add_scatter(x=price_series.index, y=price_series, name=f'{price_series.name}.Price', yaxis='y1')
            fig.add_bar(x=return_series.index, y=return_series, name=f'{price_series.name}.Return', yaxis='y2')
            fig.add_scatter(x=low_pass_price.index, y=low_pass_price, name=f'{price_series.name}.Price.LP_{threshold}', yaxis='y1')
            fig.add_scatter(x=high_pass_price.index, y=high_pass_price, name=f'{price_series.name}.Price.HP_{threshold}', yaxis='y3')
            fig.add_scatter(x=synthetic_price.index, y=synthetic_price, name=f'{price_series.name}.Return.Synth', yaxis='y1')

            fig.update_layout(
                yaxis=dict(
                    title=f"{price_series.name}.Price",
                    titlefont=dict(
                        color="#1f77b4"
                    ),
                    tickfont=dict(
                        color="#1f77b4"
                    )
                ),
                yaxis2=dict(
                    title=f"{price_series.name}.Return",
                    titlefont=dict(
                        color="#ff7f0e"
                    ),
                    tickfont=dict(
                        color="#ff7f0e"
                    ),
                    anchor="free",
                    overlaying="y",
                    side="left",
                    position=0.15
                ),
                yaxis3=dict(
                    title=f"{price_series.name}.Noise",
                    titlefont=dict(
                        color="#d62728"
                    ),
                    tickfont=dict(
                        color="#d62728"
                    ),
                    anchor="x",
                    overlaying="y",
                    side="right"
                )
            )

            fig.show()

        return synthetic_price
