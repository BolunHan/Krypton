import os
from typing import Union, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from . import GLOBAL_LOGGER

LOGGER = GLOBAL_LOGGER.getChild(os.path.basename(os.path.splitext(__file__)[0]))
__all__ = ['TechnicalAnalysis']


# noinspection PyPep8Naming
class TechnicalAnalysis(object):

    def __getattribute__(self, name):
        LOGGER.warning(DeprecationWarning('[TechnicalAnalysis] depreciated! Use [Factors.Common_Factors] instead!'), stacklevel=2)
        super().__getattribute__(name)

    @classmethod
    def BOLL_value(
            cls,
            close_price: Union[pd.DataFrame, pd.Series],
            interval: int = 1,
            window: int = 26
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Calculate BOLL value of given serial or data frame
        :param close_price: close price DataFrame or Serial. Each column are the serial of close price
        :param interval: calculate interval
        :param window: window parameter
        :return: serial or data frame
        """
        close_index = close_price.index
        close_df = close_price.reset_index(drop=True)
        # noinspection SpellCheckingInspection
        close_df = close_df.fillna(method='ffill')

        boll_value_list = []

        for i in range(interval):
            target_df = close_df.iloc[i::interval]

            mean = target_df.rolling(window=window).mean()
            std = target_df.rolling(window=window).std()
            boll = (target_df - mean) / std

            boll_value_list.append(boll)

        boll_value = pd.concat(boll_value_list, axis=0, sort=False).sort_index()
        boll_value.index = close_index

        return boll_value

    # noinspection PyPep8Naming,DuplicatedCode
    @staticmethod
    def MACD_value(
            close_price: Union[pd.DataFrame, pd.Series],
            interval: int = 1,
            window: Optional[Dict[str, int]] = None,
            flag: str = 'bar'
    ) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Calculate MACD value of given serial or data frame
        :param close_price: close price DataFrame or Serial. Each column are the serial of close price
        :param interval: calculate interval
        :param window: window parameter, Default is {'short': 12, 'long': 26, 'diff': 9}
        :param flag: return which value of MACD
        :return:
        """
        if window is None:
            window = {'short': 12, 'long': 26, 'diff': 9}

        close_index = close_price.index
        close_df = close_price.reset_index(drop=True)
        # noinspection SpellCheckingInspection
        close_df = close_df.fillna(method='ffill')

        dif_list = []
        dea_list = []
        bar_list = []

        for i in range(interval):
            target_df: Union[pd.DataFrame, pd.Series] = close_df.iloc[i::interval]

            ema_short = target_df.ewm(span=window.get('short', 12), adjust=False).mean()
            ema_long = target_df.ewm(span=window.get('long', 26), adjust=False).mean()
            dif = ema_short - ema_long
            dea = dif.ewm(span=window.get('diff', 9), adjust=False).mean()
            bar = 2 * (dif - dea)

            dif_list.append(dif)
            dea_list.append(dea)
            bar_list.append(bar)

        dif = pd.concat(dif_list, axis=0, sort=False).sort_index()
        dea = pd.concat(dea_list, axis=0, sort=False).sort_index()
        bar = pd.concat(bar_list, axis=0, sort=False).sort_index()

        dif.index = close_index
        dea.index = close_index
        bar.index = close_index

        if flag.upper() == 'BAR':
            return bar
        elif flag.upper() == 'DIF':
            return dif
        elif flag.upper() == 'DEA':
            return dea
        elif flag.upper() == 'FULL':
            return dif, dea, bar
        else:
            return bar

    # noinspection PyPep8Naming, DuplicatedCode
    @staticmethod
    def KDJ_value(
            close_price: Union[pd.DataFrame, pd.Series],
            high_price: Union[pd.DataFrame, pd.Series],
            low_price: Union[pd.DataFrame, pd.Series],
            interval: int = 1,
            window: Optional[Dict[str, int]] = None,
            flag: str = 'J'
    ) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pd.Series, pd.Series, pd.Series]]:
        """
            Calculate KDJ value of given serial or data frame
            :param close_price: close price DataFrame or Serial. Each column are the serial of close price
            :param high_price: HIGH price DataFrame or Serial. Each column are the serial of close price
            :param low_price: LOW price DataFrame or Serial. Each column are the serial of close price
            :param interval: calculate interval
            :param window: window parameter, Default is {'RSV_Window': 9, 'K_Window': 3, 'D_Window': 3}
            :param flag: return which value of KDJ
            :return:
            """
        assert len(close_price) == len(high_price) == len(low_price), 'Alignment error!'
        assert close_price.index[0] == high_price.index[0] == low_price.index[0], 'Alignment error!'

        if window is None:
            window = {'RSV_Window': 9, 'K_Window': 3, 'D_Window': 3}

        close_index = close_price.index
        close_df = close_price.reset_index(drop=True)
        high_df = high_price.reset_index(drop=True)
        low_df = low_price.reset_index(drop=True)

        # noinspection SpellCheckingInspection
        close_df = close_df.fillna(method='ffill')
        # noinspection SpellCheckingInspection
        high_df = high_df.fillna(method='ffill')
        # noinspection SpellCheckingInspection
        low_df = low_df.fillna(method='ffill')

        k_list = []
        d_list = []
        j_list = []

        for i in range(interval):
            target_close_df: Union[pd.DataFrame, pd.Series] = close_df.iloc[i::interval]
            target_high_df: Union[pd.DataFrame, pd.Series] = high_df.rolling(window=interval).max().iloc[i::interval]
            target_low_df: Union[pd.DataFrame, pd.Series] = low_df.rolling(window=interval).min().iloc[i::interval]

            window_high = target_high_df.rolling(window=window.get('RSV_Window', 9)).max()
            window_low = target_low_df.rolling(window=window.get('RSV_Window', 9)).min()

            rsv = (target_close_df - window_low) / (window_high - window_low) * 100

            df_k = rsv.ewm(com=window.get('K_Window', 3) - 1).mean()
            df_d = df_k.ewm(com=window.get('D_Window', 3) - 1).mean()
            df_j = 3 * df_k - 2 * df_d

            k_list.append(df_k)
            d_list.append(df_d)
            j_list.append(df_j)

        k = pd.concat(k_list, axis=0, sort=False).sort_index()
        d = pd.concat(d_list, axis=0, sort=False).sort_index()
        j = pd.concat(j_list, axis=0, sort=False).sort_index()

        k.index = close_index
        d.index = close_index
        j.index = close_index

        if flag.upper() == 'K':
            return k
        elif flag.upper() == 'D':
            return d
        elif flag.upper() == 'J':
            return j
        elif flag.upper() == 'FULL':
            return k, d, j
        else:
            return j

    # noinspection PyPep8Naming, DuplicatedCode
    @staticmethod
    def CCI_value(
            close_price: Union[pd.DataFrame, pd.Series],
            high_price: Union[pd.DataFrame, pd.Series],
            low_price: Union[pd.DataFrame, pd.Series],
            interval: int = 1,
            window: Optional[Dict[str, int]] = None
    ) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pd.Series, pd.Series, pd.Series]]:
        """
            Calculate CCI value of given serial or data frame
            :param close_price: close price DataFrame or Serial. Each column are the serial of close price
            :param high_price: HIGH price DataFrame or Serial. Each column are the serial of close price
            :param low_price: LOW price DataFrame or Serial. Each column are the serial of close price
            :param interval: calculate interval
            :param window: window parameter, Default is {'constant': 0.015, 'span': 14}
            :return:
            """
        assert len(close_price) == len(high_price) == len(low_price), 'Alignment error!'
        assert close_price.index[0] == high_price.index[0] == low_price.index[0], 'Alignment error!'

        if window is None:
            window = {'constant': 0.015, 'span': 14}

        close_index = close_price.index
        close_df = close_price.reset_index(drop=True)
        high_df = high_price.reset_index(drop=True)
        low_df = low_price.reset_index(drop=True)

        # noinspection SpellCheckingInspection
        close_df = close_df.fillna(method='ffill')
        # noinspection SpellCheckingInspection
        high_df = high_df.fillna(method='ffill')
        # noinspection SpellCheckingInspection
        low_df = low_df.fillna(method='ffill')

        cci_list = []

        for i in range(interval):
            target_close_df: Union[pd.DataFrame, pd.Series] = close_df.iloc[i::interval]
            target_high_df: Union[pd.DataFrame, pd.Series] = high_df.rolling(window=interval).max().iloc[i::interval]
            target_low_df: Union[pd.DataFrame, pd.Series] = low_df.rolling(window=interval).min().iloc[i::interval]

            tp = (target_high_df + target_low_df + target_close_df) / 3
            ma = tp.rolling(window=window.get('span', 0.015)).mean()
            md = tp.rolling(window=window.get('span', 0.015)).std()
            target_cci = (tp - ma) / (window.get('constant', 0.015) * md)

            cci_list.append(target_cci)

        cci = pd.concat(cci_list, axis=0, sort=False).sort_index()
        cci.index = close_index

        return cci

    # noinspection DuplicatedCode
    @staticmethod
    def dispersion(
            trade_notional: pd.DataFrame,
            calculation_period: int = 5,
            alignment_method: str = 'rank'
    ) -> pd.DataFrame:
        notional_proportion_data_frame = trade_notional.div(np.sum(trade_notional, axis=1), axis=0)

        b = {}
        i = 0
        while i < len(notional_proportion_data_frame):

            x_s = []
            y_s = []

            if alignment_method == 'rank':
                for j in range(min(calculation_period, i + 1)):
                    y = [np.log(k) for k in notional_proportion_data_frame.iloc[i + j - min(calculation_period, i + 1) + 1] if k > 0]
                    y.sort(reverse=True)

                    x = [k + 1 for k in range(len(y))]

                    x_s.extend(x)
                    y_s.extend(y)
            elif alignment_method == 'ticker':
                proportion = notional_proportion_data_frame.iloc[i - min(calculation_period, i + 1) + 1: i].sum()
                y_s = [np.log(k) for k in proportion if k > 0]
                y_s.sort(reverse=True)
                x_s = [k + 1 for k in range(len(y_s))]
            # noinspection PyBroadException
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_s, y_s)

                b[notional_proportion_data_frame.index[i]] = slope
            except Exception as _:
                b[notional_proportion_data_frame.index[i]] = float('nan')

            i += 1

        result = pd.DataFrame(data={'Dispersion': b})

        return result

    @staticmethod
    def moving_average(
            close_price: Union[pd.DataFrame, pd.Series],
            interval: int = 1,
            window: int = 5
    ):
        """
        Calculate MA value of given serial or data frame
        :param close_price: close price DataFrame or Serial. Each column are the serial of close price
        :param interval: calculate interval
        :param window: window parameter
        :return: serial or data frame
        """
        close_index = close_price.index
        close_df = close_price.reset_index(drop=True)
        # noinspection SpellCheckingInspection
        close_df = close_df.fillna(method='ffill')

        ma_value_list = []

        for i in range(interval):
            target_df = close_df.iloc[i::interval]

            mean = target_df.rolling(window=window).mean()

            ma_value_list.append(mean)

        ma_value = pd.concat(ma_value_list, axis=0, sort=False).sort_index()
        ma_value.index = close_index

        return ma_value

    # noinspection PyPep8Naming, DuplicatedCode
    @staticmethod
    def ASI_value(
            open_price: Union[pd.DataFrame, pd.Series],
            close_price: Union[pd.DataFrame, pd.Series],
            high_price: Union[pd.DataFrame, pd.Series],
            low_price: Union[pd.DataFrame, pd.Series],
            interval: int = 11
    ) -> pd.DataFrame:

        assert len(open_price) == len(close_price) == len(high_price) == len(low_price), 'Alignment error!'
        assert open_price.index[0] == close_price.index[0] == high_price.index[0] == low_price.index[0], 'Alignment error!'

        close_index = close_price.index
        open_df = open_price.reset_index(drop=True)
        close_df = close_price.reset_index(drop=True)
        high_df = high_price.reset_index(drop=True)
        low_df = low_price.reset_index(drop=True)

        # noinspection SpellCheckingInspection
        open_df = open_df.fillna(method='ffill')
        # noinspection SpellCheckingInspection
        close_df = close_df.fillna(method='ffill')
        # noinspection SpellCheckingInspection
        high_df = high_df.fillna(method='ffill')
        # noinspection SpellCheckingInspection
        low_df = low_df.fillna(method='ffill')

        asi_list = []

        for i in range(interval):
            target_open_df: Union[pd.DataFrame, pd.Series] = open_df.iloc[i::interval]
            target_close_df: Union[pd.DataFrame, pd.Series] = close_df.iloc[i::interval]
            target_high_df: Union[pd.DataFrame, pd.Series] = high_df.rolling(window=interval).max().iloc[i::interval]
            target_low_df: Union[pd.DataFrame, pd.Series] = low_df.rolling(window=interval).min().iloc[i::interval]

            target_si = 50 * (
                    target_close_df.shift(periods=1) - target_close_df
                    + 0.5 * (target_close_df.shift(periods=1) - target_open_df.shift(periods=1))
                    + 0.25 * (target_close_df - target_open_df)
            ) * pd.concat((target_high_df.shift(periods=1) - target_close_df), (target_low_df.shift(periods=1) - target_close_df)).max(level=0) / (target_high_df - target_low_df)

            r_1 = target_high_df - target_close_df.shift(periods=1) - 0.5 * (target_low_df - target_close_df.shift(periods=1)) + 0.25 * (target_close_df.shift(periods=1) - target_open_df.shift(periods=1))
            r_2 = target_low_df - target_close_df.shift(periods=1) - 0.5 * (target_high_df - target_close_df.shift(periods=1)) + 0.25 * (target_close_df.shift(periods=1) - target_open_df.shift(periods=1))
            r_3 = target_high_df - target_low_df + 0.25 * (target_close_df.shift(periods=1) - target_open_df.shift(periods=1))

            r_1 = r_1.where(target_close_df.shift(periods=1) < target_low_df)
            r_2 = r_2.where(target_close_df.shift(periods=1) > target_high_df)
            r = r_1.fillna(r_2).fillna(r_3)

            target_si = target_si / r

            asi_list.append(target_si.cumsum())

        asi = pd.concat(asi_list, axis=0, sort=False).sort_index()
        asi.index = close_index

        return asi

    @staticmethod
    def Volatility(
            close_price: Union[pd.DataFrame, pd.Series],
            interval: int = 1,
            window=5,
            multiplier=244
    ):
        close_index = close_price.index
        close_df = close_price.reset_index(drop=True)
        # noinspection SpellCheckingInspection
        close_df = close_df.fillna(method='ffill')

        std_value_list = []

        for i in range(interval):
            target_df = close_df.iloc[i::interval]

            std = target_df.pct_change().rolling(window=window).std() * np.sqrt(multiplier / interval)

            std_value_list.append(std)

        std_value = pd.concat(std_value_list, axis=0, sort=False).sort_index()
        std_value.index = close_index

        return std_value
