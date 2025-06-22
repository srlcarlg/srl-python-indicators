"""
Volume Profile
=====
Python version of Volume Profile (v2.0) developed for cTrader Trading Platform

Improvements:
    - Parallel processing of each profile interval
Additional Features:
    - DistributionData.Open and OHLC_No_Avg (high performance)
    - Buy vs Sell = Sum, Subtract and Divide total values of each side
    - Delta = Total, Min, Max Delta
Python/C# author:
    - srlcarlg
"""
import math
from copy import deepcopy
from math import sqrt
from enum import Enum
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
import mplfinance as mpf


class DistributionData(Enum):
    OHLC = 1
    OHLC_No_Avg = 2
    Open = 3
    High = 4
    Low = 5
    Close = 6
    Uniform_Distribution = 7
    Uniform_Presence = 8
    Parabolic_Distribution = 9
    Triangular_Distribution = 10

class VolumeProfile:
    def __init__(self, df_ohlcv: pd.DataFrame, df_ticks: pd.DataFrame | None,
                 row_height: float, interval: pd.Timedelta, distribution: DistributionData = DistributionData.OHLC_No_Avg):
        """
        Create Volume Profiles with Bars volume or Ticks data  \n
        Backtest version.

        Usage
        ------
        >>> from volume_profile import VolumeProfile
        >>> # Use df_ohlcv 'volume' bars as source
        >>> vp = VolumeProfile(df_ohlcv, None, row_height, pd.Timedelta(hours=4), DistributionData.OHLC_No_Avg)
        >>> # or use df_ticks as volume source
        >>> vp = VolumeProfile(df_ohlcv, df_ticks, row_height, pd.Timedelta(hours=4))
        >>> # plot with mplfinance
        >>> vp.plot('delta')
        >>> # get dataframes of each interval with all profile modes
        >>> df_intervals, df_profiles = vp.all()
        >>> # or get specific mode
        >>> vp.normal(), vp.buy_sell(), vp.delta()
        >>> # to access each ohlc_interval and its profile:
        >>> df_intervals[0] # [1], [2], etc...
        >>> df_profiles[0] # [1], [2], etc...

        Parameters
        ----------
        df_ohlcv : dataframe
            * index/datetime, open, high, low, close, volume \n
            * "datetime": If is not present, the index will be used.
            * "volume": If df_ticks is used, this column will be ignored.
        df_ticks : dataframe
            * If it's not None, ticks data will be used for Volume Profile, it should have:
            * datetime index or 'datetime' column
            * 'close' column (ticks price)
        row_height : float
            Cannot be less than or equal to 0.00000...
        interval : pd.Timedelta
            Interval for each profile, can be Minutes, Hours, Days, Weekly...
        distribution : DistributionData
            Distribution method for df_ohlcv bars volume
        """
        if 'datetime' not in df_ohlcv.columns:
            df_ohlcv["datetime"] = df_ohlcv.index
        if df_ticks is not None:
            if 'datetime' not in df_ticks.columns:
                df_ticks["datetime"] = df_ticks.index

        _expected = ['open', 'high', 'low', 'close']
        if df_ticks is None:
            _expected.append('volume')
        for column in _expected:
            if column not in df_ohlcv.columns:
                raise ValueError(f"'{column}' column from the expected {_expected} doesn't exist!")

        self._df_ohlcv = df_ohlcv
        self._df_ticks = df_ticks
        self._row_height = row_height
        self._interval = interval
        self._distribution = distribution
        df = df_ohlcv

        dfs = []
        first_date = df['datetime'].iat[0].normalize()  # any datetime to 00:00:00
        first_interval_date = first_date + self._interval
        first_interval_df = df[df['datetime'] < first_interval_date]
        dfs.append(first_interval_df)

        last_date = df['datetime'].tail(1).values[0]
        current_date = first_interval_date
        while current_date < last_date:
            start_interval_date = current_date
            end_interval_date = start_interval_date + self._interval
            interval_df = df.loc[(df['datetime'] >= start_interval_date) & (df['datetime'] < end_interval_date)]

            dfs.append(interval_df)
            current_date = end_interval_date

        def parallel_process_dataframes(dfs_list):
            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                results = pool.map(self._create_vp, dfs_list)
            return results

        self._intervals_dfs = dfs
        self._interval_profiles = parallel_process_dataframes(dfs)

    def normal(self):
        """
        Return all intervals dataframes of ohlcv + vp_profile normal mode
        vp_profile contains the following (3) columns with prefix 'vp_':
            * vp_['datetime', 'prices', 'normal', 'normal_total']
        """
        normal_dfs = [n[0] for n in self._interval_profiles]
        return [self._intervals_dfs, normal_dfs]

    def buy_sell(self):
        """
        Return all intervals dataframes of ohlcv + vp_profile buy_sell mode
        vp_profile contains the following (9) columns with prefix 'vp_':
            * vp_['datetime', 'prices', 'buy', 'sell', 'buy_value', 'sell_value', 'bs_sum', 'bs_subtract', 'bs_divide']
        """
        buy_sell_dfs = [bs[1] for bs in self._interval_profiles]
        return [self._intervals_dfs, buy_sell_dfs]

    def delta(self):
        """
        Return all intervals dataframes of ohlcv + vp_profile delta mode
        vp_profile contains the following (6) columns with prefix 'vp_':
            * vp_['datetime', 'prices', 'delta', 'delta_total', 'delta_min', 'delta_max']
        """
        delta_dfs = [d[2] for d in self._interval_profiles]
        return [self._intervals_dfs, delta_dfs]

    def all(self):
        """
        Return all intervals dataframes of ohlc + all vp_profile modes

        >>> df_intervals, df_profiles = vp.all()
        >>> # to access each ohlc_interval and its profile mode:
        >>> df_intervals[0]
        >>> df_profiles[0][0] # [0]: normal
        >>> df_profiles[0][1] # [1]: buy_sell
        >>> df_profiles[0][2] # [2]: delta
        """
        return [self._intervals_dfs, self._interval_profiles]

    def plot(self, mode: str = 'delta'):
        """
        Plot all intervals of df_ohlcv + vp_profile with mplfinance.
        :param mode: Volume mode to show
        """
        _profiles = ['normal', 'buy_sell', 'delta']
        if mode not in _profiles:
            raise ValueError(f"Only {_profiles} options are valid.")

        def parallel_process_dataframes():
            num_processes = cpu_count()
            _mode_profile = self.delta()[1] if mode == 'delta' else \
                self.buy_sell()[1] if mode == 'buy_sell' else self.normal()[1]
            with Pool(processes=num_processes) as pool:
                results = pool.starmap(self._mpf_workaround, zip(self._intervals_dfs, _mode_profile))
            return results
        dfs = parallel_process_dataframes()

        # join all interval profiles to get original df_ohlcv with vp_scatter
        df_ohlcv_vp = pd.concat(dfs, ignore_index=True)
        df_ohlcv_vp.index = df_ohlcv_vp['datetime']

        # total vp_scatter columns
        numbers = df_ohlcv_vp.columns.str.extract(r'(\d+)', expand=False)
        total_add = int(numbers[len(numbers) - 1])
        apd = []
        if mode == 'normal':
            apd = [mpf.make_addplot(df_ohlcv_vp[f'vp_scatter_{i}'], color='deepskyblue', alpha=0.3) for i in range(total_add)]
        else:
            for i in range(total_add):
                negative_column = f'vp_scatter_negative_{i}'
                if all([np.isnan(x) for x in df_ohlcv_vp[negative_column]]): continue # avoid empty columns (all np.NaN)
                apd.append(mpf.make_addplot(df_ohlcv_vp[negative_column], color='r', alpha=0.3))

                positive_column = f'vp_scatter_positive_{i}'
                if all([np.isnan(x) for x in df_ohlcv_vp[positive_column]]): continue # avoid empty columns (all np.NaN)
                apd.append(mpf.make_addplot(df_ohlcv_vp[positive_column], color='g', alpha=0.3 if mode != 'buy_sell' else 0.9))

        # type='scatter' is very heavy for massive data
        # The default add_plot 'line' type is pretty optimized
        s = mpf.make_mpf_style(base_mpl_style='seaborn-v0_8', rc={'axes.grid': False})
        mpf.plot(df_ohlcv_vp, type='candle', style=s, addplot=apd, warn_too_much_data=len(df_ohlcv_vp) + 1,
                 title=f"Volume Profile \n {mode} \n",
                 figsize=(12.5,6),
                 returnfig=True,
                 scale_padding=0.2)

        mpf.show()

    def _mpf_workaround(self, df_interval: pd.DataFrame, df_profile: pd.DataFrame):
        """
        Like in C# version a rule of three is used to plot the histograms,
        but instead of datetime(ms) the max_index of each interval is used.
        From there the math adjusts the histograms.
            max_volume    max_index(int)
               x             ?(int)
        """
        from warnings import simplefilter
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

        max_index = len(df_interval)
        max_volume = 0
        column_name = ''
        if 'vp_normal' in df_profile.columns:
            max_volume = df_profile['vp_normal'].max()
            column_name = 'vp_normal'
        if 'vp_delta' in df_profile.columns:
            max_volume = df_profile['vp_delta'].abs().max()
            column_name = 'vp_delta'
        if 'vp_buy' in df_profile.columns:
            max_volume = df_profile['vp_sell'].max()
            column_name = 'vp_buy'

        if column_name == 'vp_normal':
            for i in range(len(df_profile)):
                first = df_profile[column_name].iat[i] * max_index
                result = math.ceil(first / max_volume)

                df_interval[f'vp_scatter_{i}'] = np.NaN
                for price_index in range(result):
                    if price_index >= max_index:
                        break
                    df_interval[f'vp_scatter_{i}'].iat[price_index] = df_profile['vp_prices'].iat[i]
        elif column_name == 'vp_delta':
            for i in range(len(df_profile[column_name])):
                value = df_profile[column_name].iat[i]
                first = abs(value) * max_index
                result = math.ceil(first / max_volume)

                df_interval[f'vp_scatter_positive_{i}'] = np.NaN
                df_interval[f'vp_scatter_negative_{i}'] = np.NaN
                for price_index in range(result):
                    if price_index >= max_index:
                        break
                    if value < 0:
                        df_interval[f'vp_scatter_negative_{i}'].iat[price_index] = df_profile['vp_prices'].iat[i]
                    else:
                        df_interval[f'vp_scatter_positive_{i}'].iat[price_index] = df_profile['vp_prices'].iat[i]
        else:
            for i in range(len(df_profile[column_name])):
                value = df_profile[column_name].iat[i]
                first = value * math.ceil(max_index / 2)
                result = math.ceil(first / max_volume)

                df_interval[f'vp_scatter_positive_{i}'] = np.NaN
                for price_index in range(result):
                    if price_index >= max_index:
                        break
                    df_interval[f'vp_scatter_positive_{i}'].iat[price_index] = df_profile['vp_prices'].iat[i]

            for i in range(len(df_profile['vp_sell'])):
                value = df_profile['vp_sell'].iat[i]
                first = value * max_index
                result = math.ceil(first / max_volume)

                df_interval[f'vp_scatter_negative_{i}'] = np.NaN
                for price_index in range(result):
                    if price_index >= max_index:
                        break
                    df_interval[f'vp_scatter_negative_{i}'].iat[price_index] = df_profile['vp_prices'].iat[i]

        return df_interval

    def _create_vp(self, df_interval):
        interval_lowest = df_interval['low'].min()
        interval_highest = df_interval['high'].max()
        interval_open = df_interval['open'].iat[0]
        
        interval_segments = []
        prev_segment = interval_open
        while prev_segment >= (interval_lowest - self._row_height):
            interval_segments.append(prev_segment)
            prev_segment = abs(prev_segment - self._row_height)
        prev_segment = interval_open
        while prev_segment <= (interval_highest + self._row_height):
            interval_segments.append(prev_segment)
            prev_segment = abs(prev_segment + self._row_height)
        
        interval_segments.sort()
        
        normal = {'datetime': [], 'prices': [], 'values': [], 'total_value': 0.0}
        buy_sell = {'datetime': [], 'prices': [], 'vp_buy': [], 'vp_sell': [],
                    'value_buy': 0.0, 'value_sell': 0.0,
                    'value_sum': 0.0, 'value_subtract': 0.0, 'value_divide': 0.0}
        delta = {'datetime': [], 'prices': [], 'values': [],
                 'total_delta': 0.0, 'min_delta': 0.0, 'max_delta': 0.0}

        # I really thought that this 'multiple variables assign' was right and ended up doing the same for dicts['values']
        # but after some headaches and Stackoverflow researches, it's meaning that:
        # "all variables will share the same reference (value)"... *cough *cough pointers here?!
        # since datetime and price segments are equal in any volume mode, this behaviour is welcome.
        normal['datetime'], buy_sell['datetime'], delta['datetime'] = \
            [ [df_interval['datetime'].iat[0]] * len(interval_segments) ] * 3
        normal['prices'], buy_sell['prices'], delta['prices'] = [interval_segments] * 3
        # Imagine the headache to debug such silently unexpected behavior
        normal['values'], buy_sell['vp_buy'], buy_sell['vp_sell'], delta['values'] = \
            (deepcopy( [0.0] * len(interval_segments) ) for _ in range(4))

        def _add_volume(index: int, volume_i: float, is_up_i: bool):
            normal['values'][index] = normal['values'][index] + volume_i

            buy_sell['vp_buy' if is_up_i else 'vp_sell'][index] = \
                buy_sell['vp_buy' if is_up_i else 'vp_sell'][index] + volume_i

            prev_delta_i = sum(delta['values'])

            buy = buy_sell['vp_buy'][index]
            sell = buy_sell['vp_sell'][index]
            if buy != 0 and sell != 0:
                delta['values'][index] = delta['values'][index] + (buy - sell)
            elif buy != 0 and sell == 0:
                delta['values'][index] = delta['values'][index] + buy
            elif buy == 0 and sell != 0:
                delta['values'][index] = delta['values'][index] + (-sell)

            current_delta = sum(delta['values'])
            if prev_delta_i > current_delta:
                delta['min_delta'] = prev_delta_i
            if prev_delta_i < current_delta:
                delta['max_delta'] = prev_delta_i

        if self._df_ticks is not None:
            start = df_interval['datetime'].head(1).values[0]
            end = df_interval['datetime'].tail(1).values[0]
            ticks_interval = self._df_ticks.loc[(self._df_ticks['datetime'] >= start) & (self._df_ticks['datetime'] <= end)]

            calculate_len = len(ticks_interval)
            for i in range(calculate_len):
                tick = ticks_interval['close'].iat[i]
                prev_tick = ticks_interval['close'].iat[i - 1]
                for idx in range(len(interval_segments)):
                    row = interval_segments[idx]
                    prev_row = interval_segments[idx - 1]
                    if (tick >= prev_row) and (tick <= row):
                        normal['values'][idx] = normal['values'][idx] + 1

                        if tick > prev_tick:
                            buy_sell['vp_buy'][idx] = buy_sell['vp_buy'][idx] + 1
                        elif tick < prev_tick:
                            buy_sell['vp_sell'][idx] = buy_sell['vp_sell'][idx] + 1
                        elif tick == prev_tick:
                            buy_sell['vp_buy'][idx] = buy_sell['vp_buy'][idx] + 1
                            buy_sell['vp_sell'][idx] = buy_sell['vp_sell'][idx] + 1

                        prev_delta_i = sum(delta['values'])

                        buy = buy_sell['vp_buy'][idx]
                        sell = buy_sell['vp_sell'][idx]
                        delta['values'][idx] = delta['values'][idx] + (buy - sell)

                        current_delta = sum(delta['values'])
                        if prev_delta_i > current_delta:
                            delta['min_delta'] = prev_delta_i
                        if prev_delta_i < current_delta:
                            delta['max_delta'] = prev_delta_i
        else:
            calculate_len = len(df_interval)
            for i in range(calculate_len):
                open = df_interval['open'].iat[i]
                high = df_interval['high'].iat[i]
                low = df_interval['low'].iat[i]
                close = df_interval['close'].iat[i]
                volume = df_interval['volume'].iat[i]
                is_up = close >= open
                if self._distribution == DistributionData.OHLC or self._distribution == DistributionData.OHLC_No_Avg:
                    avg_vol = (volume / (open + high + low + close / 4)) if self._distribution == \
                                                                            DistributionData.OHLC else volume
                    for idx in range(len(interval_segments)):
                        row = interval_segments[idx]
                        if is_up:
                            if (row <= open) and (row >= low):
                                _add_volume(idx, avg_vol, is_up)
                            if (row <= high) and (row >= low):
                                _add_volume(idx, avg_vol, is_up)
                            if (row <= high) and (row >= close):
                                _add_volume(idx, avg_vol, is_up)
                        else:
                            if (row >= open) and (row <= high):
                                _add_volume(idx, avg_vol, is_up)
                            if (row <= high) and (row >= low):
                                _add_volume(idx, avg_vol, is_up)
                            if (row >= low) and (row <= close):
                                _add_volume(idx, avg_vol, is_up)
                elif self._distribution == DistributionData.Open or self._distribution == DistributionData.High or \
                        self._distribution == DistributionData.Low or self._distribution == DistributionData.Close:
                    for idx in range(len(interval_segments)):
                        row = interval_segments[idx]
                        prev_row = interval_segments[idx - 1]
                        if self._distribution == DistributionData.Open:
                            if (row >= open) and (prev_row <= open):
                                _add_volume(idx, volume, is_up)
                        elif self._distribution == DistributionData.High:
                            if (row >= high) and (prev_row <= high):
                                _add_volume(idx, volume, is_up)
                        elif self._distribution == DistributionData.Low:
                            if (row >= low) and (prev_row <= low):
                                _add_volume(idx, volume, is_up)
                        else:
                            if (row >= close) and (prev_row <= close):
                                _add_volume(idx, volume, is_up)
                elif self._distribution == DistributionData.Uniform_Distribution:
                    hl = abs(high - low)
                    uni_vol = volume / hl
                    for idx in range(len(interval_segments)):
                        row = interval_segments[idx]
                        if (row >= low) and (row <= high):
                            _add_volume(idx, uni_vol, is_up)
                elif self._distribution == DistributionData.Uniform_Presence:
                    uni_presence = 1
                    for idx in range(len(interval_segments)):
                        row = interval_segments[idx]
                        if (row >= low) and (row <= high):
                            _add_volume(idx, uni_presence, is_up)
                elif self._distribution == DistributionData.Parabolic_Distribution:
                    hl2 = abs(high - low) / 2
                    hl2_sqtr = sqrt(hl2)
                    # C# version had a 'float division by zero' by doing hl2_sqtr / hl2_sqtr
                    # In both languages (C#, Python), I not sure if the Parabolic implementation is correct.
                    final = hl2_sqtr / hl2
                    parabolic_vol = volume / final
                    for idx in range(len(interval_segments)):
                        row = interval_segments[idx]
                        if (row >= low) and (row <= high):
                            _add_volume(idx, parabolic_vol, is_up)
                elif self._distribution == DistributionData.Triangular_Distribution:
                    hl = abs(high - low)
                    hl2 = hl / 2
                    hl_minus = hl - hl2
                    # I modified the one_step and second_step because multiplying itself by 2 and dividing by 2 doesn't make sense
                    # In both languages (C#, Python), I not sure if the Triangular implementation is correct.
                    one_step = hl2 * hl_minus / 2
                    second_step = hl_minus * hl / 2
                    final = one_step + second_step
                    triangular_volume = volume / final
                    for idx in range(len(interval_segments)):
                        row = interval_segments[idx]
                        if (row >= low) and (row <= high):
                            _add_volume(idx, triangular_volume, is_up)

        normal['total_value'] = sum(normal['values'])

        buy_sell['value_buy'] = sum(buy_sell['vp_buy'])
        buy_sell['value_sell'] = sum(buy_sell['vp_sell'])
        buy_sell['value_sum'] = buy_sell['value_buy'] + buy_sell['value_sell']
        buy_sell['value_subtract'] =  buy_sell['value_buy'] - buy_sell['value_sell']
        if buy_sell['value_buy'] != 0 and buy_sell['value_sell'] != 0:
            buy_sell['value_divide'] =  buy_sell['value_buy'] / buy_sell['value_sell']

        delta['total_delta'] = sum(delta['values'])

        normal_df = pd.DataFrame(normal)
        normal_df.rename(columns={'datetime': 'vp_datetime',
                                  'prices': 'vp_prices',
                                  'values': 'vp_normal',
                                  'total_value': 'vp_normal_total'}, inplace=True)

        buy_sell_df = pd.DataFrame(buy_sell)
        buy_sell_df.rename(columns={'datetime': 'vp_datetime',
                                    'prices': 'vp_prices',
                                    'value_buy': 'vp_buy_value',
                                    'value_sell': 'vp_sell_value',
                                    'value_sum': 'vp_bs_sum',
                                    'value_subtract': 'vp_bs_subtract',
                                    'value_divide': 'vp_bs_divide'}, inplace=True)

        delta_df = pd.DataFrame(delta)
        delta_df.rename(columns={'datetime': 'vp_datetime',
                                  'prices': 'vp_prices',
                                  'values': 'vp_delta',
                                  'total_delta': 'vp_delta_total',
                                  'min_delta': 'vp_delta_min',
                                  'max_delta': 'vp_delta_max'}, inplace=True)

        for _df in [normal_df, buy_sell_df, delta_df]:
            _df.drop(_df.head(1).index, inplace=True)
            _df.drop(_df.tail(1).index, inplace=True)

        return [normal_df, buy_sell_df, delta_df]
