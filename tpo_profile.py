"""
TPO Profile
=====
Python version of TPO Profile (v2.0) developed for cTrader Trading Platform

Improvements:
    - Parallel processing of each profile interval
Python/C# author:
    - srlcarlg
"""
import math
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
import mplfinance as mpf

class TpoProfile:
    def __init__(self, df_ohlc: pd.DataFrame, row_height: float, interval: pd.Timedelta):
        """
        Create TPO Profiles from any charts  \n
        Backtest version.

        Usage
        ------
        >>> from tpo_profile import TpoProfile
        >>> tpo = TpoProfile(df_ohlc, row_height, pd.Timedelta(hours=4))
        >>> # plot with mplfinance
        >>> tpo.plot()
        >>> # get dataframes of each interval
        >>> df_intervals, df_profiles = tpo.profiles()

        Parameters
        ----------
        df_ohlc : dataframe
            * index/datetime, open, high, low, close, volume \n
            * "datetime": If is not present, the index will be used.
        row_height : float
            Cannot be less than or equal to 0.00000...
        interval : pd.Timedelta
            Interval for each profile, can be Minutes, Hours, Days, Weekly...
        """
        if 'datetime' not in df_ohlc.columns:
            df_ohlc["datetime"] = df_ohlc.index
        _expected = ['open', 'high', 'low', 'close']
        for column in _expected:
            if column not in df_ohlc.columns:
                raise ValueError(f"'{column}' column from the expected {_expected} doesn't exist!")

        self._row_height = row_height
        self._interval = interval
        df = df_ohlc

        dfs = []
        first_date = df['datetime'].iat[0].normalize() # any datetime to 00:00:00
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
                results = pool.map(self._create_tpo, dfs_list)
            return results

        self._intervals_dfs = dfs
        self._interval_profiles = parallel_process_dataframes(dfs)

    def profiles(self):
        """
        Return all intervals dataframes of ohlc + tpo_profile
        tpo_profile contains the following (3) columns with prefix 'tpo_':
            * tpo_['datetime', 'prices', 'values']

        >>> df_intervals, df_profiles = tpo.profiles()
        >>> # to access each ohlc_interval and its profile:
        >>> df_intervals[0] # [1], [2], etc...
        >>> df_profiles[0] # [1], [2], etc...
        """
        return [self._intervals_dfs, self._interval_profiles]

    def plot(self):
        """
        Plot all intervals of df_ohlc + tpo_profiles with mplfinance.
        """
        def parallel_process_dataframes():
            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                results = pool.starmap(self._mpf_workaround, zip(self._intervals_dfs, self._interval_profiles))
            return results
        dfs = parallel_process_dataframes()

        # join all interval profiles to get original df_ohlc with tpo_scatter
        df_ohlc_tpo = pd.concat(dfs, ignore_index=True)
        df_ohlc_tpo.index = df_ohlc_tpo['datetime']

        # total tpo_scatter columns
        numbers = df_ohlc_tpo.columns.str.extract(r'(\d+)', expand=False)
        total_add = int(numbers[len(numbers) - 1])
        apd = [mpf.make_addplot(df_ohlc_tpo[f'tpo_scatter_{i}'], color='deepskyblue', alpha=0.5) for i in range(total_add)]

        s = mpf.make_mpf_style(base_mpl_style='seaborn-v0_8', rc={'axes.grid': False})
        mpf.plot(df_ohlc_tpo, type='candle', style=s, addplot=apd, warn_too_much_data=len(df_ohlc_tpo) + 1,
                 title="TPO Profile",
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
        max_volume = df_profile['tpo_values'].max()

        for i in range(len(df_profile)):
            first = df_profile['tpo_values'].iat[i] * max_index
            result = math.ceil(first / max_volume)
            df_interval[f'tpo_scatter_{i}'] = np.NaN
            for price_index in range(result):
                if price_index >= max_index:
                    break
                df_interval[f'tpo_scatter_{i}'].iat[price_index] = df_profile['tpo_prices'].iat[i]

        return df_interval

    def _create_tpo(self, df_interval):
        profile = {'tpo_datetime': [], 'tpo_prices': [], 'tpo_values': []}
        
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
        
        profile['tpo_datetime'] = [df_interval['datetime'].iat[0]] * len(interval_segments)
        profile['tpo_prices'] = interval_segments
        profile['tpo_values'] = [0.0] * len(interval_segments)

        calculate_len = len(df_interval)
        for i in range(calculate_len):
            bar_high = df_interval['high'].iat[i]
            bar_low = df_interval['low'].iat[i]

            # v = vertical
            total_v_letters = 0
            for row in interval_segments:
                if (row < bar_high) and (row > bar_low):
                    total_v_letters += 1

            bar_prev_segment = bar_high
            for no_use in range(total_v_letters):
                for idx in range(len(interval_segments)):
                    prev_row = interval_segments[idx - 1]
                    row = interval_segments[idx]
                    if (bar_prev_segment >= prev_row) and (bar_prev_segment <= row):
                        profile['tpo_values'][idx] = profile['tpo_values'][idx] + 1
                        break

                bar_prev_segment = abs(bar_prev_segment - self._row_height)

        df_profile = pd.DataFrame(profile)
        df_profile.drop(df_profile.head(1).index, inplace=True)
        df_profile.drop(df_profile.tail(1).index, inplace=True)
        df_profile.index = df_profile['tpo_datetime']

        return df_profile