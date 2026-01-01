"""
Volume Profile
=====
Python version of Volume Profile (v2.0) developed for cTrader Trading Platform

Features from revision 1 (after Order Flow Aggregated development)
    - Mini/Weekly/Monthly Profiles
    - Fixed-Range Profiles
    - Shared Segments
Additional Features => that will be implemented to C# version... sometime next year (2026)
    - HVN/LVN Detection + Levels for [Normal, Delta] modes only
Improvements:
    - Parallel processing of each profile interval
    - Numpy Arrays, where possible.
Python/C# author:
    - srlcarlg
"""
import itertools
import math
from copy import deepcopy
from math import sqrt
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
import mplfinance as mpf

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from models_utils.profile_models import DistributionData, SegmentsInterval, ExtraProfile, ProfileFilter
from models_utils.profile_utils import get_intervals_list, create_shared_segments, get_segments, get_prefix, \
    volume_nodes_filter


class VolumeProfile:
    def __init__(self, df_ohlcv: pd.DataFrame, df_ticks: pd.DataFrame | None,
                 row_height: float, interval: pd.Timedelta,
                 distribution: DistributionData = DistributionData.OHLC_No_Avg,
                 profile_filter: ProfileFilter | None = None,
                 segments_interval: SegmentsInterval = SegmentsInterval.Daily,
                 with_plotly_columns: bool = True):
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
        >>> # plot with plotly
        >>> vp.plot_ly('delta')

        >>> # get dataframes of each interval with all profile modes
        >>> df_intervals, df_profiles = vp.all()
        >>> # or get specific mode
        >>> vp.normal(), vp.buy_sell(), vp.delta()

        >>> # to access each ohlc_interval and its profile:
        >>> df_intervals[0] # [1], [2], etc...
        >>> df_profiles[0] # [1], [2], etc...

        >>> # change parameters for filters
        >>> from models_utils.profile_models import ProfileFilter, ProfileSmooth, ProfileNode
        >>> params_nodes = ProfileFilter(ProfileSmooth.Gaussian, ProfileNode.LocalMinMax, strong_only=False, ...)
        >>> params_nodes.levels(61.8, 23.6) # set percentages for Symmetric Bands

        >>> vp = VolumeProfile(df_ohlcv, None, row_height, pd.Timedelta(hours=4), DistributionData.OHLC_No_Avg, params_nodes, ...)
        Parameters
        ----------
        df_ohlcv : dataframe
            * index/datetime, open, high, low, close, volume
            * "datetime": If is not present, the index will be used.
            * "volume": If df_ticks is used, this column will be ignored.
        df_ticks : dataframe
            * If it's not None, ticks data will be used for Volume Profile, it should have:
            * datetime index or 'datetime' column
            * 'close' column (tick price)
        row_height : float
            Cannot be less than or equal to 0.00000...
        interval : pd.Timedelta
            Interval for each profile, can be Minutes, Hours, Days, Weekly...
        distribution : DistributionData
            Distribution method for df_ohlcv bars volume
        profile_filter : ProfileFilter
            Parameters for HVN/LVN Detection + Levels
        segments_interval : SegmentsInterval
            Interval to calculate the price-segments that will be shared among all profiles.

            The df_ohlcv should provide, at least, one starting point for Daily/Weekly/Monthly interval, example:

            * 1 Day => Daily
            * 1 Monday => Weekly
            * First Weekday (monday) => Monthly
            * If SegmentsInterval.FromProfile, each profile will have its own segments calculated by its interval
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

        if with_plotly_columns:
            df_ohlcv['plotly_int_index'] = range(len(df_ohlcv))

        self._df_ohlcv = df_ohlcv
        self._df_ohlcv_index = df_ohlcv.index.to_numpy()
        self._df_ticks = df_ticks
        self._row_height = row_height
        self._distribution = distribution
        self._profile_filter = profile_filter if isinstance(profile_filter, ProfileFilter) else ProfileFilter()
        self._segments_interval = segments_interval
        if segments_interval != SegmentsInterval.From_Profile:
            self._shared_segments = create_shared_segments(df_ohlcv, row_height, segments_interval)
        self._with_plotly_columns = with_plotly_columns

        def parallel_process_profiles(list_of_dfs: list):
            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                results = pool.map(self._create_vp, list_of_dfs)
            return results

        df_list = get_intervals_list(df_ohlcv, interval)
        self._interval_dfs = df_list
        self._interval_profiles = parallel_process_profiles(df_list)

    def _parallel_process_extra(self, list_of_dfs: list, extra_profile: ExtraProfile):
        num_processes = cpu_count()
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(self._create_vp, zip(list_of_dfs, itertools.repeat(extra_profile)))
        return results

    def normal(self, extra_profile = ExtraProfile.No, extra_only: bool = False,
               mini_interval: pd.Timedelta = pd.Timedelta(hours=2), fixed_dates: list = ()):
        """
        Return all intervals dataframes of ohlcv + normal mode

        >>> from volume_profile import VolumeProfile
        >>> vp = VolumeProfile(...)

        >>> df_intervals, df_profiles = vp.normal()

        >>> # to access each ohlcv_interval and its profile:
        >>> df_intervals[0] # [1], [2], etc...
        >>> df_profiles[0] # [1], [2], etc...

        >>> # get extra-profile with main profile (above)
        >>> df_intervals, df_profiles, df_extra_intervals, df_extra_profiles = vp.normal(ExtraProfile.Mini..., extra_only=False)

        >>> # get mini-profiles (only)
        >>> vp.normal(ExtraProfile.Mini, True, mini_interval=pd.Timedelta(hours=4))

        >>> # get weekly/monthly (only)
        >>> vp.normal(ExtraProfile.Weekly, True) # or ExtraProfile.Monthly

        >>> # get fixed-range profiles (only)
        >>> dates = [ ('2025-05-15 01:40:00', '2025-05-15 16:41:00'), ('start_date', 'end_date'), ...]
        >>> vp.normal(ExtraProfile.Fixed, True, fixed_dates=dates)
        """

        normal_dfs = [n[0] for n in self._interval_profiles]
        if extra_profile == ExtraProfile.No:
            return self._interval_dfs, normal_dfs
        else:
            match extra_profile:
                case ExtraProfile.Mini:
                    extra_intervals, extra_profiles = self._mini(mini_interval)
                case ExtraProfile.Weekly:
                    extra_intervals, extra_profiles = self._weekly()
                case ExtraProfile.Monthly:
                    extra_intervals, extra_profiles = self._monthly()
                case _:
                    extra_intervals, extra_profiles = self._fixed(fixed_dates)

            extra_profiles = [extra_profiles[i][0] for i in range(len(extra_profiles))]

            return (extra_intervals, extra_profiles) if extra_only else \
                   (self._interval_dfs, normal_dfs, extra_intervals, extra_profiles)

    def buy_sell(self, extra_profile = ExtraProfile.No, extra_only: bool = False,
                 mini_interval: pd.Timedelta = pd.Timedelta(hours=2), fixed_dates: list = ()):
        """
        Return all intervals dataframes of ohlcv + buy_sell mode

        >>> from volume_profile import VolumeProfile
        >>> vp = VolumeProfile(...)

        >>> df_intervals, df_profiles = vp.buy_sell()
        >>> # to access each ohlcv_interval and its profile:
        >>> df_intervals[0] # [1], [2], etc...
        >>> df_profiles[0] # [1], [2], etc...

        >>> # same usage from vp.normal() for extra-profiles
        """

        buy_sell_dfs = [bs[1] for bs in self._interval_profiles]
        if extra_profile == ExtraProfile.No:
            return self._interval_dfs, buy_sell_dfs
        else:
            match extra_profile:
                case ExtraProfile.Mini:
                    extra_intervals, extra_profiles = self._mini(mini_interval)
                case ExtraProfile.Weekly:
                    extra_intervals, extra_profiles = self._weekly()
                case ExtraProfile.Monthly:
                    extra_intervals, extra_profiles = self._monthly()
                case _:
                    extra_intervals, extra_profiles = self._fixed(fixed_dates)

            extra_profiles = [extra_profiles[i][1] for i in range(len(extra_profiles))]

            return (extra_intervals, extra_profiles) if extra_only else \
                (self._interval_dfs, buy_sell_dfs, extra_intervals, extra_profiles)

    def delta(self, extra_profile = ExtraProfile.No, extra_only: bool = False,
              mini_interval: pd.Timedelta = pd.Timedelta(hours=2), fixed_dates: list = ()):
        """
        Return all intervals dataframes of ohlcv + delta mode

        >>> from volume_profile import VolumeProfile
        >>> vp = VolumeProfile(...)

        >>> df_intervals, df_profiles = vp.delta()
        >>> # to access each ohlcv_interval and its profile:
        >>> df_intervals[0] # [1], [2], etc...
        >>> df_profiles[0] # [1], [2], etc...

        >>> # same usage from vp.normal() for extra-profiles
        """

        delta_dfs = [d[2] for d in self._interval_profiles]
        if extra_profile == ExtraProfile.No:
            return self._interval_dfs, delta_dfs
        else:
            match extra_profile:
                case ExtraProfile.Mini:
                    extra_intervals, extra_profiles = self._mini(mini_interval)
                case ExtraProfile.Weekly:
                    extra_intervals, extra_profiles = self._weekly()
                case ExtraProfile.Monthly:
                    extra_intervals, extra_profiles = self._monthly()
                case _:
                    extra_intervals, extra_profiles = self._fixed(fixed_dates)

            extra_profiles = [extra_profiles[i][2] for i in range(len(extra_profiles))]

            return (extra_intervals, extra_profiles) if extra_only else \
                (self._interval_dfs, delta_dfs, extra_intervals, extra_profiles)

    def all(self, extra_profile = ExtraProfile.No, extra_only: bool = False,
            mini_interval: pd.Timedelta = pd.Timedelta(hours=2), fixed_dates: list = ()):
        """
        Return all intervals dataframes of ohlc + all modes

        >>> from volume_profile import VolumeProfile
        >>> vp = VolumeProfile(...)

        >>> df_intervals, df_profiles = vp.all()
        >>> # to access each ohlc_interval and its profile mode:
        >>> df_intervals[0]
        >>> df_profiles[0][0] # [0]: normal
        >>> df_profiles[0][1] # [1]: buy_sell
        >>> df_profiles[0][2] # [2]: delta

        >>> # get extra-profile with main profile (above)
        >>> df_intervals, df_profiles, df_extra_intervals, df_extra_profiles = vp.all(ExtraProfile.Mini..., extra_only=False)

        >>> # get extra-profiles (only)
        >>> df_extra_intervals, df_extra_profiles = vp.all(ExtraProfile.Mini, True, mini_interval=pd.Timedelta(hours=4))
        >>> # etc..
        """
        if extra_profile == ExtraProfile.No:
            return self._interval_dfs, self._interval_profiles
        else:
            match extra_profile:
                case ExtraProfile.Mini:
                    extra_intervals, extra_profiles = self._mini(mini_interval)
                case ExtraProfile.Weekly:
                    extra_intervals, extra_profiles = self._weekly()
                case ExtraProfile.Monthly:
                    extra_intervals, extra_profiles = self._monthly()
                case _:
                    extra_intervals, extra_profiles = self._fixed(fixed_dates)

            return (extra_intervals, extra_profiles) if extra_only else \
                (self._interval_dfs, self._interval_profiles, extra_intervals, extra_profiles)

    def _mini(self, mini_interval: pd.Timedelta = pd.Timedelta(hours=4)):
        df_intervals = get_intervals_list(self._df_ohlcv, mini_interval)
        df_profiles = self._parallel_process_extra(df_intervals, ExtraProfile.Mini)
        return df_intervals, df_profiles

    def _weekly(self):
        if self._segments_interval == SegmentsInterval.Daily:
            raise ValueError(f"segments_interval should be >= Weekly, Monthly or From_Profile")

        df_intervals = get_intervals_list(self._df_ohlcv, pd.Timedelta(weeks=1))
        df_profiles = self._parallel_process_extra(df_intervals, ExtraProfile.Weekly)
        return df_intervals, df_profiles

    def _monthly(self):
        if  self._segments_interval in [SegmentsInterval.Daily, SegmentsInterval.Weekly]:
            raise ValueError(f"segments_interval should be >= Monthly or From_Profile")
        df_intervals = get_intervals_list(self._df_ohlcv, pd.DateOffset(months=1))
        df_profiles = self._parallel_process_extra(df_intervals, ExtraProfile.Monthly)
        return df_intervals, df_profiles

    def _fixed(self, fixed_dates: list):
        df = self._df_ohlcv

        df_intervals = []
        for start_date, end_date in fixed_dates:
            interval_df = df.loc[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
            df_intervals.append(interval_df)
        df_profiles = self._parallel_process_extra(df_intervals, ExtraProfile.Fixed)

        return df_intervals, df_profiles

    def plot(self, mode: str = 'delta', extra_profile: ExtraProfile = ExtraProfile.No, extra_only: bool = False,
             mini_interval: pd.Timedelta = pd.Timedelta(hours=2), fixed_dates: list = ()):
        """
        Plot all intervals of Volume Profiles with 'mplfinance'.

        >>> from volume_profile import VolumeProfile
        >>> vp = VolumeProfile(...)

        >>> # plot the main interval => tpo.profiles()
        >>> vp.plot('normal')

        >>> # plot mini-profiles
        >>> vp.plot('delta', ExtraProfile.Mini, mini_interval=pd.Timedelta(hours=4))

        >>> # plot weekly/monthly
        >>> vp.plot('normal', ExtraProfile.Weekly) # or ExtraProfile.Monthly

        >>> # plot fixed-range profiles
        >>> dates = [ ('2025-05-15 01:40:00', '2025-05-15 16:41:00'), ('start_date', 'end_date'), ...]
        >>> vp.plot('delta', ExtraProfile.Fixed, fixed_dates=dates)

        >>> # plot only the respective extra-profile
        >>> vp.plot(...extra_only=True)
        """
        _profiles = ['normal', 'buy_sell', 'delta']
        if mode not in _profiles:
            raise ValueError(f"Only {_profiles} options are valid.")

        def parallel_process_profiles():
            mode_profile = self.normal()[1] if mode == 'normal' else \
                           self.buy_sell()[1] if mode == 'buy_sell' else \
                           self.delta()[1]

            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                results = pool.starmap(self._mpf_workaround, zip(self._interval_dfs, mode_profile))
            return results

        def parallel_process_extra():
            match extra_profile:
                case ExtraProfile.Mini:
                    extra_intervals, extra_profiles = \
                        self.normal(ExtraProfile.Mini, True, mini_interval) if mode == 'normal' else \
                        self.buy_sell(ExtraProfile.Mini, True, mini_interval) if mode == 'buy_sell' else \
                        self.delta(ExtraProfile.Mini, True, mini_interval)
                case ExtraProfile.Weekly:
                    extra_intervals, extra_profiles = \
                        self.normal(ExtraProfile.Weekly, True) if mode == 'normal' else \
                        self.buy_sell(ExtraProfile.Weekly, True) if mode == 'buy_sell' else \
                        self.delta(ExtraProfile.Weekly, True)
                case ExtraProfile.Monthly:
                    extra_intervals, extra_profiles = \
                        self.normal(ExtraProfile.Monthly, True) if mode == 'normal' else \
                        self.buy_sell(ExtraProfile.Monthly, True) if mode == 'buy_sell' else \
                        self.delta(ExtraProfile.Monthly, True)
                case _:
                    extra_intervals, extra_profiles = \
                        self.normal(ExtraProfile.Fixed, True, fixed_dates= fixed_dates) if mode == 'normal' else \
                        self.buy_sell(ExtraProfile.Fixed, True, fixed_dates= fixed_dates) if mode == 'buy_sell' else \
                        self.delta(ExtraProfile.Fixed, True, fixed_dates= fixed_dates)

            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                extra_results = pool.starmap(self._mpf_workaround, zip(extra_intervals, extra_profiles, itertools.repeat(extra_profile)))
            return extra_results

        # join all interval profiles to get original df_ohlcv with vp_scatter
        df_list = parallel_process_profiles()
        df_ohlcv_vp = pd.concat(df_list, ignore_index=True)
        df_ohlcv_vp.index = df_ohlcv_vp['datetime']

        if extra_profile != ExtraProfile.No:
            # Get extra-profile scatters
            extra_list = parallel_process_extra()
            extra_df = pd.concat(extra_list, ignore_index=True)
            extra_df.index = extra_df['datetime']

            if extra_profile == ExtraProfile.Fixed:
                df_start = self._df_ohlcv.loc[self._df_ohlcv['datetime'] < fixed_dates[0][0]]
                df_end = self._df_ohlcv.loc[self._df_ohlcv['datetime'] > fixed_dates[-1][1]]
                # retrieve the remaining df_ohlc rows
                df_to_join = [df_start, extra_df, df_end]
                extra_df = pd.concat(df_to_join, ignore_index=True)
                extra_df.index = extra_df['datetime']

            # drop same columns before join()
            extra_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'datetime'], inplace=True)
            if self._with_plotly_columns:
                extra_df.drop(columns=['plotly_int_index'], inplace=True)

            df_ohlcv_vp = df_ohlcv_vp.join(extra_df, how='outer')

        # total vp_scatter columns
        numbers = df_ohlcv_vp.columns.str.extract(r'(\d+)', expand=False)
        total_add = int(numbers[len(numbers) - 1])

        # remove columns with all NaN values
        df_ohlcv_vp.dropna(axis=1, how='all', inplace=True)

        prefix = get_prefix(extra_profile, True)
        apd = []
        if mode == 'normal':
            apd = [mpf.make_addplot(df_ohlcv_vp[f'vp_scatter_{i}'], color='deepskyblue', alpha=0.5) \
                   for i in range(total_add) if f'vp_scatter_{i}' in df_ohlcv_vp.columns]

            if extra_only and extra_profile != ExtraProfile.No:
                apd = []

            apd += [mpf.make_addplot(df_ohlcv_vp[f'{prefix}_scatter_{i}'], color='orange', alpha=0.5) \
                   for i in range(total_add) if f'{prefix}_scatter_{i}' in df_ohlcv_vp.columns]
        else:
            apd += [mpf.make_addplot(df_ohlcv_vp[f'vp_scatter_negative_{i}'], color='r', alpha=0.5) \
                   for i in range(total_add) if f'vp_scatter_negative_{i}' in df_ohlcv_vp.columns]

            apd += [mpf.make_addplot(df_ohlcv_vp[f'vp_scatter_positive_{i}'], color='g', alpha=0.5 if mode != 'buy_sell' else 0.9) \
                   for i in range(total_add) if f'vp_scatter_positive_{i}' in df_ohlcv_vp.columns]

            if extra_only and extra_profile != ExtraProfile.No:
                apd = []

            apd += [mpf.make_addplot(df_ohlcv_vp[f'{prefix}_scatter_negative_{i}'], color='orange', alpha=0.7 if mode != 'buy_sell' else 0.5) \
                   for i in range(total_add) if f'{prefix}_scatter_negative_{i}' in df_ohlcv_vp.columns]

            apd += [mpf.make_addplot(df_ohlcv_vp[f'{prefix}_scatter_positive_{i}'], color='deepskyblue', alpha=0.7 if mode != 'buy_sell' else 0.9) \
                   for i in range(total_add) if f'{prefix}_scatter_positive_{i}' in df_ohlcv_vp.columns]

        plot_kwargs = {}
        if extra_profile == ExtraProfile.Fixed:
            list_value = [item for sublist in fixed_dates for item in sublist]
            value = dict(vlines=list_value, linestyle='dotted')
            plot_kwargs = {'vlines': value}

        # type='scatter' is very heavy for massive data
        # The default add_plot 'line' type is pretty optimized
        s = mpf.make_mpf_style(base_mpl_style='seaborn-v0_8', rc={'axes.grid': False})
        mpf.plot(df_ohlcv_vp, type='candle', style=s, addplot=apd, warn_too_much_data=len(df_ohlcv_vp) + 1,
                 title=f"Volume Profile \n {mode} \n",
                 figsize=(12.5,6),
                 returnfig=True,
                 scale_padding=0.2, **plot_kwargs)

        mpf.show()

    def plot_ly(self, mode: str = 'delta',
                extra_profile: ExtraProfile = ExtraProfile.No, extra_only: bool = False,
                mini_interval: pd.Timedelta = pd.Timedelta(hours=2), fixed_dates: list = (),
                show_numbers: bool = False,
                nodes: bool = True, nodes_source: str ='hvn', nodes_levels: bool = True,
                chart: str = 'candle', renderer: str = 'default', width: int = 1200, height: int = 800):
        """
        Plot all intervals of Volume Profiles with 'plotly'.

        >>> from volume_profile import VolumeProfile
        >>> vp = VolumeProfile(...)

        >>> # plot the main interval => tpo.profiles()
        >>> vp.plot_ly('normal')

        >>> # plot mini-profiles
        >>> vp.plot_ly('delta', ExtraProfile.Mini, mini_interval=pd.Timedelta(hours=4))

        >>> # plot weekly/monthly
        >>> vp.plot_ly('normal', ExtraProfile.Weekly) # or ExtraProfile.Monthly

        >>> # plot fixed-range profiles
        >>> dates = [ ('2025-05-15 01:40:00', '2025-05-15 16:41:00'), ('start_date', 'end_date'), ...]
        >>> vp.plot_ly('delta', ExtraProfile.Fixed, fixed_dates=dates)

        >>> # plot HVN/LVN for ['normal', 'delta] modes only
        >>> vp.plot_ly(...nodes=True, nodes_source='hvn', nodes_levels=True)

        >>> # plot only the respective extra-profile
        >>> vp.plot_ly(...extra_only=True)
        """
        _profiles = ['normal', 'buy_sell', 'delta']
        _charts = ['candle', 'ohlc']
        _node_sources = ['hvn', 'lvn', 'hvn_raw', 'lvn_raw']

        input_values = [mode, chart, nodes_source]
        input_validation = [_profiles, _charts, _node_sources]
        for value, validation in zip(input_values, input_validation):
            if value not in validation:
                raise ValueError(f"Only {validation} options are valid.")

        df = self._df_ohlcv.copy()

        prefix = 'plotly'
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.0)

        trace_chart = go.Ohlc(x=df[f'{prefix}_int_index'],
                              open=df['open'],
                              high=df['high'],
                              low=df['low'],
                              close=df['close'], opacity=0.5) if chart == 'ohlc' else \
                      go.Candlestick(x=df[f'{prefix}_int_index'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'], opacity=0.4)
        fig.add_trace(trace_chart, row=1, col=1)

        # Some beautiful colors from plotply - colorscales wiki
        # But everything will be transparent :D
        color = [[0.0, "rgba(165,0,38, 0.0)"],
                 [0.1, "rgba(215,48,39, 0.0)"],
                 [0.2, "rgba(244,109,67, 0.0)"],
                 [0.3, "rgba(253,174,97, 0.0)"],
                 [0.4, "rgba(254,224,144, 0.0)"],
                 [0.5, "rgba(224,243,248, 0.0)"],
                 [0.6, "rgba(171,217,233, 0.0)"],
                 [0.7, "rgba(116,173,209, 0.0)"],
                 [0.8, "rgba(69,117,180, 0.0)"],
                 [1.0, "rgba(49,54,149, 0.0)"]]

        def parallel_process_dataframes():
            mode_profile = self.normal()[1] if mode == 'normal' else \
                           self.buy_sell()[1] if mode == 'buy_sell' else \
                           self.delta()[1]

            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                results = pool.starmap(self._plotly_workaround, zip(self._interval_dfs, mode_profile))
            return results

        def parallel_process_extra():
            extra_intervals, extra_profiles = \
                self.normal(extra_profile, True, mini_interval=mini_interval, fixed_dates=fixed_dates) \
                    if mode == 'normal' else \
                self.buy_sell(extra_profile, True, mini_interval=mini_interval, fixed_dates=fixed_dates) \
                    if mode == 'buy_sell' else \
                self.delta(extra_profile, True, mini_interval=mini_interval, fixed_dates=fixed_dates)

            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                extra_results = pool.starmap(self._plotly_workaround, zip(extra_intervals, extra_profiles, itertools.repeat(extra_profile)))
            return extra_results

        to_plot = parallel_process_dataframes()
        to_plot = pd.concat(to_plot, ignore_index=True)

        extra_to_plot = []
        if extra_profile != ExtraProfile.No:
            extra_to_plot = parallel_process_extra()
            extra_to_plot = pd.concat(extra_to_plot, ignore_index=True)

        extra_prefix = get_prefix(extra_profile, True)
        _names = ['vp'] if extra_profile == ExtraProfile.No else \
                 [extra_prefix] if extra_only else ['vp', extra_prefix]

        for name in _names:
            vp_prefix = f'{prefix}_{name}_{mode}'
            if name != 'vp':
                to_plot = extra_to_plot

            prices = to_plot[f'{vp_prefix}_prices'].to_numpy()
            bases = to_plot[f'{vp_prefix}_base_index'].to_numpy()
            bases_end = to_plot[f'{vp_prefix}_end_index'].to_numpy()
            plot_vp_values = to_plot[f'{vp_prefix}_values'].to_numpy()
            plot_buy_values = to_plot[f'{vp_prefix}_buy_values'].to_numpy()
            plot_sell_values = to_plot[f'{vp_prefix}_sell_values'].to_numpy()
            vp_original_values = to_plot[f'{vp_prefix}_original_values'].to_numpy()
            vp_original_buy_values = to_plot[f'{vp_prefix}_original_buy_values'].to_numpy()
            vp_original_sell_values = to_plot[f'{vp_prefix}_original_sell_values'].to_numpy()
            volume_nodes_colors = to_plot[f'{vp_prefix}_{nodes_source}_colors'].to_numpy()
            volume_nodes_levels = to_plot[f'{vp_prefix}_{nodes_source}_lvls'].to_numpy()

            for idx in range(len(to_plot)):
                y_column = prices[idx]
                x_column = plot_vp_values[idx]
                base_index = bases[idx]
                original_values = vp_original_values[idx]
                coloring = '#00BFFF'

                if mode == 'normal':
                    if nodes:
                        coloring = volume_nodes_colors[idx]
                    fig.add_trace(
                        go.Bar(y=y_column,
                               x=x_column,
                               orientation='h',
                               marker=dict(
                                   color=coloring,
                                   opacity=0.7
                               ), base=base_index), row=1, col=1)
                else:
                    if mode == 'delta' and nodes:
                        if nodes:
                            coloring = volume_nodes_colors[idx]
                        fig.add_trace(
                            go.Bar(y=y_column,
                                   x=x_column, # same as 'normal' mode
                                   orientation='h',
                                   marker=dict(
                                       color=coloring,
                                       opacity=0.7
                                   ), base=base_index), row=1, col=1)
                    else:
                        x_column = plot_buy_values[idx]
                        fig.add_trace(
                            go.Bar(y=y_column,
                                   x=x_column,
                                   orientation='h',
                                   marker=dict(
                                       color='deepskyblue',
                                       opacity=0.6
                                   ), base=base_index), row=1, col=1)

                        x_column = plot_sell_values[idx]
                        fig.add_trace(
                            go.Bar(y=y_column,
                                   x=x_column,
                                   orientation='h',
                                   marker=dict(
                                       color='crimson',
                                       opacity=0.6
                                   ), base=base_index), row=1, col=1)

                if mode in ['normal', 'delta'] and nodes_levels:
                    x_axis = bases_end[idx] - base_index
                    _levels = volume_nodes_levels[idx]
                    for price in _levels:
                        fig.add_trace(
                            go.Bar(y=[price],
                                   x=[x_axis],
                                   orientation='h',
                                   marker=dict(
                                       color='red',
                                       opacity=0.8
                                   ), base=base_index), row=1, col=1)

                if mode in ['normal', 'delta'] and show_numbers:
                    fig.add_trace(
                        go.Heatmap(
                            x=[base_index - 0.5] * len(x_column),
                            y=y_column,
                            z=original_values,
                            text=original_values,
                            colorscale=color,
                            showscale=False,  # remove numbers from show_legend=False column
                            texttemplate="%{text}",
                            textfont={
                                "size": 11,
                                "color": 'black',
                                "family": "Courier New"},
                        ), row=1, col=1)
                elif mode == 'buy_sell' and show_numbers:
                    original_values = vp_original_buy_values[idx]
                    fig.add_trace(
                        go.Heatmap(
                            x=[base_index - 0.5] * len(x_column),
                            y=y_column,
                            z=original_values,
                            text=original_values,
                            colorscale=color,
                            showscale=False,  # remove numbers from show_legend=False column
                            texttemplate="%{text}",
                            textfont={
                                "size": 11,
                                "color": 'black',
                                "family": "Courier New"},
                        ), row=1, col=1)

                    original_values = vp_original_sell_values[idx]
                    fig.add_trace(
                        go.Heatmap(
                            x=[base_index + 0.5] * len(x_column),
                            y=y_column,
                            z=original_values,
                            text=original_values,
                            colorscale=color,
                            showscale=False,  # remove numbers from show_legend=False column
                            texttemplate="%{text}",
                            textfont={
                                "size": 11,
                                "color": 'black',
                                "family": "Courier New"},
                        ), row=1, col=1)

        fig.update_layout(
            title=f"Volume Profile: {mode}",
            height=800,
            # IMPORTANT!
            # Allows bars(histograms) to diverge from the center (0)
            # from plotply horizontal bars chart wiki
            barmode='relative', # or overlay
            xaxis_rangeslider_visible=False
        )
        fig.update_traces(
            showlegend=False
        )
        if renderer != 'default':
            if renderer in ['svg', 'png', 'jpeg']:
                fig.show(renderer=renderer, width=width, height=height)
            else:
                fig.show(renderer=renderer)
        else:
            fig.show()

    def _mpf_workaround(self, df_interval: pd.DataFrame, df_profile: pd.DataFrame,
                        extra_profile: ExtraProfile = ExtraProfile.No):
        """
        Like in C# version a rule of three is used to plot the histograms,
        but instead of datetime(ms) the max_index of each interval is used.
        From there the math adjusts the histograms.
            max_volume    max_index(int)
               x             ?(int)
        """
        from warnings import simplefilter
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

        prefix = get_prefix(extra_profile, True)
        normal_name, delta_name = f'{prefix}_normal', f'{prefix}_delta'

        max_index = len(df_interval)
        max_volume = 0
        column_name = ''
        if normal_name in df_profile.columns:
            max_volume = df_profile[normal_name].max()
            column_name = normal_name
        if delta_name in df_profile.columns:
            max_volume = df_profile[delta_name].abs().max()
            column_name = delta_name
        if f'{prefix}_buy' in df_profile.columns:
            max_volume = df_profile[f'{prefix}_sell'].max()
            column_name = f'{prefix}_buy'

        vp_prices = df_profile[f'{prefix}_prices'].to_numpy() # speeed
        column_array = df_profile[column_name].to_numpy()  # speeed
        calculate_len = len(column_array)

        if column_name == normal_name:
            for i in range(calculate_len):
                first = column_array[i] * max_index
                result = math.ceil(first / max_volume)

                scatter_name = f'{prefix}_scatter_{i}'
                df_interval[scatter_name] = np.NaN
                scatter_arr = df_interval[scatter_name].to_numpy()  # speeed

                for bar_index in range(result):
                    if bar_index >= max_index:
                        break
                    scatter_arr[bar_index] = vp_prices[i]

                df_interval[scatter_name] = scatter_arr

        elif column_name == delta_name:
            for i in range(calculate_len):
                value = column_array[i]
                first = abs(value) * max_index
                result = math.ceil(first / max_volume)

                scatter_positive_name = f'{prefix}_scatter_positive_{i}'
                scatter_negative_name = f'{prefix}_scatter_negative_{i}'

                df_interval[scatter_positive_name] = np.NaN
                df_interval[scatter_negative_name] = np.NaN
                scatter_positive_arr = df_interval[scatter_positive_name].to_numpy()
                scatter_negative_arr = df_interval[scatter_negative_name].to_numpy()

                for price_index in range(result):
                    if price_index >= max_index:
                        break
                    if value > 0:
                        scatter_positive_arr[price_index] = vp_prices[i]
                    else:
                        scatter_negative_arr[price_index] = vp_prices[i]

                df_interval[scatter_positive_name] = scatter_positive_arr
                df_interval[scatter_negative_name] = scatter_negative_arr
        else:
            for i in range(calculate_len):
                value = column_array[i]
                first = value * math.ceil(max_index / 2)
                result = math.ceil(first / max_volume)

                scatter_positive_name = f'{prefix}_scatter_positive_{i}'
                df_interval[scatter_positive_name] = np.NaN
                scatter_positive_arr = df_interval[scatter_positive_name].to_numpy()

                for price_index in range(result):
                    if price_index >= max_index:
                        break
                    scatter_positive_arr[price_index] = vp_prices[i]

                df_interval[scatter_positive_name] = scatter_positive_arr

            column_array = df_profile[f'{prefix}_sell'].to_numpy()  # speeed
            calculate_len = len(column_array)
            for i in range(calculate_len):
                value = column_array[i]
                first = value * max_index
                result = math.ceil(first / max_volume)

                scatter_negative_name = f'{prefix}_scatter_negative_{i}'
                df_interval[scatter_negative_name] = np.NaN
                scatter_negative_arr = df_interval[scatter_negative_name].to_numpy()

                for price_index in range(result):
                    if price_index >= max_index:
                        break
                    scatter_negative_arr[price_index] = vp_prices[i]

                df_interval[scatter_negative_name] = scatter_negative_arr

        return df_interval

    def _plotly_workaround(self, df_interval: pd.DataFrame, df_profile: pd.DataFrame,
                           extra_profile: ExtraProfile = ExtraProfile.No):
        """
        Same logic of _mpf_workaround.
        """
        prefix = get_prefix(extra_profile, True)
        normal_name, delta_name, buy_sell_name = f'{prefix}_normal', f'{prefix}_delta', f'{prefix}_buy_sell'

        max_index = len(df_interval)
        max_volume = 0
        column_name = ''
        if normal_name in df_profile.columns:
            max_volume = df_profile[normal_name].max()
            column_name = normal_name
        if delta_name in df_profile.columns:
            max_volume = df_profile[delta_name].abs().max()
            column_name = delta_name
        if f'{prefix}_buy' in df_profile.columns:
            max_volume = df_profile[f'{prefix}_sell'].max()
            column_name = buy_sell_name

        vp_prices = df_profile[f'{prefix}_prices'].to_numpy() # speeed
        column_array = df_profile[column_name].to_numpy()  if column_name != buy_sell_name else np.empty(1) # speeed
        calculate_len = len(column_array)

        index_array = self._df_ohlcv_index
        base_idx = np.where(index_array == df_profile[f'{prefix}_datetime'].iat[0])[0]
        end_idx = np.where(index_array == df_interval[f'datetime'].iat[-1])[0]
        # with pandas index (self._df_ohlc_index is not an array)
        # base_idx = self._df_ohlcv_index.get_loc(df_profile[f'{prefix}_datetime'].iat[0])
        # end_idx = self._df_ohlcv_index.get_loc(df_interval[f'datetime'].iat[-1])

        vp_prefix = f'plotly_{column_name}'
        vp_proportioned = {
            f"{vp_prefix}_base_index": [base_idx[0]],
            f"{vp_prefix}_end_index": [end_idx[0]],
            f"{vp_prefix}_prices": [vp_prices],
            f"{vp_prefix}_hvn_colors": [np.empty(1)],
            f"{vp_prefix}_hvn_lvls": [np.empty(1)],
            f"{vp_prefix}_hvn_raw_colors": [np.empty(1)],
            f"{vp_prefix}_hvn_raw_lvls": [np.empty(1)],
            f"{vp_prefix}_lvn_colors": [np.empty(1)],
            f"{vp_prefix}_lvn_lvls": [np.empty(1)],
            f"{vp_prefix}_lvn_raw_colors": [np.empty(1)],
            f"{vp_prefix}_lvn_raw_lvls": [np.empty(1)],
            f"{vp_prefix}_values": [np.empty(calculate_len)],
            f"{vp_prefix}_buy_values": [np.empty(calculate_len)],
            f"{vp_prefix}_sell_values": [np.empty(calculate_len)],
            f"{vp_prefix}_original_values": [column_array],
            f"{vp_prefix}_original_buy_values": [df_profile[f'{prefix}_buy'].to_numpy() \
                                                if column_name == buy_sell_name else np.empty(1)],
            f"{vp_prefix}_original_sell_values": [df_profile[f'{prefix}_sell'].to_numpy() \
                                                 if column_name == buy_sell_name else np.empty(1)]
        }

        if column_name == normal_name:
            for i in range(calculate_len):
                first = column_array[i] * math.ceil(max_index / 2)
                result = math.ceil(first / max_volume)

                vp_proportioned[f'{vp_prefix}_values'][0][i] = result
        elif column_name == delta_name:
            for i in range(calculate_len):
                value = column_array[i]
                first = abs(value) * math.ceil(max_index / 2)
                result = math.ceil(first / max_volume)

                if value > 0:
                    vp_proportioned[f'{vp_prefix}_buy_values'][0][i] = result
                    vp_proportioned[f'{vp_prefix}_sell_values'][0][i] = 0
                else:
                    vp_proportioned[f'{vp_prefix}_sell_values'][0][i] = result
                    vp_proportioned[f'{vp_prefix}_buy_values'][0][i] = 0

                # to used by hvn/lvn
                vp_proportioned[f'{vp_prefix}_values'][0][i] = result
        else:
            column_array = df_profile[f'{prefix}_buy'].to_numpy()
            calculate_len = len(column_array)
            vp_proportioned[f"{vp_prefix}_buy_values"] = [np.empty(calculate_len)]
            for i in range(calculate_len):
                value = column_array[i]
                first = abs(value) * math.ceil(max_index / 4)
                result = math.ceil(first / max_volume)

                vp_proportioned[f'{vp_prefix}_buy_values'][0][i] = result

            column_array = df_profile[f'{prefix}_sell'].to_numpy()
            calculate_len = len(column_array)
            vp_proportioned[f"{vp_prefix}_sell_values"] = [np.empty(calculate_len)]
            for i in range(calculate_len):
                value = column_array[i]
                first = abs(value) * math.ceil(max_index / 2)
                result = math.ceil(first / max_volume)

                vp_proportioned[f'{vp_prefix}_sell_values'][0][i] = result

        if column_name in [normal_name, delta_name]:
            name = normal_name if column_name == normal_name else delta_name

            # hvn + bands
            d = df_profile[f'{name}_hvn_mask']
            colors = np.where(d == 1, 'aqua',
                     np.where(d == 2, 'gold',
                     np.where(d == 3, 'aqua', '#00BFFF')))
            vp_proportioned[f"{vp_prefix}_hvn_colors"][0] = colors

            # levels (with bands)
            d = df_profile[f'{name}_hvn_levels'].iat[0]
            hvn_mask = [idx for tpl in d for idx in tpl]
            vp_proportioned[f"{vp_prefix}_hvn_lvls"][0] = np.array(hvn_mask)

            # raw hvn
            d = df_profile[f'{name}_hvn_raw_mask']
            colors = np.where(d == 1, 'gold', '#00BFFF')
            vp_proportioned[f"{vp_prefix}_hvn_raw_colors"][0] = colors
            # raw levels
            vp_proportioned[f"{vp_prefix}_hvn_raw_lvls"][0] = df_profile[f'{name}_hvn_raw_levels'].iat[0]

            # lvn + bands
            d = df_profile[f'{name}_lvn_mask']
            colors = np.where(d == 1, 'blue',
                     np.where(d == 2, 'red',
                     np.where(d == 3, 'blue', '#00BFFF')))
            vp_proportioned[f"{vp_prefix}_lvn_colors"][0] = colors

            # levels (with bands)
            d = df_profile[f'{name}_lvn_levels'].iat[0]
            lvn_mask = [idx for tpl in d for idx in tpl]
            vp_proportioned[f"{vp_prefix}_lvn_lvls"][0] = np.array(lvn_mask)

            # raw lvn
            d = df_profile[f'{name}_lvn_raw_mask']
            colors = np.where(d == 1, 'red', '#00BFFF')
            vp_proportioned[f"{vp_prefix}_lvn_raw_colors"][0] = colors
            # raw levels
            vp_proportioned[f"{vp_prefix}_lvn_raw_lvls"][0] = df_profile[f'{name}_lvn_raw_levels'].iat[0]

        return pd.DataFrame(vp_proportioned)

    def _create_vp(self, df_interval: pd.DataFrame, extra_profile: ExtraProfile = ExtraProfile.No):
        interval_date = df_interval['datetime'].iat[0]
        interval_open = df_interval['open'].iat[0]
        interval_highest = df_interval['high'].max()
        interval_lowest = df_interval['low'].min()

        interval_segments = get_segments(interval_date, interval_open, interval_highest, interval_lowest,
                                         self._row_height, self._segments_interval, self._shared_segments)
        len_segments = len(interval_segments)

        vp_datetime = np.full(len_segments, interval_date, dtype=np.ndarray)
        vp_prices = np.array(interval_segments)
        normal_profile, buy_profile, sell_profile, delta_profile = \
            (deepcopy(np.zeros(len_segments, dtype=np.int64)) for _ in range(4))
        # array because of _add_volume function
        min_delta, max_delta = np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)

        def _add_volume(index: int, volume_i: float, is_up_i: bool):
            normal_profile[index] += volume_i

            if is_up_i:
                buy_profile[index] += volume_i
            else:
                sell_profile[index] += volume_i

            prev_delta_i = sum(delta_profile)

            buy = buy_profile[index]
            sell = sell_profile[index]
            if buy != 0 and sell != 0:
                delta_profile[index] += (buy - sell)
            elif buy != 0 and sell == 0:
                delta_profile[index] += buy
            elif buy == 0 and sell != 0:
                delta_profile[index] += (-sell)

            current_delta = sum(delta_profile)
            if prev_delta_i > current_delta:
                min_delta[0] = prev_delta_i
            if prev_delta_i < current_delta:
                max_delta[0] = prev_delta_i

        if self._df_ticks is not None:
            start = df_interval['datetime'].head(1).values[0]
            end = df_interval['datetime'].tail(1).values[0]
            ticks_interval = self._df_ticks.loc[(self._df_ticks['datetime'] >= start) & (self._df_ticks['datetime'] <= end)]
            ticks_array = ticks_interval['close'].to_numpy() # speeed

            calculate_len = len(ticks_array)
            for i in range(calculate_len):
                tick = ticks_array[i]
                prev_tick = ticks_array[i - 1]
                for idx in range(len_segments):
                    row = interval_segments[idx]
                    prev_row = interval_segments[idx - 1]
                    if (tick >= prev_row) and (tick <= row):
                        normal_profile[idx] += 1

                        if tick > prev_tick:
                            buy_profile[idx] += 1
                        elif tick < prev_tick:
                            sell_profile[idx] += 1
                        elif tick == prev_tick:
                            buy_profile[idx] += 1
                            sell_profile[idx] += 1

                        prev_delta_i = sum(delta_profile)

                        buy = buy_profile[idx]
                        sell = sell_profile[idx]
                        delta_profile[idx] += (buy - sell)

                        current_delta = sum(delta_profile)
                        if prev_delta_i > current_delta:
                            min_delta[0] = prev_delta_i
                        if prev_delta_i < current_delta:
                            max_delta[0] = prev_delta_i
        else:
            # speeed
            _ohlcv = ['open', 'high', 'low', 'close', 'volume']
            open_arr, high_arr, low_arr, close_arr, volume_arr = (df_interval[name].to_numpy() for name in _ohlcv)

            calculate_len = len(open_arr)
            for i in range(calculate_len):
                open, high, low, close, volume = open_arr[i], high_arr[i], low_arr[i], close_arr[i], volume_arr[i]
                is_up = close >= open

                if self._distribution == DistributionData.OHLC or self._distribution == DistributionData.OHLC_No_Avg:
                    avg_vol = (volume / (open + high + low + close / 4)) \
                              if self._distribution == DistributionData.OHLC else volume
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

        # Remove first row, it's always 0
        normal_profile = normal_profile[1:]
        buy_profile = buy_profile[1:]
        sell_profile = sell_profile[1:]
        delta_profile = delta_profile[1:]

        def _to_array(value):
            return np.full(len_segments, value, dtype=np.int64)

        def _to_same_length(list_of_tuples):
            # same as _to_array, repeat the values for each row.
            return [list_of_tuples] * len_segments

        # normal
        normal_value = sum(normal_profile)
        normal_value = _to_array(normal_value)

        # normal - HVN/LVN
        (n_hvn, n_hvn_idx, n_hvn_list, 
         n_hvn_raw, n_hvn_raw_idx, n_hvn_raw_list,
         n_lvn, n_lvn_idx, n_lvn_list, 
         n_lvn_raw, n_lvn_raw_idx, n_lvn_raw_list) = \
            volume_nodes_filter(normal_profile, vp_prices, self._profile_filter)

        _values = [n_hvn, n_lvn, n_hvn_idx, n_lvn_idx,
                   n_hvn_raw, n_lvn_raw, n_hvn_raw_idx, n_lvn_raw_idx]
        (n_hvn, n_lvn, 
         n_hvn_idx, n_lvn_idx, 
         n_hvn_raw, n_lvn_raw, 
         n_hvn_raw_idx, n_lvn_raw_idx) = (_to_same_length(value) for value in _values)

        # buy_sell
        value_buy, value_sell = sum(buy_profile), sum(sell_profile)
        value_sum, value_subtract = value_buy + value_sell, value_buy - value_sell
        value_divide = 0
        if value_buy != 0 and value_sell != 0:
            value_divide = value_buy / value_sell

        _values = [value_buy, value_sell, value_sum, value_subtract, value_divide]
        value_buy, value_sell, value_sum, value_subtract, value_divide = (_to_array(value) for value in _values)

        # delta
        delta_value = sum(delta_profile)
        subtract_delta = min_delta[0] - max_delta[0]

        _values = [delta_value, min_delta[0], max_delta[0], subtract_delta]
        delta_value, min_delta, max_delta, subtract_delta = (_to_array(value) for value in _values)

        # delta - HVN/LVN
        (d_hvn, d_hvn_idx, d_hvn_list,
         d_hvn_raw, d_hvn_raw_idx, d_hvn_raw_list,
         d_lvn, d_lvn_idx, d_lvn_list,
         d_lvn_raw, d_lvn_raw_idx, d_lvn_raw_list) = \
            volume_nodes_filter(abs(delta_profile), vp_prices, self._profile_filter)

        _values = [d_hvn, d_lvn,
                   d_hvn_idx, d_lvn_idx,
                   d_hvn_raw, d_lvn_raw,
                   d_hvn_raw_idx, d_lvn_raw_idx]
        (d_hvn, d_lvn,
         d_hvn_idx, d_lvn_idx,
         d_hvn_raw, d_lvn_raw,
         d_hvn_raw_idx, d_lvn_raw_idx) = (_to_same_length(value) for value in _values)

        # as above so below
        prefix = get_prefix(extra_profile, True)

        normal_tuple = (vp_datetime, vp_prices, normal_profile, normal_value,
                        n_hvn, n_hvn_idx, n_hvn_list,
                        n_hvn_raw, n_hvn_raw_idx, n_hvn_raw_list,
                        n_lvn, n_lvn_idx, n_lvn_list,
                        n_lvn_raw, n_lvn_raw_idx, n_lvn_raw_list)
        normal_df = pd.DataFrame(zip(*normal_tuple), columns=[
            f'{prefix}_datetime', f'{prefix}_prices', f'{prefix}_normal', f'{prefix}_normal_total',
            f'{prefix}_normal_hvn_levels', f'{prefix}_normal_hvn_idx', f'{prefix}_normal_hvn_mask',
            f'{prefix}_normal_hvn_raw_levels', f'{prefix}_normal_hvn_raw_idx', f'{prefix}_normal_hvn_raw_mask',
            f'{prefix}_normal_lvn_levels', f'{prefix}_normal_lvn_idx', f'{prefix}_normal_lvn_mask',
            f'{prefix}_normal_lvn_raw_levels', f'{prefix}_normal_lvn_raw_idx', f'{prefix}_normal_lvn_raw_mask',
        ])

        buy_sell_tuple = (vp_datetime, vp_prices, buy_profile, sell_profile, value_buy, value_sell, value_sum, value_subtract, value_divide)
        buy_sell_df = pd.DataFrame(zip(*buy_sell_tuple), columns=[
            f'{prefix}_datetime', f'{prefix}_prices', f'{prefix}_buy', f'{prefix}_sell', f'{prefix}_buy_value',
            f'{prefix}_sell_value', f'{prefix}_bs_sum', f'{prefix}_bs_subtract', f'{prefix}_bs_divide']
        )

        delta_tuple = (vp_datetime, vp_prices, delta_profile, delta_value,
                       min_delta, max_delta, subtract_delta,
                       d_hvn, d_hvn_idx, d_hvn_list,
                       d_hvn_raw, d_hvn_raw_idx, d_hvn_raw_list,
                       d_lvn, d_lvn_idx, d_lvn_list,
                       d_lvn_raw, d_lvn_raw_idx, d_lvn_raw_list)
        delta_df = pd.DataFrame(zip(*delta_tuple), columns=[
            f'{prefix}_datetime', f'{prefix}_prices', f'{prefix}_delta', f'{prefix}_delta_total',
            f'{prefix}_delta_min', f'{prefix}_delta_max', f'{prefix}_delta_subtract',
            f'{prefix}_delta_hvn_levels', f'{prefix}_delta_hvn_idx', f'{prefix}_delta_hvn_mask',
            f'{prefix}_delta_hvn_raw_levels', f'{prefix}_delta_hvn_raw_idx', f'{prefix}_delta_hvn_raw_mask',
            f'{prefix}_delta_lvn_levels', f'{prefix}_delta_lvn_idx', f'{prefix}_delta_lvn_mask',
            f'{prefix}_delta_lvn_raw_levels', f'{prefix}_delta_lvn_raw_idx', f'{prefix}_delta_lvn_raw_mask']
        )

        return normal_df, buy_sell_df, delta_df