"""
TPO Profile
=====
Python version of TPO Profile (v2.0) developed for cTrader Trading Platform

Features from revision 1 (after Order Flow Aggregated development)
    - Mini/Weekly/Monthly Profiles
    - Fixed-Range Profiles
    - Shared Segments
Additional Features => that will be implemented to C# version... sometime next year (2026)
    - HVN/LVN Detection + Levels
Improvements:
    - Parallel processing of each profile interval
    - Numpy Arrays, where possible.
Python/C# author:
    - srlcarlg
"""
import itertools
import math
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
import mplfinance as mpf

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from models_utils.profile_models import SegmentsInterval, ExtraProfile, ProfileFilter
from models_utils.profile_utils import get_intervals_list, create_shared_segments, get_segments, get_prefix, volume_nodes_filter


class TpoProfile:
    def __init__(self, df_ohlc: pd.DataFrame, row_height: float, interval: pd.Timedelta,
                 profile_filter: ProfileFilter | None = None,
                 segments_interval: SegmentsInterval = SegmentsInterval.Daily):
        """
        Create TPO Profiles from any OHLC chart!  \n
        Backtest version.

        Usage
        ------
        >>> from tpo_profile import TpoProfile, SegmentsInterval
        >>> tpo = TpoProfile(df_ohlc, row_height, pd.Timedelta(hours=4), None, SegmentsInterval.Daily)

        >>> # plot with mplfinance
        >>> tpo.plot()

        >>> # plot with plotly
        >>> tpo.plot_ly()

        >>> # get dataframes of each interval
        >>> df_intervals, df_profiles = tpo.profiles()
        >>> # Alternatively => tpo.mini(mini_interval) / tpo.weekly() / tpo.monthly()

        >>> # change parameters for filters
        >>> from models_utils.profile_models import ProfileFilter, ProfileSmooth, ProfileNode
        >>> params_nodes = ProfileFilter(ProfileSmooth.Gaussian, ProfileNode.LocalMinMax, strong_only=False, ...)
        >>> params_nodes.levels(61.8, 23.6) # set percentages for Symmetric Bands

        >>> tpo = TpoProfile(df_ohlc, row_height, pd.Timedelta(hours=4), params_nodes, ...)
        Parameters
        ----------
        df_ohlc : dataframe
            * index/datetime, open, high, low, close, volume
            * "datetime": If is not present, the index will be used.
        row_height : float
            Cannot be less than or equal to 0.00000...
        interval : pd.Timedelta
            Interval for each profile, can be Minutes, Hours, Days, Weekly.
        profile_filter : ProfileFilter
            Parameters for HVN/LVN Detection + Levels
        segments_interval : SegmentsInterval
            Interval to calculate the price-segments that will be shared among all profiles.

            The df_ohlc should provide, at least, one starting point for Daily/Weekly/Monthly interval, example:

            * 1 Day => Daily
            * 1 Monday => Weekly
            * First Weekday (monday) => Monthly
            * If SegmentsInterval.FromProfile, each profile will have its own segments calculated by its interval
        """
        if 'datetime' not in df_ohlc.columns:
            df_ohlc["datetime"] = df_ohlc.index
        _expected = ['open', 'high', 'low', 'close']
        for column in _expected:
            if column not in df_ohlc.columns:
                raise ValueError(f"'{column}' column from the expected {_expected} doesn't exist!")

        self._df_ohlc = df_ohlc
        self._df_ohlc_index = df_ohlc.index.to_numpy()
        self._row_height = row_height
        self._segments_interval = segments_interval
        self._profile_filter = profile_filter if isinstance(profile_filter, ProfileFilter) else ProfileFilter()
        self._shared_segments = dict()
        if segments_interval != SegmentsInterval.From_Profile:
            self._shared_segments = create_shared_segments(df_ohlc, row_height, segments_interval)

        def parallel_process_profiles(list_of_dfs: list):
            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                results = pool.map(self._create_tpo, list_of_dfs)
            return results

        df_list = get_intervals_list(df_ohlc, interval)
        self._interval_dfs = df_list
        self._interval_profiles = parallel_process_profiles(df_list)

    def _parallel_process_extra(self, list_of_dfs: list, extra_profile: ExtraProfile):
        num_processes = cpu_count()
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(self._create_tpo, zip(list_of_dfs, itertools.repeat(extra_profile)))
        return results

    def profiles(self):
        """
        Return all dataframes of ohlc + tpo_profile from (main) interval

        >>> from tpo_profile import TpoProfile
        >>> tpo = TpoProfile(...)

        >>> df_intervals, df_profiles = tpo.profiles()
        >>> # to access each ohlc_interval and its profile:
        >>> df_intervals[0] # [1], [2], etc...
        >>> df_profiles[0] # [1], [2], etc...
        """
        return self._interval_dfs, self._interval_profiles

    def mini(self, mini_interval: pd.Timedelta = pd.Timedelta(hours=4)):
        """
        Create/Return all dataframes of df_ohlc + mini_profiles

        >>> from tpo_profile import TpoProfile
        >>> tpo = TpoProfile(...)

        >>> df_intervals, df_profiles = tpo.mini(pd.Timedelta(hours=4))
        >>> # to access each ohlc_interval and its profile:
        >>> df_intervals[0] # [1], [2], etc...
        >>> df_profiles[0] # [1], [2], etc...
        """
        df_intervals = get_intervals_list(self._df_ohlc, mini_interval)
        df_profiles = self._parallel_process_extra(df_intervals, ExtraProfile.Mini)
        return df_intervals, df_profiles

    def weekly(self):
        """
        Create/Return all dataframes of df_ohlc + weekly_profiles

        >>> from tpo_profile import TpoProfile
        >>> tpo = TpoProfile(...)

        >>> df_intervals, df_profiles = tpo.weekly()
        >>> # to access each ohlc_interval and its profile:
        >>> df_intervals[0] # [1], [2], etc...
        >>> df_profiles[0] # [1], [2], etc...
        """
        if self._segments_interval == SegmentsInterval.Daily:
            raise ValueError(f"segments_interval should be >= Weekly, Monthly or From_Profile")

        df_intervals = get_intervals_list(self._df_ohlc, pd.Timedelta(weeks=1))
        df_profiles = self._parallel_process_extra(df_intervals, ExtraProfile.Weekly)
        return df_intervals, df_profiles

    def monthly(self):
        """
        Create/Return all dataframes of df_ohlc + monthly_profiles


        >>> from tpo_profile import TpoProfile
        >>> tpo = TpoProfile(...)

        >>> df_intervals, df_profiles = tpo.monthly()
        >>> # to access each ohlc_interval and its profile:
        >>> df_intervals[0] # [1], [2], etc...
        >>> df_profiles[0] # [1], [2], etc...
        """
        if  self._segments_interval in [SegmentsInterval.Daily, SegmentsInterval.Weekly]:
            raise ValueError(f"segments_interval should be >= Monthly or From_Profile")
        df_intervals = get_intervals_list(self._df_ohlc, pd.DateOffset(months=1))
        df_profiles = self._parallel_process_extra(df_intervals, ExtraProfile.Monthly)
        return df_intervals, df_profiles

    def fixed(self, fixed_dates: list):
        """
        Create/Return Fixed-Range profiles.

        >>> from tpo_profile import TpoProfile
        >>> tpo = TpoProfile(...)

        >>> dates = [ ('2025-05-15 01:40:00', '2025-05-15 16:41:00'), ('start_date', 'end_date'), ...]
        >>> df_intervals, df_profiles = tpo.fixed(fixed_dates)
        >>> # to access each ohlc_interval and its profile:
        >>> df_intervals[0] # [1], [2], etc...
        >>> df_profiles[0] # [1], [2], etc...

        For plotting, the range-date tuples must be in ascending order, because:
         - The start_date of the FIRST range-profile
         - The end_date of the LAST range-profile
         - will be used to retrieve the remaining df_ohlc rows
        """
        df = self._df_ohlc

        df_intervals = []
        for start_date, end_date in fixed_dates:
            interval_df = df.loc[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
            df_intervals.append(interval_df)
        df_profiles = self._parallel_process_extra(df_intervals, ExtraProfile.Fixed)

        return df_intervals, df_profiles

    def plot(self, extra_profile: ExtraProfile = ExtraProfile.No, extra_only: bool = False,
             mini_interval: pd.Timedelta = pd.Timedelta(hours=2), fixed_dates: list = ()):
        """
        Plot all intervals of TPO profiles with 'mplfinance'.

        >>> from tpo_profile import TpoProfile
        >>> tpo = TpoProfile(...)

        >>> # plot the main interval => tpo.profiles()
        >>> tpo.plot()

        >>> # plot mini-profiles
        >>> tpo.plot(ExtraProfile.Mini, mini_interval=pd.Timedelta(hours=4))

        >>> # plot weekly/monthly
        >>> tpo.plot(ExtraProfile.Weekly) # or ExtraProfile.Monthly

        >>> # plot fixed-range profiles
        >>> dates = [ ('2025-05-15 01:40:00', '2025-05-15 16:41:00'), ('start_date', 'end_date'), ...]
        >>> tpo.plot(ExtraProfile.Fixed, fixed_dates=dates)

        >>> # plot only the respective extra-profile
        >>> tpo.plot(...extra_only=True)
        """
        def parallel_process_profiles():
            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                results = pool.starmap(self._mpf_workaround, zip(self._interval_dfs, self._interval_profiles))
            return results

        def parallel_process_extra():
            match extra_profile:
                case ExtraProfile.Mini:
                    extra_intervals, extra_profiles = self.mini(mini_interval)
                case ExtraProfile.Weekly:
                    extra_intervals, extra_profiles = self.weekly()
                case ExtraProfile.Monthly:
                    extra_intervals, extra_profiles = self.monthly()
                case _:
                    extra_intervals, extra_profiles = self.fixed(fixed_dates)

            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                extra_results = pool.starmap(self._mpf_workaround, zip(extra_intervals, extra_profiles, itertools.repeat(extra_profile)))
            return extra_results

        # join all interval profiles to get original df_ohlc with tpo_scatter
        df_list = parallel_process_profiles()
        df_ohlc_tpo = pd.concat(df_list, ignore_index=True)
        df_ohlc_tpo.index = df_ohlc_tpo['datetime']

        if extra_profile != ExtraProfile.No:
            # Get extra-profile scatters
            extra_list = parallel_process_extra()
            extra_df = pd.concat(extra_list, ignore_index=True)
            extra_df.index = extra_df['datetime']

            if extra_profile == ExtraProfile.Fixed:
                df_start = self._df_ohlc.loc[self._df_ohlc['datetime'] < fixed_dates[0][0]]
                df_end = self._df_ohlc.loc[self._df_ohlc['datetime'] > fixed_dates[-1][1]]
                # retrieve the remaining df_ohlc rows
                df_to_join = [df_start, extra_df, df_end]
                extra_df = pd.concat(df_to_join, ignore_index=True)
                extra_df.index = extra_df['datetime']

            # drop same columns before join()
            extra_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'datetime'], inplace=True)

            df_ohlc_tpo = df_ohlc_tpo.join(extra_df, how='outer')

        # total tpo_scatter columns
        numbers = df_ohlc_tpo.columns.str.extract(r'(\d+)', expand=False)
        total_add = int(numbers[len(numbers) - 1])

        # remove columns with all NaN values
        df_ohlc_tpo.dropna(axis=1, how='all', inplace=True)

        apd = [mpf.make_addplot(df_ohlc_tpo[f'tpo_scatter_{i}'], color='deepskyblue', alpha=0.5) \
               for i in range(total_add) if f'tpo_scatter_{i}' in df_ohlc_tpo.columns]
        if extra_only and extra_profile != ExtraProfile.No:
            apd = []

        plot_kwargs = {}
        if extra_profile != ExtraProfile.No:
            prefix = get_prefix(extra_profile)
            apd += [mpf.make_addplot(df_ohlc_tpo[f'{prefix}_scatter_{i}'], color='orange', alpha=0.5) \
                   for i in range(total_add) if f'{prefix}_scatter_{i}' in df_ohlc_tpo.columns]

            if extra_profile == ExtraProfile.Fixed:
                list_value = [item for sublist in fixed_dates for item in sublist]
                value = dict(vlines=list_value, linestyle='dotted')
                plot_kwargs = { 'vlines': value }

        s = mpf.make_mpf_style(base_mpl_style='seaborn-v0_8', rc={'axes.grid': False})
        mpf.plot(df_ohlc_tpo, type='candle', style=s, addplot=apd, warn_too_much_data=len(df_ohlc_tpo) + 1,
                 title="TPO Profile",
                 figsize=(12.5,6),
                 returnfig=True,
                 scale_padding=0.2, **plot_kwargs)
        mpf.show()

    def plot_ly(self, extra_profile: ExtraProfile = ExtraProfile.No, extra_only: bool = False,
                mini_interval: pd.Timedelta = pd.Timedelta(hours=2), fixed_dates: list = (),
                nodes: bool = True, nodes_source ='lvn', nodes_levels: bool = True,
                show_numbers: bool = False,
                chart: str = 'candle', renderer: str = 'default', width: int = 1200, height: int = 800):
        """
        Plot all intervals of TPO profiles with 'plotly'.

        >>> from tpo_profile import TpoProfile
        >>> tpo = TpoProfile(...)

        >>> # plot the main interval => tpo.profiles()
        >>> tpo.plot_ly()

        >>> # plot mini-profiles
        >>> tpo.plot_ly(ExtraProfile.Mini, mini_interval=pd.Timedelta(hours=4))

        >>> # plot weekly/monthly
        >>> tpo.plot_ly(ExtraProfile.Weekly) # or ExtraProfile.Monthly

        >>> # plot fixed-range profiles
        >>> dates = [ ('2025-05-15 01:40:00', '2025-05-15 16:41:00'), ('start_date', 'end_date'), ...]
        >>> tpo.plot_ly(ExtraProfile.Fixed, fixed_dates=dates)

        >>> # plot HVN/LVN
        >>> tpo.plot_ly(...nodes=True, nodes_source='hvn', nodes_levels=True)

        >>> # plot only the respective extra-profile
        >>> tpo.plot_ly(...extra_only=True)
        """
        _charts = ['candle', 'ohlc']
        _node_sources = ['hvn', 'lvn', 'hvn_raw', 'lvn_raw']

        input_values = [chart, nodes_source]
        input_validation = [_charts, _node_sources]
        for value, validation in zip(input_values, input_validation):
            if value not in validation:
                raise ValueError(f"Only {validation} options are valid.")

        df = self._df_ohlc.copy()
        df['plotly_int_index'] = range(len(df))

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
            mode_profile = self.profiles()[1]

            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                results = pool.starmap(self._plotly_workaround, zip(self._interval_dfs, mode_profile))
            return results

        def parallel_process_extra():
            match extra_profile:
                case ExtraProfile.Weekly:
                    extra_intervals, extra_profiles = self.weekly()
                case ExtraProfile.Monthly:
                    extra_intervals, extra_profiles = self.monthly()
                case ExtraProfile.Fixed:
                    extra_intervals, extra_profiles = self.fixed(fixed_dates=fixed_dates)
                case _:
                    extra_intervals, extra_profiles = self.mini(mini_interval)

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

        extra_prefix = get_prefix(extra_profile)
        _names = ['tpo'] if extra_profile == ExtraProfile.No else \
                 [extra_prefix] if extra_only else ['tpo', extra_prefix]

        for name in _names:
            vp_prefix = f'{prefix}_{name}'
            if name != 'tpo':
                to_plot = extra_to_plot

            prices = to_plot[f'{vp_prefix}_prices'].to_numpy()
            bases = to_plot[f'{vp_prefix}_base_index'].to_numpy()
            bases_end = to_plot[f'{vp_prefix}_end_index'].to_numpy()
            plot_vp_values = to_plot[f'{vp_prefix}_values'].to_numpy()
            vp_original_values = to_plot[f'{vp_prefix}_original_values'].to_numpy()
            volume_nodes_colors = to_plot[f'{vp_prefix}_{nodes_source}_colors'].to_numpy()
            volume_nodes_levels = to_plot[f'{vp_prefix}_{nodes_source}_lvls'].to_numpy()

            for idx in range(len(to_plot)):
                y_column = prices[idx]
                x_column = plot_vp_values[idx]
                base_index = bases[idx]
                original_values = vp_original_values[idx]
                coloring = '#00BFFF'

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

                if nodes_levels:
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

                if show_numbers:
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

        fig.update_layout(
            title=f"TPO Profile",
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

        prefix = get_prefix(extra_profile)

        tpo_prices = df_profile[f'{prefix}_prices'].to_numpy()
        tpo_values = df_profile[f'{prefix}_values'].to_numpy()

        max_index = len(df_interval)
        max_volume = tpo_values.max()

        len_profile = len(df_profile)
        for i in range(len_profile):
            first = tpo_values[i] * max_index
            result = math.ceil(first / max_volume)

            scatter_name = f'{prefix}_scatter_{i}'
            df_interval[scatter_name] = np.NaN
            scatter_arr = df_interval[scatter_name].to_numpy() # speeeed

            for bar_index in range(result):
                if bar_index >= max_index:
                    break
                scatter_arr[bar_index] = tpo_prices[i]

            df_interval[scatter_name] = scatter_arr

        return df_interval

    def _plotly_workaround(self, df_interval: pd.DataFrame, df_profile: pd.DataFrame,
                           extra_profile: ExtraProfile = ExtraProfile.No):
        """
        Same logic of _mpf_workaround
        """
        prefix = get_prefix(extra_profile)
        tpo_name = f'{prefix}_values'

        max_index = len(df_interval)
        max_volume = df_profile[tpo_name].max()

        tpo_prices = df_profile[f'{prefix}_prices'].to_numpy() # speeed
        column_array = df_profile[tpo_name].to_numpy()
        calculate_len = len(column_array)

        index_array = self._df_ohlc_index
        base_idx = np.where(index_array == df_profile[f'{prefix}_datetime'].iat[0])[0]
        end_idx = np.where(index_array == df_interval[f'datetime'].iat[-1])[0]
        # with pandas index (self._df_ohlc_index is not an array)
        # base_idx = self._df_ohlcv_index.get_loc(df_profile[f'{prefix}_datetime'].iat[0])
        # end_idx = self._df_ohlcv_index.get_loc(df_interval[f'datetime'].iat[-1])

        tpo_prefix = f'plotly_{prefix}'
        tpo_proportioned = {
            f"{tpo_prefix}_base_index": [base_idx[0]],
            f"{tpo_prefix}_end_index": [end_idx[0]],
            f"{tpo_prefix}_prices": [tpo_prices],
            f"{tpo_prefix}_values": [np.empty(calculate_len)],
            f"{tpo_prefix}_original_values": [column_array],
            f"{tpo_prefix}_hvn_colors": [np.empty(1)],
            f"{tpo_prefix}_hvn_lvls": [np.empty(1)],
            f"{tpo_prefix}_hvn_raw_colors": [np.empty(1)],
            f"{tpo_prefix}_hvn_raw_lvls": [df_profile[f'{prefix}_hvn_raw_levels'].iat[0]],
            f"{tpo_prefix}_lvn_colors": [np.empty(1)],
            f"{tpo_prefix}_lvn_lvls": [np.empty(1)],
            f"{tpo_prefix}_lvn_raw_colors": [np.empty(1)],
            f"{tpo_prefix}_lvn_raw_lvls": [df_profile[f'{prefix}_lvn_raw_levels'].iat[0]],
        }

        for i in range(calculate_len):
            first = column_array[i] * math.ceil(max_index / 2)
            result = math.ceil(first / max_volume)

            tpo_proportioned[f'{tpo_prefix}_values'][0][i] = result

        # HVN /LVN
        name = prefix

        # hvn + bands
        d = df_profile[f'{name}_hvn_mask']
        colors = np.where(d == 1, 'aqua',
                 np.where(d == 2, 'gold',
                 np.where(d == 3, 'aqua', '#00BFFF')))
        tpo_proportioned[f"{tpo_prefix}_hvn_colors"][0] = colors

        # levels (with bands)
        d = df_profile[f'{name}_hvn_levels'].iat[0]
        hvn_mask = [idx for tpl in d for idx in tpl]
        tpo_proportioned[f"{tpo_prefix}_hvn_lvls"][0] = np.array(hvn_mask)

        # raw hvn
        d = df_profile[f'{name}_hvn_raw_mask']
        colors = np.where(d == 1, 'gold', '#00BFFF')
        tpo_proportioned[f"{tpo_prefix}_hvn_raw_colors"][0] = colors

        # lvn + bands
        d = df_profile[f'{name}_lvn_mask']
        colors = np.where(d == 1, 'blue',
                 np.where(d == 2, 'red',
                 np.where(d == 3, 'blue', '#00BFFF')))
        tpo_proportioned[f"{tpo_prefix}_lvn_colors"][0] = colors

        # levels (with bands)
        d = df_profile[f'{name}_lvn_levels'].iat[0]
        lvn_mask = [idx for tpl in d for idx in tpl]
        tpo_proportioned[f"{tpo_prefix}_lvn_lvls"][0] = np.array(lvn_mask)

        # raw lvn
        d = df_profile[f'{name}_lvn_raw_mask']
        colors = np.where(d == 1, 'red', '#00BFFF')
        tpo_proportioned[f"{tpo_prefix}_lvn_raw_colors"][0] = colors

        return pd.DataFrame(tpo_proportioned)

    def _create_tpo(self, df_interval: pd.DataFrame, extra_profile: ExtraProfile = ExtraProfile.No):
        interval_date = df_interval['datetime'].iat[0]
        interval_open = df_interval['open'].iat[0]
        interval_highest = df_interval['high'].max()
        interval_lowest = df_interval['low'].min()

        interval_segments = get_segments(interval_date, interval_open, interval_highest, interval_lowest,
                                         self._row_height, self._segments_interval, self._shared_segments)
        len_segments = len(interval_segments)

        tpo_datetime = np.full(len_segments, interval_date, dtype=np.ndarray)
        tpo_prices = interval_segments
        tpo_values = np.zeros(len_segments, dtype=np.int64)

        # np.shares_memory() = True
        high_arr = df_interval['high'].to_numpy()
        low_arr = df_interval['low'].to_numpy()
        calculate_len = len(low_arr)

        for i in range(calculate_len):
            bar_high = high_arr[i]
            bar_low = low_arr[i]

            # v = vertical
            total_v_letters = 0
            for row in interval_segments:
                if (row < bar_high) and (row > bar_low):
                    total_v_letters += 1

            bar_prev_segment = bar_high
            for no_use in range(total_v_letters):
                for idx in range(len_segments):
                    prev_row = interval_segments[idx - 1]
                    row = interval_segments[idx]
                    if (bar_prev_segment >= prev_row) and (bar_prev_segment <= row):
                        tpo_values[idx] += 1
                        break

                bar_prev_segment = abs(bar_prev_segment - self._row_height)

        # Remove first row, it's always 0
        tpo_values = tpo_values[1:]

        # HVN/LVN
        (_hvn, _hvn_idx, _hvn_list,
         _hvn_raw, _hvn_raw_idx, _hvn_raw_list,
         _lvn, _lvn_idx, _lvn_list,
         _lvn_raw, _lvn_raw_idx, _lvn_raw_list)= \
            volume_nodes_filter(tpo_values, tpo_prices, self._profile_filter)

        def _to_same_length(list_of_tuples):
            # repeat the values for each row.
            return [list_of_tuples] * len_segments

        _values = [_hvn, _lvn, _hvn_idx, _lvn_idx,
                   _hvn_raw, _lvn_raw, _hvn_raw_idx, _lvn_raw_idx]
        (_hvn, _lvn, _hvn_idx, _lvn_idx,
         _hvn_raw, _lvn_raw, _hvn_raw_idx, _lvn_raw_idx) = (_to_same_length(value) for value in _values)

        prefix = get_prefix(extra_profile)
        tpo_tuple = (tpo_datetime, tpo_prices, tpo_values,
                    _hvn, _hvn_idx, _hvn_list,
                    _hvn_raw, _hvn_raw_idx, _hvn_raw_list,
                    _lvn, _lvn_idx, _lvn_list,
                     _lvn_raw, _lvn_raw_idx, _lvn_raw_list)

        df_profile = pd.DataFrame(zip(*tpo_tuple), columns=[
            f'{prefix}_datetime', f'{prefix}_prices', f'{prefix}_values',
            f'{prefix}_hvn_levels', f'{prefix}_hvn_idx', f'{prefix}_hvn_mask',
            f'{prefix}_hvn_raw_levels', f'{prefix}_hvn_raw_idx', f'{prefix}_hvn_raw_mask',
            f'{prefix}_lvn_levels', f'{prefix}_lvn_idx', f'{prefix}_lvn_mask',
            f'{prefix}_lvn_raw_levels', f'{prefix}_lvn_raw_idx', f'{prefix}_lvn_raw_mask',
        ])

        df_profile.index = df_profile[f'{prefix}_datetime']

        return df_profile