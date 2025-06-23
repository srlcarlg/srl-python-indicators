"""
Order Flow Aggregated
=====
Python proof-of-concept indicator, yet to be developed for cTrader trading platform!

Actually, it's a conjunction of Volume Profile (Ticks) + Order Flow Ticks indicators.
* Volume Profile (intervals/values) for Aggregate Order Flow data
* Volume Profile (segmentation) to calculate the Order Flow of each bar

This 'combination' gives the quality that others footprint/order-flow software have:
 * Aligned Rows for all bars on the chart, or - in our case, at the given interval.
 * Possibility to create a truly [Volume, Delta] Bubbles chart

It's means that Order Flow Ticks is wrong / no longer useful? Absolutely not! Think about it:
 * With ODF_Ticks you get -> exactly <- what happened _inside a bar_, it's like looking
    at a microstructure (ticks) through a microscope (bar segments) using optical zoom (bar)
 * With ODF_Aggregated you get a -> structured view <- of what happened _inside the bars_, it's like looking
    at a microstructure (ticks) through a filter lens (VP segments) of a microscope (VP values) using digital zoom (VP interval)

In other words:
 * Order Flow Ticks - raw detail
 * Order Flow Aggregated - compressed detail

Improvements:
    - Parallel processing of each interval
    - Parallel processing of each bar
Additional Features:
    Nothing, yet.
Python author:
    - srlcarlg
"""
import math
from copy import deepcopy
from multiprocessing import cpu_count, Pool

import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

class OrderFlowAggregated:
    def __init__(self, df_ohlc: pd.DataFrame, df_ticks: pd.DataFrame,
                 row_height: float, interval: pd.Timedelta = pd.Timedelta(days=1),
                 is_open_time: bool = True, with_plotly_columns: bool = True):
        """
        Create Order Flow and Volume Profile ticks data to any OHLC chart! \n
        (Candles, Renko, Range, Ticks) \n
        Backtest version.

        Usage
        ------
        >>> odf_agg = OrderFlowAggregated(df_ohlc, df_ticks, row_height, pd.Timedelta(days=1))
        >>> # Plot with plotly
        >>> odf_agg.plot(mode='delta')
        >>> # Volume Profile
        >>> odf_agg.normal_vp(), odf_agg.buy_sell_vp(), odf_agg.delta_vp(), odf_agg.all_vp()
        >>> # Order Flow Ticks
        >>>  odf_agg.normal_odf(), odf_agg.buy_sell_odf(), odf_agg.delta_odf(), odf_agg.all_odf()

        Parameters
        ----------
        df_ohlc : dataframe
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
        is_open_time : bool
            Specify if the index/datetime of df_ohlc is the OpenTime or CloseTime of each bar
        with_plotly_columns : bool
            Return 'plotly_[...]' columns used in self.plot(), if False it will not be possible to plot.
        """
        if 'datetime' not in df_ohlc.columns:
            df_ohlc["datetime"] = df_ohlc.index
        if 'datetime' not in df_ticks.columns:
                df_ticks["datetime"] = df_ticks.index

        _expected = ['open', 'high', 'low', 'close']
        for column in _expected:
            if column not in df_ohlc.columns:
                raise ValueError(f"'{column}' column from the expected {_expected} doesn't exist!")

        self._df_ohlcv = df_ohlc
        self._df_ticks = df_ticks
        self._row_height = row_height
        self._interval = interval
        self._is_open_time = is_open_time
        self._with_plotly_columns = with_plotly_columns

        if is_open_time:
            df_ohlc['end_time'] = df_ohlc['datetime'].shift(-1)
        else:
            df_ohlc['start_time'] = df_ohlc['datetime'].shift(1)

        # For plotly
        if with_plotly_columns:
            df_ohlc['plotly_int_index'] = range(len(df_ohlc))

        # Volume Profile
        df = df_ohlc
        dfs_list = []
        first_date = df['datetime'].iat[0].normalize()  # any datetime to 00:00:00
        first_interval_date = first_date + self._interval
        first_interval_df = df[df['datetime'] < first_interval_date]
        dfs_list.append(first_interval_df)

        last_date = df['datetime'].tail(1).values[0]
        current_date = first_interval_date
        while current_date < last_date:
            start_interval_date = current_date
            end_interval_date = start_interval_date + self._interval
            interval_df = df.loc[(df['datetime'] >= start_interval_date) & (df['datetime'] < end_interval_date)]

            dfs_list.append(interval_df)
            current_date = end_interval_date

        def parallel_process_dataframes(list_df):
            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                results = pool.map(self._create_vp, list_df)
            return results
        interval_profiles = parallel_process_dataframes(dfs_list)

        self._intervals_dfs = dfs_list
        self._interval_profiles = interval_profiles

        # Order Flow Ticks
        # concat vp_prices column as single list to each interval row (bar)
        vp_prices_dfs = [n[0]['vp_prices'] for n in interval_profiles]
        for i in range(len(dfs_list)):
            df_interval = dfs_list[i].copy()
            df_interval['vp_interval_prices'] = [vp_prices_dfs[i].to_list()] * len(dfs_list[i])
            dfs_list[i] = df_interval

        def parallel_process_bars(df_with_vp_prices):
            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                results = pool.map(self._create_odf, df_with_vp_prices.to_dict(orient='records'))
            return results
        # join all interval profiles to get original df_ohlcv
        df_ohlcv_vp = pd.concat(dfs_list, ignore_index=True)
        df_ohlcv_vp.index = df_ohlcv_vp['datetime']

        profiles_each_bar = parallel_process_bars(df_ohlcv_vp)

        normal_bars = [pd.DataFrame(lst[0]) for lst in profiles_each_bar]
        buy_sell_bars = [pd.DataFrame(lst[1]) for lst in profiles_each_bar]
        delta_bars = [pd.DataFrame(lst[2]) for lst in profiles_each_bar]

        normal_df = pd.concat(normal_bars, ignore_index=True)
        buy_sell_df = pd.concat(buy_sell_bars, ignore_index=True)
        delta_df = pd.concat(delta_bars, ignore_index=True)
        for _df in [normal_df, buy_sell_df, delta_df]:
            _df.index = _df['datetime']
            _df.drop(columns=['datetime'], inplace=True)

        # profile_prices is equal for all modes
        normal_df.rename(columns={'profile': 'normal_profile',
                                  'profile_prices': 'profiles_prices',
                                  'value': 'normal_value'}, inplace=True)

        buy_sell_df.rename(columns={'profile_prices': 'profiles_prices',

                                    'value_buy': 'buy_value',
                                    'value_sell': 'sell_value',
                                    'value_sum': 'bs_sum',
                                    'value_subtract': 'bs_subtract',
                                    'value_divide': 'bs_divide'}, inplace=True)

        delta_df.rename(columns={'profile': 'delta_profile',
                                 'profile_prices': 'profiles_prices',
                                 'value': 'delta'}, inplace=True)
        delta_df['cumulative_delta'] = delta_df['delta'].cumsum()

        self._df_ohlcv = df_ohlc
        self._normal_df = normal_df
        self._buy_sell_df = buy_sell_df
        self._delta_df = delta_df

    def normal_odf(self):
        """
        Get df_ohlc with 'Normal' mode.
            * ['normal_profile', 'profiles_prices', 'normal_value', 'strength_normal_value']
        If with_plotly_columns is True:
            * 'plotly_[...]' 5 total columns
        :return: pandas.DataFrame
        """
        df = pd.concat([self._df_ohlcv, self._normal_df], axis=1)
        df.index = df['datetime']
        return df

    def normal_vp(self):
        """
        Return all intervals dataframes of ohlcv + vp_profile normal mode
        vp_profile contains the following (3) columns with prefix 'vp_':
            * vp_['datetime', 'prices', 'normal', 'normal_total']
        """
        normal_dfs = [n[0] for n in self._interval_profiles]
        return [self._intervals_dfs, normal_dfs]

    def buy_sell_odf(self):
        """
        Get df_ohlc with 'Buy_Sell' mode.
            * ['buy_profile', 'sell_profile', 'profiles_prices', 'buy_value', 'sell_value', 'bs_sum', 'bs_subtract', 'bs_divide', 'strength_bs_sum', 'strength_bs_subtract', 'strength_bs_divide']
        If with_plotly_columns is True:
            * 'plotly_[...]' 17 total columns
        :return: pandas.DataFrame
        """
        df = pd.concat([self._df_ohlcv, self._buy_sell_df], axis=1)
        df.index = df['datetime']
        return df

    def buy_sell_vp(self):
        """
        Return all intervals dataframes of ohlcv + vp_profile buy_sell mode
        vp_profile contains the following (9) columns with prefix 'vp_':
            * vp_['datetime', 'prices', 'buy', 'sell', 'buy_value', 'sell_value', 'bs_sum', 'bs_subtract', 'bs_divide']
        """
        buy_sell_dfs = [bs[1] for bs in self._interval_profiles]
        return [self._intervals_dfs, buy_sell_dfs]

    def delta_odf(self):
        """
        Get df_ohlc with 'Delta' mode.
            * ['delta_profile', 'profiles_prices', 'delta', 'cumulative_delta', 'min_delta', 'max_delta', 'strength_delta']
        If with_plotly_columns is True:
            * 'plotly_[...]' 13 total columns
        :return: pandas.DataFrame
        :return: pandas.DataFrame
        """
        df = pd.concat([self._df_ohlcv, self._delta_df], axis=1)
        df.index = df['datetime']
        return df

    def delta_vp(self):
        """
        Return all intervals dataframes of ohlcv + vp_profile delta mode
        vp_profile contains the following (6) columns with prefix 'vp_':
            * vp_['datetime', 'prices', 'delta', 'delta_total', 'delta_min', 'delta_max']
        """
        delta_dfs = [d[2] for d in self._interval_profiles]
        return [self._intervals_dfs, delta_dfs]

    def all_odf(self):
        """
        Get df_ohlc with all modes (normal, buy_sell, delta)
        If with_plotly_columns is True:
            * 'plotly_[...]' 35 total columns
        :return: pandas.DataFrame
        """
        df = pd.concat([self._df_ohlcv, self._normal_df, self._buy_sell_df, self._delta_df], axis=1)
        # Drop duplicated columns (we have 3 profiles_prices, keep 1)
        df = df.loc[: , ~df.columns.duplicated()]
        df.index = df['datetime']
        return df

    def all_vp(self):
        """
        Return all intervals dataframes of ohlc + all vp_profile modes

        >>> df_intervals, df_profiles = odf_agg.all()
        >>> # to access each ohlc_interval and its profile mode:
        >>> df_intervals[0]
        >>> df_profiles[0][0] # [0]: normal
        >>> df_profiles[0][1] # [1]: buy_sell
        >>> df_profiles[0][2] # [2]: delta
        """
        return [self._intervals_dfs, self._interval_profiles]

    def plot(self, iloc_value: int=15, mode: str = 'delta', view: str = 'profile',
             chart: str = 'ohlc', show_profiles: bool = False, show_odf = True,
             renderer: str = 'default', width: int = 1200, height: int = 800):
        """

        Parameters
        ----------
        iloc_value : int
            First nº rows to be plotted
        mode : str
            'normal', 'buy_sell', 'delta'
        view : str
            'divided' or 'profile'
        chart : str
            'ohlc' or 'candle'
        show_profiles : bool
            Plot all interval volume profiles
        show_odf : bool
            Plot order flow ticks of each bar
        renderer : str
            * Change 'plotply' renderer if anything goes wrong, use:
            * **static**: svg, png and jpeg
            * **interactive**: notebook, plotly_mimetype
            * **html_interactive**: browser, iframe
            * or visit plotply.py renderers wiki
        width : int
            for static renderer
        height : int
            for static renderer
        """
        _profiles = ['normal', 'buy_sell', 'delta']
        if mode not in _profiles:
            raise ValueError(f"Only {_profiles} options are valid.")
        _views = ['profile', 'divided']
        if view not in _views:
            raise ValueError(f"Only {_views} options are valid.")
        _charts = ['candle', 'ohlc']
        if chart not in _charts:
            raise ValueError(f"Only {_charts} options are valid.")
        df = self.normal_odf() if mode == 'normal' else self.buy_sell_odf() if mode == 'buy_sell' else self.delta_odf()
        df = df.iloc[:iloc_value]

        prefix = 'plotly'
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.0)
        if chart == 'ohlc':
            fig.add_trace(go.Ohlc(x=df[f'{prefix}_int_index'],
                                  open=df['open'],
                                  high=df['high'],
                                  low=df['low'],
                                  close=df['close'], opacity=0.5), row=1, col=1)
        else:
            fig.add_trace(go.Candlestick(x=df[f'{prefix}_int_index'],
                                  open=df['open'],
                                  high=df['high'],
                                  low=df['low'],
                                  close=df['close'], opacity=0.3), row=1, col=1)

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

        # Histograms
        # 'base=index' makes the trick of divided/profiles views
        # Profile view needs (max_side * 2 in self._create_odf) because
        # we're moving the initial value to the left max_side (-0.245 or 0.3)
        if show_odf:
            for index in range(len(df)):
                step = 0.245 if chart == 'candle' else 0.3
                base_index = index if view == 'divided' else index - step
                if mode == 'normal':
                    x_column = f'{prefix}_{mode}_{chart}_histogram'
                    if str(df[x_column].iat[index]) == 'nan':
                        continue

                    fig.add_trace(
                        go.Bar(y=df['profiles_prices'].iat[index], x=df[x_column].iat[index],
                               orientation='h', base=index - step,
                               marker=dict(
                                   color='#00BFFF',
                                   opacity = 0.4
                               )), row=1, col=1)
                else:
                    # Buy
                    x_buy_column = f'{prefix}_buy_{chart}_histogram' if view == 'divided' else f'{prefix}_buy_profile_{chart}_histogram'
                    if mode == 'delta':
                        x_buy_column = f'{prefix}_delta_buy_profile_{chart}_histogram' \
                            if view == 'profile' else f'{prefix}_delta_buy_{chart}_histogram'

                    if str(df[x_buy_column].iat[index]) == 'nan':
                        continue

                    fig.add_trace(
                        go.Bar(y=df['profiles_prices'].iat[index], x=df[x_buy_column].iat[index], marker=dict(
                            color='deepskyblue',
                            opacity=0.3 if view == 'divided' else 0.4
                        ),
                        orientation='h', base=base_index), row=1, col=1)

                    # Sell
                    x_sell_column = f'{prefix}_sell_{chart}_histogram' if view == 'divided' else f'{prefix}_sell_profile_{chart}_histogram'
                    if mode == 'delta':
                        x_sell_column = f'{prefix}_delta_sell_profile_{chart}_histogram' \
                            if view == 'profile' else f'{prefix}_delta_sell_{chart}_histogram'

                    if str(df[x_sell_column].iat[index]) == 'nan':
                        continue

                    fig.add_trace(
                        go.Bar(y=df['profiles_prices'].iat[index], x=df[x_sell_column].iat[index], marker=dict(
                            color='crimson',
                            opacity=0.3
                        ),
                       orientation='h', base=base_index), row=1, col=1)

            # Numbers
            for index in range(len(df)):
                if mode == 'normal':
                    x_column = f'{prefix}_{mode}_{chart}_numbers'
                    if str(df[x_column].iat[index]) == 'nan':
                        continue

                    fig.add_trace(
                        go.Heatmap(
                            x=df[x_column].iat[index],
                            y=df['profiles_prices'].iat[index],
                            z=df[f'{mode}_profile'].iat[index],
                            text=df[f'{mode}_profile'].iat[index],
                            colorscale=color,
                            showscale=False, # remove numbers from show_legend=False column
                            texttemplate="%{text}",
                            textfont={
                                "size": 11,
                                "color": 'black',
                                "family": "Courier New"},
                        ), row=1, col=1)
                else:
                    # Buy
                    x_buy_column = f'{prefix}_buy_{chart}_numbers' if view == 'divided' else f'{prefix}_buy_profile_{chart}_numbers'
                    if mode == 'delta':
                        x_buy_column = f'{prefix}_delta_profile_{chart}_numbers' if view == 'profile' else f'{prefix}_delta_{chart}_numbers'

                    if str(df[x_buy_column].iat[index]) == 'nan':
                        continue

                    value_column = 'delta_profile' if mode == 'delta' else 'buy_profile'
                    fig.add_trace(
                        go.Heatmap(
                            x=df[x_buy_column].iat[index],
                            y=df['profiles_prices'].iat[index],
                            z=df[value_column].iat[index],
                            text=df[value_column].iat[index],
                            colorscale=color,
                            showscale=False, # remove numbers from show_legend=False column
                            texttemplate="%{text}",
                            textfont={
                                "size": 11,
                                "color": 'black',
                                "family": "Courier New"},
                        ), row=1, col=1)

                    # Sell
                    x_sell_column = f'{prefix}_sell_{chart}_numbers' if view == 'divided' else f'{prefix}_sell_profile_{chart}_numbers'
                    if mode == 'delta':
                        x_sell_column = f'{prefix}_delta_profile_{chart}_numbers' if view == 'profile' else f'{prefix}_delta_{chart}_numbers'

                    if str(df[x_sell_column].iat[index]) == 'nan':
                        continue

                    value_column = 'delta_profile' if mode == 'delta' else 'sell_profile'
                    if mode == 'buy_sell':
                        fig.add_trace(
                            go.Heatmap(
                                x=df[x_sell_column].iat[index],
                                y=df['profiles_prices'].iat[index],
                                z=df[value_column].iat[index],
                                text=df[value_column].iat[index],
                                colorscale=color,
                                showscale=False, # remove numbers from show_legend=False column
                                texttemplate="%{text}",
                                textfont={
                                    "size": 11,
                                    "color": 'black',
                                    "family": "Courier New"},
                            ), row=1, col=1)

            # Cumulative delta = above the candle
            if mode == 'delta':
                df['high_y'] = df['high'] + (self._row_height * 1.5)
                fig.add_trace(go.Scatter(x=df['plotly_int_index'], y=df['high_y'], text=df['cumulative_delta'],
                                         textposition='middle center',
                                         textfont=dict(size=11, color='blue'),
                                         mode='text'), row=1, col=1)

            # Mode value = below the candle
            df['low_y'] = df['low'] - (self._row_height * 2)
            column_below = 'normal_value' if mode == 'normal' else 'bs_subtract' if mode == 'buy_sell' else 'delta'
            fig.add_trace(go.Scatter(x=df['plotly_int_index'], y=df['low_y'], text=df[column_below], textposition='middle center',
                          textfont=dict(size=11, color='green'),
                          mode='text'), row=1, col=1)

        # Volume Profile
        if show_profiles:
            def parallel_process_dataframes():
                num_processes = cpu_count()
                _mode_profile = self.delta_vp()[1] if mode == 'delta' else \
                    self.buy_sell_vp()[1] if mode == 'buy_sell' else self.normal_vp()[1]
                with Pool(processes=num_processes) as pool:
                    results = pool.starmap(self._plotly_workaround, zip(self._intervals_dfs, _mode_profile))
                return results

            vps_to_plot = parallel_process_dataframes()
            vps_to_plot = pd.concat(vps_to_plot, ignore_index=True)

            for idx in range(len(vps_to_plot)):
                y_column = vps_to_plot['plotly_vp_prices'].iat[idx]
                x_column = vps_to_plot['plotly_vp_values'].iat[idx]
                base_index = vps_to_plot['plotly_vp_base_index'].iat[idx]
                original_values = vps_to_plot['plotly_vp_original_values'].iat[idx]
                if mode == 'normal':
                    fig.add_trace(
                        go.Bar(y=y_column,
                               x=x_column,
                               orientation='h',
                               marker=dict(
                                   color='#00BFFF',
                                   opacity=0.4
                               ), base=base_index), row=1, col=1)
                else:
                    x_column = vps_to_plot['plotly_vp_buy_values'].iat[idx]
                    fig.add_trace(
                        go.Bar(y=y_column,
                               x=x_column,
                               orientation='h',
                               marker=dict(
                                   color='deepskyblue',
                                   opacity=0.4
                               ), base=base_index), row=1, col=1)

                    x_column = vps_to_plot['plotly_vp_sell_values'].iat[idx]
                    fig.add_trace(
                        go.Bar(y=y_column,
                               x=x_column,
                               orientation='h',
                               marker=dict(
                                   color='crimson',
                                   opacity=0.4
                               ), base=base_index), row=1, col=1)

                if mode == 'normal' or mode == 'delta':
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
                else:
                    original_values = vps_to_plot['plotly_vp_original_buy_values'].iat[idx]
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

                    original_values = vps_to_plot['plotly_vp_original_sell_values'].iat[idx]
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
            title=f"Order Flow Aggregated: {mode}/{view}<br>Volume Profile: {mode}",
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

    def _plotly_workaround(self, df_interval: pd.DataFrame, df_profile: pd.DataFrame):
        """
        Like in mplfinance plot version a rule of three is used to plot the histograms.
            max_volume    max_index(int)
               x             ?(int)
        """
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

        vp_proportioned = {
            "plotly_vp_base_index": [self._df_ohlcv.index.get_loc(df_profile['vp_datetime'].iat[0])],
            "plotly_vp_prices": [df_profile['vp_prices'].to_list()],
            "plotly_vp_values": [[]],
            "plotly_vp_buy_values": [[]],
            "plotly_vp_sell_values": [[]],
            "plotly_vp_original_values": [df_profile[column_name].to_list()],
            "plotly_vp_original_buy_values": [df_profile['vp_buy'].to_list() if column_name == 'vp_buy' else []],
            "plotly_vp_original_sell_values": [df_profile['vp_sell'].to_list() if column_name == 'vp_buy' else []]
        }
        if column_name == 'vp_normal':
            for i in range(len(df_profile)):
                first = df_profile[column_name].iat[i] * math.ceil(max_index / 2)
                result = math.ceil(first / max_volume)

                vp_proportioned['plotly_vp_values'][0].append(result)

        elif column_name == 'vp_delta':
            for i in range(len(df_profile[column_name])):
                value = df_profile[column_name].iat[i]
                first = abs(value) * math.ceil(max_index / 2)
                result = math.ceil(first / max_volume)

                if value > 0:
                    vp_proportioned['plotly_vp_buy_values'][0].append(result)
                    vp_proportioned['plotly_vp_sell_values'][0].append(0)
                else:
                    vp_proportioned['plotly_vp_sell_values'][0].append(result)
                    vp_proportioned['plotly_vp_buy_values'][0].append(0)
        else:
            for i in range(len(df_profile[column_name])):
                value = df_profile[column_name].iat[i]
                first = abs(value) * math.ceil(max_index / 4)
                result = math.ceil(first / max_volume)

                vp_proportioned['plotly_vp_buy_values'][0].append(result)

            for i in range(len(df_profile['vp_sell'])):
                value = df_profile[column_name].iat[i]
                first = abs(value) * math.ceil(max_index / 2)
                result = math.ceil(first / max_volume)

                vp_proportioned['plotly_vp_sell_values'][0].append(result)

        return pd.DataFrame(vp_proportioned)

    def _create_vp(self, df_interval):
        """
        Only VP bars has been removed and highest margin is kept, so any updates or future additional features added to
        volume_profile.py should works without issue.
        """
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
            [[df_interval['datetime'].iat[0]] * len(interval_segments)] * 3
        normal['prices'], buy_sell['prices'], delta['prices'] = [interval_segments] * 3
        # Imagine the headache to debug such silently unexpected behavior
        normal['values'], buy_sell['vp_buy'], buy_sell['vp_sell'], delta['values'] = \
            (deepcopy([0.0] * len(interval_segments)) for _ in range(4))

        # VP ticks
        start = df_interval['datetime'].head(1).values[0]
        end = df_interval['datetime'].tail(1).values[0]
        ticks_interval = self._df_ticks.loc[
            (self._df_ticks['datetime'] >= start) & (self._df_ticks['datetime'] <= end)]

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

        normal['total_value'] = sum(normal['values'])

        buy_sell['value_buy'] = sum(buy_sell['vp_buy'])
        buy_sell['value_sell'] = sum(buy_sell['vp_sell'])
        buy_sell['value_sum'] = buy_sell['value_buy'] + buy_sell['value_sell']
        buy_sell['value_subtract'] = buy_sell['value_buy'] - buy_sell['value_sell']
        if buy_sell['value_buy'] != 0 and buy_sell['value_sell'] != 0:
            buy_sell['value_divide'] = buy_sell['value_buy'] / buy_sell['value_sell']

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
            # _df.drop(_df.tail(1).index, inplace=True) # We need this 'highest' margin

        # shift(-1) on vp_values to match the cumulative sum of each row from Order Flow Ticks
        # still, the sum of some row values can diverge from 1 to 30 ticks
        # I would say that the overall accuracy is about 95,5%
        normal_df['vp_normal'] = normal_df['vp_normal'].shift(-1)
        normal_df.fillna(0, inplace=True)

        delta_df['vp_delta'] = delta_df['vp_delta'].shift(-1)
        delta_df.fillna(0, inplace=True)

        buy_sell_df['vp_buy'] = buy_sell_df['vp_buy'].shift(-1)
        buy_sell_df['vp_sell'] = buy_sell_df['vp_sell'].shift(-1)
        buy_sell_df.fillna(0, inplace=True)

        return [normal_df, buy_sell_df, delta_df]

    def _create_odf(self, df_ohlc):
        """
        Only interval_segments logic has been modified, so any updates or future additional features added to
        order_flow_ticks.py should works without issue.
        """
        df_ohlc_row = pd.DataFrame([df_ohlc])

        # Start - Modified Logic
        bar_high = df_ohlc_row['high'].iat[0] + (self._row_height / 2)
        bar_low = df_ohlc_row['low'].iat[0] - (self._row_height / 2)
        list_vp_segments = df_ohlc_row['vp_interval_prices'].iat[0]
        # Add another margin to the highest margin so that nearby bars can get these segments
        list_vp_segments.append(list_vp_segments[len(list_vp_segments) - 1] + self._row_height)

        interval_segments = []
        for idx in range(len(list_vp_segments)):
            row = list_vp_segments[idx]
            if bar_low <= row:
                interval_segments.append(row)
            if bar_high < row:
                break

        interval_segments.sort()
        # End - Modified Logic

        # Yes... pointers here too
        # Since datetime and price segments are equal in any volume mode, this behaviour is welcome.
        bar_date = df_ohlc_row['datetime'].iat[0]
        list_zeros = [0.0] * len(interval_segments)

        normal = {'datetime': bar_date, 'profile': [deepcopy(list_zeros)], 'profile_prices': [interval_segments], 'value': 0.0}
        buy_sell = {'datetime': bar_date, 'buy_profile': [deepcopy(list_zeros)], 'sell_profile': [deepcopy(list_zeros)],
                    'profile_prices': [interval_segments],
                    'value_buy': 0.0, 'value_sell': 0.0,
                    'value_sum': 0.0, 'value_subtract': 0.0, 'value_divide': 0.0}
        delta = {'datetime': bar_date, 'profile': [deepcopy(list_zeros)], 'profile_prices': [interval_segments],
                'value': 0.0, 'cumulative_delta': 0.0,
                'min_delta': 0.0, 'max_delta': 0.0}

        # VP Ticks
        if self._is_open_time:
            start_time = df_ohlc_row['datetime'].iat[0]
            end_time = df_ohlc_row['end_time'].iat[0]
        else:
            start_time = df_ohlc_row['start_time'].iat[0]
            end_time = df_ohlc_row['datetime'].iat[0]

        def vp_ticks(bar_start_time, bar_end_time):
            bar_index = 0
            df_ticks = self._df_ticks.loc[(self._df_ticks['datetime'] >= bar_start_time) & (self._df_ticks['datetime'] <= bar_end_time)]

            calculate_len = len(df_ticks)
            for i_tk in range(calculate_len):
                # tick_date = df_ticks['datetime'].iat[i_tk]
                # if tick_date < bar_start_time or tick_date > bar_end_time:
                #     if tick_date > bar_end_time: break
                #     else: continue

                tick = df_ticks['close'].iat[i_tk]
                prev_tick = df_ticks['close'].iat[i_tk - 1]
                for idx in range(len(interval_segments)):
                    row = interval_segments[idx]
                    prev_row = interval_segments[idx - 1]
                    if (tick >= prev_row) and (tick <= row):
                        normal['profile'][bar_index][idx] = normal['profile'][bar_index][idx] + 1

                        if tick > prev_tick:
                            buy_sell['buy_profile'][bar_index][idx] = buy_sell['buy_profile'][bar_index][idx] + 1
                        elif tick < prev_tick:
                            buy_sell['sell_profile'][bar_index][idx] = buy_sell['sell_profile'][bar_index][idx] + 1
                        elif tick == prev_tick:
                            buy_sell['buy_profile'][bar_index][idx] = buy_sell['buy_profile'][bar_index][idx] + 1
                            buy_sell['sell_profile'][bar_index][idx] = buy_sell['sell_profile'][bar_index][idx] + 1

                        prev_delta_i = sum(delta['profile'][bar_index])

                        buy = buy_sell['buy_profile'][bar_index][idx]
                        sell = buy_sell['sell_profile'][bar_index][idx]
                        delta['profile'][bar_index][idx] = delta['profile'][bar_index][idx] + (buy - sell)

                        current_delta = sum(delta['profile'][bar_index])
                        if prev_delta_i > current_delta:
                            delta['min_delta'] = prev_delta_i
                        elif prev_delta_i < current_delta:
                            delta['max_delta'] = prev_delta_i

        vp_ticks(start_time, end_time)

        normal['value'] = sum(normal['profile'][0])

        buy_sell['value_buy'] = sum(buy_sell['buy_profile'][0])
        buy_sell['value_sell'] = sum(buy_sell['sell_profile'][0])

        buy_sell['value_sum'] = buy_sell['value_buy'] + buy_sell['value_sell']
        buy_sell['value_subtract'] = buy_sell['value_buy'] - buy_sell['value_sell']

        delta['value'] = sum(delta['profile'][0])

        if buy_sell['value_buy'] != 0 and buy_sell['value_sell'] != 0:
            buy_sell['value_divide'] = (buy_sell['value_buy'] / buy_sell['value_sell'])

        # Remove first row from 'profile' since it's the variable assign value (0.0)
        if buy_sell['buy_profile'][0]:
            buy_sell['buy_profile'][0].pop(0)
            buy_sell['sell_profile'][0].pop(0)
        for lst in [normal, delta]:
            if lst['profile'][0]:
                lst['profile'][0].pop(0)

        if not self._with_plotly_columns:
            return [normal, buy_sell, delta]

        # for Plotly = 'X' integer index
        # The idea is to avoid using pandas.apply method in self.plot(),
        # since we need to get the proportion of each row in order to define
        # the X position of histogram/number of each bar.
        def add_plotly_row(dict_name: str, max_side: float, side_center: float, chart: str):
            plot = {
                f'plotly_normal_ohlc_histogram': [ [] ],
                f'plotly_normal_candle_histogram': [[]],

                f'plotly_normal_ohlc_numbers': [ [] ],
                f'plotly_normal_candle_numbers': [ [] ],
                f'plotly_delta_ohlc_numbers': [ [] ],
                f'plotly_delta_candle_numbers': [ [] ],

                f'plotly_delta_buy_ohlc_histogram': [ [] ],
                f'plotly_delta_sell_ohlc_histogram': [ [] ],

                f'plotly_delta_buy_candle_histogram': [ [] ],
                f'plotly_delta_sell_candle_histogram': [ [] ],
            }
            plot_profile = {
                f'plotly_delta_profile_ohlc_numbers': [ [] ],
                f'plotly_delta_profile_candle_numbers': [ [] ],

                f'plotly_delta_buy_profile_ohlc_histogram': [ [] ],
                f'plotly_delta_sell_profile_ohlc_histogram': [ [] ],

                f'plotly_delta_buy_profile_candle_histogram': [ [] ],
                f'plotly_delta_sell_profile_candle_histogram': [ [] ],
            }
            bar_index = df_ohlc_row['plotly_int_index'].iat[0]
            side_center_left = bar_index - side_center
            side_center_right = bar_index + side_center
            max_side_profile = max_side * 2

            if dict_name == 'normal':
                max_value = max(normal['profile'][0] if normal['profile'][0] else [1])
                for value in normal['profile'][0]:
                    first = value * max_side_profile
                    result = (first / max_value) if (first != 0) else 0
                    plot[f'plotly_normal_{chart}_histogram'][0].append(result)
                    plot[f'plotly_normal_{chart}_numbers'][0].append(bar_index)

                # Remove keys with empty list
                # to avoid .update() overwriting behavior on the 2º call of this function
                filter_plot = {key: value for key,value in plot.items() if value[0]}
                normal.update(filter_plot)
            else:
                max_list = [value for value in delta['profile'][0] if value > 0]
                buy_max_value = max(max_list if max_list else [1])

                min_list = [value for value in delta['profile'][0] if value < 0]
                sell_max_value = min(min_list if min_list else [1])
                for value in delta['profile'][0]:
                    # buy - division by zero doesn't happen... yet!
                    first = (value if value > 0 else 0) * max_side
                    result = first / buy_max_value
                    plot[f'plotly_delta_buy_{chart}_histogram'][0].append(result)

                    first = (value if value > 0 else 0) * max_side_profile
                    result_profile = first / (buy_max_value
                        if buy_max_value > abs(sell_max_value) else abs(sell_max_value))
                    plot_profile[f'plotly_delta_buy_profile_{chart}_histogram'][0].append(result_profile)

                    # sell - division by zero doesn't happen... yet!
                    first = (-value if value < 0 else 0) * max_side
                    result = first / sell_max_value
                    plot[f'plotly_delta_sell_{chart}_histogram'][0].append(result)

                    first = (-value if value < 0 else 0) * max_side_profile
                    result_profile = first / (buy_max_value
                        if buy_max_value > abs(sell_max_value) else abs(sell_max_value))
                    plot_profile[f'plotly_delta_sell_profile_{chart}_histogram'][0].append(abs(result_profile))

                    # numbers
                    plot[f'plotly_delta_{chart}_numbers'][0].append(side_center_right if value > 0 else side_center_left)
                    plot_profile[f'plotly_delta_profile_{chart}_numbers'][0].append(bar_index)

                filter_plot = {key: value for key,value in plot.items() if value[0]}
                delta.update(filter_plot)
                filter_plot = {key: value for key,value in plot_profile.items() if value[0]}
                delta.update(filter_plot)

        def add_plotly_row_bs(max_side: float, side_center: float, chart: str):
            plot = {
                f'plotly_buy_ohlc_histogram': [ [] ],
                f'plotly_buy_ohlc_numbers': [ [] ],
                f'plotly_buy_candle_histogram': [ [] ],
                f'plotly_buy_candle_numbers': [ [] ],

                f'plotly_sell_ohlc_histogram': [ [] ],
                f'plotly_sell_ohlc_numbers': [ [] ],
                f'plotly_sell_candle_histogram': [ [] ],
                f'plotly_sell_candle_numbers': [ [] ],
            }
            plot_profile = {
                f'plotly_buy_profile_ohlc_histogram': [ [] ],
                f'plotly_buy_profile_candle_histogram': [ [] ],
                f'plotly_buy_profile_ohlc_numbers': [ [] ],
                f'plotly_buy_profile_candle_numbers': [ [] ],

                f'plotly_sell_profile_ohlc_histogram': [ [] ],
                f'plotly_sell_profile_candle_histogram': [ [] ],
                f'plotly_sell_profile_ohlc_numbers': [ [] ],
                f'plotly_sell_profile_candle_numbers': [ [] ],
            }
            bar_index = df_ohlc_row['plotly_int_index'].iat[0]
            side_center_left = bar_index - side_center
            side_center_right = bar_index + side_center
            max_side_profile = max_side * 2

            buy_max_value = max(buy_sell['buy_profile'][0] if buy_sell['buy_profile'][0] else [1])
            for value in buy_sell['buy_profile'][0]:
                first = value * max_side
                result = (first / buy_max_value) if (first != 0) else 0
                plot[f'plotly_buy_{chart}_histogram'][0].append(result)

                first = value * (max_side_profile / 2)
                result = (first / buy_max_value) if (first != 0) else 0
                plot_profile[f'plotly_buy_profile_{chart}_histogram'][0].append(result)

                plot[f'plotly_buy_{chart}_numbers'][0].append(side_center_right)
                # center right = left align
                plot_profile[f'plotly_buy_profile_{chart}_numbers'][0].append(bar_index + 0.1)

            sell_max_value = max(buy_sell['sell_profile'][0] if buy_sell['sell_profile'][0] else [1])
            for value in buy_sell['sell_profile'][0]:
                first = value * (-max_side)
                result = (first / sell_max_value) if (first != 0) else 0

                plot[f'plotly_sell_{chart}_histogram'][0].append(result)

                first = value * (-max_side_profile)
                result = (first / sell_max_value) if (first != 0) else 0
                plot_profile[f'plotly_sell_profile_{chart}_histogram'][0].append(abs(result))

                plot[f'plotly_sell_{chart}_numbers'][0].append(side_center_left)
                # center right = right align
                if chart == 'candle':
                    plot_profile[f'plotly_sell_profile_{chart}_numbers'][0].append(bar_index + 0.2)
                else:
                    plot_profile[f'plotly_sell_profile_{chart}_numbers'][0].append(bar_index + 0.25)

            filter_plot = {key: value for key,value in plot.items() if value[0]}
            buy_sell.update(filter_plot)
            filter_plot = {key: value for key,value in plot_profile.items() if value[0]}
            buy_sell.update(filter_plot)

        for name in ['normal', 'delta', 'buy_sell']:
            # OHLC chart (max_side) = 0.3 / each side (center) = 0.15
            # Candle chart (max_side) = 0.245 / each side (center) = 0.125
            add_plotly_row(name, 0.3, 0.15, 'ohlc')
            add_plotly_row(name, 0.245, 0.125, 'candle')
            if name == 'buy_sell':
                add_plotly_row_bs(0.3, 0.15, 'ohlc')
                add_plotly_row_bs(0.245, 0.125, 'candle')

        return [normal, buy_sell, delta]