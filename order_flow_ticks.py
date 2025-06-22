"""
Order Flow Ticks
=====
Python version of Order Flow Ticks (v2.0) developed for cTrader Trading Platform

Improvements:
    - Parallel processing for each bar
Additional Features:
    - Buy vs Sell = Divide total values of each side
    - Delta = Min/Max Delta
Python/C# author:
    - srlcarlg
"""

from copy import deepcopy
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go


class OrderFlowTicks:
    def __init__(self, df_ohlc: pd.DataFrame, df_ticks: pd.DataFrame, row_height: float, is_open_time: bool = True,
                 with_plotly_columns: bool = True, ema_period: int = 5):
        """
        Create Order Flow Ticks data to any OHLC chart! \n
        (Candles, Renko, Range, Ticks) \n
        Backtest version.

        Usage
        ------
        >>> from order_flow_ticks import OrderFlowTicks
        >>> odft = OrderFlowTicks(df_ohlc, df_ticks, row_height)
        >>> # plot with plotly
        >>> odft.plot(mode='delta')
        >>> # get df_ohlc with all modes
        >>> df = odft.all()
        >>> # or get specific mode
        >>> odft.normal(), odft.buy_sell(), odft.delta()

        Parameters
        ----------
        df_ohlc : dataframe
            * index/datetime, open, high, low, close \n
            * "datetime": If is not present, the index will be used.
        df_ticks : dataframe
            * It should have:
            * datetime index or 'datetime' column
            * 'close' column (ticks price)
        row_height : float
            Cannot be less than or equal to 0.00000...
        is_open_time : bool
            Specify if the index/datetime of df_ohlc is the OpenTime or CloseTime of each bar
        with_plotly_columns : bool
            Return 'plotly_[...]' columns used in self.plot(), if False it will not be possible to plot.
        ema_period : int
            EMA period for Strength Filter
        """
        if 'datetime' not in df_ohlc.columns:
            df_ohlc["datetime"] = df_ohlc.index
        if 'datetime' not in df_ticks.columns:
            df_ticks["datetime"] = df_ticks.index
        _expected = ['open', 'high', 'low', 'close']
        for column in _expected:
            if column not in df_ohlc.columns:
                raise ValueError(f"'{column}' column from the expected {_expected} doesn't exist!")

        self._row_height = row_height
        self._df_ticks = df_ticks
        self._is_open_time = is_open_time
        self._with_plotly_columns = with_plotly_columns
        self._ema_period = ema_period

        if is_open_time:
            df_ohlc['end_time'] = df_ohlc['datetime'].shift(-1)
        else:
            df_ohlc['start_time'] = df_ohlc['datetime'].shift(1)

        df_ohlc.dropna(inplace=True)

        # For plotly
        if with_plotly_columns:
            df_ohlc['plotly_int_index'] = range(len(df_ohlc))

        def parallel_process_dataframes(df):
            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                results = pool.map(self._create_odf, df.to_dict(orient='records'))
            return results

        profiles_each_bar = parallel_process_dataframes(df_ohlc)

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

        normal_df = self._strength_filter(normal_df, "normal_value")
        buy_sell_df = self._strength_filter(buy_sell_df, "bs_sum")
        buy_sell_df = self._strength_filter(buy_sell_df, "bs_subtract")
        buy_sell_df = self._strength_filter(buy_sell_df, "bs_divide")
        delta_df = self._strength_filter(delta_df, "delta")

        self._df_ohlcv = df_ohlc
        self._normal_df = normal_df
        self._buy_sell_df = buy_sell_df
        self._delta_df = delta_df

    def normal(self):
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

    def buy_sell(self):
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

    def delta(self):
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

    def all(self):
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

    def plot(self, iloc_value: int=15, mode: str = 'delta', view: str = 'profile',
             chart: str = 'ohlc', renderer: str = 'default',
             width: int = 1200, height: int = 800):
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

        df = self.normal() if mode == 'normal' else self.buy_sell() if mode == 'buy_sell' else self.delta()
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

        # Histograms
        # 'base=index' makes the trick of divided/profiles views
        # Profile view needs (max_side * 2 in self._create_odf) because
        # we're moving the initial value to the left max_side (-0.245 or 0.3)
        for index in range(len(df)):
            step = 0.245 if chart == 'candle' else 0.3
            base_index = index if view == 'divided' else index - step
            if mode == 'normal':
                x_column = f'{prefix}_{mode}_{chart}_histogram'
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

                fig.add_trace(
                    go.Bar(y=df['profiles_prices'].iat[index], x=df[x_sell_column].iat[index], marker=dict(
                        color=' crimson',
                        opacity=0.3
                    ),
                   orientation='h', base=base_index), row=1, col=1)

        # Numbers
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

        for index in range(len(df)):
            if mode == 'normal':
                x_column = f'{prefix}_{mode}_{chart}_numbers'
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

        fig.update_layout(
            title=f'Order Flow Ticks: {mode}/{view}',
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

    def _strength_filter(self, df, column_name: str):
        ema_period = self._ema_period
        # EMA
        values = df[column_name].copy()
        values.iloc[:ema_period - 1] = np.NaN
        values.iloc[ema_period - 1] = values.iloc[0:ema_period].mean()
        df[f'ema_{column_name}'] = values.ewm(span=ema_period).mean()
        # Filter
        df[f'filter_{column_name}'] = df[column_name] / df[f'ema_{column_name}']
        df[f'strength_{column_name}'] = np.where(df[f'filter_{column_name}'] > 2, 4,
                                 np.where(df[f'filter_{column_name}'] > 1.5, 3,
                                 np.where(df[f'filter_{column_name}'] > 1, 2,
                                 np.where(df[f'filter_{column_name}'] > 0.5, 1, 0))))

        df.drop(columns=[f'ema_{column_name}', f'filter_{column_name}'], inplace=True)
        return df

    def _create_odf(self, df_ohlc):
        df_ohlc_row = pd.DataFrame([df_ohlc])

        interval_lowest = df_ohlc_row['low'].iat[0]
        interval_highest = df_ohlc_row['high'].iat[0]
        interval_open = df_ohlc_row['open'].iat[0]
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

        # Volume Profile Ticks
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