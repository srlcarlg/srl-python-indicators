"""
Order Flow Ticks
=====
Python version of Order Flow Ticks (v2.0) developed for cTrader Trading Platform


Features from Order Flow Ticks + Aggregated (C#):
- Bubbles Chart
    - [Delta / Subtract / Change] Sources
    - HeatMap Coloring
    - Momentum Coloring [fading, positive/negative]
    - Ultra Levels [positive/negative]
- Tick Spike Filter
    - Spike Levels
    - Spike Chart
    - HeatMap Coloring
    - Positive/Negative Coloring
- Misc
    - Custom MAs
    - Add Strength Filter to Subtract Delta

Additional Features => that will be implemented to C# version... sometime next year (2026)
    - ['sum_delta' / 'delta_bs_sum'] sources to "Spike Filter" and "Bubbles Chart"
    - FilterType.[SoftMax_Power / L2Norm / MinMax] filters alternatives to Bubbles Chart
    - [L1Norm, SoftMax_Power] filters alternatives to Tick Spike
    - FilterRatio.[Fixed / Percentage] to both

Improvements:
    - Parallel processing for each bar.
    - Numpy arrays, where possible.
    - Vectorized operations, almost everything.

Python/C# author:
    - srlcarlg
"""
from copy import deepcopy
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from custom_mas import get_ma, get_stddev
from models_utils.odf_models import *
from models_utils.odf_utils import touches_bubbles, touches_spikes, rolling_percentile, l1norm, l1norm_profile, l2norm, \
    power_softmax, power_softmax_profile


class OrderFlowTicks:
    def __init__(self, df_ohlc: pd.DataFrame, df_ticks: pd.DataFrame, row_height: float,
                 strength_filter: StrengthFilter | None = None,
                 spike_filter: SpikeFilter | None = None,
                 bubbles_filter: BubblesFilter | None = None,
                 is_open_time: bool = True,
                 with_plotly_columns: bool = True):
        """
        Create "Order Flow Ticks" from any OHLC chart! \n
        (Candles, Renko, Range, Ticks) \n
        Backtest version.

        Usage
        -----
        >>> from order_flow_ticks import OrderFlowTicks
        >>> odft = OrderFlowTicks(df_ohlc, df_ticks, row_height)

        >>> # plot with plotly
        >>> odft.plot(mode='delta')

        >>> # get df_ohlc with all modes
        >>> df = odft.all()

        >>> # or get specific mode
        >>> odft.normal(), odft.buy_sell(), odft.delta()

        >>> # change parameters for filters
        >>> from models_utils.odf_models import *
        >>> params_strength = StrengthFilter(MAType.Exponential, 5, 1.5, 2)
        >>> params_spike = SpikeFilter(SpikeFilterType.MA, FilterRatio.Percentage, MAType.Exponential, ...)
        >>> params_bubbles = BubblesFilter(FilterType.MA, FilterRatio.Percentage, MAType.Exponential, ...)

        >>> odft = OrderFlowTicks(df_ohlc, df_ticks, row_height, params_strength, params_spike, params_bubbles)

        >>> # or get multiples df_ohlc dataframes from different parameters
        >>> params_strength = StrengthFilter(MAType.Weighted, 5, 2, 4)
        >>> odft.normal(params_strength), odft.buy_sell(params_strength), odft.delta(params_strength)

        >>> # delta only
        >>> params_spike = SpikeFilter(SpikeFilterType.L1Norm, FilterRatio.Percentage, MAType.Weighted, ...)
        >>> params_spike.levels(max_count=2)

        >>> params_bubbles = BubblesFilter(FilterType.SoftMax_Power, FilterRatio.Percentage, MAType.Triangular, ...)
        >>> params_bubbles.levels(UltraBubblesLevel.High_Low, UltraBubblesBreak.Close_Only, 2)

        >>> odft.delta(params_strength, params_spike, params_bubbles)


        Parameters
        ----------
        df_ohlc : dataframe
            * index/datetime, open, high, low, close
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
        self._ticks_datetime = df_ticks['datetime'].to_numpy()
        self._ticks_close = df_ticks['close'].to_numpy()

        self._strength_filter = strength_filter if isinstance(strength_filter, StrengthFilter) else StrengthFilter()
        self._spike_filter = spike_filter if isinstance(spike_filter, SpikeFilter) else SpikeFilter()
        self._bubbles_filter = bubbles_filter if isinstance(bubbles_filter, BubblesFilter) else BubblesFilter()

        self._is_open_time = is_open_time
        self._with_plotly_columns = with_plotly_columns

        if is_open_time:
            df_ohlc['end_time'] = df_ohlc['datetime'].shift(-1)
        else:
            df_ohlc['start_time'] = df_ohlc['datetime'].shift(1)

        # drop NaN from .shift() operation
        df_ohlc.dropna(inplace=True)

        # For plotly
        if with_plotly_columns:
            df_ohlc['plotly_int_index'] = range(len(df_ohlc))

        def parallel_process_dataframes(df):
            num_processes = cpu_count()
            with Pool(processes=num_processes) as pool:
                results = pool.map(self._create, df.to_dict(orient='records'))
            return results

        profiles_each_bar = parallel_process_dataframes(df_ohlc)

        normal_bars = (lst[0] for lst in profiles_each_bar)
        buy_sell_bars = (lst[1] for lst in profiles_each_bar)
        delta_bars = (lst[2] for lst in profiles_each_bar)

        normal_df = pd.DataFrame.from_records(normal_bars)
        buy_sell_df = pd.DataFrame.from_records(buy_sell_bars)
        delta_df = pd.DataFrame.from_records(delta_bars)
        for _df in [normal_df, buy_sell_df, delta_df]:
            _df.index = _df['datetime']
            _df.drop(columns=['datetime'], inplace=True)

        delta_df['delta_change'] = delta_df['delta'] + delta_df['delta'].shift(1)
        # fill NaN from .shift() operation
        delta_df['delta_change'].fillna(delta_df['delta'], inplace=True)

        self._df_ohlcv = df_ohlc
        self._normal_df = normal_df
        self._buy_sell_df = buy_sell_df
        self._delta_df = delta_df

        # creating new columns by vectorized operations to [spike, bubbles]-sources
        from warnings import simplefilter
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    def normal(self, strength_filter: StrengthFilter | None = None):
        """
        Get df_ohlc with 'Normal' mode.
        """
        normal_df = self._normal_df.copy()
        self._strength(normal_df, "normal_value", strength_filter)

        df = pd.concat([self._df_ohlcv, normal_df], axis=1)
        df.index = df['datetime']
        return df

    def buy_sell(self, strength_filter: StrengthFilter | None = None):
        """
        Get df_ohlc with 'Buy_Sell' mode.
        """
        buy_sell_df = self._buy_sell_df.copy()
        self._strength(buy_sell_df, "bs_sum", strength_filter)
        self._strength(buy_sell_df, "bs_subtract", strength_filter)

        df = pd.concat([self._df_ohlcv, buy_sell_df], axis=1)
        df.index = df['datetime']
        return df

    def delta(self, strength_filter: StrengthFilter | None = None, spike_filter: SpikeFilter | None = None,
              bubbles_filter: BubblesFilter | None = None,):
        """
        Get df_ohlc with 'Delta' mode.
        """
        delta_df = self._delta_df.copy()
        self._strength(delta_df, "delta", strength_filter)
        self._strength(delta_df, "subtract_delta", strength_filter)
        self._tick_spike(delta_df, spike_filter)
        self._bubbles_chart(delta_df, self._df_ohlcv, bubbles_filter)

        df = pd.concat([self._df_ohlcv, delta_df], axis=1)
        df.index = df['datetime']

        self._spike_levels_count(df, spike_filter)
        self._bubbles_levels_count(df, bubbles_filter)
        return df

    def all(self):
        """
        Get df_ohlc with all modes (normal, buy_sell, delta)
        """
        df = pd.concat([self._df_ohlcv, self._normal_df, self._buy_sell_df, self._delta_df], axis=1)
        # Drop duplicated columns (we have 3 profile_prices, keep 1)
        df = df.loc[: , ~df.columns.duplicated()]
        df.index = df['datetime']
        return df

    def plot(self, iloc_value: int | list = 15, mode: str = 'delta', view: str = 'profile',
             spike_plot: SpikePlot | None = None,
             chart: str = 'ohlc', renderer: str = 'default',
             width: int = 1200, height: int = 800):
        """

        Parameters
        ----------
        df: pd.DataFrame
            If None, the dataframe from [normal, buy_sell, delta] methods of the current ODF instance will be used
        iloc_value : int
            First nº rows to be plotted
        mode : str
            'normal', 'buy_sell', 'delta'
        view : str
            'divided' or 'profile'
        spike_plot : SpikePlot
            Spike Filter parameters
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
        _views = ['profile', 'divided']
        _charts = ['candle', 'ohlc']

        input_values = [mode, view, chart]
        input_validation = [_profiles, _views, _charts]
        for value, validation in zip(input_values, input_validation):
            if value not in validation:
                raise ValueError(f"Only {validation} options are valid.")

        k = spike_plot if isinstance(spike_plot, SpikePlot) else SpikePlot()

        df = self.normal() if mode == 'normal' else \
             self.buy_sell() if mode == 'buy_sell' else \
             self.delta()

        if type(iloc_value) is int:
            df = df.iloc[:iloc_value]
        else:
            df = df.iloc[iloc_value[0]:iloc_value[1]]
            df[f'plotly_int_index'] = range(len(df))

        df_len = len(df)

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

        # TODO np.arrays
        # Histograms
        # 'base=index' makes the trick of divided/profiles views
        # Profile view needs (max_side * 2 in self._create) because
        # we're moving the initial value to the left max_side (-0.245 or 0.3)
        for index in range(df_len):
            step = 0.245 if chart == 'candle' else 0.3
            base_index = index if view == 'divided' else index - step

            y_prices = df['profile_prices'].iat[index]
            len_prices = len(y_prices)

            if mode == 'normal':
                x_column = f'{prefix}_{mode}_{chart}_histogram'
                fig.add_trace(
                    go.Bar(y=y_prices, x=df[x_column].iat[index],
                           orientation='h', base=index - step,
                           marker=dict(
                               color='#00BFFF',
                               opacity = 0.4
                           )), row=1, col=1)
            else:
                # Spike Chart
                if k.spike_chart and mode == 'delta':
                    spike_prefix = f'{prefix}_spike_chart_{k.spike_source}'
                    coloring_column = f'{spike_prefix}_{k.spike_chart_coloring}_color'

                    base_index = index - 0.4
                    fig.add_trace(
                        go.Bar(y=y_prices, x=[0.8 for _ in range(len_prices - 1)],
                               orientation='h', base=base_index,
                               marker=dict(
                                   color=df[coloring_column].iat[index],
                                   opacity=0.4
                               )), row=1, col=1)
                else:
                    chart_suffix = f'{chart}_histogram' \
                                   if view == 'divided' else \
                                   f'profile_{chart}_histogram'

                    # Buy
                    buy_middle = 'delta_buy' if mode == 'delta' else 'buy'
                    x_buy_column = f'{prefix}_{buy_middle}_{chart_suffix}'

                    color = 'deepskyblue'
                    if mode == 'delta':
                        color = df[f'{prefix}_spike_{k.spike_source}_buy_color'].iat[index] if k.spike else 'deepskyblue'

                    fig.add_trace(
                        go.Bar(y=y_prices, x=df[x_buy_column].iat[index], marker=dict(
                            color=color,
                            opacity=0.3 if view == 'divided' else 0.4
                        ),
                        orientation='h', base=base_index), row=1, col=1)

                    # Sell
                    sell_middle = 'delta_sell' if mode == 'delta' else 'sell'
                    x_sell_column = f'{prefix}_{sell_middle}_{chart_suffix}'

                    color = 'crimson'
                    if mode == 'delta':
                        color = df[f'{prefix}_spike_{k.spike_source}_sell_color'].iat[index] if k.spike else 'crimson'

                    fig.add_trace(
                        go.Bar(y=y_prices, x=df[x_sell_column].iat[index], marker=dict(
                            color=color,
                            opacity=0.3
                        ),
                       orientation='h', base=base_index), row=1, col=1)

                # Spike Levels
                if k.spike_levels and mode == 'delta':
                    lvl_prefix = 'spike_levels'
                    levels = df[f'{lvl_prefix}_price_{k.spike_source}'].iat[index]

                    len_lvl = len(levels)
                    if len_lvl == 0:
                        continue

                    break_at = abs(index - df[f'{lvl_prefix}_break_at_{k.spike_source}'].iat[index])
                    base_index = index + step

                    # remove the y2 of all levels
                    # the Y size is automatically set by plotly
                    levels = np.delete(levels, 1, axis=1)

                    levels_coloring = df[f'{prefix}_{lvl_prefix}_{k.spike_source}_{k.spike_levels_coloring}_color'].iat[index]

                    # 2d array to 1d array is not suitable due to:
                    # plotly automatic space between y values
                    for idx in range(len_lvl):
                        fig.add_trace(
                            go.Bar(y=levels[idx], x=[break_at], marker=dict(
                                color=levels_coloring[idx],
                                opacity=0.3,
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

        for index in range(df_len):
            y_prices = df['profile_prices'].iat[index]

            if mode in ['normal', 'delta']:
                chart_suffix = f'{chart}_numbers' if mode != 'delta' else \
                               f'profile_{chart}_numbers' if view == 'profile' else \
                               f'{chart}_numbers'

                x_column = f'{prefix}_{mode}_{chart_suffix}'
                z_text = df[f'{mode}_profile'].iat[index]

                # workaround to display without recalculating everything
                x_values = df[x_column].iat[index] if type(iloc_value) is int else \
                           [v - iloc_value[0] for v in df[x_column].iat[index]]
                fig.add_trace(
                    go.Heatmap(
                        x=x_values,
                        y=y_prices,
                        z=z_text,
                        text=z_text,
                        colorscale=color,
                        showscale=False, # remove numbers from show_legend=False column
                        texttemplate="%{text}",
                        textfont={
                            "size": 11,
                            "color": 'black',
                            "family": "Courier New"},
                    ), row=1, col=1)

                if mode == 'delta' and k.spike_strength:
                    z_text = df[f'spike_profile_{k.spike_source}'].iat[index]
                    x_axis = [arr + 0.5 for arr in x_values]
                    fig.add_trace(
                        go.Heatmap(
                            x=x_axis,
                            y=y_prices,
                            z=z_text,
                            text=z_text,
                            colorscale=color,
                            showscale=False, # remove numbers from show_legend=False column
                            # texttemplate="%{text}",
                            texttemplate="<= %{text}",
                            textfont={
                                "size": 11,
                                "color": 'black',
                                "family": "Courier New"},
                        ), row=1, col=1)
            else:
                chart_suffix = f'profile_{chart}_numbers' if view == 'profile' else f'{chart}_numbers'

                # Buy
                x_buy_column = f'{prefix}_buy_{chart_suffix}'
                z_text = df['buy_profile'].iat[index]

                x_buy_values = df[x_buy_column].iat[index] if type(iloc_value) is int else \
                               [v - iloc_value[0] for v in df[x_buy_column].iat[index]]
                fig.add_trace(
                    go.Heatmap(
                        x=x_buy_values,
                        y=y_prices,
                        z=z_text,
                        text=z_text,
                        colorscale=color,
                        showscale=False, # remove numbers from show_legend=False column
                        texttemplate="%{text}",
                        textfont={
                            "size": 11,
                            "color": 'black',
                            "family": "Courier New"},
                    ), row=1, col=1)

                # Sell
                x_sell_column = f'{prefix}_sell_{chart_suffix}'
                z_text = df['sell_profile'].iat[index]

                x_sell_values = df[x_sell_column].iat[index] if type(iloc_value) is int else \
                               [v - iloc_value[0] for v in df[x_sell_column].iat[index]]
                fig.add_trace(
                    go.Heatmap(
                        x=x_sell_values,
                        y=y_prices,
                        z=z_text,
                        text=z_text,
                        colorscale=color,
                        showscale=False, # remove numbers from show_legend=False column
                        texttemplate="%{text}",
                        textfont={
                            "size": 11,
                            "color": 'black',
                            "family": "Courier New"},
                    ), row=1, col=1)

        # Delta Change = above the candle
        if mode == 'delta':
            df['high_y'] = df['high'] + (self._row_height * 1)
            fig.add_trace(
                go.Scatter(
                    x=df['plotly_int_index'], y=df['high_y'], text=df['delta_change'],
                    textposition='middle center',
                    textfont=dict(size=11, color='blue'),
                    mode='text'
                ), row=1, col=1)

        # Mode value = below the candle
        df['low_y'] = df['low'] - (self._row_height * 2)
        column_below = 'normal_value' if mode == 'normal' else \
                       'bs_subtract' if mode == 'buy_sell' else \
                       'delta'
        fig.add_trace(
            go.Scatter(
                x=df['plotly_int_index'], y=df['low_y'], text=df[column_below], textposition='middle center',
                textfont=dict(size=11, color='green'),
                mode='text'
            ), row=1, col=1)

        spike_info = ""
        if mode == 'delta' and (k.spike_chart or k.spike):
            mode_column = {
                'delta': 'delta',
                'sum': 'sum_delta',
                'bs_sum': 'delta_bs_sum',
            }
            f = self._spike_filter
            ratio_text = f'/ {f.p_period} period' if f.filter_ratio == FilterRatio.Percentage else ""

            spike_info = f"Source: {mode_column[k.spike_source]} <br>" \
                         f"Filter: {f.filter_type.name} / {f.ma_period} period <br>" \
                         f"Ratio: {f.filter_ratio.name} {ratio_text}"

        fig.update_layout(
            title=f'Order Flow Ticks: {mode}/{view} <br>' + spike_info,
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

    def plot_bubbles(self, iloc_value: int | list = 15, delta_source: str = 'delta', coloring: str = 'heatmap',
                     strength: bool = False, levels: bool = False, levels_coloring: str = 'plusminus',
                     renderer: str = 'default',
                     width: int = 1200, height: int = 800):
        _sources = ['delta', 'subtract', 'sum', 'change', 'bs_sum']
        if delta_source not in _sources:
            raise ValueError(f"Only {_sources} options are valid.")
        _colors = ['heatmap', 'fading', 'plusminus']
        if coloring not in _colors:
            raise ValueError(f"Only {_colors} options are valid.")

        df = self.delta()
        if type(iloc_value) is int:
            df = df.iloc[:iloc_value]
        else:
            df = df.iloc[iloc_value[0]:iloc_value[1]]
            df[f'plotly_int_index'] = range(len(df))

        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.0)
        prefix = 'plotly'
        x_values = df[f'{prefix}_int_index']
        y_values = df['close']

        # close line
        fig.add_trace(go.Scatter(x=x_values,
                                 y=y_values,
                                 mode='lines',
                                 opacity=1), row=1, col=1)
        # delta value
        mode_column = {
            'delta' : 'delta',
            'subtract' : 'subtract_delta',
            'sum': 'sum_delta',
            'change' : 'delta_change',
            'bs_sum': 'delta_bs_sum',
        }
        name = mode_column[delta_source]
        fig.add_trace(go.Scatter(x=x_values,
                                 y=y_values,
                                 mode='text',
                                 text=df[name],
                                 textfont=dict(size=11, color='black'),
                                 opacity=1), row=1, col=1)
        # strength value
        if strength:
            df['close_y'] = y_values - (self._row_height / 4)
            fig.add_trace(go.Scatter(x=x_values,
                                     y=df['close_y'],
                                     mode='text',
                                     text=df[f'bubbles_{delta_source}_strength'],
                                     textposition='bottom center',
                                     textfont=dict(size=9, color='blue'),
                                     opacity=1), row=1, col=1)
        # bubbles
        bubbles_prefix = f'{prefix}_bubbles'
        fig.add_trace(go.Scatter(x=x_values,
                                 y=y_values,
                                 mode='markers',
                                 marker=dict(
                                     color=df[f'{bubbles_prefix}_{delta_source}_{coloring}_color'],
                                     size=df[f'{bubbles_prefix}_{delta_source}_size'],
                                 ),
                                 opacity=0.5), row=1, col=1)
        # ultra levels
        if levels:
            lvl_prefix = f'bubbles_levels_{delta_source}'
            for index in range(len(df)):
                # TODO np.arrays
                level = df[f'{lvl_prefix}_high_to_low'].iat[index]
                if level == 0:
                    continue

                break_at = abs(index - df[f'{lvl_prefix}_break_at'].iat[index])
                lvl_color = df[f'{bubbles_prefix}_{delta_source}_plusminus_color'].iat[index]  \
                            if levels_coloring == 'plusminus' else levels_coloring

                for i in range(2):
                    fig.add_trace(
                        go.Bar(y=[level[i]], x=[break_at], marker=dict(
                            color=lvl_color,
                            opacity=0.3
                        ),
                        orientation='h', base=index), row=1, col=1)

        b = self._bubbles_filter
        ratio_text = f'/ {b.p_period} period' if b.filter_ratio == FilterRatio.Percentage else ""
        fig.update_layout(
            title=f"Order Flow Ticks: {mode_column[delta_source]} <br>"
                  f"Filter: {b.filter_type.name} / {b.ma_period} period <br>"
                  f"Ratio: {b.filter_ratio.name} {ratio_text}",
            height=800,
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

    def _strength(self, df, column_name: str, s_params: StrengthFilter | None = None):
        s = s_params if isinstance(s_params, StrengthFilter) else self._strength_filter

        df[f'abs_{column_name}'] = abs(df[column_name])
        df['strength_ma'] = get_ma(df[f'abs_{column_name}'].to_numpy(), s.ma_type, s.ma_period)

        df[f'strength_{column_name}_filter'] = df[column_name] / df[f'strength_ma']

        d = df[f'strength_{column_name}_filter']
        df[f'strength_{column_name}'] = np.where(d >= s.ratio_2, 2,
                                        np.where(d >= s.ratio_1, 1, 0))

        df.drop(columns=[f'abs_{column_name}', f'strength_ma'], inplace=True)
        return df

    def _tick_spike(self, df: pd.DataFrame, t_params: SpikeFilter | None = None):
        t = t_params if isinstance(t_params, SpikeFilter) else self._spike_filter

        _delta_name = ['delta', 'sum', 'bs_sum']
        _delta_columns = ['delta', 'sum_delta', 'delta_bs_sum']
        for name, column_name in zip(_delta_name, _delta_columns):

            df[f'{name}_abs'] = abs(df[column_name])
            b_prefix = f'spike_{name}'

            match t.filter_type:
                case SpikeFilterType.MA | SpikeFilterType.StdDev:
                    df[f'{b_prefix}_ma'] = get_ma(df[f'{name}_abs'].to_numpy(), t.ma_type, t.ma_period)
                    df[f'{b_prefix}_filter'] = df[f'{b_prefix}_ma']
                    if t.filter_type == SpikeFilterType.StdDev:
                        df[f'{b_prefix}_stddev'] = get_stddev(df[f'{name}_abs'].to_numpy(), df[f'{b_prefix}_ma'], t.ma_period)
                        df[f'{b_prefix}_filter'] = df[f'{b_prefix}_stddev']
                        df.drop(columns=[f'{b_prefix}_stddev'], inplace=True)
                    df.drop(columns=[f'{b_prefix}_ma'], inplace=True)
                case SpikeFilterType.L1Norm:
                    df[f'{b_prefix}_filter'] = df[f'{name}_abs'].rolling(t.ma_period).apply(l1norm, raw=True)
                    df[f'{b_prefix}_filter'] = df[f'{b_prefix}_filter'] * 100
                case SpikeFilterType.SoftMax_Power:
                    df[f'{b_prefix}_filter'] = df[f'{name}_abs'].rolling(t.ma_period).apply(power_softmax, raw=True)
                    df[f'{b_prefix}_filter'] = df[f'{b_prefix}_filter'] * 100

            # The code below simulates the "Segments loop" of OrderFlow C# method
            # vectorized for speeed hehehe

            abs_name = 'delta_profile_abs'
            df[abs_name] = [abs(arr) for arr in df['delta_profile']]

            if t.filter_type == SpikeFilterType.L1Norm:
                df[abs_name] = [l1norm_profile(arr) for arr in df[abs_name]]
                df[abs_name] = [np.round(arr * 100, 2) for arr in df[abs_name]]
            elif t.filter_type == SpikeFilterType.SoftMax_Power:
                df[abs_name] = [power_softmax_profile(arr) for arr in df[abs_name]]
                df[abs_name] = [np.round(arr * 100, 2) for arr in df[abs_name]]
            # The overall output seems to be the "same" for both methods, after the divide operation below.

            # divide each row-profile by normalized delta
            profile_name = f'spike_profile_{name}'
            df[profile_name] = df[abs_name] / df[f'{b_prefix}_filter']
            df[profile_name] = [np.round(arr, 2) for arr in df[profile_name]]

            if t.filter_ratio == FilterRatio.Percentage:
                """
                simple math, normalize the values to 0~1, just:
                    - calculate the sum of all elements absolute value
                    - divide each element by the sum
                    - aka L1 normalization
                added MA to get the values >= 100%, as well as, percentile-like behavior of bubbles chart.
                """
                pct_name = 'pct_filter'
                df[pct_name] = [sum(abs(arr)) for arr in df[profile_name]]
                df[pct_name] = df[pct_name] = get_ma(df[pct_name].to_numpy(), t.ma_type, t.p_period)

                df[profile_name] = df[profile_name] / df[pct_name]
                df[profile_name] = [np.round(arr * 100, 1) for arr in df[profile_name]]

            # Spike Chart => heatmap
            df[f'spike_chart_{name}'] = [np.where(arr < t.lowest, 0,
                                         np.where(arr < t.low, 1,
                                         np.where(arr < t.average, 2,
                                         np.where(arr < t.high, 3,
                                         np.where(arr >= t.ultra, 4, 4))))) for arr in df[profile_name]] \
                                    if t.filter_ratio == FilterRatio.Fixed else \
                                        [np.where(arr < t.lowest_pct, 0,
                                         np.where(arr < t.low_pct, 1,
                                         np.where(arr < t.average_pct, 2,
                                         np.where(arr < t.high_pct, 3,
                                         np.where(arr >= t.ultra_pct, 4, 4))))) for arr in df[profile_name]]

            df[f'spike_plusminus_{name}'] = [arr > 0 for arr in df['delta_profile']]
            # df[f'spike_plusminus_{name}'] = [arr.astype(int) for arr in df['spike_plusminus']]

            # Spike Levels
            """
            The actual spike prices.
            Some trick => group the segments by two like:
                [1, 2, 3, 4] => [(1,2), (2,3), (3,4)]
                to match with delta_profile length.
            """
            # [ [y1, y2], [y1, y2], [y1, y2], ...]
            df['price_util'] = [np.array([arr[:-1], arr[1:]]).T for arr in df['profile_prices']]
            # Since they have the same length, filter the 'spike_chart' values (>= average)
            # [ True, False, False, ...]
            df['mask_util'] = [arr >= 2 for arr in df[f'spike_chart_{name}']]
            # Then filter the segments 2d array using the boolean 1d array
            # [ [y1, y2], ...]
            df[f'spike_levels_price_{name}'] = [arr[mask] for arr, mask in zip(df['price_util'], df['mask_util'])]
            # get its strength
            # [ 4, ...]
            df[f'spike_levels_strength_{name}'] = [arr[mask] for arr, mask in zip(df[f'spike_chart_{name}'], df['mask_util'])]
            # and its sign too
            # [ True, ...]
            df[f'spike_levels_sign_is_plus_{name}'] = [arr[mask] > 0 for arr, mask in
                                               zip(df[f'spike_plusminus_{name}'], df['mask_util'])]

            if not self._with_plotly_columns:
                df.drop(columns=[f'{name}_abs', 'price_util', 'mask_util'], inplace=True)
                continue

            prefix = 'plotly_spike'

            # Spike
            # Instead of creating a new scatter for each spike, just color the histograms
            df[f'{prefix}_{name}_buy_color'] = [np.where(arr, 'gold', 'deepskyblue') for arr in df['mask_util']]
            df[f'{prefix}_{name}_sell_color'] = [np.where(arr, 'gold', 'crimson') for arr in df['mask_util']]

            # Spike Chart
            prefix_chart = f'{prefix}_chart_{name}'
            df[f'{prefix_chart}_heatmap_color'] = [np.where(arr == 0, 'rgb(0,255,255)',
                                                   np.where(arr == 1, '#FFFFFF',
                                                   np.where(arr == 2, 'rgb(255,255,0)',
                                                   np.where(arr == 3, 'rgb(255,192,0)',
                                                   np.where(arr == 4, 'rgb(255,0,0)', 'rgb(255,0,0)'))))) \
                                                   for arr in df[f'spike_chart_{name}']]

            is_plus = [arr.astype(int) for arr in df[f'spike_plusminus_{name}']]
            df[f'{prefix_chart}_plusminus_color'] = [np.where(arr == 1, 'deepskyblue', 'crimson') for arr in is_plus]

            # Spike Levels
            prefix_lvl = f'{prefix}_levels_{name}'
            df[f'{prefix_lvl}_heatmap_color'] = [np.where(arr == 0, 'rgb(0,255,255)',
                                                 np.where(arr == 1, '#FFFFFF',
                                                 np.where(arr == 2, 'rgb(255,255,0)',
                                                 np.where(arr == 3, 'rgb(255,192,0)',
                                                 np.where(arr == 4, 'rgb(255,0,0)', 'rgb(255,0,0)'))))) \
                                                 for arr in df[f'spike_levels_strength_{name}']]

            is_plus = [arr.astype(int) for arr in df[f'spike_levels_sign_is_plus_{name}']]
            df[f'{prefix_lvl}_plusminus_color'] = [np.where(arr == 1, 'deepskyblue', 'crimson') for arr in is_plus]

            df.drop(columns=[f'{name}_abs', 'price_util', 'mask_util'], inplace=True)

        return df

    def _bubbles_chart(self, df: pd.DataFrame, df_ohlc: pd.DataFrame, b_params: BubblesFilter | None = None):
        b = b_params if isinstance(b_params, BubblesFilter) else self._bubbles_filter

        # utils for vectorized operations
        df['is_up_util'] = df_ohlc['close'] > df_ohlc['open']
        df['is_up_util'] = df['is_up_util'].astype(int)
        df['y1_util'] = np.where(df['is_up_util'] > 0, df_ohlc['high'], df_ohlc['low'])
        df['y2_util'] = np.where(df['is_up_util'] > 0, df_ohlc['low'], df_ohlc['high'])
        df['close_util'] = [(y1, y2) for y1, y2 in zip(df['y1_util'], df_ohlc['close'])]
        df['open_util'] = [(y1, y2) for y1, y2 in zip(df['y2_util'], df_ohlc['open'])]
        df['hl_util'] = [(y1, y2) for y1, y2 in zip(df_ohlc['low'], df_ohlc['high'])]

        _delta_name = ['delta', 'subtract', 'sum', 'change', 'bs_sum']
        _delta_columns = ['delta', 'subtract_delta', 'sum_delta', 'delta_change', 'delta_bs_sum']
        for name, column_name in zip(_delta_name, _delta_columns):

            df[f'{name}_abs'] = abs(df[column_name])
            b_prefix = f'bubbles_{name}'

            match b.filter_type:
                case FilterType.MA | FilterType.StdDev | FilterType.Both:
                    df[f'{b_prefix}_ma'] = get_ma(df[f'{name}_abs'].to_numpy(), b.ma_type, b.ma_period)
                    df[f'{b_prefix}_filter'] = df[f'{b_prefix}_ma']
                    if b.filter_type in [FilterType.StdDev, FilterType.Both]:
                        df[f'{b_prefix}_stddev'] = get_stddev(df[f'{name}_abs'].to_numpy(), df[f'{b_prefix}_ma'], b.ma_period)
                        if b.filter_type == FilterType.Both:
                            df[f'{b_prefix}_filter'] = (df[f'{name}_abs'] - df[f'{b_prefix}_ma']) / df[f'{b_prefix}_stddev']
                        else:
                            df[f'{b_prefix}_filter'] = df[f'{b_prefix}_stddev']
                        df.drop(columns=[f'{b_prefix}_stddev'], inplace=True)
                    df.drop(columns=[f'{b_prefix}_ma'], inplace=True)
                case FilterType.SoftMax_Power:
                    df[f'{b_prefix}_filter'] = df[f'{name}_abs'].rolling(b.ma_period).apply(power_softmax, raw=True)
                case FilterType.L2Norm:
                    df[f'{b_prefix}_filter'] = df[f'{name}_abs'].rolling(b.ma_period).apply(l2norm, raw=True)
                case FilterType.MinMax:
                    series = df[f'{name}_abs']
                    roll_min = series.rolling(b.ma_period).min()
                    roll_max = series.rolling(b.ma_period).max()
                    df[f'{b_prefix}_filter'] = (series - roll_min) / (roll_max - roll_min)

            # The code below simulates the "Segments loop" of OrderFlow C# method
            # vectorized for speeed hehehe

            # divide total x-delta by normalized x-delta (ma/stddev)
            if b.filter_type in [FilterType.MA, FilterType.StdDev]:
                df[f'{b_prefix}_strength'] = df[f'{name}_abs'] / df[f'{b_prefix}_filter']
            else:
                df[f'{b_prefix}_strength'] = df[f'{b_prefix}_filter']

            df[f'{b_prefix}_strength'] = round(df[f'{b_prefix}_strength'], 2)
            if b.filter_ratio == FilterRatio.Percentage:
                pctile = df[f'{b_prefix}_strength'].rolling(b.p_period).apply(rolling_percentile, raw=True)
                df[f'{b_prefix}_strength'] = round(pctile, 1)

            # heatmap + bubbles size
            d = df[f'{b_prefix}_strength']
            df[f'{b_prefix}_chart'] = np.where(d < b.lowest, 0,
                                      np.where(d < b.low, 1,
                                      np.where(d < b.average, 2,
                                      np.where(d < b.high, 3,
                                      np.where(d >= b.ultra, 4, 4))))) \
                                if b.filter_ratio == FilterRatio.Fixed else \
                                    np.where(d < b.lowest_pct, 0,
                                    np.where(d < b.low_pct, 1,
                                    np.where(d < b.average_pct, 2,
                                    np.where(d < b.high_pct, 3,
                                    np.where(d >= b.ultra_pct, 4, 4)))))
            # fading
            df[f'{b_prefix}_fading'] = df[column_name] < df[column_name].shift(1)
            # positive/negative
            df[f'{b_prefix}_plusminus'] = df[column_name] > 0

            # ultra levels
            lvl_prefix = f'bubbles_levels_{name}'
            df[f'{lvl_prefix}_is_ultra'] = df[f'{b_prefix}_chart'] == 4
            df[f'{lvl_prefix}_is_ultra'] = df[f'{lvl_prefix}_is_ultra'].astype(int)
            # y1, y2
            d = df[f'{lvl_prefix}_is_ultra']
            df[f'{lvl_prefix}_high_to_low'] = np.where(d > 0, df['hl_util'], 0)
            df[f'{lvl_prefix}_high_or_low_close'] = np.where(d > 0, df['close_util'], 0)
            df[f'{lvl_prefix}_high_or_low_open'] = np.where(d > 0, df['open_util'], 0)

            df.drop(columns=[f'{name}_abs'], inplace=True)

            if not self._with_plotly_columns:
                continue

            plt_prefix = f'plotly_{b_prefix}'

            d = df[f'{b_prefix}_chart']
            df[f'{plt_prefix}_size'] = np.where(d == 0, 20,
                                       np.where(d == 1, 35,
                                       np.where(d == 2, 50,
                                       np.where(d == 3, 65,
                                       np.where(d == 4, 80, 80)))))
            df[f'{plt_prefix}_heatmap_color'] = np.where(d == 0, 'rgb(0,255,255)',
                                                np.where(d == 1, '#FFFFFF',
                                                np.where(d == 2, 'rgb(255,255,0)',
                                                np.where(d == 3, 'rgb(255,192,0)',
                                                np.where(d == 4, 'rgb(255,0,0)', 'rgb(255,0,0)')))))
            d = df[f'{b_prefix}_fading'].astype(int)
            df[f'{plt_prefix}_fading_color'] = np.where(d == 1, 'crimson', 'deepskyblue')

            d = df[f'{b_prefix}_plusminus'].astype(int)
            df[f'{plt_prefix}_plusminus_color'] = np.where(d == 0, 'crimson', 'deepskyblue')

        df.drop(columns=['is_up_util', 'y1_util', 'y2_util', 'open_util', 'hl_util', 'close_util'], inplace=True)

    def _spike_levels_count(self, df: pd.DataFrame, t_params: SpikeFilter | None = None):
        t = t_params if isinstance(t_params, SpikeFilter) else self._spike_filter

        _delta_name = ['delta', 'sum', 'bs_sum']
        for name in _delta_name:
            # TODO np.arrays
            df[f'spike_levels_break_at_{name}'] = np.NaN

            current_levels = []
            for i in range(len(df)):
                o = df['open'].iat[i]
                h = df['high'].iat[i]
                l = df['low'].iat[i]
                c = df['close'].iat[i]
                price_levels = df[f'spike_levels_price_{name}'].iat[i]

                # check touches for all active levels
                for level in current_levels:
                    if not level.is_active:
                        continue

                    if touches_spikes(o, h, l, c, level.top, level.bottom):
                        level.touch_count += 1

                    if level.touch_count >= t.max_count:
                        level.is_active = False
                        df[f'spike_levels_break_at_{name}'].iat[level.level_idx] = i

                if len(price_levels) != 0:
                    # at least 1 level = 2D np.array
                    # [ [y1, y2], [y1, y2], etc...]
                    for k in range(len(price_levels)):
                        y1 = price_levels[k][0]
                        y2 = price_levels[k][1]

                        top = max(y1, y2)
                        bottom = min(y1, y2)

                        info = LevelInfo(top, bottom, i)
                        current_levels.append(info)

        return df

    def _bubbles_levels_count(self, df: pd.DataFrame, b_params: BubblesFilter | None = None):
        b = b_params if isinstance(b_params, BubblesFilter) else self._bubbles_filter

        _delta_name = ['delta', 'subtract', 'change']
        df_len = len(df)
        for name in _delta_name:
            # TODO np.arrays
            lvl_prefix = f'bubbles_levels_{name}'
            df[f'{lvl_prefix}_break_at'] = np.NaN

            current_levels = []
            for i in range(df_len):
                o = df['open'].iat[i]
                h = df['high'].iat[i]
                l = df['low'].iat[i]
                c = df['close'].iat[i]

                match b.level_size:
                    case UltraBubblesLevel.HighOrLow_Open:
                        price_level = df[f'{lvl_prefix}_high_or_low_open'].iat[i]
                    case UltraBubblesLevel.HighOrLow_Close:
                        price_level = df[f'{lvl_prefix}_high_or_low_close'].iat[i]
                    case _:
                        price_level = df[f'{lvl_prefix}_high_to_low'].iat[i]

                # check touches for all active levels
                for level in current_levels:
                    if not level.is_active:
                        continue

                    if touches_bubbles(b, o, h, l, c, level.top, level.bottom):
                        level.touch_count += 1

                    if level.touch_count >= b.max_count:
                        level.is_active = False
                        df[f'{lvl_prefix}_break_at'].iat[level.level_idx] = i

                if price_level != 0:
                    # 1 level = 1 tuple
                    y1 = price_level[0]
                    y2 = price_level[1]

                    top = max(y1, y2)
                    bottom = min(y1, y2)

                    info = LevelInfo(top, bottom, i)
                    current_levels.append(info)

        return

    def _create_segments(self, interval_open: float, interval_highest: float, interval_lowest: float):
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

        return np.array(interval_segments)

    def _create(self, ohlc_row: dict):
        interval_open = ohlc_row['open']
        interval_highest = ohlc_row['high']
        interval_lowest = ohlc_row['low']

        interval_segments = self._create_segments(interval_open, interval_highest, interval_lowest)
        len_segments = len(interval_segments)

        odf_prices = np.array(interval_segments)
        normal_profile, buy_profile, sell_profile, delta_profile = \
            (deepcopy(np.zeros(len_segments, dtype=np.int64)) for _ in range(4))
        # array because of vp_ticks function
        min_delta, max_delta = np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)

        # Volume Profile Ticks
        if self._is_open_time:
            start_time = ohlc_row['datetime']
            end_time = ohlc_row['end_time']
        else:
            start_time = ohlc_row['start_time']
            end_time = ohlc_row['datetime']

        # ticks_interval = df_ticks.loc[(df_ticks['datetime'] >= start_time) & (df_ticks['datetime'] <= end_time)]
        # ticks_array = ticks_interval['close'].to_numpy() # speeed
        ticks_array = self._ticks_close[(self._ticks_datetime >= start_time) & (self._ticks_datetime <= end_time)]

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

        # normal
        normal_value = sum(normal_profile)

        # buy_sell
        value_buy, value_sell = sum(buy_profile), sum(sell_profile)
        value_sum, value_subtract = value_buy + value_sell, value_buy - value_sell
        value_divide = 0
        if value_buy != 0 and value_sell != 0:
            value_divide = round(value_buy / value_sell, 3)

        # delta
        delta_value = sum(delta_profile)
        subtract_delta = min_delta[0] - max_delta[0]
        sum_delta = abs(min_delta[0]) + abs(max_delta[0])

        delta_buy_value = sum([n for n in delta_profile if n > 0])
        delta_sell_value = sum([n for n in delta_profile if n < 0])

        delta_bs = delta_buy_value + abs(delta_sell_value)
        delta_bs = max(1, delta_bs) # clamp it
        delta_buy_pct = round((delta_buy_value * 100) / delta_bs)
        delta_sell_pct = round((delta_sell_value * 100) / delta_bs)

        delta_divide = 0
        if delta_buy_value != 0 and delta_sell_value != 0:
            delta_divide = round(delta_buy_value / abs(delta_sell_value), 3)

        # remove the first element, it's always 0
        normal_profile = normal_profile[1:]
        buy_profile = buy_profile[1:]
        sell_profile = sell_profile[1:]
        delta_profile = delta_profile[1:]

        bar_date = ohlc_row['datetime']
        normal = {'datetime': bar_date,
                  'profile_prices': odf_prices,
                  'normal_profile': normal_profile,
                  'normal_value': normal_value}
        buy_sell = {'datetime': bar_date,
                    'profile_prices': odf_prices,
                    'buy_profile': buy_profile,
                    'sell_profile': sell_profile,
                    'buy_value': value_buy, 'sell_value': value_sell,
                    'bs_sum': value_sum, 'bs_subtract': value_subtract, 'bs_divide': value_divide}
        delta = {'datetime': bar_date,
                 'profile_prices': odf_prices,
                 'delta_profile': delta_profile,
                 'delta_buy': delta_buy_value,
                 'delta_buy_pct': delta_buy_pct,
                 'delta_sell': delta_sell_value,
                 'delta_sell_pct': delta_sell_pct,
                 'delta_bs_sum': delta_bs,
                 'delta_bs_divide': delta_divide,
                 'delta': delta_value,
                 'min_delta': min_delta[0], 'max_delta': max_delta[0],
                 'subtract_delta': subtract_delta, 'sum_delta': sum_delta}

        if not self._with_plotly_columns:
            return normal, buy_sell, delta

        # for Plotly = 'X' integer index
        # The idea is to avoid using pandas.apply method in self.plot(),
        # since we need to get the proportion of each row in order to define
        # the X position of histogram/number of each bar.
        def add_plotly_row(dict_name: str, max_side: float, side_center: float, chart: str):
            plot = {
                f'plotly_normal_ohlc_histogram': [],
                f'plotly_normal_ohlc_numbers': [],

                f'plotly_normal_candle_histogram': [],
                f'plotly_normal_candle_numbers': [],

                f'plotly_delta_buy_ohlc_histogram': [],
                f'plotly_delta_buy_candle_histogram': [],

                f'plotly_delta_sell_ohlc_histogram': [],
                f'plotly_delta_sell_candle_histogram': [],

                f'plotly_delta_ohlc_numbers': [],
                f'plotly_delta_candle_numbers': [],
            }
            plot_profile = {
                f'plotly_delta_profile_ohlc_numbers': [],
                f'plotly_delta_profile_candle_numbers': [],

                f'plotly_delta_buy_profile_ohlc_histogram': [],
                f'plotly_delta_buy_profile_candle_histogram': [],

                f'plotly_delta_sell_profile_ohlc_histogram': [],
                f'plotly_delta_sell_profile_candle_histogram': [],
            }
            bar_index = ohlc_row['plotly_int_index']
            side_center_left = bar_index - side_center
            side_center_right = bar_index + side_center
            max_side_profile = max_side * 2

            if dict_name == 'normal':
                max_value = max(normal['normal_profile'] if normal['normal_profile'].size > 0 else [1])
                for value in normal['normal_profile']:
                    first = value * max_side_profile
                    result = (first / max_value) if (first != 0) else 0
                    plot[f'plotly_normal_{chart}_histogram'].append(result)
                    plot[f'plotly_normal_{chart}_numbers'].append(bar_index)

                # Remove keys with empty list
                # to avoid .update() overwriting behavior on the 2º call of this function
                filter_plot = {key: value for key, value in plot.items() if value}
                normal.update(filter_plot)
            else:
                max_list = [value for value in delta['delta_profile'] if value > 0]
                buy_max_value = max(max_list if max_list else [1])

                min_list = [value for value in delta['delta_profile'] if value < 0]
                sell_max_value = min(min_list if min_list else [1])
                for value in delta['delta_profile']:
                    # buy - division by zero doesn't happen... yet!
                    first = (value if value > 0 else 0) * max_side
                    result = first / buy_max_value
                    plot[f'plotly_delta_buy_{chart}_histogram'].append(result)

                    first = (value if value > 0 else 0) * max_side_profile
                    result_profile = first / (buy_max_value
                        if buy_max_value > abs(sell_max_value) else abs(sell_max_value))
                    plot_profile[f'plotly_delta_buy_profile_{chart}_histogram'].append(result_profile)

                    # sell - division by zero doesn't happen... yet!
                    first = (-value if value < 0 else 0) * max_side
                    result = first / sell_max_value
                    plot[f'plotly_delta_sell_{chart}_histogram'].append(result)

                    first = (-value if value < 0 else 0) * max_side_profile
                    result_profile = first / (buy_max_value
                        if buy_max_value > abs(sell_max_value) else abs(sell_max_value))
                    plot_profile[f'plotly_delta_sell_profile_{chart}_histogram'].append(abs(result_profile))

                    # numbers
                    plot[f'plotly_delta_{chart}_numbers'].append(side_center_right if value > 0 else side_center_left)
                    plot_profile[f'plotly_delta_profile_{chart}_numbers'].append(bar_index)

                filter_plot = {key: value for key, value in plot.items() if value}
                delta.update(filter_plot)
                filter_plot = {key: value for key, value in plot_profile.items() if value}
                delta.update(filter_plot)

        def add_plotly_row_bs(max_side: float, side_center: float, chart: str):
            plot = {
                f'plotly_buy_ohlc_histogram': [],
                f'plotly_buy_ohlc_numbers': [],
                f'plotly_buy_candle_histogram': [],
                f'plotly_buy_candle_numbers': [],

                f'plotly_sell_ohlc_histogram': [],
                f'plotly_sell_ohlc_numbers': [],
                f'plotly_sell_candle_histogram': [],
                f'plotly_sell_candle_numbers': [],
            }
            plot_profile = {
                f'plotly_buy_profile_ohlc_histogram': [],
                f'plotly_buy_profile_candle_histogram': [],
                f'plotly_buy_profile_ohlc_numbers': [],
                f'plotly_buy_profile_candle_numbers': [],

                f'plotly_sell_profile_ohlc_histogram': [],
                f'plotly_sell_profile_candle_histogram': [],
                f'plotly_sell_profile_ohlc_numbers': [],
                f'plotly_sell_profile_candle_numbers': [],
            }
            bar_index = ohlc_row['plotly_int_index']
            side_center_left = bar_index - side_center
            side_center_right = bar_index + side_center
            max_side_profile = max_side * 2

            buy_max_value = max(buy_sell['buy_profile'] if buy_sell['buy_profile'].size > 0 else [1])
            for value in buy_sell['buy_profile']:
                first = value * max_side
                result = (first / buy_max_value) if (first != 0) else 0
                plot[f'plotly_buy_{chart}_histogram'].append(result)

                first = value * (max_side_profile / 2)
                result = (first / buy_max_value) if (first != 0) else 0
                plot_profile[f'plotly_buy_profile_{chart}_histogram'].append(result)

                plot[f'plotly_buy_{chart}_numbers'].append(side_center_right)
                # center right = left align
                plot_profile[f'plotly_buy_profile_{chart}_numbers'].append(bar_index + 0.1)

            sell_max_value = max(buy_sell['sell_profile'] if buy_sell['sell_profile'].size > 0 else [1])
            for value in buy_sell['sell_profile']:
                first = value * (-max_side)
                result = (first / sell_max_value) if (first != 0) else 0

                plot[f'plotly_sell_{chart}_histogram'].append(result)

                first = value * (-max_side_profile)
                result = (first / sell_max_value) if (first != 0) else 0
                plot_profile[f'plotly_sell_profile_{chart}_histogram'].append(abs(result))

                plot[f'plotly_sell_{chart}_numbers'].append(side_center_left)
                # center right = right align
                if chart == 'candle':
                    plot_profile[f'plotly_sell_profile_{chart}_numbers'].append(bar_index + 0.2)
                else:
                    plot_profile[f'plotly_sell_profile_{chart}_numbers'].append(bar_index + 0.25)

            filter_plot = {key: value for key, value in plot.items() if value}
            buy_sell.update(filter_plot)
            filter_plot = {key: value for key, value in plot_profile.items() if value}
            buy_sell.update(filter_plot)

        for name in ['normal', 'delta', 'buy_sell']:
            # OHLC chart (max_side) = 0.3 / each side (center) = 0.15
            # Candle chart (max_side) = 0.245 / each side (center) = 0.125
            add_plotly_row(name, 0.3, 0.15, 'ohlc')
            add_plotly_row(name, 0.245, 0.125, 'candle')
            if name == 'buy_sell':
                add_plotly_row_bs(0.3, 0.15, 'ohlc')
                add_plotly_row_bs(0.245, 0.125, 'candle')

        return normal, buy_sell, delta
