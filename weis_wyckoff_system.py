"""
Weis & Wyckoff System
=====
Python version of Weis & Wyckoff System (v2.0) developed for cTrader Trading Platform

David H. Weis and Richard Wyckoff concepts for any OHLC Chart

It's intended to be used together with the renkodf/rangedf package
or any OHLC chart with date/volume data

Features from revision 1 (after Order Flow Aggregated development)
    - Support to [Candles, Heikin-Ash, Tick, Range] Charts
    - Improved ZigZag => MTF support + [Percentage, Pips, NoLag_HighLow] Modes, ATR mode hasn't been implemented
    - Custom MAs
    - Strength Filters (MA/StdDev/Both, Normalized_Emphasized)

Additional Features => that will be implemented to C# version... sometime next year (2026)
    - [L1Norm] filter alternative to StrengthFilter
    - FilterRatio.[Fixed / Percentage]

Python/C# author:
    - srlcarlg
Original author of code concepts (before revision 1) in Pinescript/TradingView:
    - `akutsusho - Numbers-Renko 数字練行足 <https://www.tradingview.com/script/9BKOIhdl-Numbers-Renko/>`_
About the concept/style of the indicators:
    - David Weis - Weis on Wyckoff Renko Charts \n
    - `YouTube <https://www.youtube.com/watch?v=wfRwiU2D_Fs>`_ \n
    - `Vimeo <https://vimeo.com/394541866>`_
"""

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from custom_mas import get_ma, get_stddev
from models_utils.ww_models import FilterType, StrengthFilter, WavesMode, ZigZagInit, WavesInit, FilterRatio
from models_utils.ww_utils import reversal_logic, zigzag_logic, rolling_percentile, l1norm


class WeisWyckoffSystem:
    def __init__(self, df_ohlcv: pd.DataFrame | None = None,
                 df_htf: pd.DataFrame | None = None,
                 df_ltf: pd.DataFrame | None = None,
                 strength_filter: StrengthFilter | None = None,
                 waves_init: WavesInit | None = None,
                 zigzag_init: ZigZagInit | None = None):
        """
        David H. Weis and Richard Wyckoff analysis on OHLCV Chart. \n
        Backtest version.

        Usage
        ------

        >>> from weis_wyckoff_system import WeisWyckoffSystem
        >>> wws = WeisWyckoffSystem(df_ohlcv...)
        >>> df = wws.full_analysis()
        >>> # or specific analysis
        >>> df_waves = wws.weis_waves_analysis()
        >>> df_wyckoff = wws.wyckoff_analysis()

        >>> # or
        >>> wws = WeisWyckoffSystem()
        >>> df_waves = wws.weis_waves_analysis(df_ohlcv)
        >>> df_wyckoff = wws.wyckoff_analysis(df_ohlcv)
        """
        if df_ohlcv is not None:
            if 'datetime' not in df_ohlcv.columns:
                df_ohlcv["datetime"] = df_ohlcv.index
            if 'close' not in df_ohlcv.columns:
                raise ValueError("Column 'close' doesn't exist!")
            if 'volume' not in df_ohlcv.columns:
                raise ValueError("Column 'volume' doesn't exist!")
        if df_htf is not None:
            if 'datetime' not in df_htf.columns:
                df_htf["datetime"] = df_htf.index
        if df_ltf is not None:
            if 'datetime' not in df_ltf.columns:
                df_ltf["datetime"] = df_ltf.index

        self._df_ohlcv = df_ohlcv
        self._df_htf = df_htf
        self._df_ltf = df_ltf
        self._strength_filter = strength_filter if isinstance(strength_filter, StrengthFilter) else StrengthFilter()
        self._waves_init = waves_init if isinstance(waves_init, WavesInit) else WavesInit()
        self._zigzag_init = zigzag_init if isinstance(zigzag_init, ZigZagInit) else ZigZagInit()

        if df_ohlcv is not None:
            if self._waves_init.is_open_time:
                df_ohlcv['end_time'] = df_ohlcv['datetime'].shift(-1)
            else:
                df_ohlcv['start_time'] = df_ohlcv['datetime'].shift(1)

    def full_analysis(self, df: pd.DataFrame | None = None,
                      df_htf: pd.DataFrame | None = None,
                      df_ltf: pd.DataFrame | None = None,
                      strength_filter: StrengthFilter | None = None,
                      waves_init: WavesInit | None = None,
                      zigzag_init: ZigZagInit | None = None):
        """
        Performs Weis Waves and Wyckoff analysis.
        """
        df = self.weis_waves_analysis(df, df_htf, df_ltf, waves_init, zigzag_init)
        df = self.wyckoff_analysis(df, strength_filter)
        return  df

    def weis_waves_analysis(self, df: pd.DataFrame | None = None,
                            df_htf: pd.DataFrame | None = None,
                            df_ltf: pd.DataFrame | None = None,
                            waves_init: WavesInit | None = None,
                            zigzag_init: ZigZagInit | None = None):
        """
        Performs Weis Waves analysis on the given dataframe.

        The following (17) columns will be added:
            * 'trendline'
            * 'turning_point'
        Waves value:
            * 'end_wave'
            * 'wave_volume'
            * 'wave_effort_result'
            * 'wave_price'
            * 'wave_time'
        Yellow coloring:
            * 'large_effort_result'
            * 'large_wave'
        Comparison marks:
            * 'effort_result_vs_same_direction'
            * 'effort_result_vs_previous'
            * 'wave_vs_same_direction'
            * 'wave_vs_previous'
        Wave time(float format):
            * 'wave_time_ms'    milliseconds
            * 'wave_time_sec'   seconds
            * 'wave_time_min'   minutes
            * 'wave_time_hour'  hours

        Parameters
        ----------
        df : pd.DataFrame
            If None, the df_ohlcv from WWS instance will be used, same for remaining parameters.
        df_htf : pd.DataFrame
            Higher Timeframe OHLC for ZigZag, will be used if provided.
        df_ltf : pd.DataFrame
            Lower Timeframe OHLC for ZigZag.NoLag_HighLow with PriorityMode.Auto, will be used if provided.
        waves_init : WavesInit
            self-explanatory
        zigzag_init : ZigZagInit
            self-explanatory
        """
        wi = waves_init if isinstance(waves_init, WavesInit) else self._waves_init
        zz = zigzag_init if isinstance(zigzag_init, ZigZagInit) else self._zigzag_init

        # WavesInit should be reset in order to use another df_ohlcv timeframe
        wi.reset_waves()

        if df is None:
            df = self._df_ohlcv
        else:
            if 'datetime' not in df.columns:
                df["datetime"] = df.index
            if wi.is_open_time:
                df['end_time'] = df['datetime'].shift(-1)
            else:
                df['start_time'] = df['datetime'].shift(1)

        df_htf = self._df_htf if df_htf is None else df_htf
        df_ltf = self._df_ltf if df_ltf is None else df_ltf

        if df_htf is not None:
            if 'datetime' not in df_htf.columns:
                df_htf["datetime"] = df_htf.index
        if df_ltf is not None:
            if 'datetime' not in df_ltf.columns:
                df_ltf["datetime"] = df_ltf.index

        df['trendline'] = np.NaN
        df['turning_point'] = np.NaN
        # Waves values
        df['end_wave'] = np.NaN
        df['wave_volume'] = np.NaN
        df['wave_effort_result'] = np.NaN
        # Yellow coloring
        df['large_effort_result'] = 0
        df['large_wave'] = 0
        # Comparison marks
        df['effort_result_vs_same_direction'] = np.NaN
        df['effort_result_vs_previous'] = np.NaN
        df['wave_vs_same_direction'] = np.NaN
        df['wave_vs_previous'] = np.NaN
        # Other waves - price
        df['wave_price'] = np.NaN
        # Other waves - time
        df['wave_time'] = pd.Timedelta(milliseconds=0)
        df['wave_time_ms'] = np.NaN
        df['wave_time_sec'] = np.NaN
        df['wave_time_min'] = np.NaN
        df['wave_time_hour'] = np.NaN

        df_len = len(df)
        for i in range(df_len):
            if i + 1 >= df_len:
                break

            if wi.waves_mode == WavesMode.Reversal:
                reversal_logic(df, i, wi)
            else:
                low_i = df['low'].iat[i]
                high_i = df['high'].iat[i]
                prev_low_i = df['low'].iat[i - 1]
                prev_high_i = df['high'].iat[i - 1]

                if df_htf is not None:
                    date_i = df['datetime'].iat[i]
                    htf_prices = df_htf.loc[df_htf['datetime'] >= date_i].head(1)
                    if len(htf_prices) > 0:
                        low_i = htf_prices['low'].iat[0]
                        high_i = htf_prices['high'].iat[0]

                    prev_date_i = df['datetime'].iat[i - 1]
                    htf_prev_prices = df_htf.loc[df_htf['datetime'] >= prev_date_i].head(1)
                    if len(htf_prev_prices) > 0:
                        prev_low_i = htf_prev_prices['low'].iat[0]
                        prev_high_i = htf_prev_prices['high'].iat[0]

                price_tuple = (low_i, high_i, prev_low_i, prev_high_i)
                zigzag_logic(zz, df, df_htf, df_ltf, wi, i, price_tuple)

        return df

    def wyckoff_analysis(self, df: pd.DataFrame | None = None,
                         strength_filter: StrengthFilter | None = None):
        """
        Performs Wyckoff analysis on the given dataframe.

        The following (9) columns will be added:
            * 'volume_filter'
            * 'time_filter'
            * 'volume_strength'
            * 'time_strength'
        Bar time:
            * 'bar_time'
            * 'bar_time_ms'    milliseconds
            * 'bar_time_sec'   seconds
            * 'bar_time_min'   minutes
            * 'bar_time_hour'  hours

        Parameters
        ----------
        df : pd.DataFrame
            If None, the df_ohlcv from WWS instance will be used, same for remaining parameters.
        strength_filter : StrengthFilter
            self-explanatory
        """
        if df is None:
            df = self._df_ohlcv
        f = strength_filter if isinstance(strength_filter, StrengthFilter) else self._strength_filter

        # Bar time
        time_bars = pd.DataFrame(df.index, index=df.index)
        prev_time = time_bars if f.is_open_time else time_bars.shift(1)
        curr_time = time_bars.shift(-1) if f.is_open_time else time_bars
        df['bar_time'] = (curr_time - prev_time)
        df['bar_time_ms'] = (curr_time - prev_time) / pd.Timedelta(milliseconds=1)
        df['bar_time_sec'] = (curr_time - prev_time) / pd.Timedelta(seconds=1)
        df['bar_time_min'] = (curr_time - prev_time) / pd.Timedelta(minutes=1)
        df['bar_time_hour'] = (curr_time - prev_time) / pd.Timedelta(hours=1)

        # Filters
        match f.filter_type:
            case FilterType.MA | FilterType.StdDev | FilterType.Both:
                df['volume_ma'] = get_ma(df['volume'].to_numpy(), f.ma_type, f.ma_period)
                df['time_ma'] = get_ma(df['bar_time_ms'].to_numpy(), f.ma_type, f.ma_period)

                df['volume_filter'] = df['volume'] / df['volume_ma']
                df['time_filter'] = (df['bar_time_ms'] / df['time_ma'])
                if f.filter_type in [FilterType.StdDev, FilterType.Both]:
                    df['volume_stddev'] = get_stddev(df['volume'].to_numpy(), df['volume_ma'], f.ma_period)
                    df['time_stddev'] = get_stddev(df['bar_time_ms'].to_numpy(), df['time_ma'], f.ma_period)

                    if f.filter_type == FilterType.StdDev:
                        df['volume_filter'] = df['volume'] / df['volume_stddev']
                        df['time_filter'] = (df['bar_time_ms'] / df['time_stddev'])
                    else:
                        df['volume_filter'] = (df['volume'] - df['volume_ma']) / df['volume_stddev']
                        df['time_filter'] = (df['bar_time_ms'] - df['time_ma']) / df['time_stddev']

                    df.drop(columns=['volume_stddev', 'time_stddev'], inplace=True)
                df.drop(columns=['volume_ma', 'time_ma'], inplace=True)
            case FilterType.Normalized_Emphasized:
                # volume
                df['volume_avg'] = df['volume'].rolling(f.n_period).mean()
                df['volume_normalized'] = df['volume'] / df['volume_avg']
                df['volume_pct'] = (df['volume_normalized'] * 100) - 100
                df['volume_pct_multiplier'] = df['volume_pct'] * f.n_multiplier
                # time
                df['time_avg'] = df['bar_time_ms'].rolling(f.n_period).mean()
                df['time_normalized'] = df['bar_time_ms'] / df['time_avg']
                df['time_pct'] = (df['time_normalized'] * 100) - 100
                df['time_pct_multiplier'] = df['time_pct'] * f.n_multiplier
                # final
                df['volume_filter'] = df['volume_pct_multiplier']
                df['time_filter'] = df['time_pct_multiplier']

                df.drop(columns=['volume_avg', 'volume_normalized', 'volume_pct', 'volume_pct_multiplier',
                                 'time_avg', 'time_normalized', 'time_pct', 'time_pct_multiplier',], inplace=True)
            case FilterType.L1Norm:
                df['volume_filter'] = df['volume'].rolling(f.ma_period).apply(l1norm, raw=True)
                df['time_filter'] = df['bar_time_ms'].rolling(f.ma_period).apply(l1norm, raw=True)

        df['volume_filter'] = abs(df['volume_filter'])
        df['time_filter'] = abs(df['time_filter'])
        df['volume_filter'] = round(df['volume_filter'], 2)
        df['time_filter'] = round(df['time_filter'], 2)
        if f.filter_ratio == FilterRatio.Percentage and f.filter_type != FilterType.Normalized_Emphasized:
            pctile = df['volume_filter'].rolling(f.n_period).apply(rolling_percentile, raw=True)
            df['volume_filter'] = round(pctile, 1)

            pctile = df['time_filter'].rolling(f.n_period).apply(rolling_percentile, raw=True)
            df['time_filter'] = round(pctile, 1)

        # Strength
        d = df['volume_filter']
        if f.filter_type == FilterType.Normalized_Emphasized:
            df['volume_strength'] = np.where(d < f.lowest_pct, 0,
                                    np.where(d < f.low_pct, 1,
                                    np.where(d < f.average_pct, 2,
                                    np.where(d < f.high_pct, 3,
                                    np.where(d >= f.ultra_pct, 4, 4)))))
        else:
            df['volume_strength'] = np.where(d < f.lowest, 0,
                                    np.where(d < f.low, 1,
                                    np.where(d < f.average, 2,
                                    np.where(d < f.high, 3,
                                    np.where(d >= f.ultra, 4, 4))))) \
                                if f.filter_ratio == FilterRatio.Fixed else \
                                    np.where(d < f.lowest_pctile, 0,
                                    np.where(d < f.low_pctile, 1,
                                    np.where(d < f.average_pctile, 2,
                                    np.where(d < f.high_pctile, 3,
                                    np.where(d >= f.ultra_pctile, 4, 4))))) \


        d = df['time_filter']
        if f.filter_type == FilterType.Normalized_Emphasized:
            df['time_strength'] = np.where(d < f.lowest_pct, 0,
                                  np.where(d < f.low_pct, 1,
                                  np.where(d < f.average_pct, 2,
                                  np.where(d < f.high_pct, 3,
                                  np.where(d >= f.ultra_pct, 4, 4)))))
        else:
            df['time_strength'] = np.where(d < f.lowest, 0,
                                  np.where(d < f.low, 1,
                                  np.where(d < f.average, 2,
                                  np.where(d < f.high, 3,
                                  np.where(d >= f.ultra, 4, 4))))) \
                                if f.filter_ratio == FilterRatio.Fixed else \
                                    np.where(d < f.lowest_pctile, 0,
                                    np.where(d < f.low_pctile, 1,
                                    np.where(d < f.average_pctile, 2,
                                    np.where(d < f.high_pctile, 3,
                                    np.where(d >= f.ultra_pctile, 4, 4)))))

        return df

    def plot(self, df: pd.DataFrame | None = None,  iloc_value: int | list = 15,
                wyckoff_only: bool = False, bar_time: bool = True, bar_volume: bool = True,
                bar_strength: bool = False, turning_point: bool = False,
                chart: str = 'candle', renderer: str = 'default',
                width: int = 1200, height: int = 800):
        """
        Plot with plotly.

        Waves information:
            - (_) => Volume
            - [_] => Effort vs Result
            - 12p => Price
            - 12m => Time

        Parameters
        ----------
        df : pd.DataFrame
            If None, the df_ohlcv from WWS instance will be used
        iloc_value : int | list
            - if int => first nº values \n
            - if list => [start_index, end_index] \n
            - Used if 'df' parameter is None
        wyckoff_only : bool
            self-explanatory
        bar_time : bool
            self-explanatory
        bar_volume : bool
            self-explanatory
        bar_strength : bool
            Debug 'volume_filter' column
        turning_point : bool
            Show turning point of ZigZag
        """
        _charts = ['candle', 'ohlc']

        input_values = [chart]
        input_validation = [_charts]
        for value, validation in zip(input_values, input_validation):
            if value not in validation:
                raise ValueError(f"Only {validation} options are valid.")

        if df is None:
            df = self.wyckoff_analysis() if wyckoff_only else self.full_analysis()

            if type(iloc_value) is int:
                df = df.iloc[:iloc_value]
            else:
                df = df.iloc[iloc_value[0]:iloc_value[1]]

        df = df.copy()

        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.0)

        df[f'plotly_int_index'] = range(len(df))
        x_column_index = df[f'plotly_int_index']
        trace_chart = go.Ohlc(x=x_column_index,
                              open=df['open'],
                              high=df['high'],
                              low=df['low'],
                              close=df['close'], opacity=0.4) if chart == 'ohlc' else \
                      go.Candlestick(x=x_column_index,
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'], opacity=0.4)

        # Set fill and wick colors
        trace_chart.increasing.fillcolor = 'rgba(255,255,255, 0.0)'
        trace_chart.increasing.line.color = 'rgba(62,62,64, 0.5)'
        trace_chart.decreasing.fillcolor = 'rgba(255,255,255, 0.0)'
        trace_chart.decreasing.line.color = 'rgba(62,62,64, 0.5)'

        fig.add_trace(trace_chart, row=1, col=1)

        # Wyckoff Bars = Volume coloring
        df['is_up'] = df['close'] > df['open']
        df['is_up'] = df['is_up'].astype(int)
        df['bars_coloring'] = np.where((df['large_effort_result'] == 1), 'yellow',
                            np.where((df['volume_strength'] == 4) & (df['is_up'] == 1), 'rgba(29,137,52, 0.8)',
                            np.where((df['volume_strength'] == 4) & (df['is_up'] == 0), 'rgba(224,1,6, 0.8)',
                            np.where((df['volume_strength'] == 3) & (df['is_up'] == 1), 'rgba(161,246,161, 0.8)',
                            np.where((df['volume_strength'] == 3) & (df['is_up'] == 0),'rgba(250,102,129, 0.8)',
                            np.where(df['volume_strength'] == 2, 'rgba(217,217,217, 0.8)',
                            np.where(df['volume_strength'] == 1, 'rgba(143,144,146, 0.8)', 'rgba(62,62,64, 0.8)'))))))) \
                        if not wyckoff_only else \
                            np.where((df['volume_strength'] == 4) & (df['is_up'] == 1), 'rgba(29,137,52, 0.8)',
                            np.where((df['volume_strength'] == 4) & (df['is_up'] == 0), 'rgba(224,1,6, 0.8)',
                            np.where((df['volume_strength'] == 3) & (df['is_up'] == 1), 'rgba(161,246,161, 0.8)',
                            np.where((df['volume_strength'] == 3) & (df['is_up'] == 0),'rgba(250,102,129, 0.8)',
                            np.where(df['volume_strength'] == 2, 'rgba(217,217,217, 0.8)',
                            np.where(df['volume_strength'] == 1, 'rgba(143,144,146, 0.8)', 'rgba(62,62,64, 0.8)'))))))

        # Instead of creating multiples OHLC/Candlestick traces for each color,
        # just create a histogram (Bar type) between Open/Close and... color it!
        # idea from order_flow_ticks => spike chart
        for index in range(len(df)):
            open_i = df['open'].iat[index]
            close_i = df['close'].iat[index]

            is_up = open_i > close_i
            coloring = df['bars_coloring'].iat[index]

            # Due to automatic size between values, create one histogram for each bar.
            # vertical for [y1, y2]-like approach.
            fig.add_trace(
                go.Bar(y=[abs(close_i - open_i)], x=[index],
                   orientation='v', base=[close_i if is_up else open_i],
                   marker=dict(
                       color=coloring,
                       opacity = 0.9
                   )), row=1, col=1)

        df['hl2'] = (df['close'] + df['open']) / 2

        # Wyckoff Bars = Time display
        if bar_time and not bar_strength:
            df_time = timedelta_to_decimal_string(df['bar_time'])
            df_time['time_human'] = df_time['time_human'].str[:3]
            fig.add_trace(go.Scatter(x=x_column_index,
                                     y=df['hl2'],
                                     mode='text',
                                     text=df_time['time_human'],
                                     textposition='bottom center',
                                     textfont=dict(size=9, color='blue'),
                                     opacity=1), row=1, col=1)

        # Wyckoff Bars = Volume display
        if bar_volume:
            fig.add_trace(go.Scatter(x=x_column_index,
                                     y=df['hl2'],
                                     mode='text',
                                     text=df['volume'],
                                     textposition='top center',
                                     textfont=dict(size=9, color='blue'),
                                     opacity=1), row=1, col=1)
        if bar_strength:
            fig.add_trace(go.Scatter(x=x_column_index,
                                     y=df['hl2'],
                                     mode='text',
                                     text=df['volume_filter'],
                                     textposition='bottom center',
                                     textfont=dict(size=9, color='blue'),
                                     opacity=1), row=1, col=1)
        if not wyckoff_only:
            # Weis Waves
            df['end_wave_fill'] = df['end_wave'].fillna(0)
            df['close_1'] = np.where(df['end_wave_fill'] != 0, df['trendline'], np.NaN)

            # Volume
            df['wave_volume_string'] = '(' + df['wave_volume'].astype(str) + ') | ' # some space
            fig.add_trace(go.Scatter(x=x_column_index,
                                     y=df['close_1'],
                                     mode='text',
                                     text=df['wave_volume_string'],
                                     textposition='top left',
                                     textfont=dict(size=9, color='blue'),
                                     opacity=1), row=1, col=1)

            # Effort vs Result
            df['wave_effort_result_string'] = '[' + df['wave_effort_result'].astype(str) + ']'
            fig.add_trace(go.Scatter(x=x_column_index,
                                     y=df['close_1'],
                                     mode='text',
                                     text=df['wave_effort_result_string'],
                                     textposition='top right',
                                     textfont=dict(size=9, color='blue'),
                                     opacity=1), row=1, col=1)


            # Price
            df['wave_price_string'] = df['wave_price'].astype(str) + 'p | ' # some space
            fig.add_trace(go.Scatter(x=x_column_index,
                                     y=df['close_1'],
                                     mode='text',
                                     text=df['wave_price_string'],
                                     textposition='bottom left',
                                     textfont=dict(size=9, color='blue'),
                                     opacity=1), row=1, col=1)

            # Time
            waves_df_time = timedelta_to_decimal_string(df['wave_time'])
            waves_df_time['wave_time_human'] = waves_df_time['time_human'].str[:3]
            fig.add_trace(go.Scatter(x=x_column_index,
                                     y=df['close_1'],
                                     mode='text',
                                     text=waves_df_time['wave_time_human'],
                                     textposition='bottom right',
                                     textfont=dict(size=9, color='blue'),
                                     opacity=1), row=1, col=1)

            # ZigZag
            fig.add_trace(go.Scatter(x=x_column_index,
                                     y=df['trendline'],
                                     mode='lines',
                                     marker=dict(
                                         color='aqua',
                                         size=1,
                                     ),
                                     # IMPORTANT!
                                     # Connect prices points
                                     connectgaps=True,
                                     opacity=1), row=1, col=1)

            if turning_point:
                fig.add_trace(go.Scatter(x=x_column_index,
                                         y=df['turning_point'],
                                         mode='markers',
                                         marker=dict(
                                             color='red',
                                             size=4,
                                         ),
                                         opacity=1), row=1, col=1)

        fig.update_layout(
            title=f"Weis & Wyckoff System",
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


def timedelta_to_human_readable(df_timedelta: pd.Series):
    df_comp = df_timedelta.dt.components
    df_len = len(df_comp)
    times_strings = []
    for i in range(df_len):
        if str(df_comp['days'].iat[i]) == 'nan':
            times_strings.append('0ms')
            continue
        parts = []
        if df_comp['days'].iat[i] != 0: parts.append(f"{int(df_comp['days'].iat[i])}d")
        if df_comp['hours'].iat[i] != 0: parts.append(f"{int(df_comp['hours'].iat[i])}h")
        if df_comp['minutes'].iat[i] != 0: parts.append(f"{int(df_comp['minutes'].iat[i])}m")
        if df_comp['seconds'].iat[i] != 0: parts.append(f"{int(df_comp['seconds'].iat[i])}s")
        if df_comp['milliseconds'].iat[i] != 0: parts.append(f"{int(df_comp['milliseconds'].iat[i])}ms")
        times_strings.append(''.join(parts) or '0ms')

    return pd.DataFrame(times_strings, columns=['time_human'])

def timedelta_to_decimal_string(df_timedelta: pd.Series):
    df_comp = df_timedelta.dt.components
    df_len = len(df_comp)
    times_strings = []
    for i in range(df_len):
        if str(df_comp['days'].iat[i]) == 'nan':
            times_strings.append('0ms')
            continue
        parts = []
        if df_comp['days'].iat[i] != 0: parts.append(f"{int(df_comp['days'].iat[i]):0{2}d}d")
        if df_comp['hours'].iat[i] != 0: parts.append(f"{int(df_comp['hours'].iat[i]):0{2}d}h")
        if df_comp['minutes'].iat[i] != 0: parts.append(f"{int(df_comp['minutes'].iat[i]):0{2}d}m")
        if df_comp['seconds'].iat[i] != 0: parts.append(f"{int(df_comp['seconds'].iat[i]):0{2}d}s")
        if df_comp['milliseconds'].iat[i] != 0: parts.append(f"{int(df_comp['milliseconds'].iat[i]):0{2}d}ms")
        times_strings.append(''.join(parts) or '0ms')

    return pd.DataFrame(times_strings, columns=['time_human'])