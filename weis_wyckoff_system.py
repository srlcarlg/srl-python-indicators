"""
Weis & Wyckoff System
=====
David H. Weis and Richard Wyckoff concepts on Renko Chart

Python version of Weis & Wyckoff System (v2.0) developed for cTrader Trading Platform
with corrections from C# source

It is intended to be used together with the renkodf package
or Renko OHLC charts with date/volume data

Python/C# author:
    - srlcarlg
Original author of code concepts in Pinescript/TradingView:
    - `akutsusho - Numbers-Renko 数字練行足 <https://www.tradingview.com/script/9BKOIhdl-Numbers-Renko/>`_
About the concept/style of the indicators:
    - David Weis - Weis on Wyckoff Renko Charts \n
    - `YouTube <https://www.youtube.com/watch?v=wfRwiU2D_Fs>`_ \n
    - `Vimeo <https://vimeo.com/394541866>`_
"""
import numpy as np
import pandas as pd
import mplfinance as mpf

class WeisWyckoffSystem:
    def __init__(self, df_ohlcv: pd.DataFrame | None, large_wave_ratio: float = 1.5, ema_period: int = 5):
        """
        David H. Weis and Richard Wyckoff analysis on Renko OHLCV Chart. \n
        Backtest version.

        Usage
        ------

        >>> from weis_wyckoff_system import WeisWyckoffSystem
        >>> wws = WeisWyckoffSystem(df_ohlcv)
        >>> df = wws.full_analysis()
        >>> # or
        >>> df_waves = wws.weis_waves_analysis()
        >>> df_wyckoff = wws.wyckoff_analysis()
        >>> # or
        >>> wws = WeisWyckoffSystem()
        >>> df_waves = wws.weis_waves_analysis(df_ohlcv)
        >>> df_wyckoff = wws.wyckoff_analysis(df_ohlcv)

        Parameters
        ----------
        :param df_ohlcv:
            OHLCV with datetime index
        :param large_wave_ratio:
            Ratio value for Large (Volume, Effort vs Result) Waves
        :param ema_period:
            EMA period for Wyckoff Analysis
        """
        if df_ohlcv is not None:
            if 'datetime' not in df_ohlcv.columns:
                df_ohlcv["datetime"] = df_ohlcv.index
            if 'close' not in df_ohlcv.columns:
                raise ValueError("Column 'close' doesn't exist!")
            if 'volume' not in df_ohlcv.columns:
                raise ValueError("Column 'volume' doesn't exist!")

        self._df = df_ohlcv
        self._large_wave_ratio = large_wave_ratio
        self._ema_period = ema_period

        # Weis Waves
        self._prev_waves_volume: list = [0, 0, 0, 0]
        self._prev_waves_vol_div_renko: list = [0.0, 0.0, 0.0, 0.0]
        self._prev_cumulative_Up: list = [0, 0]
        self._prev_cumulative_Down: list = [0, 0]

        # same_wave = previous same direction wave
        # div = volume divide(/) renko
        self._prev_same_wave_div_Up: float = 0.0
        self._prev_same_wave_div_Down: float = 0.0
        # vol = cumulative volume
        self._prev_same_wave_vol_Up: float = 0.0
        self._prev_same_wave_vol_Down: float = 0.0

        # Wyckoff
        # No need global variable

    def full_analysis(self):
        """
        Performs Weis Waves and Wyckoff analysis
        :return: df
        """
        df = self.weis_waves_analysis()
        df = self.wyckoff_analysis(df)
        return  df

    def weis_waves_analysis(self, df: pd.DataFrame | None = None):
        """
        Performs Weis Waves analysis on the given dataframe. \n
        Designed for RENKO CHART ONLY. \n

        The following (16) columns will be added:
            * 'trendline'
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

        :param df: dataframe
            If None, the df_ohlcv from WWS instance will be used
        :return: df
        """
        if df is None:
            df = self._df

        df['trendline'] = np.NaN
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
        # Other waves
        df['wave_price'] = np.NaN

        df['wave_time'] = pd.Timedelta(milliseconds=0)
        df['wave_time_ms'] = np.NaN
        df['wave_time_sec'] = np.NaN
        df['wave_time_min'] = np.NaN
        df['wave_time_hour'] = np.NaN

        trend_start_index = 0
        df_len = len(df)

        def _direction_changed(index: int):
            if index + 1 >= df_len:
                return False

            dynamic_current = df['close'].iat[index] > df['open'].iat[index]
            next_is_up = df['close'].iat[index + 1] > df['open'].iat[index + 1]
            next_is_down = df['close'].iat[index + 1] < df['open'].iat[index + 1]
            prev_is_up = df['close'].iat[index - 1] > df['open'].iat[index - 1]
            prev_is_down = df['close'].iat[index - 1] < df['open'].iat[index - 1]

            return (prev_is_up and dynamic_current and next_is_down) or (prev_is_down and dynamic_current and next_is_down) \
                or (prev_is_down and not dynamic_current and next_is_up) or (prev_is_up and not dynamic_current and next_is_up)

        def _set_extremum(index: int):
            # End of wave
            if not _direction_changed(index):
                return False
            df['end_wave'].iat[index] = df['close'].iat[index]

            dynamic_current = df['close'].iat[index] > df['open'].iat[index]
            if dynamic_current:
                self._calculate_waves(df, 1, trend_start_index, index, True)
            else:
                self._calculate_waves(df, -1, trend_start_index, index, True)

            return True

        def _move_extremum(index: int):
            df[ 'trendline'].iat[index] = df['open'].iat[index]

            dynamic_current = df['close'].iat[index] > df['open'].iat[index]
            if dynamic_current:
                self._calculate_waves(df, 1, trend_start_index, index, False)
            else:
                self._calculate_waves(df, -1, trend_start_index, index, False)

            return _set_extremum(index)

        # "Zig Zag" for Renko (only)
        # It's not a true zigzag indicator, since Renko chart is already a 'filter' itself
        # We just need to know when the Trend ends = Reversal Renko
        # and the Trendline is just the Open price of each brick
        # so, C# version can be simplified.
        start_trend = False
        for i in range(df_len):
            if start_trend:
                trend_start_index = i
            if i + 1 >= df_len:
                break
            start_trend = _move_extremum(i)

        return df

    def _calculate_waves(self, df: pd.DataFrame,  direction: int, first_brick_index: int, last_brick_index: int, direction_changed: bool):
        def cumul_volume():
            if first_brick_index == last_brick_index:
                return df['volume'].iat[last_brick_index]
            volume = 0
            # (first_brick - 1) because of python's range behavior
            # first=1 and last=2: [1]
            # first=1 and last=3: [1, 2]
            for i in range(first_brick_index, last_brick_index + 1):
                volume += df['volume'].iat[i]
            return volume

        def cumul_renko():
            if first_brick_index == last_brick_index:
                return 1
            renko_count = 0
            # (first_brick - 1) because of python's range behavior
            # first=1 and last=2: [1]
            # first=1 and last=3: [1, 2]
            for i in range(first_brick_index, last_brick_index + 1):
                renko_count += 1
            return renko_count

        def cumul_price(is_up: bool):
            price = df['low'].iat[first_brick_index] - df['high'].iat[last_brick_index] if is_up else \
                df['high'].iat[first_brick_index] - df['low'].iat[last_brick_index]
            return abs(round(price, 5))

        def cumul_time():
            # first_brick - 1 because renkodf datetime is the CloseTime of bar
            prev_time = df['datetime'].iat[first_brick_index - 1]
            if first_brick_index == last_brick_index:
                prev_time = df['datetime'].iat[first_brick_index - 1]
            curr_time = df['datetime'].iat[last_brick_index]

            df['wave_time'].iat[last_brick_index] = (curr_time - prev_time)
            df['wave_time_ms'].iat[last_brick_index] = (curr_time - prev_time) / pd.Timedelta(milliseconds=1)
            df['wave_time_sec'].iat[last_brick_index] = (curr_time - prev_time) / pd.Timedelta(seconds=1)
            df['wave_time_min'].iat[last_brick_index] = (curr_time - prev_time) / pd.Timedelta(minutes=1)
            df['wave_time_hour'].iat[last_brick_index] = (curr_time - prev_time) / pd.Timedelta(hours=1)

        def other_waves(is_up: bool):
            df['wave_price'].iat[last_brick_index] = cumul_price(is_up)
            cumul_time()

        cumulative_volume = cumul_volume()
        cumulative_renko = cumul_renko()
        cumulative_vol_div_renko = round(cumulative_volume / cumulative_renko, 1)

        df['wave_volume'].iat[last_brick_index] = cumulative_volume
        df['wave_effort_result'].iat[last_brick_index] = cumulative_vol_div_renko

        if direction == 1:
            next_is_down = df['close'].iat[last_brick_index + 1] < df['open'].iat[last_brick_index + 1]
            prev_is_down = df['close'].iat[last_brick_index - 1] < df['open'].iat[last_brick_index - 1]

            end_wave = direction_changed
            if end_wave:
                self._effort_vs_result_analysis(df, last_brick_index, cumulative_vol_div_renko, end_wave,True)
                self._waves_analysis(df, last_brick_index, cumulative_volume, True)
                other_waves(True)
                self._set_previous_waves(cumulative_volume, cumulative_vol_div_renko, prev_is_down, next_is_down, True,
                                         direction_changed)

        elif direction == -1:
            next_is_up = df['close'].iat[last_brick_index + 1] > df['open'].iat[last_brick_index + 1]
            prev_is_up = df['close'].iat[last_brick_index - 1] > df['open'].iat[last_brick_index - 1]

            end_wave = direction_changed
            if end_wave:
                self._effort_vs_result_analysis(df, last_brick_index, cumulative_vol_div_renko, end_wave,False)
                self._waves_analysis(df, last_brick_index, cumulative_volume, False)
                other_waves(False)
                self._set_previous_waves(cumulative_volume, cumulative_vol_div_renko, prev_is_up, next_is_up, False,
                                         direction_changed)

        return df

    def _effort_vs_result_analysis(self, df, index, cumul_vol_div_renko: float, is_end_wave: bool, is_up: bool):
        def large_effort_result():
            have_zero = False
            for i, value in enumerate(self._prev_waves_vol_div_renko):
                if value == 0.0:
                    have_zero = True
                    break
            if have_zero:
                return 0
            return 1 if (cumul_vol_div_renko + sum(self._prev_waves_vol_div_renko)) / 5 * self._large_wave_ratio < cumul_vol_div_renko else 0

        # 1 = greater, -1 = lesser
        # Right comparison mark in C# version is wrongly implemented, instead of use the previous wave value
        # it's using the same direction wave value as the Left comparison mark.
        if is_up:
            df['effort_result_vs_same_direction'].iat[index] = 1 if cumul_vol_div_renko > self._prev_same_wave_div_Up else -1
            df['effort_result_vs_previous'].iat[index] = 1 if cumul_vol_div_renko > self._prev_cumulative_Down[1] else -1
        else:
            df['effort_result_vs_same_direction'].iat[index] = 1 if cumul_vol_div_renko > self._prev_same_wave_div_Down else -1
            df['effort_result_vs_previous'].iat[index] = 1 if cumul_vol_div_renko > self._prev_cumulative_Up[1] else -1

        if is_end_wave:
            df['large_effort_result'].iat[index] = large_effort_result()

    def _waves_analysis(self, df, index, cumul_vol: float, is_up: bool):
        # 1 = greater, -1 = lesser
        # Right comparison mark in C# version is wrongly implemented, instead of use the previous wave value
        # it's using the same direction wave value as the Left comparison mark.
        if is_up:
            df['wave_vs_same_direction'].iat[index] = 1 if cumul_vol > self._prev_same_wave_vol_Up else -1
            df['wave_vs_previous'].iat[index] = 1 if cumul_vol > self._prev_cumulative_Down[0] else -1
        else:
            df['wave_vs_same_direction'].iat[index] = 1 if cumul_vol > self._prev_same_wave_vol_Down else -1
            df['wave_vs_previous'].iat[index] = 1 if cumul_vol > self._prev_cumulative_Up[0] else -1

        df['large_wave'].iat[index] = 1 if (cumul_vol + sum(self._prev_waves_volume)) / 5 * self._large_wave_ratio < cumul_vol else 0

    def _set_previous_waves(self, cumul_vol: float, cumul_vol_div_renko: float, prev_is_dynamic: bool, next_is_dynamic: bool, is_up: bool, direction_changed: bool):

        condition_trend = not prev_is_dynamic and direction_changed and next_is_dynamic
        condition_ranging = prev_is_dynamic and direction_changed and next_is_dynamic

        def set_ranging():
            # Effort vs Result Analysis
            curr_wave = self._prev_waves_vol_div_renko
            new_wave = [curr_wave[1], curr_wave[2], curr_wave[3], cumul_vol_div_renko]
            self._prev_waves_vol_div_renko = new_wave

            # Large Weis Wave Analysis
            curr_wave = self._prev_waves_volume
            new_wave = [curr_wave[1], curr_wave[2], curr_wave[3], cumul_vol]
            self._prev_waves_volume = new_wave

        # Exclude the most old wave, keep the 3 others and add current Wave value for most recent Wave
        # Set previous accumulated UP WAVE
        if is_up:
            # (prevIsDown && DirectionChanged && nextIsDown);
            if condition_ranging:
                set_ranging()
            # (prevIsUp && DirectionChanged && nextIsDown)
            elif condition_trend:
                prev_cumul_vol_up = self._prev_cumulative_Up[0]
                prev_cumul_vol_div_renko_up = self._prev_cumulative_Up[1]
                self._prev_same_wave_div_Up = prev_cumul_vol_div_renko_up
                self._prev_same_wave_vol_Up = prev_cumul_vol_up

                # Effort vs Result Analysis
                curr_wave = self._prev_waves_vol_div_renko
                new_wave = [curr_wave[1], curr_wave[2], curr_wave[3], prev_cumul_vol_div_renko_up]
                self._prev_waves_vol_div_renko = new_wave

                # Large Weis Wave Analysis
                curr_wave = self._prev_waves_volume
                new_wave = [curr_wave[1], curr_wave[2], curr_wave[3], prev_cumul_vol_up]
                self._prev_waves_volume = new_wave

            self._prev_cumulative_Up = [cumul_vol, cumul_vol_div_renko]

        # Set previous accumulated DOWN WAVE
        else:
            # (prevIsUp && DirectionChanged && nextIsUp);
            if condition_ranging:
                set_ranging()
            # (prevIsDown && DirectionChanged && nextIsUp);
            elif condition_trend:
                prev_cumul_vol_down = self._prev_cumulative_Down[0]
                prev_cumul_vol_div_renko_down = self._prev_cumulative_Down[1]
                self._prev_same_wave_div_Down = prev_cumul_vol_div_renko_down
                self._prev_same_wave_vol_Down = prev_cumul_vol_down
                # Effort vs Result Analysis
                curr_wave = self._prev_waves_vol_div_renko
                new_wave = [curr_wave[1], curr_wave[2], curr_wave[3], prev_cumul_vol_div_renko_down]
                self._prev_waves_vol_div_renko = new_wave

                # Large Weis Wave Analysis
                curr_wave = self._prev_waves_volume
                new_wave = [curr_wave[1], curr_wave[2], curr_wave[3], prev_cumul_vol_down]
                self._prev_waves_volume = new_wave

            self._prev_cumulative_Down = [cumul_vol, cumul_vol_div_renko]

    def wyckoff_analysis(self, df: pd.DataFrame | None = None):
        """
        Performs Wyckoff analysis on the given dataframe. \n
        Can be used with Range charts \n
        The following (7) columns will be added:
            * 'bar_time'
            * 'strength_volume'
            * 'strength_time'
        Bar time(float format):
            * 'bar_time_ms'    milliseconds
            * 'bar_time_sec'   seconds
            * 'bar_time_min'   minutes
            * 'bar_time_hour'  hours

        :param df: dataframe
            If None, the df_ohlcv from WWS instance will be used
        :return: df
        """
        if df is None:
            df = self._df
        ema_period = self._ema_period

        # EMA Volume
        volume = df['volume'].copy()
        volume.iloc[:ema_period - 1] = np.NaN
        volume.iloc[ema_period - 1] = volume.iloc[0:ema_period].mean()
        df['ema_volume'] = volume.ewm(span=ema_period).mean()

        # Bar time
        time_bars = pd.DataFrame(df.index, index=df.index)
        prev_time = time_bars.shift(1)
        curr_time = time_bars
        df['bar_time'] = (curr_time - prev_time)
        df['bar_time_ms'] = (curr_time - prev_time) / pd.Timedelta(milliseconds=1)
        df['bar_time_sec'] = (curr_time - prev_time) / pd.Timedelta(seconds=1)
        df['bar_time_min'] = (curr_time - prev_time) / pd.Timedelta(minutes=1)
        df['bar_time_hour'] = (curr_time - prev_time) / pd.Timedelta(hours=1)

        # EMA Time
        time = df['bar_time_ms'].copy()
        time.iloc[:ema_period - 1] = np.NaN
        time.iloc[ema_period - 1] = time.iloc[0:ema_period].mean()
        df['ema_time'] = time.ewm(span=ema_period).mean()

        # Volume Filter
        df['filter_volume'] = df['volume'] / df['ema_volume']
        df['strength_volume'] = np.where(df['filter_volume'] > 2, 4,
                                np.where(df['filter_volume'] > 1.5, 3,
                                np.where(df['filter_volume'] > 1, 2,
                                np.where(df['filter_volume'] >  0.5, 1, 0))))
        # Time Filter
        df['filter_time'] = (df['bar_time'] / df['ema_time']) / pd.Timedelta(milliseconds=1)
        df['strength_time'] = np.where(df['filter_time'] > 2, 4,
                              np.where(df['filter_time'] > 1.5, 3,
                              np.where(df['filter_time'] > 1, 2,
                              np.where(df['filter_time'] >  0.5, 1, 0))))

        for name in ['volume', 'time']:
            df.drop(columns=[f'ema_{name}', f'filter_{name}'], inplace=True)

        return df

    def plot(self, iloc_value: int = 50, string_decimals: int = 3):
        """
        Plot full_analysis() with mplfinance. \n
        Designed for RENKO CHART ONLY. \n

        Parameters
        ----------
        iloc_value : int
            * If positive: First nº rows will be plotted
            * If negative: Last nº rows will be plotted
        string_decimals : int
            * Max characters to show on custom scatter mplfinance plot.
            * Time values are always 3 characters
            * if 3: 90 -> 090, 5 -> 005, etc...
            * if 4: 90 -> 0090, 5 -> 0005, etc...
        """
        if string_decimals < 3 or string_decimals > 4:
            raise ValueError("string_decimals cannot be '< 3' and '> 4'")

        print("Performing Full Analysis...")
        df = self.full_analysis()

        print("Calculating plots...")
        if iloc_value < 0:
            df = df.copy().iloc[iloc_value:]
        else:
            df =  df.copy().iloc[:iloc_value]

        # Wyckoff Bars = Volume coloring
        df['is_up'] = df['close'] > df['open']
        df['is_up'] = df['is_up'].astype(int)
        df['bars_coloring'] = np.where((df['large_effort_result'] == 1), 'yellow',
                            np.where((df['strength_volume'] == 4) & (df['is_up'] == 1), '#1D8934',
                            np.where((df['strength_volume'] == 4) & (df['is_up'] == 0), '#E00106',
                            np.where((df['strength_volume'] == 3) & (df['is_up'] == 1), '#A1F6A1',
                            np.where((df['strength_volume'] == 3) & (df['is_up'] == 0),'#FA6681',
                            np.where(df['strength_volume'] == 2, '#D9D9D9',
                            np.where(df['strength_volume'] == 1, '#8F9092', '#3E3E40')))))))
        coloring = df['bars_coloring'].to_list()

        # Wyckoff Bars = Volume display
        df['markers_volume'] = '$' + df['volume'].astype(int).apply(lambda x: int_to_decimal_string(x, string_decimals)) + '$'
        volume_markers = df['markers_volume'].to_list()

        # Wyckoff Bars = Time display
        df_time = timedelta_to_decimal_string(df['bar_time'])
        df_time['time_human'] = '$' + df_time['time_human'].str[:3] + '$'
        time_markers = df_time['time_human'].to_list()


        # Weis Waves = Effort vs Result
        df['markers_EvsR'] = '$' + df['wave_effort_result'].apply(lambda x: float_to_decimal_string(x, string_decimals)) + '$'
        waves_effvsres_markers = df['markers_EvsR'].to_list()

        # Weis Waves = Volume
        df['markers_volume'] = '$' + df['wave_volume'].apply(lambda x: float_to_decimal_string(x, string_decimals)) + '$'
        waves_volume_markers = df['markers_volume'].to_list()

        # Weis Waves = Price
        df['markers_price'] = '$' + df['wave_price'].apply(lambda x: float_to_decimal_string(x, string_decimals)) + '$'
        waves_price_markers = df['markers_price'].to_list()

        # Weis Waves = Time
        waves_df_time = timedelta_to_decimal_string(df['wave_time'])
        waves_df_time['wave_time_human'] = '$' + waves_df_time['time_human'].str[:3] + '$'
        waves_time_markers = waves_df_time['wave_time_human'].to_list()

        # Plot positions
        brick_size = abs(df['close'].iat[1] - df['open'].iat[1])
        df['hl2'] = (df['close'] + df['open']) / 2
        df['hl2_up'] = df['hl2'] + (brick_size / 4)
        # close +- brick_size / position_step
        df['close_1'] = df['close'] + ((df['close'] - df['open']) / 6)
        df['close_2'] = df['close'] + ((df['close'] - df['open']) / 2.5)
        df['close_3'] = df['close'] + ((df['close'] - df['open']) / 1.5)
        df['close_4'] = df['close'] + ((df['close'] - df['open']) / 1)
        # Just the end of wave
        df['end_wave_fill'] = df['end_wave'].fillna(0)
        df['close_1'] = np.where(df['end_wave_fill'] != 0, df['close_1'], np.NaN)
        df['close_2'] = np.where(df['end_wave_fill'] != 0, df['close_2'], np.NaN)
        df['close_3'] = np.where(df['end_wave_fill'] != 0, df['close_3'], np.NaN)
        df['close_4'] = np.where(df['end_wave_fill'] != 0, df['close_4'], np.NaN)

        size = 150 if string_decimals == 3 else 250
        size_times = 150 if string_decimals == 3 else 130
        ap0 = [mpf.make_addplot(df['close_1'], color='skyblue', markersize=size, type='scatter', marker=waves_volume_markers),
               mpf.make_addplot(df['close_2'], color='coral', markersize=size, type='scatter', marker=waves_effvsres_markers),
               mpf.make_addplot(df['close_3'], color='white', markersize=size, type='scatter', marker=waves_price_markers),
               mpf.make_addplot(df['close_4'], color='lightgray', markersize=size_times, type='scatter', marker=waves_time_markers),
               mpf.make_addplot(df['hl2_up'], color='skyblue', markersize=size, type='scatter', marker=volume_markers),
               mpf.make_addplot(df['hl2'], color='lightgray', markersize=size_times, type='scatter', marker=time_markers)]

        print('Completed! mpf.plot() was called, wait a moment!')
        mpf.plot(df, type='candle', style='nightclouds', volume=True, marketcolor_overrides=coloring, addplot=ap0,
                 title="Weis & Wyckoff System",
                 figsize=(12.5,6),
                 returnfig=True,
                 scale_padding=0.2)

        # ax.text have some limitations:
        #   * cannot plot massive texts
        #   * text axis change when zooming
        # a workaround with scatter custom markers is a more suitable approach
        mpf.show()

    def plot_wyckoff(self, iloc_value: int = 50, string_decimals: int = 3, plot_time: bool = True):
        """
        Plot wyckoff_analysis() with mplfinance. \n
        Can be used with Range chart. \n

        Parameters
        ----------
        iloc_value : int
            * If positive: First nº rows will be plotted
            * If negative: Last nº rows will be plotted
        string_decimals : int
            * Max characters to show on custom scatter mplfinance plot.
            * Time values are always 3 characters
            * if 3: 90 -> 090, 5 -> 005, etc...
            * if 4: 90 -> 0090, 5 -> 0005, etc...
        """
        if string_decimals < 3 or string_decimals > 4:
            raise ValueError("string_decimals cannot be '< 3' and '> 4'")

        print("Performing Wyckoff Analysis...")
        df = self.wyckoff_analysis()

        print("Calculating plots...")
        if iloc_value < 0:
            df = df.copy().iloc[iloc_value:]
        else:
            df =  df.copy().iloc[:iloc_value]

        # Wyckoff Bars = Volume coloring
        df['is_up'] = df['close'] > df['open']
        df['is_up'] = df['is_up'].astype(int)
        df['bars_coloring'] = np.where((df['strength_volume'] == 4) & (df['is_up'] == 1), '#1D8934',
                            np.where((df['strength_volume'] == 4) & (df['is_up'] == 0), '#E00106',
                            np.where((df['strength_volume'] == 3) & (df['is_up'] == 1), '#A1F6A1',
                            np.where((df['strength_volume'] == 3) & (df['is_up'] == 0),'#FA6681',
                            np.where(df['strength_volume'] == 2, '#D9D9D9',
                            np.where(df['strength_volume'] == 1, '#8F9092', '#3E3E40'))))))
        coloring = df['bars_coloring'].to_list()

        # Wyckoff Bars = Volume display
        df['markers_volume'] = '$' + df['volume'].astype(int).apply(lambda x: int_to_decimal_string(x, string_decimals)) + '$'
        volume_markers = df['markers_volume'].to_list()

        # Wyckoff Bars = Time display
        time_markers = []
        if plot_time:
            df_time = timedelta_to_decimal_string(df['bar_time'])
            df_time['time_human'] = '$' + df_time['time_human'].str[:3] + '$'
            time_markers = df_time['time_human'].to_list()

        # Plot positions
        brick_size = abs(df['close'].iat[1] - df['open'].iat[1])
        df['hl2'] = (df['close'] + df['open']) / 2
        df['hl2_up'] = df['hl2'] + (brick_size / 4)

        size = 150 if string_decimals == 3 else 250
        size_times = 150 if string_decimals == 3 else 130
        ap0 = [mpf.make_addplot(df['hl2_up'], color='skyblue', markersize=size, type='scatter', marker=volume_markers)]
        if plot_time:
            ap0.extend([mpf.make_addplot(df['hl2'], color='lightgray', markersize=size_times,
                        type='scatter', marker=time_markers)])

        print('Completed! mpf.plot() was called, wait a moment!')
        mpf.plot(df, type='candle', style='nightclouds', volume=True, marketcolor_overrides=coloring, addplot=ap0,
                 title="Weis & Wyckoff System",
                 figsize=(12.5,6),
                 returnfig=True,
                 scale_padding=0.2)
        mpf.show()

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

def int_to_decimal_string(number: int, decimals: int):
    return f'{number:0{decimals}d}'

def float_to_decimal_string(number: float, decimals: int):
    nb_str = str(number)[::-1].find('.') == -1
    if nb_str -1:
        return f'{int(number):0{decimals}d}'
    return f'{number:0<{decimals-2}.1f}'