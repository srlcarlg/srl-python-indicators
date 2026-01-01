import numpy as np
import pandas as pd
from models_utils.ww_models import ZigZagMode, Direction, ZigZagInit, WavesInit, WavesMode, PriorityMode


# Reversal Logic (Renko only)
def reversal_logic(df: pd.DataFrame, index: int, wi: WavesInit):
        is_up = df['close'].iat[index] > df['open'].iat[index]
        is_up = Direction.UP if is_up else Direction.DOWN

        # show current wave
        calculate_waves(df, None, wi, is_up, wi.trend_start_index, index, False)
        df['trendline'].iat[index] = df['open'].iat[index]

        if not _reversal_direction_changed(df, index):
            return

        # end of direction
        df['end_wave'].iat[index] = df['close'].iat[index]
        calculate_waves(df, None, wi, is_up, wi.trend_start_index, index, True)
        wi.trend_start_index = index + 1

        value_poit = 'high' if is_up else 'low'
        df['turning_point'].iat[index] = df[value_poit].iat[index]

def _reversal_direction_changed(df: pd.DataFrame, index: int):
    dynamic_current = df['close'].iat[index] > df['open'].iat[index]
    next_is_up = df['close'].iat[index + 1] > df['open'].iat[index + 1]
    next_is_down = df['close'].iat[index + 1] < df['open'].iat[index + 1]
    prev_is_up = df['close'].iat[index - 1] > df['open'].iat[index - 1]
    prev_is_down = df['close'].iat[index - 1] < df['open'].iat[index - 1]

    return (prev_is_up and dynamic_current and next_is_down) or (prev_is_down and dynamic_current and next_is_down) \
        or (prev_is_down and not dynamic_current and next_is_up) or (prev_is_up and not dynamic_current and next_is_up)

# ZigZag Logic
def zigzag_logic(zz: ZigZagInit, df: pd.DataFrame,
                  df_htf: pd.DataFrame | None,
                  df_ltf: pd.DataFrame | None,
                  wi: WavesInit,
                  index: int, price_tuple: tuple):
    low, high, prev_low, prev_high = price_tuple

    # first index
    if zz.extremum_price == 0:
        zz.extremum_price = high
        zz.extremum_index = index

    if zz.mode == ZigZagMode.NoLag_HighLow and zz.no_lag_priority != PriorityMode.No:
        if zz.no_lag_priority == PriorityMode.Skip or _nolag_both_is_pivot(zz, df, df_htf, df_ltf, wi, index, price_tuple):
            return

    if zz.direction == Direction.DOWN:
        if low <= zz.extremum_price:
            _move_extremum(zz, df, df_htf, wi, index, low)
        elif _zigzag_direction_changed(zz, price_tuple):
            _set_extremum(zz, df, df_htf, wi, index, high, False)
            zz.direction = Direction.UP
    else:
        if high >= zz.extremum_price:
            _move_extremum(zz, df, df_htf, wi, index, high)
        elif _zigzag_direction_changed(zz, price_tuple):
            _set_extremum(zz, df, df_htf, wi, index, low, False)
            zz.direction = Direction.DOWN

def _zigzag_direction_changed(zz: ZigZagInit, price_tuple: tuple):
    low, high, prev_low, prev_high = price_tuple
    match zz.mode:
        case ZigZagMode.Percentage:
            if zz.direction == Direction.DOWN:
                return high >= zz.extremum_price * (1.0 + zz.pct_value * 0.01)
            else:
                return low <= zz.extremum_price * (1.0 - zz.pct_value * 0.01)
        case ZigZagMode.NoLag_HighLow:
            both_is_pivot = high > prev_high and low < prev_low
            high_is_pivot = high > prev_high and low >= prev_low
            low_is_pivot = low < prev_low and high <= prev_high
            if both_is_pivot:
                return False
            return low_is_pivot if zz.direction == Direction.UP else high_is_pivot
        case _:
            value = zz.pips_value * zz.symbol_pipsize
            if zz.direction == Direction.DOWN:
                return abs(zz.extremum_price - high) >= value
            else:
                return abs(low - zz.extremum_price) >= value

def _move_extremum(zz: ZigZagInit, df: pd.DataFrame, df_htf: pd.DataFrame, wi: WavesInit,
                   idx: int, price: float):
    df['trendline'].iat[zz.extremum_index] = np.NaN
    _set_extremum(zz, df, df_htf, wi, idx, price, True)

def _set_extremum(zz: ZigZagInit, df: pd.DataFrame, df_htf: pd.DataFrame | None, wi: WavesInit,
                  idx: int, price: float, is_move: bool):
    if is_move:
        # show current wave
        calculate_waves(df, df_htf, wi, zz.direction, wi.trend_start_index, idx, False)
    else:
        # end of direction
        df['end_wave'].iat[zz.extremum_index] = zz.extremum_price
        calculate_waves(df, df_htf, wi, zz.direction, wi.trend_start_index, idx, True)
        wi.trend_start_index = zz.extremum_index + 1

        # current idx is the turning point
        column_name = 'high' if zz.direction == Direction.UP else 'low'
        df['turning_point'].iat[idx] = df[column_name].iat[idx]

    if df_htf is not None and is_move:
        # Same workaround to remove the behavior of shift(nº) when moving the extremum price at
        # custom timeframe source (higher timeframe)
        is_up = zz.direction == Direction.UP
        column_name = 'high' if is_up else 'low'
        extreme_price = df[column_name].iat[zz.extremum_index]
        current_price = df[column_name].iat[idx]

        condition = current_price <= extreme_price if is_up else current_price >= extreme_price
        zz.extremum_index = zz.extremum_index if condition else idx
    else:
        zz.extremum_index = idx

    zz.extremum_price = price

    df['trendline'].iat[zz.extremum_index] = zz.extremum_price

def _nolag_both_is_pivot(zz: ZigZagInit, df: pd.DataFrame, df_htf: pd.DataFrame | None, df_ltf: pd.DataFrame,
                        wi: WavesInit, index: int, price_tuple: tuple):
    low, high, prev_low, prev_high = price_tuple

    both_is_pivot = high > prev_high and low < prev_low
    if not both_is_pivot or zz.no_lag_priority != PriorityMode.Auto:
        return False
        
    start_date = df['datetime'].iat[index]
    end_date = df['datetime'].iat[index + 1]
    if df_htf is not None:
        start_date = df_htf.loc[df_htf['datetime'] >= start_date].head(1)
        start_date = start_date['datetime'].iat[0]
        end_date = df_htf.loc[df_htf['datetime'] >= end_date].head(1)
        end_date = end_date['datetime'].iat[0]

    high_is_first = _auto_priority(start_date, end_date, df_ltf, price_tuple)
    if high_is_first:
        if zz.direction == Direction.UP:
            if high > zz.extremum_price:
                # Fix => C# version is using 'extremumIndex' instead of 'index',
                df['trendline'].iat[index] = high
            _set_extremum(zz, df, None, wi, index, low, True)
            zz.direction = Direction.DOWN
    else:
        if zz.direction == Direction.DOWN:
            if low < zz.extremum_price:
                df['trendline'].iat[index] = low
            _set_extremum(zz, df, None, wi, index, high, True)
            zz.direction = Direction.UP


def _auto_priority(bar_start_time, bar_end_time, df_mtf: pd.DataFrame, price_tuple: tuple):
    low, high, prev_low, prev_high = price_tuple

    lower_tf = df_mtf.loc[(df_mtf["datetime"] >= bar_start_time) & (df_mtf["datetime"] <= bar_end_time)]

    first_is_high = False
    at_least_one = False
    for i in range(len(lower_tf)):
        if lower_tf["high"].iat[i] > prev_high:
            first_is_high = True
            at_least_one = True
            break
        if lower_tf["low"].iat[i] < prev_low:
            # Fix => C# version sets True for first_is_high
            first_is_high = False
            at_least_one = True
            break

    if not at_least_one:
        subt_high = abs(high - prev_high)
        subt_low = abs(prev_low - low)
        return subt_high >= subt_low

    return first_is_high

# Waves Logic
def calculate_waves(df: pd.DataFrame, df_htf: pd.DataFrame | None, wi: WavesInit, direction: Direction,
                     first_brick_index: int, last_brick_index: int, direction_changed: bool):

    def cumul_volume():
        if first_brick_index == last_brick_index:
            return df['volume'].iat[last_brick_index]
        # (last_brick_index + 1) because of python's range behavior
        # first=1 and last=2: [1]
        # first=1 and last=3: [1, 2]
        volume = 0
        for i in range(first_brick_index, last_brick_index + 1):
            volume += df['volume'].iat[i]
        return volume

    def cumul_renko():
        if first_brick_index == last_brick_index:
            return 1
        # (last_brick_index + 1) because of python's range behavior
        # first=1 and last=2: [1]
        # first=1 and last=3: [1, 2]
        renko_count = 0
        for i in range(first_brick_index, last_brick_index + 1):
            renko_count += 1
        return renko_count

    def cumul_price(is_up: bool):
        price = df['high'].iat[last_brick_index] - df['low'].iat[first_brick_index] if is_up else \
            df['high'].iat[first_brick_index] - df['low'].iat[last_brick_index]

        if df_htf is not None:
            first_date = df['datetime'].iat[first_brick_index]
            first_htf = df_htf.loc[df_htf['datetime'] >= first_date].head(1)
            last_date = df['datetime'].iat[last_brick_index]
            last_htf = df_htf.loc[df_htf['datetime'] >= last_date].head(1)
            if len(first_htf) > 0 and len(last_htf) > 0:
                price = last_htf['high'].iat[0] - first_htf['low'].iat[0] if is_up else \
                        first_htf['high'].iat[0] - last_htf['low'].iat[0]

        if price == 0:
            price = 1
        # TODO (maybe) price /= zz.symbol_pipsize
        return abs(round(price, 5))

    def cumul_time():
        open_column = 'datetime' if wi.is_open_time else 'start_time'
        prev_time = df[open_column].iat[first_brick_index]

        close_column = 'end_time' if wi.is_open_time else 'datetime'
        curr_time = df[close_column].iat[last_brick_index]

        df['wave_time'].iat[last_brick_index] = (curr_time - prev_time)
        df['wave_time_ms'].iat[last_brick_index] = (curr_time - prev_time) / pd.Timedelta(milliseconds=1)
        df['wave_time_sec'].iat[last_brick_index] = (curr_time - prev_time) / pd.Timedelta(seconds=1)
        df['wave_time_min'].iat[last_brick_index] = (curr_time - prev_time) / pd.Timedelta(minutes=1)
        df['wave_time_hour'].iat[last_brick_index] = (curr_time - prev_time) / pd.Timedelta(hours=1)

    def other_waves(is_up: bool):
        df['wave_price'].iat[last_brick_index] = cumul_price(is_up)
        cumul_time()

    direction_is_up = direction == Direction.UP

    cumulative_volume = cumul_volume()
    cumulative_renko_or_price = cumul_renko() if wi.is_renko_chart else cumul_price(direction_is_up)
    cumulative_vol_price = round(cumulative_volume / cumulative_renko_or_price, 1)

    df['wave_volume'].iat[last_brick_index] = cumulative_volume
    df['wave_effort_result'].iat[last_brick_index] = cumulative_vol_price

    other_waves(direction_is_up)
    _effort_vs_result_analysis(df, wi, last_brick_index, cumulative_vol_price, direction_changed, direction_is_up)
    _waves_analysis(df, wi, last_brick_index, cumulative_volume, direction_changed, direction_is_up)

    if direction_is_up:
        prev_is_down = df['close'].iat[last_brick_index - 1] < df['open'].iat[last_brick_index - 1]
        next_is_down = df['close'].iat[last_brick_index + 1] < df['open'].iat[last_brick_index + 1]
        _set_previous_waves(wi, cumulative_volume, cumulative_vol_price, prev_is_down,
                                 next_is_down, direction_is_up, direction_changed)
    else:
        prev_is_up = df['close'].iat[last_brick_index - 1] > df['open'].iat[last_brick_index - 1]
        next_is_up = df['close'].iat[last_brick_index + 1] > df['open'].iat[last_brick_index + 1]
        _set_previous_waves(wi, cumulative_volume, cumulative_vol_price, prev_is_up,
                                 next_is_up, direction_is_up, direction_changed)

def _waves_analysis(df: pd.DataFrame, wi: WavesInit, index, cumul_vol: float, is_end_wave: bool, is_up: bool):
    def large_volume():
        have_zero = False
        for i, value in enumerate(wi.prev_waves_volume):
            if value == 0.0:
                have_zero = True
                break
        if have_zero:
            return 0
        return 1 if (cumul_vol + sum(wi.prev_waves_volume)) / 5 * wi.large_wave_ratio < cumul_vol else 0
    # 1 = greater, -1 = lesser
    if is_up:
        df['wave_vs_same_direction'].iat[index] = 1 if cumul_vol > wi.prev_wave_up[0] else -1
        df['wave_vs_previous'].iat[index] = 1 if cumul_vol > wi.prev_wave_down[0] else -1
    else:
        df['wave_vs_same_direction'].iat[index] = 1 if cumul_vol > wi.prev_wave_down[0] else -1
        df['wave_vs_previous'].iat[index] = 1 if cumul_vol > wi.prev_wave_up[0] else -1

    if is_end_wave:
        df['large_wave'].iat[index]= large_volume()

def _effort_vs_result_analysis(df: pd.DataFrame, wi: WavesInit, index: int, cumul_vol_price: float,
                               is_end_wave: bool, is_up: bool):
    def large_effort_result():
        have_zero = False
        for i, value in enumerate(wi.prev_waves_EvsR):
            if value == 0.0:
                have_zero = True
                break
        if have_zero:
            return 0
        return 1 if (cumul_vol_price + sum(wi.prev_waves_EvsR)) / 5 * wi.large_wave_ratio < cumul_vol_price else 0

    # 1 = greater, -1 = lesser
    if is_up:
        df['effort_result_vs_same_direction'].iat[index] = 1 if cumul_vol_price > wi.prev_wave_up[1] else -1
        df['effort_result_vs_previous'].iat[index] = 1 if cumul_vol_price > wi.prev_wave_down[1] else -1
    else:
        df['effort_result_vs_same_direction'].iat[index] = 1 if cumul_vol_price > wi.prev_wave_down[1] else -1
        df['effort_result_vs_previous'].iat[index] = 1 if cumul_vol_price > wi.prev_wave_up[1] else -1

    if is_end_wave:
        df['large_effort_result'].iat[index] = large_effort_result()

def _set_previous_waves(wi: WavesInit, cumul_vol: float, cumul_vol_price: float, prev_is_dynamic: bool,
                        next_is_dynamic: bool, is_up: bool, direction_changed: bool):
    """
    Exclude the most old wave, keep the 3 others and add current Wave value for most recent Wave

    The previous "wrongly" implementation turns out to be a good filter,
    with the correct implementation of 5 waves, it might give too many yellow bars.
    Since it's useful, keep it.
    """

    def set_ranging():
        # Ranging or 1 renko trend pullback
        # Weis Wave Analysis
        new_wave = [wi.prev_waves_volume[1], wi.prev_waves_volume[2], wi.prev_waves_volume[3], cumul_vol]
        wi.prev_waves_volume = new_wave

        # Effort vs Result Analysis
        new_wave = [wi.prev_waves_EvsR[1], wi.prev_waves_EvsR[2], wi.prev_waves_EvsR[3], cumul_vol_price]
        wi.prev_waves_EvsR = new_wave

        if wi.ignore_ranging:
            cumul_wave = [cumul_vol, cumul_vol_price]
            if is_up: wi.prev_wave_up = cumul_wave
            else : wi.prev_wave_down = cumul_wave

    def set_trend():
        yewave = wi.yellow_waves
        cumul_wave = [cumul_vol, cumul_vol_price]
        if is_up:
            # Weis Wave Analysis
            # Fix => C# version is using _prev_wave_down for UsePrev_SameWave condition
            volume_value = cumul_vol if yewave == yewave.UseCurrent else \
                           wi.prev_wave_up[0] if yewave == yewave.UsePrev_SameWave else \
                           wi.prev_wave_down[0]
            new_wave = [wi.prev_waves_volume[1], wi.prev_waves_volume[2], wi.prev_waves_volume[3], volume_value]
            wi.prev_waves_volume = new_wave

            # Effort vs Result Analysis
            evsr_value = cumul_vol_price if yewave == yewave.UseCurrent else \
                           wi.prev_wave_up[1] if yewave == yewave.UsePrev_SameWave else \
                           wi.prev_wave_down[1]
            new_wave = [wi.prev_waves_EvsR[1], wi.prev_waves_EvsR[2], wi.prev_waves_EvsR[3], evsr_value]
            wi.prev_waves_EvsR = new_wave

            # Prev Wave
            wi.prev_wave_up = cumul_wave
        else:
            # Weis Wave Analysis
            volume_value = cumul_vol if yewave == yewave.UseCurrent else \
                           wi.prev_wave_down[0] if yewave == yewave.UsePrev_SameWave else \
                           wi.prev_wave_up[0]
            new_wave = [wi.prev_waves_volume[1], wi.prev_waves_volume[2], wi.prev_waves_volume[3], volume_value]
            wi.prev_waves_volume = new_wave

            # Effort vs Result Analysis
            evsr_value = cumul_vol_price if yewave == yewave.UseCurrent else \
                         wi.prev_wave_down[1] if yewave == yewave.UsePrev_SameWave else \
                         wi.prev_wave_up[1]
            new_wave = [wi.prev_waves_EvsR[1], wi.prev_waves_EvsR[2], wi.prev_waves_EvsR[3], evsr_value]
            wi.prev_waves_EvsR = new_wave

            # Prev Wave
            wi.prev_wave_down = cumul_wave

    if wi.waves_mode == WavesMode.ZigZag:
        if not direction_changed: return
        set_trend()
        return

    condition_trend = not prev_is_dynamic and direction_changed and next_is_dynamic
    condition_ranging = prev_is_dynamic and direction_changed and next_is_dynamic

    # Keep the current structure to understand later
    if is_up:
        # (prevIsDown && DirectionChanged && nextIsDown);
        if condition_ranging:
            set_ranging()
        # (prevIsUp && DirectionChanged && nextIsDown)
        elif condition_trend:
            set_trend()
    else:
        # (prevIsUp && DirectionChanged && nextIsUp);
        if condition_ranging:
            set_ranging()
        # (prevIsDown && DirectionChanged && nextIsUp);
        elif condition_trend:
            set_trend()


# from odf_utils
def rolling_percentile(a):
    # Percentile Rank of last element inside window
    last = a[-1]
    return np.mean(a <= last) * 100

def l1norm(window_values):
    denom = np.abs(window_values).sum()
    return window_values[-1] / denom if denom != 0 else 1