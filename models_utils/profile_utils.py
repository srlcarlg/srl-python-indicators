
import numpy as np
import pandas as pd

from models_utils.profile_models import SegmentsInterval, ExtraProfile, ProfileFilter, ProfileSmooth, ProfileNode

# Prefix
def get_prefix(extra_profile: ExtraProfile, is_vp: bool = False):
    extra_name = '' if extra_profile == ExtraProfile.No else f"_{extra_profile.name.lower()}"
    return f"vp{extra_name}" if is_vp else f"tpo{extra_name}"

# Intervals
def get_intervals_list(df: pd.DataFrame, interval: pd.Timedelta | pd.DateOffset):
    first_date = df['datetime'].iat[0].normalize() # any datetime to 00:00:00
    first_interval_date = first_date + interval
    first_interval_df = df[df['datetime'] < first_interval_date]

    df_list = [first_interval_df]

    last_date = df['datetime'].tail(1).values[0]
    current_date = first_interval_date
    while current_date < last_date:
        start_interval_date = current_date
        end_interval_date = start_interval_date + interval
        interval_df = df.loc[(df['datetime'] >= start_interval_date) & (df['datetime'] < end_interval_date)]

        df_list.append(interval_df)
        current_date = end_interval_date

    return df_list

# Segments
def create_segments(interval_open: float, interval_highest: float, interval_lowest: float, row_height: float):
    interval_segments = []

    prev_segment = interval_open
    while prev_segment >= (interval_lowest - row_height):
        interval_segments.append(prev_segment)
        prev_segment = abs(prev_segment - row_height)
    prev_segment = interval_open
    while prev_segment <= (interval_highest + row_height):
        interval_segments.append(prev_segment)
        prev_segment = abs(prev_segment + row_height)

    interval_segments.sort()

    return np.array(interval_segments)

def create_shared_segments(df_ohlc: pd.DataFrame, row_height: float, segments_interval: SegmentsInterval):
    # Get all intervals to be calculated.
    match segments_interval:
        case SegmentsInterval.Weekly:
            period_list = df_ohlc.index.to_period('W').unique()
            period_list = [period_list[i] for i in range(len(period_list))]
        case SegmentsInterval.Monthly:
            first_days_of_month = df_ohlc.index.to_period('M').unique().to_timestamp()
            first_mondays = pd.DatetimeIndex(first_days_of_month).shift(1, freq='WOM-1MON')
            period_list = [str(first_mondays[i]).split(' ')[0] for i in range(len(first_mondays))]
        case _:
            period_list = df_ohlc.index.to_period('D').unique()
            period_list = [period_list[i] for i in range(len(period_list))]

    # init dictionary with datetime keys
    segments_dict = dict.fromkeys([str(name) for name in period_list], np.empty(1))

    interval_str = segments_interval.name
    normalized_dates = df_ohlc.index.normalize()

    for period_date in period_list:
        period_str = str(period_date)

        if interval_str in ['Daily', 'Monthly']:
            start_date = period_str

            if start_date not in normalized_dates:
                raise ValueError(
                    f"Expected '{start_date}' starting point for {interval_str} segments doesn't exist!")

            interval_df = df_ohlc.loc[(normalized_dates == start_date)]
        else:
            start_date, end_date = period_str.split('/')

            if start_date not in normalized_dates:
                raise ValueError(
                    f"Expected '{start_date}' starting point for {interval_str} segments doesn't exist!")

            interval_df = df_ohlc.loc[(normalized_dates >= start_date) & (df_ohlc['datetime'] < end_date)]

        interval_open = interval_df['open'].iat[0]
        interval_highest = interval_df['high'].max()
        interval_lowest = interval_df['low'].min()

        interval_segments = create_segments(interval_open, interval_highest, interval_lowest, row_height)
        segments_dict[period_str] = interval_segments

    return segments_dict

def get_segments(interval_datetime, interval_open: float, interval_highest: float, interval_lowest: float,
                 row_height: float, segments_interval: SegmentsInterval,
                 shared_segments_dict: dict):
    if segments_interval == SegmentsInterval.From_Profile:
        return create_segments(interval_open, interval_highest, interval_lowest, row_height)
    else:
        match segments_interval:
            case SegmentsInterval.Weekly:
                key_name = interval_datetime.to_period('W')
            case SegmentsInterval.Monthly:
                # TODO: Test it
                first_days_of_month = interval_datetime.to_period('M').to_timestamp()
                first_mondays = pd.DatetimeIndex(first_days_of_month).shift(1, freq='WOM-1MON')
                key_name = [str(first_mondays[i]).split(' ')[0] for i in range(len(first_mondays))]
            case _:
                key_name = interval_datetime.to_period('D')

        key_name = str(key_name)
        interval_segments = shared_segments_dict[key_name]

        # Remove segments outside the profile range
        between = (interval_segments >= interval_lowest) & (interval_segments <= interval_highest)

    return interval_segments[between]

# HVN / LVN
def volume_nodes_filter(profile_values: np.array, profile_prices: np.array, p: ProfileFilter):

    p_normalized = _gaussian_smooth(profile_values) \
                   if p.profile_smooth == ProfileSmooth.Gaussian else \
                   _savitzky_golay(profile_values)

    # Get indexes of LVNs/HVNs
    hvns_raw, lvns_raw = [], []
    match p.profile_node:
        case ProfileNode.LocalMinMax:
            lvns_raw, hvns_raw = _find_local_minmax(p_normalized)
        case ProfileNode.Percentile:
            hvns_raw, lvns_raw = _percentile_nodes(p_normalized, p.hvn_pctile, p.lvn_pctile)
        case ProfileNode.Topology:
            peaks, valleys = _profile_topology(p_normalized)
            hvns_raw, lvns_raw = peaks, valleys

    if p.profile_node == ProfileNode.Percentile:
        hvn_groups = _group_consecutive_indexes(hvns_raw)
        lvn_groups = _group_consecutive_indexes(lvns_raw)

        hvn_prices = [profile_prices[arr_idx].tolist() for arr_idx in hvn_groups]
        lvn_prices = [profile_prices[arr_idx].tolist() for arr_idx in lvn_groups]

        hvn_list = np.zeros(len(profile_prices), dtype=np.int8)
        lvn_list = np.zeros(len(profile_prices), dtype=np.int8)

        hvn_list[hvns_raw] = 1
        hvn_raw_lvls = [profile_prices[idx] for idx in hvns_raw]

        lvn_list[lvns_raw] = 1
        lvn_raw_lvls = [profile_prices[idx] for idx in lvns_raw]

        return (hvn_prices, hvn_groups, hvn_list,
                hvn_raw_lvls, hvns_raw, hvn_list,
                lvn_prices, lvn_groups, lvn_list,
                lvn_raw_lvls, lvns_raw, lvn_list)

    # Filter it
    if p.strong_only:
        global_poc = p_normalized.max()
        hvn_pct = round(p.strong_hvn_pct / 100, 3)
        lvn_pct = round(p.strong_lvn_pct / 100, 3)
        strong_hvns = hvns_raw[p_normalized[hvns_raw] >= hvn_pct * global_poc]
        strong_lvns = lvns_raw[p_normalized[lvns_raw] <= lvn_pct * global_poc]
        hvns_raw, lvns_raw = strong_hvns, strong_lvns

    # Split profile by LVNs
    areas_between = []
    start = 0
    # start = lvns[0] if len(lvns) > 0 else 0
    for lvn in lvns_raw:
        areas_between.append((start, lvn))
        start = lvn
    areas_between.append((start, len(p_normalized)))
    # areas_between.append((start, len(p_normalized) - 1))

    # Extract mini-bells
    bells = []
    for start_index, end_index in areas_between:
        area = p_normalized[start_index:end_index]
        if len(area) == 0:
            continue

        poc_idx = start_index + np.argmax(area)
        bells.append((start_index, end_index, poc_idx))

    # Extract HVN/LVN/POC + Levels
    # [(low, center, high), ...]
    hvn_levels = []
    hvn_indexes = []

    lvn_levels = []
    lvn_indexes = []

    len_prices = len(profile_prices) - 1
    for bell in bells:
        start_idx, end_idx, poc_idx = bell
        poc_idx = min(len_prices, poc_idx)

        # HVNs/POCs + levels
        lvl_low, lvl_high = _hvn_symmetric_va(start_idx, end_idx, poc_idx, round(p.hvn_band_pct / 100, 3))
        lvl_low, lvl_high = lvl_low, min(len_prices, lvl_high)

        price_tuple = (profile_prices[lvl_low], profile_prices[poc_idx], profile_prices[lvl_high])
        hvn_levels.append(price_tuple)

        idx_tuple = (lvl_low, poc_idx, lvl_high)
        hvn_indexes.append(idx_tuple)

        # LVNs + levels
        lvl_low, lvl_high = _lvn_symmetric_band(start_idx, end_idx, round(p.lvn_band_pct / 100, 3))
        lvl_low, lvl_high = lvl_low, min(len_prices, lvl_high)

        price_tuple = (profile_prices[lvl_low], profile_prices[start_idx], profile_prices[lvl_high])
        lvn_levels.append(price_tuple)

        idx_tuple = (lvl_low, start_idx, lvl_high)
        lvn_indexes.append(idx_tuple)

    # Create a mask of False values
    # Use integer instead of boolean type to use np.where in _plotly_workaround later (without .astype)
    # Add values between 1~3 to identify if it's low/center/high of respective node.
    hvn_list =  np.zeros(len(profile_prices), dtype=np.int8)
    hvn_raw_list = np.zeros(len(profile_prices), dtype=np.int8)
    lvn_list =  np.zeros(len(profile_prices), dtype=np.int8)
    lvn_raw_list = np.zeros(len(profile_prices), dtype=np.int8)

    # list of tuples to list(integers)
    hvn_low_mask = [tpl[0] for tpl in hvn_indexes]
    hvn_center_mask = [tpl[1] for tpl in hvn_indexes]
    hvn_high_mask = [tpl[2] for tpl in hvn_indexes]

    lvn_low_mask = [tpl[0] for tpl in lvn_indexes]
    lvn_center_mask = [tpl[1] for tpl in lvn_indexes]
    lvn_high_mask = [tpl[2] for tpl in lvn_indexes]

    # old ones, just 0-1
    # hvn_mask = [idx for tpl in hvn_indexes for idx in tpl]
    # lvn_mask = [idx for tpl in lvn_indexes for idx in tpl]

    # Set the indices to True
    hvn_list[hvn_low_mask] = 1
    hvn_list[hvn_center_mask] = 2
    hvn_list[hvn_high_mask] = 3

    lvn_list[lvn_low_mask] = 1
    lvn_list[lvn_center_mask] = 2
    lvn_list[lvn_high_mask] = 3

    # Raw POCs, since the current POCs are derived from LVN splits (mini bells)
    # TODO length verification
    # hvns_raw = hvns_raw[:-1]
    hvn_raw_list[hvns_raw] = 1
    hvn_raw_lvls = [profile_prices[idx] for idx in hvns_raw]

    lvn_raw_list[lvns_raw] = 1
    lvn_raw_lvls = [profile_prices[idx] for idx in lvns_raw]

    return (hvn_levels, hvn_indexes, hvn_list,
            hvn_raw_lvls, hvns_raw, hvn_raw_list,
            lvn_levels, lvn_indexes, lvn_list,
            lvn_raw_lvls, lvns_raw, lvn_raw_list)

# Functions generated by LLM
# Smoothing
def _gaussian_smooth(arr, sigma=2):
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")

def _savitzky_golay(y, window_size=9, poly_order=3):
    """
    Savitzky–Golay filter

    y           : input signal (1D array)
    window_size : odd integer (e.g. 7, 9, 11)
    poly_order  : polynomial order (e.g. 2 or 3)
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if poly_order >= window_size:
        raise ValueError("poly_order must be < window_size")

    half = window_size // 2

    # Design matrix
    x = np.arange(-half, half + 1)
    A = np.vander(x, poly_order + 1, increasing=True)

    # Pseudoinverse
    coeffs = np.linalg.pinv(A)[0]

    # Pad signal (edge handling)
    y_padded = np.pad(y, (half, half), mode="edge")

    # Convolution
    return np.convolve(y_padded, coeffs[::-1], mode="valid")

# Volume Nodes (LVN / HVN)
def _percentile_nodes(profile, hvn_pct: int = 90, lvn_pct: int = 15):
    hvn_threshold = np.percentile(profile, hvn_pct)
    lvn_threshold = np.percentile(profile, lvn_pct)

    hvn_idx = np.where(profile >= hvn_threshold)[0]
    lvn_idx = np.where(profile <= lvn_threshold)[0]

    return hvn_idx, lvn_idx

def _find_local_minmax(arr):
    minimum = np.where((arr[1:-1] < arr[:-2]) & (arr[1:-1] < arr[2:]))[0] + 1
    maximum = np.where((arr[1:-1] > arr[:-2]) &(arr[1:-1] > arr[2:]))[0] + 1
    return minimum, maximum

def _profile_topology(profile):
    # First and second derivatives
    d1 = np.gradient(profile)
    d2 = np.gradient(d1)

    # Peak detection (HVN / POC)
    peaks = np.where(
        (np.sign(d1[:-1]) > 0) &  # rising
        (np.sign(d1[1:]) < 0) &  # falling
        (d2[1:] < 0)  # concave down
    )[0] + 1

    # Valley detection (LVN)
    valleys = np.where(
        (np.sign(d1[:-1]) < 0) &  # falling
        (np.sign(d1[1:]) > 0) &  # rising
        (d2[1:] > 0)  # concave up
    )[0] + 1

    return peaks, valleys

# Nodes Levels
# HVN => mini-bell
def _hvn_symmetric_va(start_idx: int, end_idx: int, poc_idx: int, va_pct: float = 0.70):
    # VA calculation is quite expensive, plus I don't see it's going to be useful as levels
    # Just extracting the 70% for both sides from POC (100%) is cheap and enough.
    # LLM called it as => “Symmetric POC Band”
    width = end_idx - start_idx
    half = int(width * va_pct / 2)
    return max(start_idx, poc_idx - half), min(end_idx, poc_idx + half)

# LVN => start/end of each mini-bell
def _lvn_symmetric_band(lvn, next_lvn, band_pct=0.25):
    # After some prompt pain, it's a version hvn_symmetric_va
    # just uses 25% Width of the current HVN for both sides
    # centralized into LVN, of course.
    width = next_lvn - lvn
    radius = int(width * band_pct / 2)

    # Manual fix, clamp it
    low = max(0, lvn - radius)
    high = min(next_lvn, lvn + radius)

    return low, high

# Percentile
def _group_consecutive_indexes(indices):
    if len(indices) == 0:
        return []

    groups = [[indices[0]]]
    for i in indices[1:]:
        if i == groups[-1][-1] + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups