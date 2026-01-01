from enum import Enum
from custom_mas import MAType


class StrengthFilter:
    def __init__(self, ma_type: MAType = MAType.Exponential, ma_period: int = 5,
                       ratio_1: float = 1.5, ratio_2: float = 2):
        self.ma_type = ma_type
        self.ma_period = ma_period
        self.ratio_1 = ratio_1
        self.ratio_2 = ratio_2

class FilterRatio(Enum):
    Fixed = 1,
    Percentage = 2,

class SpikeFilterType(Enum):
    MA = 1,
    StdDev = 2,
    L1Norm = 3,
    SoftMax_Power = 4,

class SpikeFilter:
    def __init__(self, filter_type: SpikeFilterType = SpikeFilterType.SoftMax_Power,
                 filter_ratio: FilterRatio = FilterRatio.Percentage,
                 ma_type: MAType = MAType.Simple,
                 ma_period: int = 20,
                 thresholds: tuple = (0.5, 1.2, 2.5, 3.5, 3.51),
                 p_period = 10,
                 percentage: tuple = (23.6, 38.2, 61.8, 100, 101)):
        """
        Tick Spike Filter settings

        Parameters
        ----------
        filter_type : SpikeFilterType
            self-explanatory.
        filter_ratio : FilterRatio
            self-explanatory.
        ma_type : MAType
            self-explanatory.
        ma_period : int
            self-explanatory, it's also used by SpikeFilterType.[L1Norm, SoftMax_Power]
        p_period : int
            for FilterRatio.Percentage
        thresholds : tuple
            for FilterRatio.Fixed, default/generic thresholds for MA/StdDev.
        percentage : tuple
            for FilterRatio.Percentage
        """
        self.filter_type = filter_type
        self.filter_ratio = filter_ratio
        # filter
        self.ma_type = ma_type
        self.ma_period = ma_period
        # thresholds - fixed
        self.lowest = thresholds[0]
        self.low = thresholds[1]
        self.average = thresholds[2]
        self.high = thresholds[3]
        self.ultra = thresholds[4]
        # percentages - percentile
        self.p_period = p_period
        self.lowest_pct = percentage[0]
        self.low_pct = percentage[1]
        self.average_pct = percentage[2]
        self.high_pct = percentage[3]
        self.ultra_pct = percentage[4]
        # ultra levels]
        self.max_count = 2

    def levels(self, max_count = 2):
        """
        Self-explanatory
        """
        self.max_count = max_count


class FilterType(Enum):
    MA = 1,
    StdDev = 2,
    Both = 3
    SoftMax_Power = 4,
    L2Norm = 5,
    MinMax = 6

class UltraBubblesBreak(Enum):
    Close_Only = 1,
    Close_plus_BarBody = 2,
    OHLC_plus_BarBody = 3

class UltraBubblesLevel(Enum):
    High_Low = 1,
    HighOrLow_Close = 2,
    HighOrLow_Open = 3,

class BubblesFilter:
    def __init__(self, filter_type: FilterType = FilterType.MA, filter_ratio: FilterRatio = FilterRatio.Percentage,
                 ma_type: MAType = MAType.Exponential,
                 ma_period: int = 20,
                 thresholds: tuple = (0.5, 1.2, 2.5, 3.5, 3.51),
                 p_period = 20,
                 percentage: tuple = (40, 70, 90, 97, 99)):
        """
        Bubbles Chart (filter) settings

        Parameters
        ----------
        filter_type : FilterType
            self-explanatory.
        filter_ratio : FilterRatio
            self-explanatory.
        ma_type : MAType
            self-explanatory.
        ma_period : int
            self-explanatory, it's also used by FilterType.[L2Norm, MinMax, SoftMax_Power]
        p_period : int
            for FilterRatio.Percentage
        thresholds : tuple
            for FilterRatio.Fixed, default/generic thresholds for MA/StdDev.
        percentage : tuple
            for FilterRatio.Percentage
        """
        self.filter_type = filter_type
        self.filter_ratio = filter_ratio
        # filter
        self.ma_type = ma_type
        self.ma_period = ma_period
        # thresholds - fixed
        self.lowest = thresholds[0]
        self.low = thresholds[1]
        self.average = thresholds[2]
        self.high = thresholds[3]
        self.ultra = thresholds[4]
        # percentages - percentile
        self.p_period = p_period
        self.lowest_pct = percentage[0]
        self.low_pct = percentage[1]
        self.average_pct = percentage[2]
        self.high_pct = percentage[3]
        self.ultra_pct = percentage[4]
        # ultra levels
        self.level_size = UltraBubblesLevel.High_Low
        self.break_at = UltraBubblesBreak.Close_Only
        self.max_count = 2

    def levels(self, level_size: UltraBubblesLevel = UltraBubblesLevel.High_Low,
               break_at = UltraBubblesBreak.Close_Only, max_count = 2):
        """
        Self-explanatory
        """
        self.level_size = level_size
        self.break_at = break_at
        self.max_count = max_count


class LevelInfo:
    def __init__(self, top: float = 0.0, bottom: float = 0.0, level_idx: int = 0, touch_count: int = 0,
                 is_active: bool = True):
        self.top = top
        self.bottom = bottom
        self.level_idx = level_idx
        self.touch_count = touch_count
        self.is_active = is_active


class SpikePlot:
    def __init__(self, spike: bool = True, spike_source: str = 'bs_sum', spike_strength: bool = False,
             spike_levels: bool = False, spike_levels_coloring = 'heatmap',
             spike_chart: bool = False, spike_chart_coloring = 'heatmap',):
        """
        Parameters to plot the Spike Filter

        Parameters
        ----------
        spike : bool
            self-explanatory.
        spike_source : str
            ['delta', 'sum', 'bs_sum']
        spike_strength : bool
            debug each value of delta-profile
        spike_levels : bool
            self-explanatory, don't use
        spike_levels_coloring : str
            ['heatmap', 'plusminus']
        spike_chart : bool
            self-explanatory.
        spike_chart_coloring : str
            ['heatmap', 'plusminus']
        """
        _sources = ['delta', 'sum', 'bs_sum']
        if spike_source not in _sources:
            raise ValueError(f"Only {_sources} options are valid.")

        _colors = ['heatmap', 'plusminus']
        if spike_levels_coloring not in _colors or spike_chart_coloring not in _colors:
            raise ValueError(f"Only {_colors} options are valid.")

        self.spike = spike
        self.spike_source = spike_source
        self.spike_strength = spike_strength
        self.spike_levels = spike_levels
        self.spike_levels_coloring = spike_levels_coloring
        self.spike_chart = spike_chart
        self.spike_chart_coloring = spike_chart_coloring