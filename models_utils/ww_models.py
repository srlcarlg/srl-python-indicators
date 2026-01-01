from enum import Enum
from custom_mas import MAType


class FilterType(Enum):
    MA = 1,
    StdDev = 2,
    Both = 3,
    Normalized_Emphasized = 4
    L1Norm = 5,

class FilterRatio(Enum):
    Fixed = 1,
    Percentage = 2,

class StrengthFilter:
    def __init__(self, filter_type: FilterType = FilterType.Normalized_Emphasized,
                 filter_ratio: FilterRatio = FilterRatio.Percentage, ma_type: MAType = MAType.Exponential,
                 ma_period: int = 5, n_period: int = 20, n_multiplier: int = 1,
                 thresholds: tuple = (0.5, 1.2, 2.5, 3.5, 3.51),
                 percentage: tuple = (23.6, 38.2, 61.8, 100, 101),
                 pctile: tuple = (40, 70, 90, 97, 99),
                 is_open_time: bool = True):
        """
        Strength settings

        Parameters
        ----------
        filter_type : FilterType
            self-explanatory.
        filter_ratio : FilterRatio
            self-explanatory.
        ma_type : MAType
            self-explanatory.
        ma_period : int
            self-explanatory, it's also used by FilterType.L1Norm
        n_period : int
            Used by FilterType.Normalized_Emphasized and FilterRatio.Percentage
        n_multiplier : int
            for FilterType.Normalized_Emphasized (only), increase 'sensibility' of each value.
        thresholds : tuple
            for FilterRatio.Fixed, default/generic thresholds for MA/StdDev.
        percentage : tuple
            for FilterType.Normalized_Emphasized (only).
        pctile : tuple
            for FilterRatio.Percentage (only).
        is_open_time : bool
            Specify if the index/datetime of df_ohlcv is the OpenTime or CloseTime of each bar.
        """
        self.filter_type = filter_type
        self.filter_ratio = filter_ratio
        self.ma_type = ma_type
        self.ma_period = ma_period
        self.is_open_time = is_open_time
        # thresholds
        self.lowest = thresholds[0]
        self.low = thresholds[1]
        self.average = thresholds[2]
        self.high = thresholds[3]
        self.ultra = thresholds[4]
        # normalized / percentile
        self.n_period = n_period
        # normalized
        self.n_multiplier = n_multiplier
        self.lowest_pct = percentage[0]
        self.low_pct = percentage[1]
        self.average_pct = percentage[2]
        self.high_pct = percentage[3]
        self.ultra_pct = percentage[4]
        # percentile
        self.lowest_pctile = pctile[0]
        self.low_pctile = pctile[1]
        self.average_pctile = pctile[2]
        self.high_pctile = pctile[3]
        self.ultra_pctile = pctile[4]


class WavesMode(Enum):
    Reversal = 1,
    ZigZag = 2

class YellowWaves(Enum):
    UseCurrent = 1,
    UsePrev_SameWave = 2,
    UsePrev_InvertWave = 3

class WavesInit:
    def __init__(self, waves_mode: WavesMode = WavesMode.ZigZag,
                 yellow_waves: YellowWaves = YellowWaves.UseCurrent,
                 large_wave_ratio: float = 1.5,
                 ignore_ranging: bool = False,
                 is_open_time: bool = True,
                 is_renko_chart: bool = False):
        """
        Waves Settings

        Parameters
        ----------
        waves_mode : WavesMode
            'Reversal' mode for Renko Chart (only)
        yellow_waves : YellowWaves
            Set which waves should be used for large wave coloring, applies for [Volume, Effort vs Result] waves
        large_wave_ratio : float
            self-explanatory
        ignore_ranging : bool
            Used by WavesMode.Reversal (only)
        is_open_time : bool
            Specify if the index/datetime of df_ohlcv is the OpenTime or CloseTime of each bar.
        is_renko_chart : bool
            self-explanatory
        """
        self.waves_mode = waves_mode
        self.yellow_waves = yellow_waves
        self.large_wave_ratio = large_wave_ratio
        self.ignore_ranging = ignore_ranging
        self.is_open_time = is_open_time
        self.is_renko_chart = is_renko_chart
        # for loop
        self.prev_waves_volume = [0, 0, 0, 0]
        self.prev_waves_EvsR = [0.0, 0.0, 0.0, 0.0]
        self.prev_wave_up = [0, 0]
        self.prev_wave_down = [0, 0]
        self.trend_start_index = 0

    def reset_waves(self):
        self.prev_waves_volume = [0, 0, 0, 0]
        self.prev_waves_EvsR = [0.0, 0.0, 0.0, 0.0]
        self.prev_wave_up = [0, 0]
        self.prev_wave_down = [0, 0]
        self.trend_start_index = 0


class ZigZagMode(Enum):
    Percentage = 1,
    Pips = 2,
    NoLag_HighLow = 3

class PriorityMode(Enum):
    No = 1,
    Auto = 2,
    Skip = 3

class Direction(Enum):
    UP = 1,
    DOWN = 2

class ZigZagInit:
    def __init__(self, zigzag_mode: ZigZagMode = ZigZagMode.NoLag_HighLow,
                 pct_value: float = 0.06, pips_value: float = 2.0,
                 no_lag_priority: PriorityMode = PriorityMode.No):
        """
        ZigZag Settings

        Parameters
        ----------
        zigzag_mode : WavesMode
            self-explanatory
        pct_value : float
            for ZigZagMode.Percentage
        pips_value : float
            for ZigZagMode.Pips
        no_lag_priority : PriorityMode
            For ZigZagMode.NoLag_HighLow, the 'df_ltf' should be provided if PriorityMode.Auto
        """
        self.mode = zigzag_mode
        self.no_lag_priority = no_lag_priority
        self.pct_value = pct_value
        self.pips_value = pips_value
        # self.symbol_pipsize = symbol_pipsize
        self.extremum_index = 0
        self.extremum_price = 0.0
        self.direction = Direction.UP