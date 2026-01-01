from enum import Enum


class DistributionData(Enum):
    OHLC = 1
    OHLC_No_Avg = 2
    Open = 3
    High = 4
    Low = 5
    Close = 6
    Uniform_Distribution = 7
    Uniform_Presence = 8
    Parabolic_Distribution = 9
    Triangular_Distribution = 10

class SegmentsInterval(Enum):
    From_Profile = 1
    Daily = 2
    Weekly = 3
    Monthly = 4

class ExtraProfile(Enum):
    No = 1,
    Mini = 2,
    Weekly = 4,
    Monthly = 5,
    Fixed = 6,


class ProfileSmooth(Enum):
    Gaussian = 1,
    Savitzky_Golay = 2

class ProfileNode(Enum):
    LocalMinMax = 1,
    Percentile = 2,
    Topology = 3

class ProfileFilter:
    def __init__(self, profile_smooth: ProfileSmooth = ProfileSmooth.Gaussian,
                 profile_node: ProfileNode = ProfileNode.LocalMinMax,
                 strong_only: bool = False, strong_hvn_pct: float = 23.6, strong_lvn_pct: float = 38.2,
                 hvn_pctile: int = 90, lvn_pctile: int = 25):
        """
        prefix => ['tpo', 'tpo_mini', ...] or  ['vp', vp_mini', ...] \n
        hvn => High Volume Node (bell-shapes between LVN splits) \n
        lvn => Low Volume Node

        'prefix_[hvn,lvn]_levels'
            - Prices of each HVN / LVN
            - [ (low, center, high), (low, center, high), ... ]
            - Low/High (symmetric bands, ex: 70% or 25% of current mini-bell) around POC or LVN
            - HVN => POC (center) derived from the current mini-bell
            - LVN => (center) is the current LVN
        'prefix_[hvn,lvn]_idx'
            - self-explanatory
            - same structure as above.
        'prefix_[hvn,lvn]_mask'
            - Zero values with the same length of profile
            - if [0... 1, 2, 3, ...0] => [low, center, high]
        'prefix_[hvn,lvn]_raw_levels':
            - Raw POCs/LVNs derived from ProfileNode.[LocalMinMax/Topology]
            - [price, price, ...]
        'prefix_[hvn,lvn]_raw_idx',
            - self-explanatory
            - same structure as above.
        'prefix_[hvn,lvn]_raw_mask'
            - Zero values with the same length of profile
            - '1' => POC or LVN
        """
        self.profile_smooth = profile_smooth
        self.profile_node = profile_node

        self.strong_only = strong_only
        self.strong_hvn_pct = strong_hvn_pct
        self.strong_lvn_pct = strong_lvn_pct

        self.hvn_pctile = hvn_pctile
        self.lvn_pctile = lvn_pctile

        self.hvn_band_pct = 61.8
        self.lvn_band_pct = 23.6

    def levels(self, hvn_band_pct: float = 61.8, lvn_band_pct: float = 23.6):
        """
        Percentages for Symmetric Bands, ex: 70% or 25% of current mini-bell around POC or LVN
        """
        self.hvn_band_pct = hvn_band_pct
        self.lvn_band_pct = lvn_band_pct
