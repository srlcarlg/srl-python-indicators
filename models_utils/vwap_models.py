from enum import Enum

class BandsType(Enum):
    No = 1
    Stddev = 2,
    Percentile = 3,
    Percentile_Asymmetric = 4,

class BandsFilter:
    def __init__(self, bands_type: BandsType = BandsType.Stddev, volume_weighted: bool = True,
                 multipliers: tuple = (1.236, 2.382, 3.618), pctile: tuple = (80, 90, 97),
                 pctile_up: tuple = (70, 85, 95), pctile_down: tuple = (70, 85, 95)):
        """
        The following (6) columns will be added: {prefix}_upper_[1,2,3], {prefix}_lower_[1,2,3],

        Uses '{prefix}_vwap_median' column to calculate the bands.

        Parameters
        ----------
        bands_type : BandsType
            self-explanatory
        volume_weighted : bool
            Use "volume weighted bands" version of the selected bands_type
        multipliers : tuple
            For BandsType.StdDev
        pctile : tuple
            For BandsType.Percentile
        pctile_up : tuple
            For BandsType.Percentile_Asymmetric
        pctile_down : tuple
            For BandsType.Percentile_Asymmetric
        """
        self.bands_type = bands_type
        self.volume_weighted = volume_weighted
        self.multipliers = multipliers
        self.pctile = pctile
        self.pctile_up = pctile_up
        self.pctile_down = pctile_down