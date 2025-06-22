import pandas as pd
import mplfinance as mpf

class MultiVwap:
    def __init__(self, df_ohlc: pd.DataFrame):
        _expected = ['open', 'high', 'low', 'close', 'volume']
        for column in _expected:
            if column not in df_ohlc.columns:
                raise ValueError(f"'{column}' column from the expected {_expected} doesn't exist!")

        self._df = df_ohlc
        self._volume = df_ohlc['volume']
        self._high = df_ohlc['high'] * df_ohlc['volume']
        self._median = ((df_ohlc['high'] + df_ohlc['low']) / 2) * df_ohlc['volume']
        self._low = df_ohlc['low'] * df_ohlc['volume']

    def plot(self, date: str = None, show_weekly: bool = True, show_monthly: bool = False):
        """
        Plot with mplfinance.

        :param date: For anchored vwap
        :param show_weekly: bool
        :param show_monthly: bool
        """
        anchored = []
        if date is not None:
            self.anchored(date)
            anchored = [mpf.make_addplot(self._df['anchored_vwap_high'], color='deepskyblue', label="anchored_high"),
                        mpf.make_addplot(self._df['anchored_vwap_median'], color='lightyellow',
                                         linestyle='dashdot', label="anchored_median"),
                        mpf.make_addplot(self._df['anchored_vwap_low'], color='orange', label="anchored_low"),]

        weekly = []
        if show_weekly is True:
            self.weekly()
            weekly = [mpf.make_addplot(self._df['weekly_vwap_high'], color='goldenrod', label="weekly"),
                      mpf.make_addplot(self._df['weekly_vwap_median'], color='goldenrod'),
                      mpf.make_addplot(self._df['weekly_vwap_low'], color='goldenrod'), ]

        monthly = []
        if show_monthly is True:
            self.monthly()
            monthly = [mpf.make_addplot(self._df['monthly_vwap_high'], color='crimson', label="monthly"),
                       mpf.make_addplot(self._df['monthly_vwap_median'], color='crimson'),
                       mpf.make_addplot(self._df['monthly_vwap_low'], color='crimson')]

        self.daily()
        daily = [mpf.make_addplot(self._df['daily_vwap_high'], color='gold', label="daily"),
               mpf.make_addplot(self._df['daily_vwap_median'], color='gold'),
               mpf.make_addplot(self._df['daily_vwap_low'], color='gold'),]

        apd = daily
        if show_weekly is True:
            apd.extend(weekly)
        if show_monthly is True:
            apd.extend(monthly)
        if date is not None:
            apd.extend(anchored)

        s = mpf.make_mpf_style(base_mpl_style='seaborn-v0_8')
        mpf.plot(self._df, type='candle', style=s, addplot=apd, warn_too_much_data=len(self._df) + 1,
                 title="Multi VWAP",
                 figsize=(12.5,6),
                 returnfig=True,
                 scale_padding=0.2)

        mpf.show()

    def anchored(self, date: str):
        """
        The following (3) columns will be added: anchored_vwap_[high, median, low]
        :param date:
        :return: pd.DataFrame
        """
        anchored_volume = self._volume[self._volume.index >= date].copy()
        anchored_high = self._high[self._high.index >= date].copy()
        anchored_median = self._median[self._median.index >= date].copy()
        anchored_low = self._low[self._low.index >= date].copy()
        self._df['anchored_vwap_high'] = anchored_high.cumsum() / anchored_volume.cumsum()
        self._df['anchored_vwap_median'] = anchored_median.cumsum() / anchored_volume.cumsum()
        self._df['anchored_vwap_low'] = anchored_low.cumsum() / anchored_volume.cumsum()
        return self._df

    def daily(self):
        """
        The following (3) columns will be added: daily_vwap_[high, median, low]
        :return: pd.DataFrame
        """
        daily_volume = self._volume.groupby(self._df.index.to_period('D')).cumsum()
        period = self._df.index.to_period('D')
        self._df['daily_vwap_high'] = self._high.groupby(period).cumsum() / daily_volume
        self._df['daily_vwap_median'] = self._median.groupby(period).cumsum() / daily_volume
        self._df['daily_vwap_low'] = self._low.groupby(period).cumsum() / daily_volume
        return self._df

    def weekly(self):
        """
        The following (3) columns will be added: weekly_vwap_[high, median, low]
        :return: pd.DataFrame
        """
        weekly_volume = self._volume.groupby(self._df.index.to_period('W')).cumsum()
        period = self._df.index.to_period('W')
        self._df['weekly_vwap_high'] = self._high.groupby(period).cumsum() / weekly_volume
        self._df['weekly_vwap_median'] = self._median.groupby(period).cumsum() / weekly_volume
        self._df['weekly_vwap_low'] = self._low.groupby(period).cumsum() / weekly_volume
        return self._df

    def monthly(self):
        """
        The following (3) columns will be added: monthly_vwap_[high, median, low]
        :return: pd.DataFrame
        """
        monthly_volume = self._volume.groupby(self._df.index.to_period('M')).cumsum()
        period = self._df.index.to_period('M')
        self._df['monthly_vwap_high'] = self._high.groupby(period).cumsum() / monthly_volume
        self._df['monthly_vwap_median'] = self._median.groupby(period).cumsum() / monthly_volume
        self._df['monthly_vwap_low'] = self._low.groupby(period).cumsum() / monthly_volume
        return self._df
