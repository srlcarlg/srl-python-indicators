"""
Multi VWAP
=====
Python version of Anchored VWAP Buttons developed for cTrader Trading Platform

Features from revision 1 (after Order Flow Aggregated development):
    - StdDev to any VWAP (one StdDev per instance)
Additional Features => that will be implemented to C# version... sometime next year (2026)
    - StdDev with volume-weighted-bands
    - Quantile Symmetric/Asymmetric Bands
        - volume-weighted-bands for both
Python/C# author:
    - srlcarlg
"""
import itertools
from multiprocessing import Pool, cpu_count

import pandas as pd
import mplfinance as mpf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from models_utils.vwap_models import BandsFilter, BandsType
from models_utils.vwap_utils import get_periods_list, stddev_bands, stddev_bands_volume, quantile_bands, \
    quantile_bands_volume, quantile_asymmetric_bands, quantile_asymmetric_bands_volume


class MultiVwap:
    def __init__(self, df_ohlcv: pd.DataFrame | None = None, bands_filter: BandsFilter | None = None):
        """
        Multi-VWAP to any OHLCV chart.

        Usage
        -----

        >>> from multi_vwap import MultiVwap
        >>> vwap = MultiVwap(df_ohlcv)
        >>> df = vwap.daily()
        >>> # Alternatively
        >>> # vwap.weekly(), vwap.monthly, vwap.anchored(date)

        >>> # or
        >>> vwap = MultiVwap()
        >>> df = vwap.daily(df_ohlcv)
        >>> # Alternatively
        >>> # vwap.weekly(df_ohlcv), vwap.monthly(df_ohlcv), vwap.anchored(date, df_ohlcv)

        Parameters
        ----------
        df_ohlcv : pd.DataFrame
            OHLCV with datetime index
        bands_filter : BandsFilter
            self-explanatory
        """
        if df_ohlcv is not None:
            _expected = ['open', 'high', 'low', 'close', 'volume']
            for column in _expected:
                if column not in df_ohlcv.columns:
                    raise ValueError(f"'{column}' column from the expected {_expected} doesn't exist!")
            if 'datetime' not in df_ohlcv.columns:
                df_ohlcv["datetime"] = df_ohlcv.index

        self._df_ohlcv = df_ohlcv
        self._bands_filter = bands_filter if isinstance(bands_filter, BandsFilter) else BandsFilter()

    def _parallel_process_profiles(self, list_of_dfs: list, bands_filter: BandsFilter, prefix_period: str):
        num_processes = cpu_count()
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(self._create_vwap,
                                   zip(list_of_dfs, itertools.repeat(bands_filter), itertools.repeat(prefix_period)))
        return results

    def plot(self, df: pd.DataFrame | None = None, date: str = None, show_weekly: bool = False, show_monthly: bool = False,
                bands_at: str = 'anchored', bands_filter: BandsFilter | None = None):
        """
        Plot with mplfinance.
        """
        _at = ['none', 'anchored', 'daily', 'weekly', 'monthly']
        if bands_at not in _at:
            raise ValueError(f"Only {_at} options are valid.")

        if df is None:
            df = self._df_ohlcv
        f = bands_filter if isinstance(bands_filter, BandsFilter) else self._bands_filter
        bands_name = f.bands_type.name if bands_at != 'none' else 'No'
        bands_select = f.bands_type

        f.bands_type = bands_select if bands_at == 'daily' else BandsType.No
        df = self.daily(df, f)
        daily = [mpf.make_addplot(df['daily_vwap_high'], color='gold', label="daily"),
                mpf.make_addplot(df['daily_vwap_median'], color='gold'),
                mpf.make_addplot(df['daily_vwap_low'], color='gold')]

        anchored = []
        if date is not None:
            f.bands_type = bands_select if bands_at == 'anchored' else BandsType.No
            df = self.anchored(date, df, f)

            anchored = [mpf.make_addplot(df['anchored_vwap_high'], color='deepskyblue', label="anchored_high"),
                        mpf.make_addplot(df['anchored_vwap_median'], color='lightyellow',
                                         linestyle='dashdot', label="anchored_median"),
                        mpf.make_addplot(df['anchored_vwap_low'], color='orange', label="anchored_low"),]

        weekly = []
        if show_weekly:
            f.bands_type = bands_select if bands_at == 'weekly' else BandsType.No
            df = self.weekly(df, f)
            weekly = [mpf.make_addplot(df['weekly_vwap_high'], color='goldenrod', label="weekly"),
                      mpf.make_addplot(df['weekly_vwap_median'], color='goldenrod'),
                      mpf.make_addplot(df['weekly_vwap_low'], color='goldenrod'), ]

        monthly = []
        if show_monthly:
            f.bands_type = bands_select if bands_at == 'monthly' else BandsType.No
            df = self.monthly(df, f)
            monthly = [mpf.make_addplot(df['monthly_vwap_high'], color='crimson', label="monthly"),
                       mpf.make_addplot(df['monthly_vwap_median'], color='crimson'),
                       mpf.make_addplot(df['monthly_vwap_low'], color='crimson')]

        apd = daily
        if show_weekly:
            apd.extend(weekly)
        if show_monthly:
            apd.extend(monthly)
        if date is not None:
            apd.extend(anchored)

        if bands_at != 'none':
            bands = [mpf.make_addplot(df[f'{bands_at}_upper_1'], color='blue'),
                    mpf.make_addplot(df[f'{bands_at}_upper_2'], color='blue'),
                    mpf.make_addplot(df[f'{bands_at}_upper_3'], color='blue'),
                    mpf.make_addplot(df[f'{bands_at}_lower_1'], color='red'),
                    mpf.make_addplot(df[f'{bands_at}_lower_2'], color='red'),
                    mpf.make_addplot(df[f'{bands_at}_lower_3'], color='red')]
            apd.extend(bands)

        s = mpf.make_mpf_style(base_mpl_style='seaborn-v0_8')
        mpf.plot(df, type='candle', style=s, addplot=apd, warn_too_much_data=len(df) + 1,
                 title="Multi VWAP \n"
                 f"Bands: {bands_name}",
                 figsize=(12.5,6),
                 returnfig=True,
                 scale_padding=0.2)

        mpf.show()

    def plot_ly(self, df: pd.DataFrame | None = None, date: str = None, show_weekly: bool = False, show_monthly: bool = False,
                bands_at: str = 'none', bands_filter: BandsFilter | None = None,
                chart: str = 'candle', renderer: str = 'default',
                width: int = 1200, height: int = 800):
        """
        Plot with plotly.
        """
        _charts = ['candle', 'ohlc']
        _at = ['none', 'anchored', 'daily', 'weekly', 'monthly']

        input_values = [chart, bands_at]
        input_validation = [_charts, _at]
        for value, validation in zip(input_values, input_validation):
            if value not in validation:
                raise ValueError(f"Only {validation} options are valid.")

        if df is None:
            df = self._df_ohlcv
        f = bands_filter if isinstance(bands_filter, BandsFilter) else self._bands_filter
        bands_name = f.bands_type.name if bands_at != 'none' else 'No'
        bands_select = f.bands_type

        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.0)

        df[f'plotly_int_index'] = range(len(df))
        x_column_index = df[f'plotly_int_index']
        trace_chart = go.Ohlc(x=x_column_index,
                              open=df['open'],
                              high=df['high'],
                              low=df['low'],
                              close=df['close'], opacity=0.8) if chart == 'ohlc' else \
                      go.Candlestick(x=x_column_index,
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'], opacity=0.6)

        fig.add_trace(trace_chart, row=1, col=1)

        # Daily VWAP
        f.bands_type = bands_select if bands_at == 'daily' else BandsType.No
        df = self.daily(df, f)
        for name in ['daily_vwap_high', 'daily_vwap_median', 'daily_vwap_low']:
            fig.add_trace(go.Scatter(x=x_column_index,
                                     y=df[name],
                                     mode='lines',
                                     marker=dict(
                                         color='gold',
                                         size=1,
                                     ),
                                     opacity=0.5), row=1, col=1)
        # Anchored
        if date is not None:
            f.bands_type = bands_select if bands_at == 'anchored' else BandsType.No
            df = self.anchored(date, df, f)

            columns = ['anchored_vwap_high', 'anchored_vwap_median', 'anchored_vwap_low']
            colors = ['deepskyblue', 'lightyellow', 'orange']
            for name, coloring in zip(columns, colors):
                fig.add_trace(go.Scatter(x=x_column_index,
                                         y=df[name],
                                         mode='lines',
                                         marker=dict(
                                             color=coloring,
                                             size=1,
                                         ),
                                         opacity=0.5), row=1, col=1)

        # Weekly
        if show_weekly:
            f.bands_type = bands_select if bands_at == 'weekly' else BandsType.No
            df = self.weekly(df, f)

            for name in ['weekly_vwap_high', 'weekly_vwap_median', 'weekly_vwap_low']:
                fig.add_trace(go.Scatter(x=x_column_index,
                                         y=df[name],
                                         mode='lines',
                                         marker=dict(
                                             color='goldenrod',
                                             size=1,
                                         ),
                                         opacity=0.5), row=1, col=1)
        # Monthly
        if show_monthly:
            f.bands_type = bands_select if bands_at == 'monthly' else BandsType.No
            df = self.monthly(df, f)

            for name in ['monthly_vwap_high', 'monthly_vwap_median', 'monthly_vwap_low']:
                fig.add_trace(go.Scatter(x=x_column_index,
                                         y=df[name],
                                         mode='lines',
                                         marker=dict(
                                             color='crimson',
                                             size=1,
                                         ),
                                         opacity=0.5), row=1, col=1)

        # Bands
        if bands_at != 'none':
            bands = [f'{bands_at}_upper_1',
                    f'{bands_at}_upper_2',
                    f'{bands_at}_upper_3',
                    f'{bands_at}_lower_1',
                    f'{bands_at}_lower_2',
                    f'{bands_at}_lower_3']
            for name in bands:
                coloring = 'blue' if name.split('_')[1] == 'upper' else 'red'
                fig.add_trace(go.Scatter(x=x_column_index,
                                         y=df[name],
                                         mode='lines',
                                         marker=dict(
                                             color=coloring,
                                             size=1,
                                         ),
                                         opacity=0.5), row=1, col=1)

        fig.update_layout(
            title=f"Multi VWAP <br>Bands: {bands_name}",
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


    def anchored(self, date: str, df_ohlcv: pd.DataFrame | None = None, bands_filter: BandsFilter | None = None):
        """
        The following (3) columns will be added: anchored_vwap_[high, median, low]
        """
        if df_ohlcv is None:
            df_ohlcv = self._df_ohlcv
        bands_filter = bands_filter if isinstance(bands_filter, BandsFilter) else self._bands_filter

        df_period = df_ohlcv[df_ohlcv.index >= date].copy()
        df_period = self._create_vwap(df_period, bands_filter, 'anchored')
        df = pd.concat([df_ohlcv, df_period], axis=1)
        df = df.loc[: , ~df.columns.duplicated()]
        return df

    def daily(self, df_ohlcv: pd.DataFrame | None = None, bands_filter: BandsFilter | None = None):
        """
        The following (3) columns will be added: daily_vwap_[high, median, low]
        """
        if df_ohlcv is None:
            df_ohlcv = self._df_ohlcv
        bands_filter = bands_filter if isinstance(bands_filter, BandsFilter) else self._bands_filter

        daily_list = get_periods_list(df_ohlcv, pd.Timedelta(days=1))
        daily_dfs = self._parallel_process_profiles(daily_list, bands_filter,'daily')
        return pd.concat(daily_dfs)

    def weekly(self, df_ohlcv: pd.DataFrame | None = None, bands_filter: BandsFilter | None = None):
        """
        The following (3) columns will be added: weekly_vwap_[high, median, low]
        """
        if df_ohlcv is None:
            df_ohlcv = self._df_ohlcv
        bands_filter = bands_filter if isinstance(bands_filter, BandsFilter) else self._bands_filter

        weekly_list = get_periods_list(df_ohlcv, pd.Timedelta(weeks=1))
        weekly_dfs = self._parallel_process_profiles(weekly_list, bands_filter, 'weekly')
        return pd.concat(weekly_dfs)

    def monthly(self, df_ohlcv: pd.DataFrame | None = None, bands_filter: BandsFilter | None = None):
        """
        The following (3) columns will be added: monthly_vwap_[high, median, low]
        """
        if df_ohlcv is None:
            df_ohlcv = self._df_ohlcv
        bands_filter = bands_filter if isinstance(bands_filter, BandsFilter) else self._bands_filter

        monthly_list = get_periods_list(df_ohlcv, pd.DateOffset(months=1))
        monthly_dfs = self._parallel_process_profiles(monthly_list, bands_filter, 'monthly')
        return pd.concat(monthly_dfs)

    def _create_vwap(self, df_period: pd.DataFrame, b: BandsFilter, prefix_period: str):
        _volume = df_period['volume']
        _high_prices = df_period['high']
        _low_prices = df_period['low']
        _median_prices = (_high_prices + _low_prices) / 2

        _high_vol = _high_prices * _volume
        _median_vol = _median_prices * _volume
        _low_vol = _low_prices * _volume

        df_period[f'{prefix_period}_vwap_high'] = _high_vol.cumsum() / _volume.cumsum()
        _median_vwap = _median_vol.cumsum() / _volume.cumsum()
        df_period[f'{prefix_period}_vwap_median'] = _median_vwap
        df_period[f'{prefix_period}_vwap_low'] = _low_vol.cumsum() / _volume.cumsum()

        if b.bands_type == BandsType.No:
            return df_period

        _median_prices = _median_prices.to_numpy()
        _median_vwap = _median_vwap.to_numpy()
        _volume = _volume.to_numpy()

        match b.bands_type:
            case BandsType.Percentile_Asymmetric:
                _pctile_up = [round(pct / 100, 3) for pct in b.pctile_up]
                _pctile_down = [round(pct / 100, 3) for pct in b.pctile_down]

                upper_1, upper_2, upper_3, lower_1, lower_2, lower_3 = \
                    quantile_asymmetric_bands_volume(_median_prices, _median_vwap, _volume, _pctile_up, _pctile_down) \
                        if b.volume_weighted else \
                    quantile_asymmetric_bands(_median_prices, _median_vwap, _pctile_up, _pctile_down)
            case BandsType.Percentile:
                _pctile = [round(pct / 100, 3) for pct in b.pctile]

                upper_1, upper_2, upper_3, lower_1, lower_2, lower_3 = \
                    quantile_bands_volume(_median_prices, _median_vwap, _volume, _pctile) \
                        if b.volume_weighted else \
                    quantile_bands(_median_prices, _median_vwap, _pctile)
            case _:
                upper_1, upper_2, upper_3, lower_1, lower_2, lower_3 = \
                    stddev_bands_volume(_median_prices, _median_vwap, _volume, b.multipliers) \
                        if b.volume_weighted else \
                    stddev_bands(_median_prices, _median_vwap, b.multipliers)

        df_period[f"{prefix_period}_upper_1"], df_period[f"{prefix_period}_upper_2"], \
            df_period[f"{prefix_period}_upper_3"], \
        df_period[f"{prefix_period}_lower_1"], df_period[f"{prefix_period}_lower_2"], \
            df_period[f"{prefix_period}_lower_3"] = upper_1, upper_2, upper_3, lower_1, lower_2, lower_3

        return df_period
