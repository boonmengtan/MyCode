"""Simple moving average crossover trading strategy implementation.

This module loads price history from the provided CSV file and generates
trade signals based on a moving average crossover combined with a
volatility filter.  The script can be executed as a standalone program
or imported to reuse its helper functions for research purposes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration values that control the behaviour of the strategy.

    Attributes
    ----------
    fast_window:
        Rolling window used for the fast moving average.
    slow_window:
        Rolling window used for the slow moving average.  This value must be
        larger than ``fast_window`` in order to avoid degenerate signals.
    volatility_window:
        Number of observations used to estimate realised volatility.  The
        volatility series is used as a filter to prevent trades when the
        market is unusually calm.
    volatility_quantile:
        Quantile of the volatility distribution that acts as a threshold for
        the filter.  A typical value of ``0.5`` keeps trades only when the
        latest volatility print is above the rolling median.
    annualisation_factor:
        Number of observations per year that should be used when
        annualising returns and volatility.  Daily data usually uses 252.
    """

    fast_window: int = 5
    slow_window: int = 20
    volatility_window: int = 10
    volatility_quantile: float = 0.5
    annualisation_factor: int = 252

    def validate(self) -> None:
        if self.fast_window <= 0:
            raise ValueError("fast_window must be positive")
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be larger than fast_window")
        if self.volatility_window <= 1:
            raise ValueError("volatility_window must be greater than 1")
        if not (0.0 <= self.volatility_quantile <= 1.0):
            raise ValueError("volatility_quantile must be between 0 and 1")
        if self.annualisation_factor <= 0:
            raise ValueError("annualisation_factor must be positive")


def load_price_history(csv_path: Path) -> pd.DataFrame:
    """Load the CSV file exported from the dashboard and keep price columns.

    Parameters
    ----------
    csv_path:
        Path to the ``Trades History_with_OHLC.csv`` file.

    Returns
    -------
    pandas.DataFrame
        A tidy data frame containing the date, ticker symbol and OHLC prices.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["DATE_USED"])
    required_columns = {"DATE_USED", "YF_SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            "CSV file is missing required columns: " + ", ".join(sorted(missing))
        )

    tidy_df = (
        df.loc[:, ["DATE_USED", "YF_SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE"]]
        .rename(columns={"DATE_USED": "date", "YF_SYMBOL": "symbol"})
        .dropna(subset=["date", "symbol", "CLOSE"])
    )
    tidy_df["symbol"] = tidy_df["symbol"].astype(str)
    tidy_df = tidy_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return tidy_df


def _apply_group_rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _compute_group_volatility(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change().rolling(window=window, min_periods=2).std(ddof=0)


def add_indicators(price_data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Append moving averages and volatility estimates to the price table."""

    config.validate()

    grouped = price_data.groupby("symbol", group_keys=False)
    enriched = price_data.copy()
    enriched["fast_ma"] = grouped["CLOSE"].transform(
        lambda s: _apply_group_rolling(s, config.fast_window)
    )
    enriched["slow_ma"] = grouped["CLOSE"].transform(
        lambda s: _apply_group_rolling(s, config.slow_window)
    )
    enriched["volatility"] = grouped["CLOSE"].transform(
        lambda s: _compute_group_volatility(s, config.volatility_window)
    )
    return enriched


def generate_signals(indicator_data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Create trading signals from the indicator table."""

    signals = indicator_data.copy()
    signals["raw_signal"] = np.sign(signals["fast_ma"] - signals["slow_ma"])

    grouped = signals.groupby("symbol", group_keys=False)
    threshold = grouped["volatility"].transform(
        lambda s: s.rolling(window=config.volatility_window, min_periods=1)
        .quantile(config.volatility_quantile)
        .bfill()
    )
    signals["volatility_filter"] = (signals["volatility"] >= threshold).astype(int)
    signals["signal"] = signals["raw_signal"] * signals["volatility_filter"]
    signals.loc[signals["signal"] == 0, "signal"] = np.nan
    signals["position"] = grouped["signal"].ffill().fillna(0)
    signals["position_change"] = signals.groupby("symbol")["position"].diff().fillna(
        signals["position"]
    )
    return signals


def compute_performance(signals: pd.DataFrame, config: StrategyConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate strategy performance metrics.

    Returns
    -------
    summary:
        Aggregated statistics per symbol.
    equity_curve:
        Detailed time-series with cumulative strategy returns.
    """

    results = signals.copy()
    grouped = results.groupby("symbol", group_keys=False)
    results["asset_return"] = grouped["CLOSE"].pct_change().fillna(0.0)
    results["strategy_return"] = (
        grouped["position"].shift(1).fillna(0.0) * results["asset_return"]
    )
    results["cumulative_return"] = grouped["strategy_return"].apply(
        lambda r: (1.0 + r).cumprod()
    )

    def _calc_summary(group: pd.DataFrame) -> Dict[str, float]:
        total_return = group["cumulative_return"].iloc[-1] - 1.0
        avg_return = group["strategy_return"].mean()
        vol = group["strategy_return"].std(ddof=0)
        ann_factor = config.annualisation_factor
        annual_return = (1.0 + avg_return) ** ann_factor - 1.0
        annual_vol = vol * np.sqrt(ann_factor)
        sharpe = np.nan if annual_vol == 0 else annual_return / annual_vol

        running_max = group["cumulative_return"].cummax()
        drawdowns = group["cumulative_return"] / running_max - 1.0
        max_drawdown = drawdowns.min()

        trades = group.loc[group["position_change"] != 0]
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "trades": float(len(trades)),
        }

    summary_dict = {symbol: _calc_summary(group) for symbol, group in grouped}
    summary = pd.DataFrame.from_dict(summary_dict, orient="index")
    summary = summary.sort_values("total_return", ascending=False)
    return summary, results


def run_strategy(csv_path: Path | str, config: StrategyConfig | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Complete pipeline that loads data, produces signals and summarises performance."""

    cfg = config or StrategyConfig()
    price_history = load_price_history(Path(csv_path))
    indicator_data = add_indicators(price_history, cfg)
    signal_table = generate_signals(indicator_data, cfg)
    summary, equity_curve = compute_performance(signal_table, cfg)
    trade_log = signal_table.loc[signal_table["position_change"] != 0, [
        "date",
        "symbol",
        "CLOSE",
        "position",
        "position_change",
    ]].copy()
    trade_log.rename(columns={"CLOSE": "close"}, inplace=True)
    return summary, equity_curve, trade_log


def _format_percentage(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:6.2f}%"


def _format_number(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:6.2f}"


def _print_report(summary: pd.DataFrame, trade_log: pd.DataFrame) -> None:
    if summary.empty:
        print("No trades were generated.")
        return

    headers = [
        "Symbol",
        "Total Return",
        "Annual Return",
        "Annual Vol",
        "Sharpe",
        "Max DD",
        "Trades",
    ]
    print("\nStrategy performance summary")
    print("-" * 80)
    print(
        f"{headers[0]:>10} | {headers[1]:>13} | {headers[2]:>13} | {headers[3]:>10} | {headers[4]:>8} | {headers[5]:>8} | {headers[6]:>6}"
    )
    print("-" * 80)
    for symbol, row in summary.iterrows():
        print(
            f"{symbol:>10} | {_format_percentage(row['total_return'])} | {_format_percentage(row['annual_return'])} | "
            f"{_format_percentage(row['annual_volatility'])} | {_format_number(row['sharpe_ratio'])} | "
            f"{_format_percentage(row['max_drawdown'])} | {int(row['trades']):6d}"
        )

    if not trade_log.empty:
        print("\nGenerated trades")
        print("-" * 80)
        for _, trade in trade_log.iterrows():
            action = "Long" if trade["position_change"] > 0 else "Exit"
            print(
                f"{trade['date'].date()}  {trade['symbol']:>6}  {action:>4} at close {trade['close']:8.2f} -> position {trade['position']:5.1f}"
            )


def main(argv: Iterable[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("Trades History_with_OHLC.csv"),
        help="Path to the CSV file exported from the dashboard.",
    )
    parser.add_argument("--fast", type=int, default=5, help="Fast moving average window.")
    parser.add_argument(
        "--slow",
        type=int,
        default=20,
        help="Slow moving average window (must be larger than fast window).",
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=10,
        help="Window used to estimate realised volatility.",
    )
    parser.add_argument(
        "--vol-quantile",
        type=float,
        default=0.5,
        help="Quantile for the volatility filter (0 disables the filter).",
    )
    parser.add_argument(
        "--annualisation",
        type=int,
        default=252,
        help="Number of observations per year (252 for daily, 52 for weekly).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = StrategyConfig(
        fast_window=args.fast,
        slow_window=args.slow,
        volatility_window=args.vol_window,
        volatility_quantile=args.vol_quantile,
        annualisation_factor=args.annualisation,
    )

    summary, _, trade_log = run_strategy(args.csv, config)
    _print_report(summary, trade_log)


if __name__ == "__main__":
    main()
