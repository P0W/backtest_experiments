"""
ETF Momentum Strategy

A portfolio momentum strategy that trades Indian ETFs based on momentum scoring
and regular rebalancing. The strategy selects top-performing ETFs and
maintains an equal-weighted portfolio with realistic trading costs.
"""

from typing import Any

import backtrader as bt
import numpy as np

from .base_strategy import BaseStrategy, StrategyConfig


class ETFMomentumStrategy(BaseStrategy):
    """
    ETF Momentum Strategy based on momentum scoring and rebalancing
    """

    params = (
        ("portfolio_size", 5),
        ("rebalance_frequency", 30),  # Days between rebalancing
        (
            "long_term_period",
            200,
        ),  # Long-term momentum period (trading bars)
        (
            "short_term_period",
            90,
        ),  # Short-term momentum period (trading bars)
        ("moving_avg_period", 20),  # Moving average filter period
        ("min_momentum_threshold", 0.0),  # Minimum momentum to consider
        ("volume_threshold", 200000),  # Minimum 30-day average daily volume
        ("use_threshold_rebalancing", False),  # Enable threshold-based rebalancing
        ("profit_threshold_pct", 10.0),  # Profit threshold % to trigger rebalancing
        ("loss_threshold_pct", -5.0),  # Loss threshold % to trigger rebalancing
        ("printlog", False),  # Whether to print log messages
    )

    def __init__(self):
        super().__init__()
        self.rebalance_counter = 0
        self.last_rebalance_date = None
        self.portfolio_holdings = {}  # Track current holdings
        self.position_entry_prices = {}  # Track entry prices for threshold calculations

        # Initialize indicators for each ETF
        self.indicators = {}
        self.log(f"Initializing ETFMomentumStrategy with {len(self.datas)} ETF feeds")

        for d in self.datas:
            etf_name = d._name
            self.log(f"Setting up indicators for ETF: {etf_name}")

            try:
                # Use BackTrader indicators for efficiency and accuracy
                self.indicators[etf_name] = {
                    "sma": bt.ind.SimpleMovingAverage(
                        d.close, period=self.p.moving_avg_period
                    ),
                    "ema": bt.ind.ExponentialMovingAverage(
                        d.close, period=self.p.moving_avg_period
                    ),
                    # Volume indicators
                    "volume_sma_30": bt.ind.SimpleMovingAverage(d.volume, period=30),
                    # Returns/momentum indicators
                    "roc_long": bt.ind.RateOfChange(
                        d.close, period=self.p.long_term_period
                    ),
                    "roc_short": bt.ind.RateOfChange(
                        d.close, period=self.p.short_term_period
                    ),
                    # High tracking for retracement filter
                    "highest": bt.ind.Highest(d.close, period=self.p.long_term_period),
                    # Volatility tracking (using standard deviation of returns)
                    "volatility": bt.ind.StandardDeviation(
                        bt.ind.PercentChange(d.close, period=1), period=30
                    ),
                }
                self.log(f"Successfully created indicators for {etf_name}")
            except Exception as e:
                self.log(f"ERROR creating indicators for {etf_name}: {e!s}")
                self.indicators[etf_name] = {
                    "sma": None,
                    "ema": None,
                    "volume_sma_30": None,
                    "roc_long": None,
                    "roc_short": None,
                    "highest": None,
                    "volatility": None,
                }

        self.log(f"Indicators initialized for {len(self.indicators)} ETFs")

    def notify_order(self, order):
        """Track actual order fills and rejections"""
        if order.status == order.Completed:
            etf_name = order.data._name
            pos = self.getposition(order.data)
            if pos.size > 0:
                self.position_entry_prices[etf_name] = float(order.executed.price)
            else:
                self.position_entry_prices.pop(etf_name, None)
        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            self.log(
                f"Order {order.getstatusname()} for {order.data._name}: size={order.size}"
            )

    def prenext(self):
        """
        Called before next() when we don't have enough data for all indicators
        but can still achieve portfolio_size with available ETFs
        """
        try:
            current_date = self.datas[0].datetime.date(0)
            current_value = self.broker.getvalue()

            # Track portfolio performance even during prenext
            self.portfolio_values.append(current_value)
            self.dates.append(current_date)

            # Check if we have enough ETFs with sufficient data to form a portfolio
            available_etfs = []

            for d in self.datas:
                etf_name = d._name

                # Check if we have sufficient data using manual calculation
                # (equivalent to checking if indicators have valid data)
                if self._has_sufficient_data(d):
                    try:
                        # Additional check for valid price data
                        current_price = d.close[0]
                        if not np.isnan(current_price) and current_price > 0:
                            available_etfs.append(etf_name)
                    except (IndexError, TypeError):
                        continue

            self.log(
                f"Prenext: {len(available_etfs)} ETFs with sufficient data available"
            )

            # Execute as soon as at least 1 ETF has warm data (unfilled slots stay in cash)
            if len(available_etfs) >= 1:
                self.log(
                    f"Prenext: Warm ETFs available ({len(available_etfs)} ready)"
                )
                self.execute_strategy()
            else:
                self.log("Prenext: Waiting for warm-up data (0 ETFs ready)")
        except Exception as e:
            self.log(f"Error in prenext(): {e!s}")
            # Continue to allow strategy to proceed even if prenext fails

    def next(self):
        """
        Main next method called by backtrader for each bar
        """
        # Call parent next() method to track portfolio performance
        super().next()

    def execute_strategy(self):
        """Execute ETF momentum strategy logic"""
        current_date = self.datas[0].datetime.date(0)

        # Check if it's time to rebalance
        if self._should_rebalance(current_date):
            self.log(f"Rebalancing portfolio on {current_date}")
            if self._rebalance_portfolio(current_date):
                self.last_rebalance_date = current_date

    def _should_rebalance(self, current_date):
        """Determine if it's time to rebalance the portfolio"""
        if self.last_rebalance_date is None:
            self.log(f"First rebalance check on {current_date}")
            return True

        days_since_rebalance = (current_date - self.last_rebalance_date).days

        # If threshold-based rebalancing is enabled, check position thresholds;
        # also re-evaluate on schedule whenever slots are unfilled (e.g. 0-ETF cash exit)
        # so the strategy can re-enter the market instead of staying permanently in cash.
        if self.p.use_threshold_rebalancing:
            if self._check_threshold_rebalance():
                self.log("Threshold-based rebalance triggered")
                return True
            open_positions = sum(
                1 for d in self.datas if self.getposition(d).size > 0
            )
            if (
                open_positions < self.p.portfolio_size
                and days_since_rebalance >= self.p.rebalance_frequency
            ):
                self.log(
                    f"Scheduled cash-slot re-evaluation triggered ({open_positions}/{self.p.portfolio_size} slots filled)"
                )
                return True
            return False

        # Time-based rebalancing check (when threshold mode is disabled)
        if days_since_rebalance >= self.p.rebalance_frequency:
            self.log(
                f"Time-based rebalance triggered: {days_since_rebalance} days >= {self.p.rebalance_frequency} threshold"
            )
            return True

        return False

    def _check_threshold_rebalance(self):
        """Check if any position has hit profit/loss thresholds"""
        threshold_hit = False
        positions_checked = 0

        for d in self.datas:
            etf_name = d._name
            position = self.getposition(d)

            if position.size != 0:
                positions_checked += 1
                if etf_name in self.position_entry_prices:
                    entry_price = self.position_entry_prices[etf_name]
                    current_price = d.close[0]

                    # Calculate P&L percentage
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100

                    self.log(
                        f"{etf_name}: Entry={entry_price:.2f}, Current={current_price:.2f}, P&L={pnl_pct:.2f}%"
                    )

                    # Check thresholds
                    if pnl_pct >= self.p.profit_threshold_pct:
                        self.log(
                            f"{etf_name}: Profit threshold hit - {pnl_pct:.2f}% >= {self.p.profit_threshold_pct}%"
                        )
                        threshold_hit = True
                    elif pnl_pct <= self.p.loss_threshold_pct:
                        self.log(
                            f"{etf_name}: Loss threshold hit - {pnl_pct:.2f}% <= {self.p.loss_threshold_pct}%"
                        )
                        threshold_hit = True
                    else:
                        self.log(
                            f"{etf_name}: Within thresholds - {self.p.loss_threshold_pct}% < {pnl_pct:.2f}% < {self.p.profit_threshold_pct}%"
                        )
                else:
                    self.log(
                        f"{etf_name}: Position exists but no entry price tracked - this should not happen!"
                    )

        if positions_checked == 0:
            self.log("No positions to check for thresholds")
        elif not threshold_hit:
            self.log(f"Checked {positions_checked} positions - no threshold hits")

        return threshold_hit

    def _rebalance_portfolio(self, current_date):
        """Rebalance the portfolio based on momentum scores"""
        try:
            self.log(f"Starting rebalance process on {current_date}")

            has_warm_data = any(self._has_sufficient_data(d) for d in self.datas)
            if not has_warm_data:
                self.log("Insufficient warm-up data across universe, skipping rebalance")
                return False

            # Calculate momentum scores for ETFs passing trend/retracement/volume filters
            momentum_scores = self._calculate_momentum_scores()
            self.log(f"Calculated momentum scores for {len(momentum_scores)} ETFs")

            # Filter ETFs that meet momentum threshold
            eligible_etfs = (
                self._filter_eligible_etfs(momentum_scores) if momentum_scores else []
            )

            if len(eligible_etfs) == 0:
                self.log(
                    "No eligible ETFs found - exiting existing positions to cash"
                )
                self._execute_rebalancing_trades([])
                return True

            if len(eligible_etfs) < self.p.portfolio_size:
                self.log(
                    f"Only {len(eligible_etfs)} eligible ETFs found (target {self.p.portfolio_size}); "
                    f"capping each at {100.0 / self.p.portfolio_size:.1f}% and holding remaining slots in cash"
                )

            # Select top ETFs
            top_etfs = eligible_etfs[: self.p.portfolio_size]
            top_etf_names = [etf["name"] for etf in top_etfs]

            self.log(f"Selected top {len(top_etf_names)} ETFs: {top_etf_names}")

            # Execute rebalancing trades
            self._execute_rebalancing_trades(top_etf_names)
            return True

        except Exception as e:
            self.log(f"Error in rebalancing: {e!s}")
            return False

    def _calculate_momentum_scores(self):
        """
        Calculate momentum scores for all ETFs using MomentumCalculator-equivalent logic
        """
        momentum_scores = {}
        total_etfs = len(self.datas)

        # Calculate adaptive weights based on market volatility (MomentumCalculator style)
        market_volatility = self._calculate_market_volatility()
        long_weight, short_weight = self._get_adaptive_momentum_weights(
            market_volatility
        )

        for d in self.datas:
            etf_name = d._name

            try:
                # Check minimum data points requirement (MomentumCalculator equivalent)
                if not self._has_sufficient_data(d):
                    self.log(f"{etf_name}: Insufficient data points")
                    continue

                # Get current values using BackTrader indicators
                try:
                    current_price = d.close[0]

                    # Get indicators for this ETF
                    indicators = self.indicators.get(etf_name, {})
                    roc_long = indicators.get("roc_long")
                    roc_short = indicators.get("roc_short")
                    volume_sma_30 = indicators.get("volume_sma_30")

                    # Check if indicators are available and have data
                    if (
                        roc_long is None
                        or roc_short is None
                        or len(roc_long) == 0
                        or len(roc_short) == 0
                    ):
                        continue

                    # bt.ind.RateOfChange already returns decimal ratio ((close / close[-n]) - 1)
                    long_return = roc_long[0]
                    short_return = roc_short[0]

                    # Require valid 30-day average volume (fail closed if missing)
                    current_volume = (
                        d.volume[0] if hasattr(d, "volume") and len(d.volume) > 0 else np.nan
                    )
                    avg_volume_30day = (
                        volume_sma_30[0]
                        if volume_sma_30 and len(volume_sma_30) > 0
                        else np.nan
                    )

                    # Skip if essential values are invalid
                    if (
                        np.isnan(long_return)
                        or np.isnan(short_return)
                        or np.isnan(current_price)
                        or current_price <= 0
                    ):
                        continue

                except (IndexError, TypeError):
                    continue

                # Calculate weighted momentum score (MomentumCalculator style)
                if short_return is not None:
                    momentum_score = (
                        long_return * long_weight + short_return * short_weight
                    )
                else:
                    momentum_score = long_return

                # Apply MomentumCalculator-style filters
                filters_passed = self._apply_momentum_filters(
                    d, current_price, avg_volume_30day
                )

                if not filters_passed:
                    self.log(f"{etf_name}: Failed momentum filters")
                    continue

                momentum_scores[etf_name] = {
                    "score": momentum_score,
                    "long_momentum": long_return,
                    "short_momentum": short_return,
                    "current_price": current_price,
                    "sma": current_price,  # Not needed for MomentumCalculator approach
                    "trend_filter": filters_passed,
                    "volume": current_volume,
                    "avg_volume": avg_volume_30day,
                    "has_long_term_data": long_return is not None,
                }

                self.log(
                    f"{etf_name}: momentum={momentum_score:.4f} (weights: {long_weight:.1f}/{short_weight:.1f}), "
                    f"long={long_return:.4f}, short={short_return:.4f}"
                )

            except Exception as e:
                self.log(f"Error calculating momentum for {etf_name}: {e!s}")
                continue

        self.log(
            f"Momentum calculation: {len(momentum_scores)}/{total_etfs} ETFs with valid scores"
        )
        return momentum_scores

    def _calculate_market_volatility(self):
        """Calculate market volatility using NIFTYBEES.NS (or first feed fallback)"""
        try:
            market_data = self._get_data_by_name("NIFTYBEES.NS") or self.datas[0]
            etf_name = market_data._name
            indicators = self.indicators.get(etf_name, {})
            volatility_indicator = indicators.get("volatility")

            if volatility_indicator and len(volatility_indicator) > 0:
                daily_vol = volatility_indicator[0]
                if not np.isnan(daily_vol):
                    return daily_vol * np.sqrt(252)

            return 0.2  # Default volatility if indicator not available
        except Exception as e:
            self.log(f"Error calculating market volatility: {e!s}")
            return 0.2

    def _get_adaptive_momentum_weights(self, market_volatility):
        """Get adaptive momentum weights based on market volatility (MomentumCalculator style)"""
        # Default weights (can be derived from exit_rank_buffer parameter)
        base_long_weight = 0.7
        base_short_weight = 0.3

        # Adaptive logic: if volatility > 20%, increase short-term weight
        if market_volatility > 0.2:
            # Higher volatility -> more short-term focus
            long_weight = 0.5
            short_weight = 0.5
            self.log(
                f"High volatility ({market_volatility:.2%}) - using balanced weights"
            )
        else:
            # Normal volatility -> standard weights
            long_weight = base_long_weight
            short_weight = base_short_weight

        return long_weight, short_weight

    def _has_sufficient_data(self, data):
        """Check if ETF has sufficient data points"""
        min_data_points = max(self.p.long_term_period + 1, 50)
        return len(data) >= min_data_points

    def _apply_momentum_filters(self, data, current_price, avg_volume_30day):
        """Apply MomentumCalculator-style filters using BackTrader indicators"""
        try:
            # 1. Retracement filter (using BackTrader Highest indicator)
            if not self._passes_retracement_filter(data, current_price):
                return False

            # 2. Moving average filter (using BackTrader EMA indicator)
            if not self._passes_ema_filter(data, current_price):
                return False

            # 3. Volume filter (using BackTrader volume SMA indicator)
            if not self._passes_volume_filter(data, avg_volume_30day):
                return False

            return True
        except Exception as e:
            self.log(f"Error in momentum filters: {e!s}")
            return False

    def _passes_retracement_filter(self, data, current_price):
        """Check if ETF passes retracement filter using BackTrader Highest indicator"""
        try:
            max_retracement = 0.25  # 25% maximum retracement

            indicators = self.indicators.get(data._name, {})
            highest_indicator = indicators.get("highest")

            if highest_indicator is None or len(highest_indicator) == 0:
                return False

            try:
                peak_price = highest_indicator[0]
                if np.isnan(peak_price) or peak_price <= 0:
                    return False

                retracement = (peak_price - current_price) / peak_price
                passed = retracement <= max_retracement

                if not passed:
                    self.log(
                        f"{data._name}: Retracement filter failed - {retracement:.2%} > {max_retracement:.2%}"
                    )

                return passed
            except (IndexError, TypeError):
                return False

        except Exception as e:
            self.log(f"Error in retracement filter for {data._name}: {e!s}")
            return False

    def _passes_ema_filter(self, data, current_price):
        """Check if current price is above EMA (using BackTrader EMA indicator)"""
        try:
            indicators = self.indicators.get(data._name, {})
            ema_indicator = indicators.get("ema")

            if ema_indicator is None or len(ema_indicator) == 0:
                return False

            try:
                ema_value = ema_indicator[0]
                if np.isnan(ema_value):
                    return False

                passed = current_price > ema_value
                if not passed:
                    self.log(
                        f"{data._name}: EMA filter failed - {current_price:.2f} <= {ema_value:.2f}"
                    )

                return passed
            except (IndexError, TypeError):
                return False

        except Exception:
            return False

    def _passes_volume_filter(self, data, avg_volume_30day=None):
        """Check if average volume meets minimum requirement using BackTrader indicator"""
        try:
            if avg_volume_30day is not None:
                if np.isnan(avg_volume_30day):
                    return False
                return avg_volume_30day >= self.p.volume_threshold

            indicators = self.indicators.get(data._name, {})
            volume_sma_30 = indicators.get("volume_sma_30")

            if volume_sma_30 and len(volume_sma_30) > 0:
                avg_volume = volume_sma_30[0]
                if not np.isnan(avg_volume):
                    return avg_volume >= self.p.volume_threshold

            return False

        except Exception as e:
            self.log(f"Error in volume filter for {data._name}: {e!s}")
            return False

    def _filter_eligible_etfs(self, momentum_scores):
        """Filter ETFs based on momentum and liquidity criteria (MomentumCalculator style)"""
        eligible_etfs = []

        for etf_name, data in momentum_scores.items():
            # Require strictly positive score when threshold is 0.0, else >= threshold
            if self.p.min_momentum_threshold == 0:
                meets_momentum_threshold = data["score"] > 0.0
            else:
                meets_momentum_threshold = (
                    data["score"] >= self.p.min_momentum_threshold
                )

            # All other filters (trend, retracement, volume) are already applied in _calculate_momentum_scores
            # This matches MomentumCalculator's approach where filtering is done during calculation

            if meets_momentum_threshold:
                eligible_etfs.append(
                    {"name": etf_name, "momentum_score": data["score"], "data": data}
                )

        # Sort by momentum score (descending) - same as MomentumCalculator
        eligible_etfs.sort(key=lambda x: x["momentum_score"], reverse=True)

        self.log(
            f"Found {len(eligible_etfs)} eligible ETFs out of {len(momentum_scores)} total"
        )

        return eligible_etfs

    def _execute_rebalancing_trades(self, target_etfs):
        """
        Execute trades to rebalance to target portfolio

        Note: All trades are executed in whole shares only to comply with
        Indian market regulations which don't allow fractional share trading.
        Each ETF slot is capped at 1 / portfolio_size (leaving unfilled slots in cash),
        and all sell orders are submitted before buy orders to avoid Margin rejections.
        """
        try:
            current_value = self.broker.getvalue()

            if current_value <= 0:
                self.log(f"Invalid portfolio value ({current_value})")
                return

            # Reserve 0.8% buffer for statutory commissions and cheat-on-close cash checks,
            # and cap each slot at 1 / portfolio_size so unfilled slots stay in cash.
            usable_value = current_value * 0.992
            target_allocation = usable_value / self.p.portfolio_size

            # Get current positions
            current_positions = {}
            for d in self.datas:
                position = self.getposition(d)
                if position.size != 0:
                    current_positions[d._name] = {
                        "size": int(position.size),
                        "price": position.price,
                        "value": position.size * d.close[0],
                        "data": d,
                    }

            target_set = set(target_etfs)

            # 1. Submit full exits first
            for etf_name, pos_info in current_positions.items():
                if etf_name not in target_set:
                    self.log(f"Exiting position in {etf_name}")
                    try:
                        self.sell(data=pos_info["data"], size=pos_info["size"])
                    except Exception as e:
                        self.log(f"Error selling {etf_name}: {e!s}")

            # Compute target deltas for target ETFs
            pending_buys = []
            pending_sells = []
            for etf_name in target_etfs:
                etf_data = self._get_data_by_name(etf_name)
                if etf_data is None:
                    self.log(f"Warning: Could not find data feed for {etf_name}")
                    continue

                try:
                    current_price = etf_data.close[0]
                    target_shares = self.calculate_position_size(
                        target_allocation, current_price
                    )
                    current_shares = int(self.getposition(etf_data).size)
                    shares_diff = target_shares - current_shares

                    if shares_diff <= -1:
                        pending_sells.append((etf_name, etf_data, abs(shares_diff)))
                    elif shares_diff >= 1:
                        pending_buys.append((etf_name, etf_data, shares_diff))
                    elif current_shares > 0:
                        # Refresh threshold baseline for retained positions needing <1 share adjustment
                        self.position_entry_prices[etf_name] = float(current_price)
                except Exception as e:
                    self.log(f"Error sizing {etf_name}: {e!s}")

            # 2. Submit partial-trim sells before buys so cash settles first in broker queue
            for etf_name, etf_data, sell_qty in pending_sells:
                try:
                    self.log(f"Selling {sell_qty} shares of {etf_name}")
                    self.sell(data=etf_data, size=sell_qty)
                except Exception as e:
                    self.log(f"Error selling {etf_name}: {e!s}")

            # 3. Submit buys last
            for etf_name, etf_data, buy_qty in pending_buys:
                try:
                    self.log(f"Buying {buy_qty} shares of {etf_name}")
                    self.buy(data=etf_data, size=buy_qty)
                except Exception as e:
                    self.log(f"Error buying {etf_name}: {e!s}")

        except Exception as e:
            self.log(f"Critical error in _execute_rebalancing_trades: {e!s}")

    def _get_data_by_name(self, etf_name):
        """Get data feed by ETF name"""
        for d in self.datas:
            if d._name == etf_name:
                return d
        return None


class ETFMomentumConfig(StrategyConfig):
    """
    Configuration for ETF Momentum Strategy
    """

    def get_parameter_grid(self) -> dict[str, list[Any]]:
        """
        Define the parameter grid for ETF momentum strategy experiments

        Total combinations: 1 x 4 x 3 x 3 x 3 x 3 x 3 x 2 x 2 x 3 x 3 = 11,664 combinations
        Max experiments needed to test fully: 11,664
        """
        return {
            "portfolio_size": [5],  # 1 option
            "rebalance_frequency": [20, 30, 45, 60],  # 4 options
            "long_term_period": [100, 180, 200],  # 3 options
            "short_term_period": [30, 60, 90],  # 3 options
            "moving_avg_period": [20, 50, 100],  # 3 options
            "min_momentum_threshold": [0.0, 0.05, 0.10],  # 3 options
            "volume_threshold": [100000, 200000],  # 2 options
            "use_threshold_rebalancing": [False, True],  # 2 options
            "profit_threshold_pct": [4.0, 5.0, 10.0],  # 3 options
            "loss_threshold_pct": [-2.5, -3.0, -5.0],  # 3 options
        }

    def get_intraday_parameter_grid(self) -> dict[str, list[Any]]:
        """
        Intraday-optimized parameter grid (shorter periods)

        Total combinations: 3 x 3 x 3 x 3 x 3 x 3 x 3 x 2 x 2 x 3 x 3 = 39,366 combinations
        Max experiments needed to test fully: 39,366
        """
        return {
            "portfolio_size": [3, 5, 7],  # 3 options
            "rebalance_frequency": [5, 10, 15],  # 3 options (Days)
            "long_term_period": [60, 90, 120],  # 3 options (Shorter for intraday)
            "short_term_period": [15, 30, 45],  # 3 options
            "moving_avg_period": [10, 20, 30],  # 3 options
            "min_momentum_threshold": [0.0, 0.05, 0.10],  # 3 options
            "volume_threshold": [100000, 200000],  # 2 options
            "use_threshold_rebalancing": [False, True],  # 2 options
            "profit_threshold_pct": [5.0, 8.0, 12.0],  # 3 options
            "loss_threshold_pct": [-2.0, -3.0, -5.0],  # 3 options
        }

    def get_default_params(self) -> dict[str, Any]:
        """
        Get default parameters for ETF momentum strategy
        """
        return {
            "portfolio_size": 5,
            "rebalance_frequency": 30,  # Reduced from 30 for more frequent rebalancing
            "long_term_period": 200,  # Reduced from 126 for earlier start
            "short_term_period": 90,  # Reduced from 30
            "moving_avg_period": 20,  # Reduced from 25
            "min_momentum_threshold": 0.0,
            "volume_threshold": 200000,
            "use_threshold_rebalancing": False,
            "profit_threshold_pct": 4,
            "loss_threshold_pct": -4.0,
        }

    def validate_params(self, params: dict[str, Any]) -> bool:
        """
        Validate ETF momentum strategy parameters
        """
        # Long-term period must be longer than short-term
        if params.get("long_term_period", 0) <= params.get("short_term_period", 0):
            return False

        # Portfolio size should be reasonable
        portfolio_size = params.get("portfolio_size", 0)
        if portfolio_size <= 0 or portfolio_size > 20:
            return False

        # Rebalance frequency should be positive
        if params.get("rebalance_frequency", 0) <= 0:
            return False

        # Moving average period should be reasonable
        ma_period = params.get("moving_avg_period", 0)
        if ma_period <= 0 or ma_period > params.get("short_term_period", 100):
            return False

        # Volume threshold should be positive
        if params.get("volume_threshold", 0) <= 0:
            return False

        # Threshold rebalancing parameters validation
        if params.get("use_threshold_rebalancing", False):
            profit_threshold = params.get("profit_threshold_pct", 0)
            loss_threshold = params.get("loss_threshold_pct", 0)

            # Profit threshold should be positive
            if profit_threshold <= 0:
                return False

            # Loss threshold should be negative
            if loss_threshold >= 0:
                return False

            # Ensure profit threshold is reasonable (between 1% and 50%)
            if profit_threshold < 1.0 or profit_threshold > 50.0:
                return False

            # Ensure loss threshold is reasonable (between -1% and -50%)
            if loss_threshold > -1.0 or loss_threshold < -50.0:
                return False

        return True

    def get_strategy_class(self) -> type:
        """
        Get the ETF momentum strategy class
        """
        return ETFMomentumStrategy

    def get_required_data_feeds(self) -> int:
        """
        ETF momentum strategy works with multiple ETF feeds
        """
        return -1  # Variable number of data feeds

    def get_composite_score_weights(self) -> dict[str, float]:
        """
        Weights optimized for ETF momentum strategy
        """
        return {"total_return": 0.4, "sharpe_ratio": 0.35, "max_drawdown": 0.25}
