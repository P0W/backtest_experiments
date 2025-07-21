"""
ETF Momentum Strategy

A portfolio momentum strategy that trades Indian ETFs based on momentum scoring
and regular rebalancing. The strategy selects top-performing ETFs and
maintains an equal-weighted portfolio with realistic trading costs.
"""

from typing import Any, Dict, List

import backtrader as bt
import numpy as np

from .base_strategy import BaseStrategy, StrategyConfig


class ETFMomentumStrategy(BaseStrategy):
    """
    ETF Momentum Strategy based on momentum scoring and rebalancing
    """

    params = (
        ("portfolio_size", 5),
        ("rebalance_frequency", 15),  # Days between rebalancing
        (
            "long_term_period",
            90,
        ),  # Long-term momentum period (days) - further reduced for earlier start
        (
            "short_term_period",
            20,
        ),  # Short-term momentum period (days) - further reduced
        ("moving_avg_period", 20),  # Moving average filter period - reduced
        (
            "exit_rank_buffer",
            2.0,
        ),  # Exit multiplier (exit if rank > portfolio_size * buffer)
        ("min_momentum_threshold", 0.0),  # Minimum momentum to consider
        ("volume_threshold", 100000),  # Minimum daily volume threshold
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
                self.indicators[etf_name] = {
                    # Price indicators
                    "sma": bt.ind.SimpleMovingAverage(
                        d.close, period=self.p.moving_avg_period
                    ),
                    # Momentum indicators
                    "roc_long": bt.ind.RateOfChange(
                        d.close, period=self.p.long_term_period
                    ),
                    "roc_short": bt.ind.RateOfChange(
                        d.close, period=self.p.short_term_period
                    ),
                    # Volume indicator
                    "volume_sma": bt.ind.SimpleMovingAverage(d.volume, period=20),
                }
                self.log(f"Successfully created indicators for {etf_name}")
            except Exception as e:
                self.log(f"ERROR creating indicators for {etf_name}: {str(e)}")
                self.indicators[etf_name] = {
                    "sma": None,
                    "roc_long": None,
                    "roc_short": None,
                    "volume_sma": None,
                }

        self.log(f"Indicators initialized for {len(self.indicators)} ETFs")

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
                indicators = self.indicators.get(etf_name, {})

                # Check if we have at least short-term momentum data available
                roc_short = indicators.get("roc_short")
                sma = indicators.get("sma")

                if roc_short is not None and sma is not None:
                    try:
                        # Check if indicators have valid data (not NaN)
                        if (
                            len(roc_short) > 0
                            and len(sma) > 0
                            and not np.isnan(roc_short[0])
                            and not np.isnan(sma[0])
                        ):
                            available_etfs.append(etf_name)
                    except (IndexError, TypeError):
                        continue

            self.log(
                f"Prenext: {len(available_etfs)} ETFs with sufficient data available"
            )

            # If we have enough ETFs to form a portfolio, execute strategy
            if len(available_etfs) >= self.p.portfolio_size:
                self.log(
                    f"Prenext: Sufficient ETFs available ({len(available_etfs)} >= {self.p.portfolio_size})"
                )
                self.execute_strategy()
            else:
                self.log(
                    f"Prenext: Waiting for more data ({len(available_etfs)} < {self.p.portfolio_size})"
                )
        except Exception as e:
            self.log(f"Error in prenext(): {str(e)}")
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
            self._rebalance_portfolio(current_date)
            self.last_rebalance_date = current_date

        # Check exit conditions for existing positions
        self._check_exit_conditions()

    def _should_rebalance(self, current_date):
        """Determine if it's time to rebalance the portfolio"""
        if self.last_rebalance_date is None:
            self.log(f"First rebalance check on {current_date}")
            return True

        # If threshold-based rebalancing is enabled, only use threshold logic
        if self.p.use_threshold_rebalancing:
            threshold_rebalance = self._check_threshold_rebalance()
            if threshold_rebalance:
                self.log("Threshold-based rebalance triggered")
                return True
            # Don't do time-based rebalancing when threshold is enabled
            return False

        # Time-based rebalancing check (only when threshold is disabled)
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        time_based_rebalance = days_since_rebalance >= self.p.rebalance_frequency

        if time_based_rebalance:
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
                    
                    self.log(f"{etf_name}: Entry={entry_price:.2f}, Current={current_price:.2f}, P&L={pnl_pct:.2f}%")
                    
                    # Check thresholds
                    if pnl_pct >= self.p.profit_threshold_pct:
                        self.log(f"{etf_name}: Profit threshold hit - {pnl_pct:.2f}% >= {self.p.profit_threshold_pct}%")
                        threshold_hit = True
                    elif pnl_pct <= self.p.loss_threshold_pct:
                        self.log(f"{etf_name}: Loss threshold hit - {pnl_pct:.2f}% <= {self.p.loss_threshold_pct}%")
                        threshold_hit = True
                    else:
                        self.log(f"{etf_name}: Within thresholds - {self.p.loss_threshold_pct}% < {pnl_pct:.2f}% < {self.p.profit_threshold_pct}%")
                else:
                    self.log(f"{etf_name}: Position exists but no entry price tracked - this should not happen!")
        
        if positions_checked == 0:
            self.log("No positions to check for thresholds")
        elif not threshold_hit:
            self.log(f"Checked {positions_checked} positions - no threshold hits")
        
        return threshold_hit

    def _rebalance_portfolio(self, current_date):
        """Rebalance the portfolio based on momentum scores"""
        try:
            self.log(f"Starting rebalance process on {current_date}")
            
            # Calculate momentum scores for all ETFs
            momentum_scores = self._calculate_momentum_scores()

            if not momentum_scores:
                self.log("No valid momentum scores calculated, skipping rebalance")
                return

            self.log(f"Calculated momentum scores for {len(momentum_scores)} ETFs")

            # Filter ETFs that meet criteria
            eligible_etfs = self._filter_eligible_etfs(momentum_scores)

            if len(eligible_etfs) == 0:
                self.log("No eligible ETFs found - cannot rebalance")
                return

            # Use available ETFs if fewer than portfolio size (but at least 1)
            target_portfolio_size = min(self.p.portfolio_size, len(eligible_etfs))
            
            if len(eligible_etfs) < self.p.portfolio_size:
                self.log(
                    f"Only {len(eligible_etfs)} eligible ETFs found, using all {target_portfolio_size} instead of target {self.p.portfolio_size}"
                )

            # Select top ETFs
            top_etfs = eligible_etfs[: self.p.portfolio_size]
            top_etf_names = [etf["name"] for etf in top_etfs]

            self.log(f"Selected top {len(top_etf_names)} ETFs: {top_etf_names}")

            # Execute rebalancing trades
            self._execute_rebalancing_trades(top_etf_names)

        except Exception as e:
            self.log(f"Error in rebalancing: {str(e)}")

    def _calculate_momentum_scores(self):
        """Calculate momentum scores for all ETFs"""
        momentum_scores = {}
        total_etfs = len(self.datas)

        for d in self.datas:
            etf_name = d._name

            try:
                # Get indicators first
                indicators = self.indicators.get(etf_name, {})
                roc_long = indicators.get("roc_long")
                roc_short = indicators.get("roc_short")
                sma = indicators.get("sma")
                volume_sma = indicators.get("volume_sma")

                # Check if minimum required indicators exist (short-term momentum and SMA)
                if roc_short is None or sma is None:
                    self.log(f"{etf_name}: Essential indicators not initialized")
                    continue

                # Get current values and check if they're valid
                try:
                    current_price = d.close[0]
                    short_momentum = roc_short[0] / 100.0
                    sma_value = sma[0]
                    current_volume = d.volume[0] if hasattr(d, "volume") else 1000000
                    avg_volume = (
                        volume_sma[0]
                        if volume_sma and len(volume_sma) > 0
                        else current_volume
                    )

                    # Try to get long-term momentum, but don't fail if not available
                    long_momentum = None
                    if roc_long is not None:
                        try:
                            long_momentum_val = roc_long[0] / 100.0
                            if not np.isnan(long_momentum_val):
                                long_momentum = long_momentum_val
                        except (IndexError, TypeError):
                            pass

                    # Skip if essential values are NaN or invalid
                    if (
                        np.isnan(short_momentum)
                        or np.isnan(current_price)
                        or np.isnan(sma_value)
                    ):
                        continue

                except (IndexError, TypeError):
                    # Not enough data yet for indicators
                    continue

                # Calculate composite momentum score
                if long_momentum is not None:
                    # Full momentum calculation when long-term data is available
                    momentum_score = 0.7 * long_momentum + 0.3 * short_momentum
                    self.log(
                        f"{etf_name}: Full momentum calculation - "
                        f"long={long_momentum:.4f}, short={short_momentum:.4f}"
                    )
                else:
                    # Use only short-term momentum when long-term isn't available yet
                    momentum_score = short_momentum
                    long_momentum = 0.0  # Set to 0 for logging consistency
                    self.log(
                        f"{etf_name}: Short-term only momentum calculation - "
                        f"short={short_momentum:.4f} (long-term data not ready)"
                    )

                # Apply trend filter (price above moving average)
                trend_filter = current_price > sma_value

                momentum_scores[etf_name] = {
                    "score": momentum_score,
                    "long_momentum": long_momentum,
                    "short_momentum": short_momentum,
                    "current_price": current_price,
                    "sma": sma_value,
                    "trend_filter": trend_filter,
                    "volume": current_volume,
                    "avg_volume": avg_volume,
                    "has_long_term_data": long_momentum is not None
                    and long_momentum != 0.0,
                }

                self.log(
                    f"{etf_name}: momentum={momentum_score:.4f}, "
                    f"trend_ok={trend_filter}"
                )

            except Exception as e:
                self.log(f"Error calculating momentum for {etf_name}: {str(e)}")
                continue

        self.log(
            f"Momentum calculation: {len(momentum_scores)}/{total_etfs} ETFs with valid scores"
        )
        return momentum_scores

    def _filter_eligible_etfs(self, momentum_scores):
        """Filter ETFs based on momentum and liquidity criteria"""
        eligible_etfs = []

        for etf_name, data in momentum_scores.items():
            # Apply filters
            meets_momentum_threshold = data["score"] >= self.p.min_momentum_threshold
            meets_trend_filter = data["trend_filter"]  # Re-enabled trend filter
            meets_volume_threshold = data["avg_volume"] >= self.p.volume_threshold

            if (
                meets_momentum_threshold
                and meets_trend_filter
                and meets_volume_threshold
            ):
                eligible_etfs.append(
                    {"name": etf_name, "momentum_score": data["score"], "data": data}
                )

        # Sort by momentum score (descending)
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
        """
        try:
            current_value = self.broker.getvalue()

            # Ensure we have valid portfolio value and target ETFs
            if current_value <= 0 or len(target_etfs) == 0:
                self.log(
                    f"Invalid portfolio value ({current_value}) or no target ETFs ({len(target_etfs)})"
                )
                return

            target_allocation = current_value / len(target_etfs)

            # Get current positions
            current_positions = {}
            for d in self.datas:
                position = self.getposition(d)
                if position.size != 0:
                    current_positions[d._name] = {
                        "size": position.size,
                        "price": position.price,
                        "value": position.size * d.close[0],
                        "data": d,
                    }

            # Exit positions not in target portfolio
            for etf_name, pos_info in current_positions.items():
                if etf_name not in target_etfs:
                    self.log(f"Exiting position in {etf_name}")
                    try:
                        self.sell(data=pos_info["data"], size=pos_info["size"])
                        # Remove from entry price tracking
                        self.position_entry_prices.pop(etf_name, None)
                        self.log(f"{etf_name}: Removed from entry price tracking (position exited)")
                    except Exception as e:
                        self.log(f"Error selling {etf_name}: {str(e)}")

            # Enter/adjust positions for target ETFs
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

                    current_position = self.getposition(etf_data)
                    current_shares = int(
                        current_position.size
                    )  # Ensure current shares is integer

                    shares_diff = target_shares - current_shares

                    if (
                        abs(shares_diff) >= 1
                    ):  # Minimum trade threshold of 1 whole share
                        if shares_diff > 0:
                            self.log(f"Buying {shares_diff} shares of {etf_name}")
                            self.buy(data=etf_data, size=shares_diff)
                            # Set entry price only for new positions, not for adding to existing ones
                            if current_shares == 0:
                                # New position
                                self.position_entry_prices[etf_name] = current_price
                                self.log(f"{etf_name}: NEW position - Entry price set to {current_price:.2f}")
                            else:
                                # Adding to existing position - calculate weighted average entry price
                                if etf_name in self.position_entry_prices:
                                    old_entry = self.position_entry_prices[etf_name]
                                    total_shares = current_shares + shares_diff
                                    weighted_avg = ((current_shares * old_entry) + (shares_diff * current_price)) / total_shares
                                    self.position_entry_prices[etf_name] = weighted_avg
                                    self.log(f"{etf_name}: ADDING to position - New weighted avg entry price: {weighted_avg:.2f} (was {old_entry:.2f})")
                                else:
                                    # This shouldn't happen but handle it
                                    self.position_entry_prices[etf_name] = current_price
                                    self.log(f"{etf_name}: Missing entry price - Setting to current: {current_price:.2f}")
                        else:
                            self.log(f"Selling {abs(shares_diff)} shares of {etf_name}")
                            self.sell(data=etf_data, size=abs(shares_diff))
                            # If completely exiting position, remove from tracking
                            if current_shares + shares_diff <= 0:
                                self.position_entry_prices.pop(etf_name, None)
                                self.log(f"{etf_name}: Removed from entry price tracking (position closed)")
                            else:
                                # Partial sell - keep the same entry price for remaining shares
                                entry_price = self.position_entry_prices.get(etf_name, current_price)
                                self.log(f"{etf_name}: Partial sell - keeping entry price {entry_price:.2f}")
                except Exception as e:
                    self.log(f"Error processing {etf_name}: {str(e)}")

        except Exception as e:
            self.log(f"Critical error in _execute_rebalancing_trades: {str(e)}")

    def _check_exit_conditions(self):
        """Check if any positions should be exited based on momentum deterioration"""
        # For now, rely on periodic rebalancing
        # Could add stop-loss or momentum deterioration checks here
        pass

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

    def get_parameter_grid(self) -> Dict[str, List[Any]]:
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
            "exit_rank_buffer": [1.5, 2.0, 2.5],  # 3 options
            "min_momentum_threshold": [0.0, 0.05, 0.10],  # 3 options
            "volume_threshold": [100000, 200000],  # 2 options
            "use_threshold_rebalancing": [False, True],  # 2 options
            "profit_threshold_pct": [4.0, 5.0, 10.0],  # 3 options
            "loss_threshold_pct": [-2.5, -3.0, -5.0],  # 3 options
        }

    def get_intraday_parameter_grid(self) -> Dict[str, List[Any]]:
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
            "exit_rank_buffer": [1.5, 2.0, 2.5],  # 3 options
            "min_momentum_threshold": [0.0, 0.05, 0.10],  # 3 options
            "volume_threshold": [100000, 200000],  # 2 options
            "use_threshold_rebalancing": [False, True],  # 2 options
            "profit_threshold_pct": [5.0, 8.0, 12.0],  # 3 options
            "loss_threshold_pct": [-2.0, -3.0, -5.0],  # 3 options
        }

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for ETF momentum strategy
        """
        return {
            "portfolio_size": 5,
            "rebalance_frequency": 30,  # Reduced from 30 for more frequent rebalancing
            "long_term_period": 200,  # Reduced from 126 for earlier start
            "short_term_period": 90,  # Reduced from 30
            "moving_avg_period": 20,  # Reduced from 25
            "exit_rank_buffer": 1.5,
            "min_momentum_threshold": 0.0,
            "volume_threshold": 200000,
            "use_threshold_rebalancing": False,
            "profit_threshold_pct": 4,
            "loss_threshold_pct": -4.0,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
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

        # Exit rank buffer should be >= 1.0
        if params.get("exit_rank_buffer", 0) < 1.0:
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

    def get_composite_score_weights(self) -> Dict[str, float]:
        """
        Weights optimized for ETF momentum strategy
        """
        return {"total_return": 0.4, "sharpe_ratio": 0.35, "max_drawdown": 0.25}
