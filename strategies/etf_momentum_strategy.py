"""
ETF Momentum Strategy

A portfolio momentum strategy that trades Indian ETFs based on momentum scoring
and regular rebalancing. The strategy selects top-performing ETFs and
maintains an equal-weighted portfolio with realistic trading costs.
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import backtrader as bt
import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, StrategyConfig


class ETFMomentumStrategy(BaseStrategy):
    """
    ETF Momentum Strategy based on momentum scoring and rebalancing
    """

    params = (
        ("portfolio_size", 5),
        ("rebalance_frequency", 15),  # Days between rebalancing  
        ("long_term_period", 90),  # Long-term momentum period (days) - further reduced for earlier start
        ("short_term_period", 20),  # Short-term momentum period (days) - further reduced
        ("moving_avg_period", 20),  # Moving average filter period - reduced
        ("exit_rank_buffer", 2.0),  # Exit multiplier (exit if rank > portfolio_size * buffer)
        ("min_momentum_threshold", 0.0),  # Minimum momentum to consider
        ("volume_threshold", 100000),  # Minimum daily volume threshold
    )

    def __init__(self):
        super().__init__()
        self.rebalance_counter = 0
        self.last_rebalance_date = None
        self.portfolio_holdings = {}  # Track current holdings
        
        # Initialize indicators for each ETF
        self.indicators = {}
        self.log(f"Initializing ETFMomentumStrategy with {len(self.datas)} ETF feeds")
        
        for d in self.datas:
            etf_name = d._name
            self.log(f"Setting up indicators for ETF: {etf_name}")
            
            try:
                self.indicators[etf_name] = {
                    # Price indicators
                    'sma': bt.ind.SimpleMovingAverage(d.close, period=self.p.moving_avg_period),
                    # Momentum indicators  
                    'roc_long': bt.ind.RateOfChange(d.close, period=self.p.long_term_period),
                    'roc_short': bt.ind.RateOfChange(d.close, period=self.p.short_term_period),
                    # Volume indicator
                    'volume_sma': bt.ind.SimpleMovingAverage(d.volume, period=20),
                }
                self.log(f"Successfully created indicators for {etf_name}")
            except Exception as e:
                self.log(f"ERROR creating indicators for {etf_name}: {str(e)}")
                self.indicators[etf_name] = {
                    'sma': None,
                    'roc_long': None, 
                    'roc_short': None,
                    'volume_sma': None,
                }

        self.log(f"Indicators initialized for {len(self.indicators)} ETFs")

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
            
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        should_rebalance = days_since_rebalance >= self.p.rebalance_frequency
        
        if should_rebalance:
            self.log(f"Time to rebalance: {days_since_rebalance} days >= {self.p.rebalance_frequency} threshold")
        
        return should_rebalance

    def _rebalance_portfolio(self, current_date):
        """Rebalance the portfolio based on momentum scores"""
        try:
            # Calculate momentum scores for all ETFs
            momentum_scores = self._calculate_momentum_scores()
            
            if not momentum_scores:
                self.log("No valid momentum scores calculated, skipping rebalance")
                return
            
            # Filter ETFs that meet criteria
            eligible_etfs = self._filter_eligible_etfs(momentum_scores)
            
            if len(eligible_etfs) < self.p.portfolio_size:
                self.log(f"Only {len(eligible_etfs)} eligible ETFs found, need {self.p.portfolio_size}")
                if len(eligible_etfs) == 0:
                    return
            
            # Select top ETFs
            top_etfs = eligible_etfs[:self.p.portfolio_size]
            top_etf_names = [etf['name'] for etf in top_etfs]
            
            self.log(f"Selected top ETFs: {top_etf_names}")
            
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
                roc_long = indicators.get('roc_long')
                roc_short = indicators.get('roc_short')
                sma = indicators.get('sma')
                volume_sma = indicators.get('volume_sma')
                
                # Check if indicators exist
                if (roc_long is None or roc_short is None or sma is None):
                    self.log(f"{etf_name}: Indicators not initialized")
                    continue
                
                # Get current values and check if they're valid
                try:
                    current_price = d.close[0]
                    long_momentum = roc_long[0] / 100.0  # Convert percentage to decimal
                    short_momentum = roc_short[0] / 100.0
                    sma_value = sma[0]
                    current_volume = d.volume[0] if hasattr(d, 'volume') else 1000000
                    avg_volume = volume_sma[0] if volume_sma and len(volume_sma) > 0 else current_volume
                    
                    # Skip if values are NaN or invalid (indicates insufficient data)
                    if (np.isnan(long_momentum) or np.isnan(short_momentum) or 
                        np.isnan(current_price) or np.isnan(sma_value)):
                        continue
                        
                except (IndexError, TypeError):
                    # Not enough data yet for indicators
                    continue
                
                # Calculate composite momentum score
                # Weight long-term more heavily, but consider short-term momentum
                momentum_score = (0.7 * long_momentum + 0.3 * short_momentum)
                
                # Apply trend filter (price above moving average)
                trend_filter = current_price > sma_value
                
                momentum_scores[etf_name] = {
                    'score': momentum_score,
                    'long_momentum': long_momentum,
                    'short_momentum': short_momentum,
                    'current_price': current_price,
                    'sma': sma_value,
                    'trend_filter': trend_filter,
                    'volume': current_volume,
                    'avg_volume': avg_volume,
                }
                
                self.log(f"{etf_name}: momentum={momentum_score:.4f}, "
                        f"long={long_momentum:.4f}, short={short_momentum:.4f}, "
                        f"trend_ok={trend_filter}")
                
            except Exception as e:
                self.log(f"Error calculating momentum for {etf_name}: {str(e)}")
                continue
        
        self.log(f"Momentum calculation: {len(momentum_scores)}/{total_etfs} ETFs with valid scores")
        return momentum_scores

    def _filter_eligible_etfs(self, momentum_scores):
        """Filter ETFs based on momentum and liquidity criteria"""
        eligible_etfs = []
        
        for etf_name, data in momentum_scores.items():
            # Apply filters
            meets_momentum_threshold = data['score'] >= self.p.min_momentum_threshold
            meets_trend_filter = data['trend_filter']
            meets_volume_threshold = data['avg_volume'] >= self.p.volume_threshold
            
            if meets_momentum_threshold and meets_trend_filter and meets_volume_threshold:
                eligible_etfs.append({
                    'name': etf_name,
                    'momentum_score': data['score'],
                    'data': data
                })
        
        # Sort by momentum score (descending)
        eligible_etfs.sort(key=lambda x: x['momentum_score'], reverse=True)
        
        self.log(f"Found {len(eligible_etfs)} eligible ETFs out of {len(momentum_scores)} total")
        
        return eligible_etfs

    def _execute_rebalancing_trades(self, target_etfs):
        """Execute trades to rebalance to target portfolio"""
        current_value = self.broker.getvalue()
        target_allocation = current_value / len(target_etfs)
        
        # Get current positions
        current_positions = {}
        for d in self.datas:
            position = self.getposition(d)
            if position.size != 0:
                current_positions[d._name] = {
                    'size': position.size,
                    'price': position.price,
                    'value': position.size * d.close[0],
                    'data': d
                }
        
        # Exit positions not in target portfolio
        for etf_name, pos_info in current_positions.items():
            if etf_name not in target_etfs:
                self.log(f"Exiting position in {etf_name}")
                self.sell(data=pos_info['data'], size=pos_info['size'])
        
        # Enter/adjust positions for target ETFs
        for etf_name in target_etfs:
            etf_data = self._get_data_by_name(etf_name)
            if etf_data is None:
                continue
            
            current_price = etf_data.close[0]
            target_shares = target_allocation / current_price
            
            current_position = self.getposition(etf_data)
            current_shares = current_position.size
            
            shares_diff = target_shares - current_shares
            
            if abs(shares_diff) > 0.01:  # Minimum trade threshold
                if shares_diff > 0:
                    self.log(f"Buying {shares_diff:.2f} shares of {etf_name}")
                    self.buy(data=etf_data, size=shares_diff)
                else:
                    self.log(f"Selling {abs(shares_diff):.2f} shares of {etf_name}")
                    self.sell(data=etf_data, size=abs(shares_diff))

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
        """
        return {
            "portfolio_size": [3, 5, 7, 10],
            "rebalance_frequency": [20, 30, 45, 60],
            "long_term_period": [180, 252, 365],
            "short_term_period": [30, 60, 90],
            "moving_avg_period": [20, 50, 100],
            "exit_rank_buffer": [1.5, 2.0, 2.5, 3.0],
            "min_momentum_threshold": [-0.05, 0.0, 0.05, 0.10],
            "volume_threshold": [50000, 100000, 200000],
        }

    def get_intraday_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Intraday-optimized parameter grid (shorter periods)
        """
        return {
            "portfolio_size": [3, 5, 7],
            "rebalance_frequency": [5, 10, 15],  # Days
            "long_term_period": [60, 90, 120],  # Shorter for intraday
            "short_term_period": [15, 30, 45],
            "moving_avg_period": [10, 20, 30],
            "exit_rank_buffer": [1.5, 2.0, 2.5],
            "min_momentum_threshold": [0.0, 0.05, 0.10],
            "volume_threshold": [100000, 200000],
        }

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for ETF momentum strategy
        """
        return {
            "portfolio_size": 5,
            "rebalance_frequency": 15,  # Reduced from 30 for more frequent rebalancing
            "long_term_period": 90,  # Reduced from 126 for earlier start
            "short_term_period": 20,  # Reduced from 30
            "moving_avg_period": 20,  # Reduced from 25
            "exit_rank_buffer": 2.0,
            "min_momentum_threshold": 0.0,
            "volume_threshold": 100000,
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
        return {
            "total_return": 0.4,
            "sharpe_ratio": 0.35,
            "max_drawdown": 0.25
        }

