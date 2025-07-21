"""
Unified Strategy Experiment Framework

This module provides a unified interface for running experiments across different trading strategies.
It supports any strategy that implements the StrategyConfig interface.
"""

import itertools
import json
import logging
import os
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import backtrader as bt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from dashboard_components import DashboardComponents
from performance_display import (PerformanceDisplayManager,
                                 SingleStrategyAnalyzer,
                                 StatisticsDisplayManager)
from strategies.base_strategy import ExperimentResult, StrategyConfig
from streak_analyzer import DetailedTradeAnalyzer, StreakAnalyzer
from utils import IndianBrokerageCommission, MarketDataLoader

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logger = logging.getLogger(__name__)


class UnifiedExperimentFramework:
    """
    Unified framework for running experiments across different trading strategies
    """

    def __init__(
        self, strategy_config: StrategyConfig, results_dir: str = "experiment_results"
    ):
        """
        Initialize the experiment framework

        Args:
            strategy_config: Strategy configuration implementing StrategyConfig interface
            results_dir: Directory to store experiment results
        """
        self.strategy_config = strategy_config
        self.results_dir = results_dir
        self.strategy_name = strategy_config.__class__.__name__.replace("Config", "")

        # Create strategy-specific results directory
        self.strategy_results_dir = os.path.join(
            results_dir, self.strategy_name.lower()
        )
        if not os.path.exists(self.strategy_results_dir):
            os.makedirs(self.strategy_results_dir)

        self.results: List[ExperimentResult] = []
        self.best_result: Optional[ExperimentResult] = None
        self.best_score = -float("inf")

    def generate_parameter_combinations(
        self, max_combinations: int = 100, interval: str = "1d"
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations for testing

        Args:
            max_combinations: Maximum number of combinations to test
            interval: Data interval to optimize parameter selection

        Returns:
            list: List of parameter dictionaries
        """
        # Use intraday-specific parameter grid for minute intervals
        if interval in ["1m", "2m", "5m", "15m", "30m"] and hasattr(
            self.strategy_config, "get_intraday_parameter_grid"
        ):
            param_grid = self.strategy_config.get_intraday_parameter_grid()
            print(f"📊 Using intraday-optimized parameter grid for {interval} interval")
        else:
            param_grid = self.strategy_config.get_parameter_grid()

        # Generate all possible combinations
        keys = param_grid.keys()
        values = param_grid.values()
        all_combinations = list(itertools.product(*values))

        # Filter out invalid combinations using strategy validation
        valid_combinations = []
        for combo in all_combinations:
            params = dict(zip(keys, combo))
            if self.strategy_config.validate_params(params):
                valid_combinations.append(params)

        # If too many combinations, sample randomly
        if len(valid_combinations) > max_combinations:
            import random

            random.seed(42)  # For reproducibility
            valid_combinations = random.sample(valid_combinations, max_combinations)

        print(
            f"🧪 Generated {len(valid_combinations)} valid parameter combinations for {self.strategy_name}"
        )
        return valid_combinations

    def prepare_data_feeds(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> List[bt.feeds.PandasData]:
        """
        Prepare data feeds for the strategy

        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval ('1d', '5m', '15m', '1h', etc.)

        Returns:
            list: List of backtrader data feeds
        """
        # Adjust date range based on interval limitations
        adjusted_start_date = start_date
        adjusted_end_date = end_date

        if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
            # Intraday data: Yahoo Finance limits to last 60 days for minute data
            max_start_date = end_date - timedelta(days=55)  # Use 55 days to be safe
            if start_date < max_start_date:
                adjusted_start_date = max_start_date
                print(
                    f"⚠️ Adjusted start date to {adjusted_start_date.strftime('%Y-%m-%d')} for {interval} interval (Yahoo Finance limitation)"
                )
        elif interval in ["1h"]:
            # Hourly data: Available for ~730 days
            max_start_date = end_date - timedelta(days=700)  # Use 700 days to be safe
            if start_date < max_start_date:
                adjusted_start_date = max_start_date
                print(
                    f"⚠️ Adjusted start date to {adjusted_start_date.strftime('%Y-%m-%d')} for {interval} interval (Yahoo Finance limitation)"
                )

        loader = MarketDataLoader()

        required_feeds = self.strategy_config.get_required_data_feeds()

        if required_feeds == -1:  # Variable number (like momentum strategy)
            data_feeds = loader.load_market_data(
                symbols=symbols,
                start_date=adjusted_start_date,
                end_date=adjusted_end_date,
                interval=interval,
                force_refresh=False,
                use_parallel=False,  # Disable parallel for experiments
            )
        elif required_feeds == 2:  # Pairs trading
            # Use first two symbols for pairs trading
            if len(symbols) < 2:
                print(
                    f"❌ Pairs trading requires at least 2 symbols, got {len(symbols)}"
                )
                return []

            data_feeds = loader.load_market_data(
                symbols=symbols[:2],
                start_date=adjusted_start_date,
                end_date=adjusted_end_date,
                interval=interval,
                force_refresh=False,
                use_parallel=False,
            )
        else:
            # Fixed number of feeds
            data_feeds = loader.load_market_data(
                symbols=symbols[:required_feeds],
                start_date=adjusted_start_date,
                end_date=adjusted_end_date,
                interval=interval,
                force_refresh=False,
                use_parallel=False,
            )

        return data_feeds

    def run_single_experiment(
        self,
        params: Dict[str, Any],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_cash: float = 1000000,
        preloaded_data_feeds: List[bt.feeds.PandasData] = None,
        interval: str = "1d",
    ) -> Optional[ExperimentResult]:
        """
        Run a single experiment with given parameters

        Args:
            params: Parameter dictionary
            symbols: List of stock symbols
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_cash: Initial cash for backtest
            preloaded_data_feeds: Pre-loaded data feeds to avoid re-loading
            interval: Data interval ('1d', '5m', '15m', '1h', etc.)

        Returns:
            ExperimentResult or None if experiment failed
        """
        experiment_start_time = time.time()

        try:
            # Create cerebro instance
            cerebro = bt.Cerebro()
            cerebro.broker.setcash(initial_cash)
            cerebro.broker.set_coc(True)

            # Apply Indian brokerage commission scheme
            commission_scheme = IndianBrokerageCommission()
            cerebro.broker.addcommissioninfo(commission_scheme)

            # Use pre-loaded data feeds if available, otherwise load fresh
            if preloaded_data_feeds:
                data_feeds = preloaded_data_feeds
            else:
                data_feeds = self.prepare_data_feeds(
                    symbols, start_date, end_date, interval
                )

            if not data_feeds:
                return None

            # Add data feeds
            for data_feed in data_feeds:
                cerebro.adddata(data_feed)

            # Get strategy class and add with parameters
            strategy_class = self.strategy_config.get_strategy_class()
            if strategy_class is None:
                return None

            # Merge default params with experiment params
            # Use intraday-specific defaults for minute intervals
            if interval in ["1m", "2m", "5m", "15m", "30m"] and hasattr(
                self.strategy_config, "get_intraday_default_params"
            ):
                default_params = self.strategy_config.get_intraday_default_params()
            else:
                default_params = self.strategy_config.get_default_params()
            final_params = {**default_params, **params}

            cerebro.addstrategy(strategy_class, **final_params)

            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

            # Add custom streak analyzer
            cerebro.addanalyzer(StreakAnalyzer, _name="streaks")
            cerebro.addanalyzer(DetailedTradeAnalyzer, _name="detailed_trades")

            # Run backtest
            # Set runonce=True for performance, assuming data feeds are aligned.
            # This may need to be False if data feeds have varying lengths.
            results = cerebro.run(runonce=True)
            result = results[0]

            # Extract metrics
            final_value = cerebro.broker.getvalue()
            total_return = ((final_value - initial_cash) / initial_cash) * 100

            sharpe_analysis = result.analyzers.sharpe.get_analysis()
            drawdown_analysis = result.analyzers.drawdown.get_analysis()
            trades_analysis = result.analyzers.trades.get_analysis()

            sharpe_ratio = sharpe_analysis.get("sharperatio", 0) or 0
            max_drawdown = drawdown_analysis.get("max", {}).get("drawdown", 0) or 0

            # Trade statistics
            total_trades = trades_analysis.get("total", {}).get("total", 0)
            won_trades = trades_analysis.get("won", {}).get("total", 0)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

            # Profit factor
            gross_profit = trades_analysis.get("won", {}).get("pnl", {}).get("total", 0)
            gross_loss = abs(
                trades_analysis.get("lost", {}).get("pnl", {}).get("total", 0)
            )
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

            # Extract enhanced metrics from custom analyzers
            streak_analysis = result.analyzers.streaks.get_analysis()
            detailed_analysis = result.analyzers.detailed_trades.get_analysis()

            max_winning_streak = streak_analysis.get("max_winning_streak", 0)
            max_losing_streak = streak_analysis.get("max_losing_streak", 0)
            avg_win = streak_analysis.get("avg_win", 0.0)
            avg_loss = streak_analysis.get("avg_loss", 0.0)
            max_win = streak_analysis.get("max_win", 0.0)
            max_loss = streak_analysis.get("max_loss", 0.0)
            consecutive_wins = streak_analysis.get("consecutive_wins", 0)
            consecutive_losses = streak_analysis.get("consecutive_losses", 0)
            even_trades = streak_analysis.get("even_trades", 0)

            # Trade length from detailed analyzer
            avg_trade_length = detailed_analysis.get("len", {}).get("avg", 0.0)

            # Calculate additional metrics
            # Annualized return (assuming 252 trading days per year)
            days_in_backtest = (end_date - start_date).days
            annualized_return = (
                ((final_value / initial_cash) ** (365.25 / days_in_backtest) - 1) * 100
                if days_in_backtest > 0
                else 0
            )

            # Expectancy calculation using TradeAnalyzer data
            # Expectancy = (Probability of Win × Average Win) + (Probability of Loss × Average Loss)
            if total_trades > 0:
                prob_win = won_trades / total_trades
                prob_loss = (total_trades - won_trades) / total_trades

                # Get average win and loss from TradeAnalyzer
                avg_win_trade = (
                    trades_analysis.get("won", {}).get("pnl", {}).get("average", 0) or 0
                )
                avg_loss_trade = (
                    trades_analysis.get("lost", {}).get("pnl", {}).get("average", 0)
                    or 0
                )

                # Calculate expectancy (expected value per trade in rupees)
                expectancy = (prob_win * avg_win_trade) + (prob_loss * avg_loss_trade)
            else:
                expectancy = 0

            # Calculate metrics dict for composite score
            metrics = {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
            }

            # Calculate composite score using strategy-specific method
            composite_score = self.strategy_config.calculate_composite_score(metrics)

            experiment_duration = time.time() - experiment_start_time

            experiment_result = ExperimentResult(
                params=params.copy(),
                final_value=final_value,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                composite_score=composite_score,
                num_data_feeds=len(data_feeds),
                strategy_name=self.strategy_name,
                experiment_duration=experiment_duration,
                trades_count=total_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                max_winning_streak=max_winning_streak,
                max_losing_streak=max_losing_streak,
                avg_win=avg_win,
                avg_loss=avg_loss,
                max_win=max_win,
                max_loss=max_loss,
                gross_profit=gross_profit,
                gross_loss=gross_loss,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                even_trades=even_trades,
                avg_trade_length=avg_trade_length,
                annualized_return=annualized_return,
                expectancy=expectancy,
            )

            # Extract and store portfolio values from TimeReturn analyzer
            timereturn_analysis = result.analyzers.timereturn.get_analysis()
            dates = list(timereturn_analysis.keys())

            # Calculate portfolio value from returns, starting with initial cash
            if dates:
                returns = pd.Series(list(timereturn_analysis.values()))
                portfolio_values = (initial_cash * (1 + returns).cumprod()).tolist()
            else:
                portfolio_values = []

            experiment_result.portfolio_values = portfolio_values
            experiment_result.dates = dates

            return experiment_result

        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            return None

    def run_experiments(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_combinations: int = 50,
        initial_cash: float = 1000000,
        use_parallel: bool = True,
        max_workers: int = 4,
        interval: str = "1d",
        use_fixed_filename: bool = True,
    ):
        """
        Run multiple experiments with different parameter combinations

        Args:
            symbols: List of stock symbols to use
            start_date: Start date for backtest (defaults to 2 years ago)
            end_date: End date for backtest (defaults to now)
            max_combinations: Maximum number of parameter combinations to test
            initial_cash: Initial cash for backtest
            use_parallel: Whether to use parallel processing
            max_workers: Number of parallel workers
            interval: Data interval ('1d', '5m', '15m', '1h', etc.)
            use_fixed_filename: If True, uses fixed filenames that get overwritten.
                               If False, uses timestamped filenames.
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(
                days=365 * 2
            )  # 2 years ago (will be adjusted in prepare_data_feeds if needed)

        print(f"🚀 Starting {self.strategy_name} Strategy Experiments")
        print("=" * 60)
        print(
            f"📊 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        print(f"⏰ Interval: {interval}")
        print(f"💰 Initial Cash: ₹{initial_cash:,.0f}")
        print(f"📈 Symbols: {len(symbols)} stocks")

        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations(
            max_combinations, interval
        )

        # 🚀 PRE-LOAD DATA ONCE to avoid repeated loading
        logger.info("Pre-loading market data for experiments...")
        data_feeds = self.prepare_data_feeds(symbols, start_date, end_date, interval)
        if not data_feeds:
            logger.error("Failed to load market data. Aborting experiments.")
            return
        logger.info(f"Data loaded successfully for {len(data_feeds)} instruments")

        # Run experiments
        self.results = []

        if use_parallel and max_workers > 1:
            print(f"🔄 Running {len(param_combinations)} experiments in parallel...")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all experiments
                future_to_params = {
                    executor.submit(
                        self.run_single_experiment,
                        params,
                        symbols,
                        start_date,
                        end_date,
                        initial_cash,
                        data_feeds,  # Pass pre-loaded data
                        interval,
                    ): params
                    for params in param_combinations
                }

                # Collect results with progress bar
                with tqdm(
                    total=len(param_combinations), desc="🧪 Running experiments"
                ) as pbar:
                    for future in as_completed(future_to_params):
                        result = future.result()
                        if result:
                            self.results.append(result)

                            # Update best result
                            if result.composite_score > self.best_score:
                                self.best_score = result.composite_score
                                self.best_result = result

                        pbar.update(1)
        else:
            print(f"🔄 Running {len(param_combinations)} experiments sequentially...")
            for params in tqdm(param_combinations, desc="🧪 Running experiments"):
                result = self.run_single_experiment(
                    params,
                    symbols,
                    start_date,
                    end_date,
                    initial_cash,
                    data_feeds,
                    interval,
                )
                if result:
                    self.results.append(result)

                    # Update best result
                    if result.composite_score > self.best_score:
                        self.best_score = result.composite_score
                        self.best_result = result

        print(f"\n✅ Completed {len(self.results)} successful experiments")

        # Save results
        self.save_results()

        # Generate visualizations automatically
        if self.results:
            self.create_visualizations(use_fixed_filename=use_fixed_filename)

        # Display summary
        self.display_results_summary()

    def save_results(self):
        """Save experiment results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_file = os.path.join(
            self.strategy_results_dir,
            f"{self.strategy_name.lower()}_results_{timestamp}.json",
        )

        # Convert results to JSON-serializable format
        results_data = []
        for result in self.results:
            result_dict = {
                "params": result.params,
                "final_value": result.final_value,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "composite_score": result.composite_score,
                "num_data_feeds": result.num_data_feeds,
                "strategy_name": result.strategy_name,
                "experiment_duration": result.experiment_duration,
                "trades_count": result.trades_count,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "max_winning_streak": result.max_winning_streak,
                "max_losing_streak": result.max_losing_streak,
                "avg_win": result.avg_win,
                "avg_loss": result.avg_loss,
                "max_win": result.max_win,
                "max_loss": result.max_loss,
                "gross_profit": result.gross_profit,
                "gross_loss": result.gross_loss,
                "consecutive_wins": result.consecutive_wins,
                "consecutive_losses": result.consecutive_losses,
                "even_trades": result.even_trades,
                "avg_trade_length": result.avg_trade_length,
                "annualized_return": getattr(result, "annualized_return", 0),
                "expectancy": getattr(result, "expectancy", 0),
            }
            results_data.append(result_dict)

        with open(json_file, "w") as f:
            json.dump(results_data, f, indent=2)

        # Save as CSV
        csv_file = os.path.join(
            self.strategy_results_dir,
            f"{self.strategy_name.lower()}_results_{timestamp}.csv",
        )
        df = self.results_to_dataframe()
        df.to_csv(csv_file, index=False)

        print(f"💾 Results saved to {json_file} and {csv_file}")

    def results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        flat_results = []
        for result in self.results:
            flat_result = result.params.copy()
            flat_result.update(
                {
                    "final_value": result.final_value,
                    "total_return": result.total_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "composite_score": result.composite_score,
                    "num_data_feeds": result.num_data_feeds,
                    "strategy_name": result.strategy_name,
                    "experiment_duration": result.experiment_duration,
                    "trades_count": result.trades_count,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "max_winning_streak": result.max_winning_streak,
                    "max_losing_streak": result.max_losing_streak,
                    "avg_win": result.avg_win,
                    "avg_loss": result.avg_loss,
                    "max_win": result.max_win,
                    "max_loss": result.max_loss,
                    "gross_profit": result.gross_profit,
                    "gross_loss": result.gross_loss,
                    "consecutive_wins": result.consecutive_wins,
                    "consecutive_losses": result.consecutive_losses,
                    "even_trades": result.even_trades,
                    "avg_trade_length": result.avg_trade_length,
                    "annualized_return": getattr(result, "annualized_return", 0),
                    "expectancy": getattr(result, "expectancy", 0),
                }
            )
            flat_results.append(flat_result)

        return pd.DataFrame(flat_results)

    def display_results_summary(self):
        """Display comprehensive summary of experiment results with all available metrics"""
        if not self.results:
            print("❌ No results to display")
            return

        df = self.results_to_dataframe()

        print(f"\n📊 {self.strategy_name.upper()} STRATEGY EXPERIMENT RESULTS")
        print("=" * 80)

        # Best result detailed breakdown
        if self.best_result:
            PerformanceDisplayManager.display_single_result_performance(
                self.best_result,
                f"🏆 BEST STRATEGY PERFORMANCE (Composite Score: {self.best_score:.3f})",
                self.best_result.params,
            )

        # Top 10 results with comprehensive metrics
        StatisticsDisplayManager.display_top_results_comparison(df)

        # Comprehensive Statistics across all experiments
        StatisticsDisplayManager.display_comprehensive_statistics(df)

        # Summary insights
        StatisticsDisplayManager.display_key_insights(df)

    def create_visualizations(self, use_fixed_filename: bool = True):
        """Create visualizations of experiment results

        Args:
            use_fixed_filename: If True, uses a fixed filename that gets overwritten.
                               If False, uses timestamped filename (default behavior).
        """
        if not self.results:
            print("❌ No results to visualize")
            return

        df = self.results_to_dataframe()

        # Set up the plotting style
        plt.style.use("seaborn-v0_8")
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(
            f"{self.strategy_name} Strategy Experiments - Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Composite Score Distribution
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(
            df["composite_score"],
            bins=30,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        ax1.set_title("Composite Score Distribution")
        ax1.set_xlabel("Composite Score")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)

        # 2. Return vs Sharpe Ratio
        ax2 = plt.subplot(2, 3, 2)
        scatter = ax2.scatter(
            df["sharpe_ratio"],
            df["total_return"],
            c=df["composite_score"],
            cmap="viridis",
            alpha=0.6,
        )
        ax2.set_title("Return vs Sharpe Ratio")
        ax2.set_xlabel("Sharpe Ratio")
        ax2.set_ylabel("Total Return (%)")
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label="Composite Score")

        # 3. Drawdown vs Return
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(df["max_drawdown"], df["total_return"], alpha=0.6, color="coral")
        ax3.set_title("Drawdown vs Return")
        ax3.set_xlabel("Max Drawdown (%)")
        ax3.set_ylabel("Total Return (%)")
        ax3.grid(True, alpha=0.3)

        # 4. Parameter Importance (correlation with score)
        ax4 = plt.subplot(2, 3, 4)
        param_importance = {}
        numeric_params = []

        for param in df.columns:
            if param not in [
                "final_value",
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
                "composite_score",
                "num_data_feeds",
                "strategy_name",
                "experiment_duration",
            ]:
                if df[param].dtype in ["int64", "float64"]:
                    correlation = df[param].corr(df["composite_score"])
                    if not pd.isna(correlation):
                        param_importance[param] = abs(correlation)
                        numeric_params.append(param)

        if param_importance:
            params = list(param_importance.keys())
            importance = list(param_importance.values())
            ax4.barh(params, importance, color="lightgreen")
            ax4.set_title("Parameter Importance")
            ax4.set_xlabel("Absolute Correlation with Score")

        # 5. Performance over time (experiment duration)
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(
            df["experiment_duration"], df["composite_score"], alpha=0.6, color="orange"
        )
        ax5.set_title("Performance vs Experiment Duration")
        ax5.set_xlabel("Experiment Duration (seconds)")
        ax5.set_ylabel("Composite Score")
        ax5.grid(True, alpha=0.3)

        # 6. Best parameters visualization (if available)
        ax6 = plt.subplot(2, 3, 6)
        if len(numeric_params) >= 2:
            # Create a correlation matrix of numeric parameters
            param_corr = df[numeric_params + ["composite_score"]].corr()
            sns.heatmap(param_corr, annot=True, cmap="coolwarm", center=0, ax=ax6)
            ax6.set_title("Parameter Correlation Matrix")
        else:
            ax6.text(
                0.5,
                0.5,
                "Insufficient numeric\nparameters for\ncorrelation analysis",
                ha="center",
                va="center",
                transform=ax6.transAxes,
            )
            ax6.set_title("Parameter Analysis")

        plt.tight_layout()

        # Save the plot
        if use_fixed_filename:
            # Use fixed filename for daily updates (overwrites previous file)
            filename = os.path.join(
                self.strategy_results_dir,
                f"{self.strategy_name.lower()}_analysis.png",
            )
        else:
            # Use timestamped filename (creates new file each time)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.strategy_results_dir,
                f"{self.strategy_name.lower()}_analysis_{timestamp}.png",
            )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"📊 Visualizations saved to {filename}")

        # plt.show()

    def _prepare_portfolio_data(
        self, result: ExperimentResult, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Prepare portfolio data for dashboard creation"""
        # Prepare data
        portfolio_values = getattr(result, "portfolio_values", [])
        dates = getattr(result, "dates", [])

        if not portfolio_values or not dates:
            # If no portfolio tracking data, create mock data for visualization structure
            dates = pd.date_range(start_date, end_date, freq="D")
            initial_value = result.final_value / (1 + result.total_return / 100)
            portfolio_values = [
                initial_value * (1 + result.total_return / 100 * i / len(dates))
                for i in range(len(dates))
            ]

        # Convert to pandas for easier manipulation
        portfolio_df = pd.DataFrame({"date": dates, "value": portfolio_values})
        portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
        portfolio_df.set_index("date", inplace=True)

        # Calculate returns
        portfolio_df["returns"] = portfolio_df["value"].pct_change()
        portfolio_df["cumulative_returns"] = (1 + portfolio_df["returns"]).cumprod() - 1

        # Calculate drawdown
        portfolio_df["peak"] = portfolio_df["value"].cummax()
        portfolio_df["drawdown"] = (
            (portfolio_df["value"] - portfolio_df["peak"]) / portfolio_df["peak"] * 100
        )

        return portfolio_df

    def _setup_dashboard_figure(self) -> plt.Figure:
        """Setup the main dashboard figure with proper styling"""
        plt.style.use("default")
        fig = plt.figure(figsize=(20, 30))
        fig.suptitle(
            f"{self.strategy_name} Strategy - Comprehensive Portfolio Dashboard",
            fontsize=20,
            fontweight="bold",
            y=0.985,
        )
        return fig

    def _save_dashboard(self, fig: plt.Figure, use_fixed_filename: bool) -> str:
        """Save the dashboard figure to file"""
        if use_fixed_filename:
            # Use fixed filename for daily updates (overwrites previous file)
            filename = os.path.join(
                self.strategy_results_dir,
                f"{self.strategy_name.lower()}_dashboard.png",
            )
        else:
            # Use timestamped filename (creates new file each time)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.strategy_results_dir,
                f"{self.strategy_name.lower()}_dashboard_{timestamp}.png",
            )

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"📊 Comprehensive portfolio dashboard saved to {filename}")
        return filename

    def create_portfolio_dashboard(
        self,
        result: ExperimentResult,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        strategy_class=None,
        use_fixed_filename: bool = True,
    ) -> str:
        """
        Create comprehensive portfolio performance dashboard

        Args:
            result: ExperimentResult with portfolio performance data
            symbols: List of symbols used
            start_date: Start date of analysis
            end_date: End date of analysis
            strategy_class: Strategy class for getting strategy instance (optional)
            use_fixed_filename: If True, uses a fixed filename that gets overwritten.
                               If False, uses timestamped filename (default behavior).

        Returns:
            str: Path to the generated dashboard file
        """
        print(
            f"📊 Creating comprehensive portfolio dashboard for {self.strategy_name}..."
        )

        # Setup figure and prepare data
        fig = self._setup_dashboard_figure()
        portfolio_df = self._prepare_portfolio_data(result, start_date, end_date)

        # Create all dashboard components using DashboardComponents
        components = DashboardComponents()

        # Row 1: Portfolio Value, Drawdown, Monthly Returns
        components.create_portfolio_value_chart(plt.subplot(5, 3, 1), portfolio_df)
        components.create_drawdown_chart(plt.subplot(5, 3, 2), portfolio_df)
        components.create_monthly_returns_heatmap(plt.subplot(5, 3, 3), portfolio_df)

        # Row 2: Returns Distribution, Rolling Sharpe, Cumulative Returns
        components.create_returns_distribution(plt.subplot(5, 3, 4), portfolio_df)
        components.create_rolling_sharpe_chart(plt.subplot(5, 3, 5), portfolio_df)
        components.create_cumulative_returns_chart(
            plt.subplot(5, 3, 6), portfolio_df, self.strategy_name
        )

        # Row 3: Performance Metrics, Risk-Return Scatter, Portfolio Summary
        components.create_performance_metrics_panel(
            plt.subplot(5, 3, 7), result, start_date, end_date, portfolio_df
        )
        components.create_risk_return_scatter(
            plt.subplot(5, 3, 8),
            result,
            start_date,
            end_date,
            portfolio_df,
            self.strategy_name,
        )
        components.create_portfolio_summary_panel(
            plt.subplot(5, 3, 9),
            self.strategy_name,
            start_date,
            end_date,
            symbols,
            result,
        )

        # Row 4: Weekly Pattern, Rolling Volatility, Underwater Plot
        components.create_weekly_returns_pattern(plt.subplot(5, 3, 10), portfolio_df)
        components.create_rolling_volatility_chart(plt.subplot(5, 3, 11), portfolio_df)
        components.create_underwater_plot(plt.subplot(5, 3, 12), portfolio_df)

        # Row 5: Parameters Panel (spanning all 3 columns)
        components.create_parameters_panel(plt.subplot(5, 1, 5), result)

        # Save and return filename
        return self._save_dashboard(fig, use_fixed_filename)

    def run_portfolio_analysis(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_cash: float = 1000000,
        params: Dict[str, Any] = None,
        interval: str = "1d",
        use_fixed_filename: bool = True,
    ) -> Optional[str]:
        """
        Run portfolio analysis and generate dashboard without experiments

        Args:
            symbols: List of stock symbols
            start_date: Analysis start date
            end_date: Analysis end date
            initial_cash: Initial portfolio value
            params: Strategy parameters (uses defaults if None)
            interval: Data interval ('1d', '5m', '15m', '1h', etc.)
            use_fixed_filename: If True, uses a fixed filename that gets overwritten.
                               If False, uses timestamped filename.

        Returns:
            Path to generated dashboard file
        """
        print(f"📊 Running portfolio analysis for {self.strategy_name} strategy...")

        # Use default params if none provided
        if params is None:
            params = self.strategy_config.get_default_params()

        # Run single experiment to get portfolio data
        result = self.run_single_experiment(
            params=params,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            interval=interval,
        )

        if result:
            # Display comprehensive performance details
            self.display_single_strategy_performance(
                result, symbols, start_date, end_date, params
            )

            # Generate comprehensive dashboard
            dashboard_file = self.create_portfolio_dashboard(
                result,
                symbols,
                start_date,
                end_date,
                use_fixed_filename=use_fixed_filename,
            )

            return dashboard_file
        else:
            print("❌ Portfolio analysis failed")
            return None

    def display_single_strategy_performance(
        self,
        result: ExperimentResult,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        params: Dict[str, Any],
    ):
        """Display comprehensive performance details for a single strategy run"""
        SingleStrategyAnalyzer.display_comprehensive_analysis(
            result, symbols, start_date, end_date, params, self.strategy_name
        )

    def get_optimal_parameters(self) -> Optional[Dict[str, Any]]:
        """Get the optimal parameters from experiments"""
        if self.best_result:
            return self.best_result.params
        else:
            print("❌ No experiments run yet. Run experiments first.")
            return None
