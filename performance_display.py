"""
Performance Display and Analysis Utilities

This module provides reusable components for displaying and analyzing
trading strategy performance results in a comprehensive format.
"""

from typing import Any, Dict, List

import pandas as pd
from tabulate import tabulate

try:
    from strategies.base_strategy import ExperimentResult
except ImportError:
    # Handle potential import issues
    ExperimentResult = None


class PerformanceDisplayManager:
    """
    Centralized manager for displaying performance metrics and analysis
    """

    @staticmethod
    def get_performance_metrics_table(result: ExperimentResult) -> List[List[str]]:
        """Generate financial performance metrics table data"""
        return [
            ["Metric", "Value"],
            ["Total Return", f"{result.total_return:.2f}%"],
            ["Annualized Return", f"{getattr(result, 'annualized_return', 0):.2f}%"],
            ["Final Portfolio Value", f"₹{result.final_value:,.2f}"],
            ["Gross Profit", f"₹{result.gross_profit:,.2f}"],
            ["Gross Loss", f"₹{result.gross_loss:,.2f}"],
            ["Net Profit", f"₹{result.gross_profit - result.gross_loss:,.2f}"],
            ["Expectancy (Per Trade)", f"₹{getattr(result, 'expectancy', 0):,.2f}"],
        ]

    @staticmethod
    def get_risk_metrics_table(result: ExperimentResult) -> List[List[str]]:
        """Generate risk metrics table data"""
        return [
            ["Metric", "Value"],
            ["Sharpe Ratio", f"{result.sharpe_ratio:.3f}"],
            ["Maximum Drawdown", f"{result.max_drawdown:.2f}%"],
            ["Profit Factor", f"{result.profit_factor:.2f}"],
            ["Maximum Single Win", f"₹{result.max_win:,.2f}"],
            ["Maximum Single Loss", f"₹{result.max_loss:,.2f}"],
            ["Average Win", f"₹{result.avg_win:,.2f}"],
            ["Average Loss", f"₹{result.avg_loss:,.2f}"],
        ]

    @staticmethod
    def get_trading_activity_table(
        result: ExperimentResult, symbols_count: int = None
    ) -> List[List[str]]:
        """Generate trading activity table data"""
        data = [
            ["Metric", "Value"],
            ["Total Trades", f"{result.trades_count}"],
            ["Win Rate", f"{result.win_rate:.2f}%"],
            [
                "Average Trade Length",
                f"{getattr(result, 'avg_trade_length', 0):.1f} days",
            ],
            ["Even Trades", f"{result.even_trades}"],
        ]
        if symbols_count:
            data.append(
                ["Data Feeds Used", f"{result.num_data_feeds} of {symbols_count}"]
            )
        else:
            data.append(["Data Feeds Used", f"{result.num_data_feeds}"])
        return data

    @staticmethod
    def get_streak_analysis_table(result: ExperimentResult) -> List[List[str]]:
        """Generate streak analysis table data"""
        return [
            ["Metric", "Value"],
            ["Maximum Winning Streak", f"{result.max_winning_streak} trades"],
            ["Maximum Losing Streak", f"{result.max_losing_streak} trades"],
            ["Current Consecutive Wins", f"{result.consecutive_wins} trades"],
            ["Current Consecutive Losses", f"{result.consecutive_losses} trades"],
        ]

    @staticmethod
    def get_parameters_table(params: Dict[str, Any]) -> List[List[str]]:
        """Generate parameters table data"""
        data = [["Parameter", "Value"]]
        for param, value in params.items():
            data.append([param.replace("_", " ").title(), value])
        return data

    @staticmethod
    def assess_performance_quality(result: ExperimentResult) -> Dict[str, str]:
        """Assess the quality of performance metrics"""
        assessments = {}

        # Sharpe ratio assessment
        if result.sharpe_ratio > 2.0:
            assessments["sharpe"] = "Excellent"
        elif result.sharpe_ratio > 1.0:
            assessments["sharpe"] = "Good"
        else:
            assessments["sharpe"] = "Poor"

        # Drawdown assessment
        if result.max_drawdown < 10:
            assessments["drawdown"] = "Low"
        elif result.max_drawdown < 20:
            assessments["drawdown"] = "Medium"
        else:
            assessments["drawdown"] = "High"

        # Win rate assessment
        if result.win_rate > 60:
            assessments["win_rate"] = "Excellent"
        elif result.win_rate > 50:
            assessments["win_rate"] = "Good"
        else:
            assessments["win_rate"] = "Poor"

        # Profit factor assessment
        if result.profit_factor > 2.0:
            assessments["profit_factor"] = "Excellent"
        elif result.profit_factor > 1.5:
            assessments["profit_factor"] = "Good"
        else:
            assessments["profit_factor"] = "Poor"

        return assessments

    @staticmethod
    def calculate_professional_score(result: ExperimentResult) -> int:
        """Calculate professional assessment score (0-9)"""
        score = 0
        if getattr(result, "annualized_return", 0) > 12:
            score += 2
        if result.sharpe_ratio > 1.0:
            score += 2
        if result.max_drawdown < 20:
            score += 2
        if result.win_rate > 50:
            score += 1
        if result.profit_factor > 1.5:
            score += 1
        if result.max_losing_streak <= 5:
            score += 1
        return score

    @staticmethod
    def get_professional_recommendation(score: int) -> str:
        """Get professional recommendation based on score"""
        if score >= 8:
            return "🏆 INSTITUTIONAL GRADE: This strategy meets professional investment standards"
        elif score >= 6:
            return "✅ INVESTMENT WORTHY: Strong strategy suitable for deployment"
        elif score >= 4:
            return "⚠️  NEEDS OPTIMIZATION: Promising but requires parameter tuning"
        else:
            return "❌ NOT RECOMMENDED: Strategy needs significant improvement"

    @classmethod
    def display_single_result_performance(
        cls, result: ExperimentResult, title: str, params: Dict[str, Any] = None
    ):
        """Display comprehensive performance for a single result"""
        print(f"\n{title}")
        print("=" * 80)

        # Financial Performance
        print("\n💰 FINANCIAL PERFORMANCE")
        print(
            tabulate(
                cls.get_performance_metrics_table(result),
                headers="firstrow",
                tablefmt="grid",
            )
        )

        # Risk Metrics
        print("\n⚖️ RISK METRICS")
        print(
            tabulate(
                cls.get_risk_metrics_table(result), headers="firstrow", tablefmt="grid"
            )
        )

        # Trading Activity
        print("\n📈 TRADING ACTIVITY")
        print(
            tabulate(
                cls.get_trading_activity_table(result),
                headers="firstrow",
                tablefmt="grid",
            )
        )

        # Streak Analysis
        print("\n🔥 STREAK ANALYSIS")
        print(
            tabulate(
                cls.get_streak_analysis_table(result),
                headers="firstrow",
                tablefmt="grid",
            )
        )

        # Parameters (if provided)
        if params:
            print("\n🎯 STRATEGY PARAMETERS")
            print(
                tabulate(
                    cls.get_parameters_table(params),
                    headers="firstrow",
                    tablefmt="grid",
                )
            )


class StatisticsDisplayManager:
    """
    Manager for displaying statistical analysis of multiple results
    """

    @staticmethod
    def display_top_results_comparison(df: pd.DataFrame):
        """Display top 10 results comparison table"""
        top_10 = df.nlargest(10, "composite_score")
        print("\n📈 TOP 10 STRATEGIES COMPARISON")
        print("=" * 80)

        display_cols = [
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "trades_count",
            "win_rate",
            "profit_factor",
            "expectancy",
            "max_winning_streak",
            "max_losing_streak",
            "composite_score",
        ]

        # Format headers efficiently
        header_map = {
            "total_return": "Tot Ret%",
            "annualized_return": "Ann Ret%",
            "sharpe_ratio": "Sharpe",
            "max_drawdown": "Max DD%",
            "trades_count": "Trades",
            "win_rate": "Win%",
            "profit_factor": "PF",
            "expectancy": "Expect₹",
            "max_winning_streak": "MaxWin",
            "max_losing_streak": "MaxLoss",
            "composite_score": "Score",
        }

        formatted_headers = [
            header_map.get(col, col.replace("_", " ").title()) for col in display_cols
        ]

        print(
            tabulate(
                top_10[display_cols].round(2),
                headers=formatted_headers,
                tablefmt="grid",
                showindex=False,
            )
        )

    @staticmethod
    def get_statistics_table(
        df: pd.DataFrame, columns: List[str], column_names: List[str]
    ) -> List[List[str]]:
        """Generate statistics table for given columns"""
        stats_data = [["Metric", "Mean", "Median", "Std", "Min", "Max", "Best 25%"]]

        for col, name in zip(columns, column_names):
            quantile_val = (
                0.75
                if col not in ["max_drawdown", "max_losing_streak", "avg_loss"]
                else 0.25
            )
            stats_data.append(
                [
                    name,
                    (
                        f"{df[col].mean():.2f}"
                        if "ratio" not in col
                        else f"{df[col].mean():.3f}"
                    ),
                    (
                        f"{df[col].median():.2f}"
                        if "ratio" not in col
                        else f"{df[col].median():.3f}"
                    ),
                    (
                        f"{df[col].std():.2f}"
                        if "ratio" not in col
                        else f"{df[col].std():.3f}"
                    ),
                    (
                        f"{df[col].min():.2f}"
                        if "ratio" not in col
                        else f"{df[col].min():.3f}"
                    ),
                    (
                        f"{df[col].max():.2f}"
                        if "ratio" not in col
                        else f"{df[col].max():.3f}"
                    ),
                    (
                        f"{df[col].quantile(quantile_val):.2f}"
                        if "ratio" not in col
                        else f"{df[col].quantile(quantile_val):.3f}"
                    ),
                ]
            )

        return stats_data

    @classmethod
    def display_comprehensive_statistics(cls, df: pd.DataFrame):
        """Display comprehensive performance statistics"""
        print("\n📊 COMPREHENSIVE PERFORMANCE STATISTICS")
        print("=" * 80)

        # Core Performance Stats
        print("\n💎 CORE PERFORMANCE METRICS")
        core_cols = [
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
        ]
        core_names = [
            "Total Return (%)",
            "Annualized Return (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
        ]
        print(
            tabulate(
                cls.get_statistics_table(df, core_cols, core_names),
                headers="firstrow",
                tablefmt="grid",
            )
        )

        # Trading Activity Stats
        print("\n🎯 TRADING ACTIVITY STATISTICS")
        trading_cols = ["trades_count", "win_rate", "profit_factor", "expectancy"]
        trading_names = [
            "Total Trades",
            "Win Rate (%)",
            "Profit Factor",
            "Expectancy (₹)",
        ]
        trading_stats = cls.get_statistics_table(df, trading_cols, trading_names)
        # Fix expectancy formatting
        for i, row in enumerate(trading_stats[1:], 1):
            if "Expectancy" in row[0]:
                for j in range(1, 7):
                    row[j] = f"{float(row[j]):.0f}"
        print(tabulate(trading_stats, headers="firstrow", tablefmt="grid"))

        # Streak Analysis Stats
        print("\n🔥 STREAK ANALYSIS STATISTICS")
        streak_cols = ["max_winning_streak", "max_losing_streak", "avg_win", "avg_loss"]
        streak_names = [
            "Max Winning Streak",
            "Max Losing Streak",
            "Average Win (₹)",
            "Average Loss (₹)",
        ]
        streak_stats = cls.get_statistics_table(df, streak_cols, streak_names)
        # Fix currency formatting for wins/losses
        for i, row in enumerate(streak_stats[1:], 1):
            if "₹" in row[0]:
                for j in range(1, 7):
                    row[j] = f"{float(row[j]):.0f}"
        print(tabulate(streak_stats, headers="firstrow", tablefmt="grid"))

    @staticmethod
    def display_key_insights(df: pd.DataFrame):
        """Display key insights and performance summary"""
        print("\n🎯 KEY INSIGHTS")
        print("=" * 80)

        strategies_above_market = len(df[df["annualized_return"] > 12])
        strategies_positive_sharpe = len(df[df["sharpe_ratio"] > 1.0])
        strategies_low_drawdown = len(df[df["max_drawdown"] < 20])

        total_strategies = len(df)
        print(
            f"• {strategies_above_market}/{total_strategies} strategies beat 12% annual benchmark ({strategies_above_market/total_strategies*100:.1f}%)"
        )
        print(
            f"• {strategies_positive_sharpe}/{total_strategies} strategies achieved Sharpe > 1.0 ({strategies_positive_sharpe/total_strategies*100:.1f}%)"
        )
        print(
            f"• {strategies_low_drawdown}/{total_strategies} strategies kept drawdown < 20% ({strategies_low_drawdown/total_strategies*100:.1f}%)"
        )
        print(f"• Best Sharpe Ratio: {df['sharpe_ratio'].max():.3f}")
        print(f"• Best Annual Return: {df['annualized_return'].max():.2f}%")
        print(f"• Lowest Drawdown: {df['max_drawdown'].min():.2f}%")


class SingleStrategyAnalyzer:
    """
    Specialized analyzer for single strategy performance with extended context
    """

    @classmethod
    def display_comprehensive_analysis(
        cls,
        result: ExperimentResult,
        symbols: List[str],
        start_date,
        end_date,
        params: Dict[str, Any],
        strategy_name: str,
    ):
        """Display comprehensive performance details for a single strategy run"""

        print(f"\n📊 {strategy_name.upper()} STRATEGY PERFORMANCE ANALYSIS")
        print("=" * 80)

        # Strategy Overview
        cls._display_strategy_overview(result, symbols, start_date, end_date)

        # Core Performance with Benchmarks
        cls._display_core_performance_with_benchmarks(result)

        # Detailed Financial Breakdown
        cls._display_detailed_financial_breakdown(result)

        # Trading Activity Analysis
        cls._display_trading_activity_analysis(result, len(symbols))

        # Streak Analysis & Psychology
        cls._display_streak_analysis_psychology(result)

        # Risk Assessment
        cls._display_comprehensive_risk_assessment(result, start_date, end_date)

        # Strategy Parameters
        cls._display_strategy_parameters(params)

        # Performance Benchmarking
        cls._display_performance_benchmarking(result)

        # Professional Assessment
        cls._display_professional_assessment(result)

    @staticmethod
    def _display_strategy_overview(
        result: ExperimentResult, symbols: List[str], start_date, end_date
    ):
        """Display strategy overview section"""
        print("\n🎯 STRATEGY OVERVIEW")
        print("=" * 80)
        backtest_days = (end_date - start_date).days
        print(
            f"📅 Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({backtest_days} days)"
        )
        print(f"📈 Universe: {len(symbols)} instruments")
        print(
            f"💰 Initial Capital: ₹{result.final_value / (1 + result.total_return / 100):,.2f}"
        )
        print(f"🏁 Final Portfolio Value: ₹{result.final_value:,.2f}")

    @staticmethod
    def _display_core_performance_with_benchmarks(result: ExperimentResult):
        """Display core performance metrics with benchmarks"""
        print("\n💎 CORE PERFORMANCE METRICS")
        perf_data = [
            ["Metric", "Value", "Benchmark"],
            ["Total Return", f"{result.total_return:.2f}%", "8-12% (Market)"],
            [
                "Annualized Return",
                f"{getattr(result, 'annualized_return', 0):.2f}%",
                "12% (Nifty)",
            ],
            ["Sharpe Ratio", f"{result.sharpe_ratio:.3f}", "> 1.0 (Good)"],
            ["Maximum Drawdown", f"{result.max_drawdown:.2f}%", "< 20% (Good)"],
            ["Composite Score", f"{result.composite_score:.3f}", "Higher Better"],
        ]
        print(tabulate(perf_data, headers="firstrow", tablefmt="grid"))

    @staticmethod
    def _display_detailed_financial_breakdown(result: ExperimentResult):
        """Display detailed financial performance breakdown"""
        print("\n💰 FINANCIAL PERFORMANCE BREAKDOWN")
        print(
            tabulate(
                PerformanceDisplayManager.get_performance_metrics_table(result),
                headers="firstrow",
                tablefmt="grid",
            )
        )

    @staticmethod
    def _display_trading_activity_analysis(
        result: ExperimentResult, symbols_count: int
    ):
        """Display trading activity analysis with quality assessments"""
        print("\n📈 TRADING ACTIVITY ANALYSIS")
        trade_data = [
            ["Metric", "Value", "Quality Assessment"],
            ["Total Trades", f"{result.trades_count}", "More = Better Sample Size"],
            ["Win Rate", f"{result.win_rate:.2f}%", "> 50% Good, > 60% Excellent"],
            [
                "Average Trade Length",
                f"{getattr(result, 'avg_trade_length', 0):.1f} days",
                "Strategy Dependent",
            ],
            ["Even Trades (Breakeven)", f"{result.even_trades}", "Should be Low"],
            [
                "Data Feeds Utilized",
                f"{result.num_data_feeds}",
                f"Out of {symbols_count} available",
            ],
        ]
        print(tabulate(trade_data, headers="firstrow", tablefmt="grid"))

    @staticmethod
    def _display_streak_analysis_psychology(result: ExperimentResult):
        """Display streak analysis with psychological insights"""
        print("\n🔥 STREAK ANALYSIS & TRADING PSYCHOLOGY")
        streak_data = [
            ["Metric", "Value", "Psychological Impact"],
            [
                "Maximum Winning Streak",
                f"{result.max_winning_streak} trades",
                "Confidence Builder",
            ],
            [
                "Maximum Losing Streak",
                f"{result.max_losing_streak} trades",
                "< 5 is Good for Psychology",
            ],
            [
                "Current Win Streak",
                f"{result.consecutive_wins} trades",
                "Current Momentum",
            ],
            [
                "Current Loss Streak",
                f"{result.consecutive_losses} trades",
                "Risk Alert if > 3",
            ],
        ]
        print(tabulate(streak_data, headers="firstrow", tablefmt="grid"))

    @staticmethod
    def _display_comprehensive_risk_assessment(
        result: ExperimentResult, start_date, end_date
    ):
        """Display comprehensive risk assessment"""
        print("\n⚖️ COMPREHENSIVE RISK ASSESSMENT")

        # Calculate additional risk metrics
        win_loss_ratio = (
            result.avg_win / abs(result.avg_loss) if result.avg_loss != 0 else 0
        )
        backtest_days = (end_date - start_date).days
        expectancy_monthly = (
            getattr(result, "expectancy", 0)
            * (result.trades_count / (backtest_days / 30.44))
            if backtest_days > 0
            else 0
        )

        risk_data = [
            ["Risk Metric", "Value", "Risk Level"],
            [
                "Maximum Drawdown",
                f"{result.max_drawdown:.2f}%",
                (
                    "Low"
                    if result.max_drawdown < 10
                    else "Medium" if result.max_drawdown < 20 else "High"
                ),
            ],
            [
                "Sharpe Ratio",
                f"{result.sharpe_ratio:.3f}",
                (
                    "Excellent"
                    if result.sharpe_ratio > 2
                    else "Good" if result.sharpe_ratio > 1 else "Poor"
                ),
            ],
            [
                "Win/Loss Ratio",
                f"{win_loss_ratio:.2f}",
                (
                    "Good"
                    if win_loss_ratio > 1.5
                    else "Average" if win_loss_ratio > 1 else "Poor"
                ),
            ],
            [
                "Profit Factor",
                f"{result.profit_factor:.2f}",
                (
                    "Excellent"
                    if result.profit_factor > 2
                    else "Good" if result.profit_factor > 1.5 else "Poor"
                ),
            ],
            [
                "Monthly Expectancy",
                f"₹{expectancy_monthly:,.0f}",
                "Expected monthly profit",
            ],
        ]
        print(tabulate(risk_data, headers="firstrow", tablefmt="grid"))

    @staticmethod
    def _display_strategy_parameters(params: Dict[str, Any]):
        """Display strategy parameters"""
        print("\n🎯 STRATEGY PARAMETERS USED")
        print(
            tabulate(
                PerformanceDisplayManager.get_parameters_table(params),
                headers="firstrow",
                tablefmt="grid",
            )
        )

    @staticmethod
    def _display_performance_benchmarking(result: ExperimentResult):
        """Display performance benchmarking"""
        print("\n🏆 PERFORMANCE BENCHMARKING")
        nifty_annual = 12.0  # Assume Nifty 50 annual return
        market_sharpe = 0.8  # Typical market Sharpe

        benchmark_data = [
            ["Comparison", "Strategy", "Benchmark", "Outperformance"],
            [
                "Annual Return",
                f"{getattr(result, 'annualized_return', 0):.2f}%",
                f"{nifty_annual:.1f}%",
                f"{getattr(result, 'annualized_return', 0) - nifty_annual:+.2f}%",
            ],
            [
                "Sharpe Ratio",
                f"{result.sharpe_ratio:.3f}",
                f"{market_sharpe:.1f}",
                f"{result.sharpe_ratio - market_sharpe:+.3f}",
            ],
            [
                "Risk (Max DD)",
                f"{result.max_drawdown:.2f}%",
                "15-25%",
                "Lower is Better",
            ],
        ]
        print(tabulate(benchmark_data, headers="firstrow", tablefmt="grid"))

    @classmethod
    def _display_professional_assessment(cls, result: ExperimentResult):
        """Display professional assessment and recommendations"""
        print("\n🎯 KEY INSIGHTS & RECOMMENDATIONS")
        print("=" * 80)

        insights = cls._generate_performance_insights(result)
        for insight in insights:
            print(f"• {insight}")

        # Professional recommendation
        print("\n🎖️ PROFESSIONAL ASSESSMENT")
        print("=" * 80)

        score = PerformanceDisplayManager.calculate_professional_score(result)
        recommendation = PerformanceDisplayManager.get_professional_recommendation(
            score
        )
        print(f"{recommendation} (Score: {score}/9)")

        print("\n📊 Dashboard saved with comprehensive visual analytics!")

    @staticmethod
    def _generate_performance_insights(result: ExperimentResult) -> List[str]:
        """Generate automated performance insights"""
        insights = []

        # Performance insights
        if getattr(result, "annualized_return", 0) > 15:
            insights.append(
                "✅ EXCELLENT: Annualized return exceeds market expectations"
            )
        elif getattr(result, "annualized_return", 0) > 12:
            insights.append("✅ GOOD: Annualized return beats market benchmark")
        else:
            insights.append("⚠️  CONCERN: Annualized return below market benchmark")

        # Sharpe insights
        if result.sharpe_ratio > 2.0:
            insights.append(
                "✅ EXCELLENT: World-class risk-adjusted returns (Sharpe > 2.0)"
            )
        elif result.sharpe_ratio > 1.0:
            insights.append("✅ GOOD: Strong risk-adjusted returns (Sharpe > 1.0)")
        else:
            insights.append("⚠️  CONCERN: Poor risk-adjusted returns (Sharpe < 1.0)")

        # Drawdown insights
        if result.max_drawdown < 10:
            insights.append("✅ EXCELLENT: Very low maximum drawdown (<10%)")
        elif result.max_drawdown < 20:
            insights.append("✅ GOOD: Reasonable maximum drawdown (<20%)")
        else:
            insights.append("⚠️  CONCERN: High maximum drawdown (>20%)")

        # Trading insights
        if result.win_rate > 60:
            insights.append(
                "✅ EXCELLENT: High win rate provides psychological comfort"
            )
        elif result.win_rate > 50:
            insights.append("✅ GOOD: Positive win rate")
        else:
            insights.append("⚠️  REVIEW: Low win rate - ensure avg win > avg loss")

        # Streak insights
        if result.max_losing_streak <= 3:
            insights.append(
                "✅ EXCELLENT: Low losing streaks reduce psychological stress"
            )
        elif result.max_losing_streak <= 5:
            insights.append("✅ GOOD: Manageable losing streaks")
        else:
            insights.append(
                "⚠️  CAUTION: High losing streaks may cause emotional trading"
            )

        return insights
