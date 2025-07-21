"""
Dashboard Components Module

This module contains individual chart creation methods for the portfolio dashboard,
following the Single Responsibility Principle for better maintainability.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DashboardComponents:
    """
    Contains individual methods for creating different dashboard components.
    Each method is responsible for a single chart or visualization.
    """

    @staticmethod
    def create_portfolio_value_chart(ax, portfolio_df):
        """Create portfolio value over time chart"""
        ax.plot(
            portfolio_df.index,
            portfolio_df["value"],
            linewidth=2,
            color="#2E86AB",
            label="Portfolio Value",
        )
        ax.fill_between(
            portfolio_df.index, portfolio_df["value"], alpha=0.3, color="#2E86AB"
        )
        ax.set_title("Portfolio Value Over Time", fontsize=14, fontweight="bold")
        ax.set_ylabel("Portfolio Value (₹)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        # Format y-axis
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda x, p: (
                    f"₹{x/100000:.1f}L" if x < 10000000 else f"₹{x/10000000:.1f}Cr"
                )
            )
        )

    @staticmethod
    def create_drawdown_chart(ax, portfolio_df):
        """Create drawdown analysis chart"""
        ax.fill_between(
            portfolio_df.index,
            portfolio_df["drawdown"],
            0,
            alpha=0.7,
            color="red",
            label="Drawdown",
        )
        ax.plot(
            portfolio_df.index, portfolio_df["drawdown"], linewidth=1, color="darkred"
        )
        ax.set_title("Drawdown Analysis", fontsize=14, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        # Add max drawdown line
        max_dd = portfolio_df["drawdown"].min()
        ax.axhline(
            y=max_dd,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"Max DD: {max_dd:.2f}%",
        )
        ax.legend()

    @staticmethod
    def create_monthly_returns_heatmap(ax, portfolio_df):
        """Create monthly returns heatmap"""
        if len(portfolio_df) > 30:  # Only create heatmap if we have enough data
            monthly_returns = (
                portfolio_df["returns"]
                .resample("M")
                .apply(lambda x: (1 + x).prod() - 1)
                * 100
            )
            monthly_returns_df = monthly_returns.to_frame()
            monthly_returns_df["year"] = monthly_returns_df.index.year
            monthly_returns_df["month"] = monthly_returns_df.index.month

            # Pivot for heatmap
            heatmap_data = monthly_returns_df.pivot(
                index="year", columns="month", values="returns"
            )

            # Create heatmap
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".1f",
                cmap="RdYlGn",
                center=0,
                ax=ax,
                cbar_kws={"label": "Monthly Return (%)"},
            )
            ax.set_title("Monthly Returns Heatmap", fontsize=14, fontweight="bold")
            ax.set_xlabel("Month")
            ax.set_ylabel("Year")
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data\nfor monthly heatmap\n(need >30 days)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Monthly Returns Heatmap", fontsize=14, fontweight="bold")

    @staticmethod
    def create_returns_distribution(ax, portfolio_df):
        """Create daily returns distribution chart"""
        returns_clean = portfolio_df["returns"].dropna()
        if len(returns_clean) > 1:
            ax.hist(
                returns_clean * 100,
                bins=50,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
                density=True,
            )
            ax.axvline(
                returns_clean.mean() * 100,
                color="red",
                linestyle="--",
                label=f"Mean: {returns_clean.mean()*100:.3f}%",
            )
            ax.axvline(
                returns_clean.median() * 100,
                color="green",
                linestyle="--",
                label=f"Median: {returns_clean.median()*100:.3f}%",
            )
            ax.set_title("Daily Returns Distribution", fontsize=14, fontweight="bold")
            ax.set_xlabel("Daily Return (%)")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data\nfor distribution",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Daily Returns Distribution", fontsize=14, fontweight="bold")

    @staticmethod
    def create_rolling_sharpe_chart(ax, portfolio_df):
        """Create rolling Sharpe ratio chart"""
        returns_clean = portfolio_df["returns"].dropna()
        if len(returns_clean) > 30:
            rolling_sharpe = (
                returns_clean.rolling(window=30).mean()
                / returns_clean.rolling(window=30).std()
                * np.sqrt(252)
            )
            ax.plot(
                rolling_sharpe.index,
                rolling_sharpe,
                linewidth=2,
                color="purple",
                label="30-Day Rolling Sharpe",
            )
            ax.axhline(
                y=1, color="green", linestyle="--", alpha=0.7, label="Sharpe = 1.0"
            )
            ax.axhline(
                y=2, color="orange", linestyle="--", alpha=0.7, label="Sharpe = 2.0"
            )
            ax.set_title(
                "Rolling Sharpe Ratio (30-Day)", fontsize=14, fontweight="bold"
            )
            ax.set_ylabel("Sharpe Ratio")
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data\nfor rolling Sharpe\n(need >30 days)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(
                "Rolling Sharpe Ratio (30-Day)", fontsize=14, fontweight="bold"
            )

    @staticmethod
    def create_cumulative_returns_chart(ax, portfolio_df, strategy_name):
        """Create cumulative returns vs benchmark chart"""
        ax.plot(
            portfolio_df.index,
            portfolio_df["cumulative_returns"] * 100,
            linewidth=2,
            color="blue",
            label=f"{strategy_name} Strategy",
        )

        # Add simple benchmark lines
        days_total = len(portfolio_df)
        years_total = days_total / 365.25
        benchmark_8 = [(8 * i / days_total * years_total) for i in range(days_total)]
        benchmark_12 = [(12 * i / days_total * years_total) for i in range(days_total)]

        ax.plot(
            portfolio_df.index,
            benchmark_8,
            linestyle="--",
            color="green",
            alpha=0.7,
            label="8% p.a. Benchmark",
        )
        ax.plot(
            portfolio_df.index,
            benchmark_12,
            linestyle="--",
            color="orange",
            alpha=0.7,
            label="12% p.a. Benchmark",
        )

        ax.set_title("Cumulative Returns vs Benchmarks", fontsize=14, fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    @staticmethod
    def create_performance_metrics_panel(
        ax, result, start_date, end_date, portfolio_df
    ):
        """Create key performance metrics panel"""
        returns_clean = portfolio_df["returns"].dropna()

        # Calculate proper annualized return using CAGR formula
        days_in_period = (end_date - start_date).days
        years_in_period = days_in_period / 365.25
        annualized_return_cagr = (
            (((1 + result.total_return / 100) ** (1 / years_in_period)) - 1) * 100
            if years_in_period > 0
            else result.total_return
        )

        metrics_data = {
            "Total Return": f"{result.total_return:.2f}%",
            "Annualized Return": f"{annualized_return_cagr:.2f}%",
            "Sharpe Ratio": f"{result.sharpe_ratio:.3f}",
            "Max Drawdown": f"{result.max_drawdown:.2f}%",
            "Volatility": (
                f"{returns_clean.std() * np.sqrt(252) * 100:.2f}%"
                if len(returns_clean) > 1
                else "N/A"
            ),
            "Win Rate": f"{getattr(result, 'win_rate', 0):.1f}%",
            "Profit Factor": f"{getattr(result, 'profit_factor', 0):.2f}",
            "Total Trades": f"{getattr(result, 'trades_count', 0)}",
            "Max Win Streak": f"{getattr(result, 'max_winning_streak', 0)}",
            "Max Loss Streak": f"{getattr(result, 'max_losing_streak', 0)}",
            "Avg Win": f"₹{getattr(result, 'avg_win', 0):.2f}",
            "Avg Loss": f"₹{getattr(result, 'avg_loss', 0):.2f}",
        }

        y_pos = 0.95
        line_height = 0.07
        for metric, value in metrics_data.items():
            ax.text(
                0.05,
                y_pos,
                f"{metric}:",
                fontweight="bold",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.text(0.6, y_pos, value, transform=ax.transAxes, fontsize=10)
            y_pos -= line_height

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Key Performance Metrics", fontsize=14, fontweight="bold")

    @staticmethod
    def create_risk_return_scatter(
        ax, result, start_date, end_date, portfolio_df, strategy_name
    ):
        """Create risk-return scatter plot"""
        returns_clean = portfolio_df["returns"].dropna()
        volatility = (
            returns_clean.std() * np.sqrt(252) * 100 if len(returns_clean) > 1 else 0
        )

        # Calculate annualized return
        days_in_period = (end_date - start_date).days
        years_in_period = days_in_period / 365.25
        annualized_return = (
            (((1 + result.total_return / 100) ** (1 / years_in_period)) - 1) * 100
            if years_in_period > 0
            else result.total_return
        )

        # Plot strategy point
        ax.scatter(
            [volatility],
            [annualized_return],
            s=200,
            c="red",
            marker="*",
            label=f"{strategy_name}",
            zorder=5,
        )

        # Add benchmark points
        benchmarks = [
            ("Conservative", 5, 8),
            ("Moderate", 10, 12),
            ("Aggressive", 15, 16),
            ("High Risk", 25, 20),
        ]

        for name, vol, ret in benchmarks:
            ax.scatter([vol], [ret], s=100, alpha=0.6, label=name)

        ax.set_xlabel("Volatility (% p.a.)")
        ax.set_ylabel("Return (% p.a.)")
        ax.set_title("Risk-Return Profile", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    @staticmethod
    def create_portfolio_summary_panel(
        ax, strategy_name, start_date, end_date, symbols, result
    ):
        """Create portfolio summary panel"""
        summary_text = f"""
            Portfolio Summary:
            ━━━━━━━━━━━━━━━━━━━━
            Strategy: {strategy_name}
            Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
            Duration: {(end_date - start_date).days} days

            Symbols: {len(symbols)}
            {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}

            Initial Value: ₹{result.final_value / (1 + result.total_return/100):,.0f}
            Final Value: ₹{result.final_value:,.0f}
            Profit/Loss: ₹{result.final_value - (result.final_value / (1 + result.total_return/100)):,.0f}

            Performance Rating: {'⭐' * min(5, max(1, int(result.sharpe_ratio + 2)))}
        """

        ax.text(
            0.05,
            0.95,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Portfolio Summary", fontsize=14, fontweight="bold")

    @staticmethod
    def create_weekly_returns_pattern(ax, portfolio_df):
        """Create weekly returns pattern chart"""
        if len(portfolio_df) > 14:
            portfolio_df["weekday"] = portfolio_df.index.dayofweek
            weekly_pattern = portfolio_df.groupby("weekday")["returns"].mean() * 100
            weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            colors = ["green" if x > 0 else "red" for x in weekly_pattern]

            bars = ax.bar(
                range(len(weekly_pattern)), weekly_pattern, color=colors, alpha=0.7
            )
            ax.set_xticks(range(len(weekly_pattern)))
            ax.set_xticklabels([weekdays[i] for i in weekly_pattern.index])
            ax.set_title(
                "Average Returns by Day of Week", fontsize=14, fontweight="bold"
            )
            ax.set_ylabel("Average Return (%)")
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for bar, value in zip(bars, weekly_pattern):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (max(weekly_pattern) - min(weekly_pattern)) * 0.01,
                    f"{value:.3f}%",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=9,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data\nfor weekly pattern",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(
                "Average Returns by Day of Week", fontsize=14, fontweight="bold"
            )

    @staticmethod
    def create_rolling_volatility_chart(ax, portfolio_df):
        """Create rolling volatility chart"""
        returns_clean = portfolio_df["returns"].dropna()
        if len(returns_clean) > 30:
            rolling_vol = returns_clean.rolling(window=30).std() * np.sqrt(252) * 100
            ax.plot(
                rolling_vol.index,
                rolling_vol,
                linewidth=2,
                color="red",
                label="30-Day Rolling Volatility",
            )
            ax.axhline(
                y=rolling_vol.mean(),
                color="blue",
                linestyle="--",
                alpha=0.7,
                label=f"Mean: {rolling_vol.mean():.1f}%",
            )
            ax.set_title("Rolling Volatility (30-Day)", fontsize=14, fontweight="bold")
            ax.set_ylabel("Volatility (% p.a.)")
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data\nfor rolling volatility",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Rolling Volatility (30-Day)", fontsize=14, fontweight="bold")

    @staticmethod
    def create_underwater_plot(ax, portfolio_df):
        """Create underwater plot (days below peak)"""
        # Calculate underwater periods (days below previous peak)
        is_underwater = portfolio_df["drawdown"] < -0.01  # More than 0.01% drawdown
        portfolio_df["days_underwater"] = (
            is_underwater.groupby((~is_underwater).cumsum()).cumcount() + 1
        ) * is_underwater

        # Plot underwater duration
        ax.fill_between(
            portfolio_df.index,
            portfolio_df["days_underwater"],
            0,
            alpha=0.7,
            color="blue",
            label="Days Underwater",
        )
        ax.plot(
            portfolio_df.index,
            portfolio_df["days_underwater"],
            linewidth=1,
            color="darkblue",
        )
        ax.set_title(
            "Underwater Plot (Days Below Peak)", fontsize=14, fontweight="bold"
        )
        ax.set_ylabel("Days Underwater")
        ax.grid(True, alpha=0.3)

        # Add statistics
        max_underwater = portfolio_df["days_underwater"].max()
        avg_underwater = portfolio_df[portfolio_df["days_underwater"] > 0][
            "days_underwater"
        ].mean()

        if max_underwater > 0:
            ax.axhline(
                y=max_underwater,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Max: {max_underwater:.0f} days",
            )
            if not pd.isna(avg_underwater):
                ax.axhline(
                    y=avg_underwater,
                    color="orange",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Avg: {avg_underwater:.0f} days",
                )

        ax.legend()

    @staticmethod
    def create_parameters_panel(ax, result):
        """Create optimal parameters panel"""
        # Create a visually appealing parameters display
        params_text = "🏆 OPTIMAL STRATEGY PARAMETERS (Best Configuration) 🏆\n"
        params_text += "=" * 80 + "\n\n"

        # Format parameters in a table-like structure
        param_items = list(result.params.items())
        # Split parameters into two columns for better layout
        mid_point = len(param_items) // 2 + len(param_items) % 2
        left_params = param_items[:mid_point]
        right_params = param_items[mid_point:]

        # Create two-column layout
        for i in range(max(len(left_params), len(right_params))):
            line = ""

            # Left column
            if i < len(left_params):
                param, value = left_params[i]
                if isinstance(value, float):
                    line += f"{param.replace('_', ' ').title():<20}: {value:<10.3f}"
                else:
                    line += f"{param.replace('_', ' ').title():<20}: {str(value):<10}"
            else:
                line += " " * 32

            line += "    "  # Spacing between columns

            # Right column
            if i < len(right_params):
                param, value = right_params[i]
                if isinstance(value, float):
                    line += f"{param.replace('_', ' ').title():<20}: {value:<10.3f}"
                else:
                    line += f"{param.replace('_', ' ').title():<20}: {str(value):<10}"

            params_text += line + "\n"

        # Add performance metrics at the bottom
        params_text += "\n" + "─" * 80 + "\n"
        params_text += "📊 PERFORMANCE SUMMARY:\n"
        params_text += f"Total Return: {result.total_return:.2f}%  |  "
        params_text += f"Sharpe Ratio: {result.sharpe_ratio:.3f}  |  "
        params_text += f"Max Drawdown: {result.max_drawdown:.2f}%  |  "
        params_text += f"Composite Score: {result.composite_score:.4f}"

        # Display the text with special formatting
        ax.text(
            0.5,
            0.5,
            params_text,
            transform=ax.transAxes,
            fontsize=11,
            ha="center",
            va="center",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=1",
                facecolor="gold",
                alpha=0.3,
                edgecolor="darkgoldenrod",
                linewidth=3,
            ),
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Add a crown emoji as title decoration
        ax.text(
            0.5,
            0.95,
            "👑 WINNING CONFIGURATION 👑",
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
        )
