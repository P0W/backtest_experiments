[![Daily Data Refresh](https://github.com/P0W/backtest_experiments/actions/workflows/refresh_data.yml/badge.svg)](https://github.com/P0W/backtest_experiments/actions/workflows/refresh_data.yml)
[![Daily ETF Dashboard](https://github.com/P0W/backtest_experiments/actions/workflows/daily-dashboard-update.yml/badge.svg)](https://github.com/P0W/backtest_experiments/actions/workflows/daily-dashboard-update.yml)

# Trading Strategy Experiment Framework

A unified framework for testing and optimizing different trading strategies using systematic parameter experiments.

## �️ Dependencies

This project uses `uv` as the package manager. Key dependencies:

- `backtrader`: Trading strategy framework
- `yfinance`: Market data
- `pandas`, `numpy`: Data processing
- `matplotlib`, `seaborn`: Visualization
- `scipy`, `statsmodels`: Statistical analysis
- `tqdm`: Progress bars
- `tabulate`: Table formatting

## 🧪 Experimental Features

### Parameter Optimization
- **Grid Search**: Systematic testing of parameter combinations
- **Validation**: Automatic parameter validation to avoid invalid combinations
- **Parallel Processing**: Multi-core execution for faster experiments
- **Smart Sampling**: Random sampling when parameter space is too large

### Results & Visualization
- **Summary Tables**: Top performers, statistics
- **Visualizations**: Performance distributions, parameter correlations
- **Export**: JSON and CSV format results
- **Optimal Parameters**: Best parameter combinations identified

## 📊 Latest ETF Dashboard

![ETF Momentum Dashboard](experiment_results/etfmomentum/etfmomentum_dashboard.png)

*Dashboard is automatically updated daily at 12:00 PM IST after market close*

## 🤖 Automated Dashboard Updates

The repository includes automated GitHub Actions workflows for daily updates:

### Daily ETF Dashboard
- **Schedule**: Runs daily at 12:00 PM IST (6:30 AM UTC) after market close
- **Action**: Updates `experiment_results/etfmomentum/etfmomentum_dashboard.png`
- **Strategy**: ETF Momentum strategy with 12 ETF universe
- **Commit**: Automatically commits updated dashboard with daily timestamp
- **Manual Trigger**: Can be triggered manually via GitHub Actions tab

### Key Features
- **Fixed Filename**: Always updates the same PNG file for consistent dashboard viewing
- **Smart Caching**: Only commits if dashboard actually changed
- **Market Timing**: Runs after Indian market close for latest data
- **Failure Handling**: Includes error handling and status reporting

## 📊 Performance Metrics

### Core Metrics
- **Total Return**: Absolute performance measurement
- **Sharpe Ratio**: Risk-adjusted returns (return per unit of risk)
- **Maximum Drawdown**: Peak-to-trough decline (risk management metric)
- **Volatility**: Standard deviation of returns
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

### Advanced Metrics
- **Sortino Ratio**: Downside deviation-adjusted returns
- **Calmar Ratio**: Annual return divided by maximum drawdown
- **Value at Risk (VaR)**: Potential loss at given confidence level
- **Beta**: Correlation with market movements
- **Alpha**: Excess return over market benchmark
- **Composite Score**: Weighted combination of all metrics


This interface ensures:
- **Consistency**: All strategies follow the same pattern
- **Validation**: Parameters are validated before execution
- **Optimization**: Systematic parameter space exploration
- **Scoring**: Custom performance evaluation logic
- **Flexibility**: Strategy-specific requirements and constraints

## 🔧 Configuration

### Experiment Settings
- **Time Period**: 1-3 years or custom date range
- **Initial Capital**: ₹5L to ₹20L or custom amount
- **Intensity**: Quick (20 combinations) to Comprehensive (100+ combinations)
- **Parallel Workers**: 2-8 workers depending on system

### Data Requirements
- Uses `yfinance` for market data
- Automatic caching for faster re-runs
- Support for Indian NSE stocks (.NS suffix)
- Support for indices (^NSEI, ^NSEBANK)

## 📁 Results Structure

Results are organized by strategy:

```
experiment_results/
├── adaptivemomentum/
│   ├── adaptivemomentum_results_20250715_123456.json
│   ├── adaptivemomentum_results_20250715_123456.csv
│   ├── adaptivemomentum_analysis_20250715_123456.png
│   └── optimal_params_20250715_123456.json
├── pairs/
│   └── ...
└── portfoliomeanreversion/
    └── ...
```



