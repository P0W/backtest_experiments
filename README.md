[![Daily Data Refresh](https://github.com/P0W/backtest_experiments/actions/workflows/refresh_data.yml/badge.svg)](https://github.com/P0W/backtest_experiments/actions/workflows/refresh_data.yml)

# Trading Strategy Experiment Framework

A unified framework for testing and optimizing different trading strategies using systematic parameter experiments.

## 🚀 Quick Start

To run experiments for any strategy:

```bash
uv run run_experiments.py
```

This will launch an interactive menu where you can:
1. Choose from available strategies
2. Select appropriate assets/stocks
3. Configure experiment parameters
4. Run optimization experiments
5. View results and visualizations

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

##  Tips for Best Results

1. **Start Small**: Begin with quick experiments to understand the strategy
2. **Quality Data**: Ensure sufficient data history for meaningful results
3. **Parameter Ranges**: Use realistic parameter ranges based on market characteristics
4. **Validation**: Always validate results on out-of-sample data
5. **Risk Management**: Consider transaction costs and slippage in real trading
6. **Multiple Timeframes**: Test strategies across different market conditions

