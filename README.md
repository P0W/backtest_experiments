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

## 📊 Available Strategies

### 1. Adaptive Momentum Strategy (`momentum`)
- **Description**: Multi-stock momentum-based trading strategy with trend following
- **Requirements**: 5-30 stocks recommended
- **Parameters**: SMA periods, ATR multipliers, momentum periods, rebalance frequency
- **Use Case**: Trending markets, directional moves

### 2. Pairs Trading Strategy (`pairs`)
- **Description**: Statistical arbitrage strategy trading correlated asset pairs
- **Requirements**: Exactly 2 correlated assets (e.g., Nifty vs Bank Nifty)
- **Parameters**: Beta windows, correlation thresholds, deviation factors
- **Use Case**: Market-neutral strategies, relative value trading

### 3. Portfolio Mean Reversion Strategy (`mean_reversion`)
- **Description**: Portfolio strategy that trades mean reversion opportunities using z-scores
- **Requirements**: 5-20 liquid stocks recommended
- **Parameters**: Lookback periods, z-score thresholds, position limits
- **Use Case**: Range-bound markets, oversold/overbought conditions

### 4. Statistical Trend Following Strategy (`stat_trend`)
- **Description**: Multi-indicator trend following using MACD, RSI, and Bollinger Bands
- **Requirements**: 5-25 stocks across different sectors
- **Parameters**: MACD settings, RSI thresholds, Bollinger Band parameters
- **Use Case**: Strong trending markets, multi-timeframe analysis

## 🏗️ Architecture

### Core Components

```
├── run_experiments.py          # Main entry point
├── experiment_framework.py     # Unified experiment framework
├── strategies/                 # Strategy configurations
│   ├── base.py                # Base classes and interfaces
│   ├── adaptive_momentum.py   # Momentum strategy config
│   └── pairs_trading.py       # Pairs trading config
├── adaptive_momentum.py        # Momentum strategy implementation
├── main.py                    # Pairs trading strategy implementation  
├── p_mv.py                    # Mean reversion strategy implementation
├── stat_trend.py              # Statistical trend strategy implementation
└── utils.py                   # Common utilities and data loading
```

### Strategy Interface

All strategies implement the `StrategyConfig` interface:

```python
class StrategyConfig(ABC):
    def get_parameter_grid(self) -> Dict[str, List[Any]]
    def get_default_params(self) -> Dict[str, Any]
    def validate_params(self, params: Dict[str, Any]) -> bool
    def get_strategy_class(self) -> type
    def get_required_data_feeds(self) -> int
    def calculate_composite_score(self, metrics: Dict[str, float]) -> float
```

## 📈 Adding New Strategies

To add a new strategy:

1. **Create the strategy class** (inheriting from `bt.Strategy`)
2. **Create a configuration class** (inheriting from `StrategyConfig`)
3. **Add to the registry** in `run_experiments.py`

Example:

```python
# my_strategy.py
class MyStrategy(bt.Strategy):
    params = (('param1', 10), ('param2', 0.5))
    # ... strategy implementation

class MyStrategyConfig(StrategyConfig):
    def get_parameter_grid(self):
        return {'param1': [5, 10, 15], 'param2': [0.3, 0.5, 0.7]}
    # ... other required methods

# Add to AVAILABLE_STRATEGIES in run_experiments.py
'my_strategy': {
    'name': 'My Custom Strategy',
    'config_class': MyStrategyConfig,
    'description': 'Description of my strategy',
    'data_requirements': 'Data requirements'
}
```

## 🧪 Experiment Features

### Parameter Optimization
- **Grid Search**: Systematic testing of parameter combinations
- **Validation**: Automatic parameter validation to avoid invalid combinations
- **Parallel Processing**: Multi-core execution for faster experiments
- **Smart Sampling**: Random sampling when parameter space is too large

### Performance Metrics
- **Total Return**: Absolute performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Risk management
- **Composite Score**: Weighted combination of metrics

### Results & Visualization
- **Summary Tables**: Top performers, statistics
- **Visualizations**: Performance distributions, parameter correlations
- **Export**: JSON and CSV format results
- **Optimal Parameters**: Best parameter combinations identified

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

## 🛠️ Dependencies

This project uses `uv` as the package manager. Key dependencies:

- `backtrader`: Trading strategy framework
- `yfinance`: Market data
- `pandas`, `numpy`: Data processing
- `matplotlib`, `seaborn`: Visualization
- `scipy`, `statsmodels`: Statistical analysis
- `tqdm`: Progress bars
- `tabulate`: Table formatting

## 💡 Tips for Best Results

1. **Start Small**: Begin with quick experiments to understand the strategy
2. **Quality Data**: Ensure sufficient data history for meaningful results
3. **Parameter Ranges**: Use realistic parameter ranges based on market characteristics
4. **Validation**: Always validate results on out-of-sample data
5. **Risk Management**: Consider transaction costs and slippage in real trading
6. **Multiple Timeframes**: Test strategies across different market conditions

## 🔄 Migration from Old System

The old `experiments.py` is preserved but the new system offers:
- ✅ **Multi-Strategy Support**: Work with any strategy, not just momentum
- ✅ **Better Organization**: Clear separation of concerns
- ✅ **Extensibility**: Easy to add new strategies
- ✅ **Consistent Interface**: Same workflow for all strategies
- ✅ **Better Results**: Strategy-specific optimizations and metrics

To migrate existing strategies, wrap them with the new `StrategyConfig` interface.