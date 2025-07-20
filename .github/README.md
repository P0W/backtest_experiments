# GitHub Actions Workflows

This repository includes automated workflows for daily trading strategy analysis.

## 🔄 Daily Data Refresh
[![Daily Data Refresh](https://github.com/P0W/backtest_experiments/actions/workflows/refresh_data.yml/badge.svg)](https://github.com/P0W/backtest_experiments/actions/workflows/refresh_data.yml)

- **Purpose**: Updates market data cache
- **Schedule**: Daily data refresh
- **Files Updated**: `data_parquet/*.parquet`

## 📊 Daily ETF Dashboard Update
[![Daily ETF Dashboard](https://github.com/P0W/backtest_experiments/actions/workflows/daily-dashboard-update.yml/badge.svg)](https://github.com/P0W/backtest_experiments/actions/workflows/daily-dashboard-update.yml)

- **Purpose**: Updates trading strategy dashboard
- **Schedule**: Daily at 12:00 PM IST (6:30 AM UTC)
- **Default Strategy**: ETF Momentum
- **Files Updated**: `experiment_results/etfmomentum/etfmomentum_dashboard.png`

### Manual Trigger Options
You can manually trigger the dashboard update with custom parameters:

- **Strategy**: etf, adaptivemomentum
- **Universe**: etf, nifty50, nifty100  
- **Start Date**: Any date (default: 2020-01-01)
- **Initial Cash**: Any amount (default: 100000)

### Features
- ✅ **Smart Change Detection**: Only commits if dashboard actually changed
- ✅ **Performance Metrics**: Extracts and displays key performance indicators
- ✅ **Error Handling**: Robust error handling with detailed logs
- ✅ **Artifact Upload**: Saves backtest logs for debugging
- ✅ **Detailed Summaries**: Rich GitHub Actions summary with performance data

## 📁 Output Files

### Daily Dashboard
- **Location**: `experiment_results/{strategy}momentum/{strategy}momentum_dashboard.png`
- **Format**: PNG image with comprehensive trading performance visualization
- **Update Frequency**: Daily (if data changes)
- **Filename**: Fixed (always overwrites same file for consistent access)

### Logs and Artifacts
- **Backtest Logs**: Available as workflow artifacts (7-day retention)
- **Performance Data**: Embedded in commit messages and workflow summaries

## 🚀 Getting Started

1. **View Latest Dashboard**: Check `experiment_results/etfmomentum/etfmomentum_dashboard.png`
2. **Monitor Workflows**: Click the badges above to see workflow status
3. **Manual Runs**: Go to Actions tab → Select workflow → "Run workflow"
4. **View Logs**: Check workflow runs for detailed execution logs

## 📈 Performance Tracking

The workflows automatically track and report:
- Total Return %
- Sharpe Ratio
- Maximum Drawdown %
- Final Portfolio Value
- Execution Status
- Change Detection
