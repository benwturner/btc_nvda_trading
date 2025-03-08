# Statistical Arbitrage Trader: BTC-NVDA Pair Trading

## Overview

This project is for educational purposes only and is not intended for real trading or investment.

This script uses historical price data from Yahoo Finance to calculate volatility, perform statistical analysis, and execute trades based on a Z-score threshold. It supports:
- **Backtesting**: Evaluate the strategy's performance over a specified period.
- **Live Trading Simulation**: Monitor real-time signals using recent data.
- **Visualization**: Plot account balance and trade P&L.

The strategy assumes a mean-reverting relationship between BTC and NVDA volatility, taking long/short positions when the spread deviates significantly from its mean.

## Prerequisites

- **Python 3.8+**
- **Required Libraries**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `yfinance`
  - `logging`

  ## Usage

The script provides two main modes: **backtesting** and **live trading simulation**. Both are executed from the `StatArbitrageTrader` class.

### Running a Backtest

1. **Edit Parameters (Optional)**:
   Open `stat_arb_trader.py` and modify the `StatArbitrageTrader` initialization parameters if desired:
   ```python
   trader = StatArbitrageTrader(
       crypto_symbol='BTC-USD',  # Cryptocurrency symbol
       stock_symbol='NVDA',      # Stock symbol
       initial_balance=10000,    # Initial USD balance per asset
       vol_window=200,           # Volatility calculation window
       reg_window=50,            # Regression window
       z_threshold=2.5,          # Z-score threshold for trades
       position_size=0.5,        # Fraction of balance per trade
       tx_fee=0.002,             # Transaction fee (0.2%)
       int_rate=0.0005           # Interest rate (0.05%)
   )

