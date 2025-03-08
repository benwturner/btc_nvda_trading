# Statistical Arbitrage Trader: BTC-NVDA Pair Trading

## Overview

This repository contains a Python-based statistical arbitrage trading strategy designed to exploit mean-reverting relationships between Bitcoin (BTC-USD) and NVIDIA (NVDA) stock prices. Leveraging historical price data from Yahoo Finance, the strategy employs quadratic regression and Z-score analysis to identify mispricings in the volatility spread between these assets, executing pair trades to capitalize on their convergence.

The code supports both **backtesting** over historical data and **live trading simulation** with real-time signals, making it a versatile tool for algorithmic trading enthusiasts and researchers. Built with performance optimizations and robust error handling, it serves as a practical example of applying statistical methods to financial markets.

## Features

- **Pair Trading Strategy**: Executes trades on BTC and NVDA based on deviations in their volatility spread.
- **Quadratic Regression**: Models the relationship between BTC and NVDA volatilities over a configurable window.
- **Z-Score Signals**: Triggers trades when the spread deviates significantly (e.g., Z > 3.0 or Z < -3.0).
- **Backtesting**: Evaluates strategy performance with metrics such as total return, win rate, and Sharpe ratio.
- **Live Trading Simulation**: Generates real-time trade signals using recent data.
- **Visualization**: Plots account balance and trade P&L for performance analysis.
- **Data Handling**: Fetches and aligns data from Yahoo Finance via `yfinance`, with options to save/load CSV files.