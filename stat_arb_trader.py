import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
import yfinance as yf
import logging
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StatArbitrageTrader:
    def __init__(self, crypto_symbol='BTC-USD', stock_symbol='NVDA', 
                 initial_balance=10000, vol_window=200, reg_window=50, 
                 z_threshold=2.5, position_size=0.5, tx_fee=0.002, int_rate=0.0005):
        """
        Initialize the statistical arbitrage trader.

        Parameters:
        -----------
        crypto_symbol : str
            Cryptocurrency symbol (e.g., 'BTC-USD')
        stock_symbol : str
            Stock symbol (e.g., 'NVDA')
        initial_balance : float
            Initial balance for each asset in USD
        vol_window : int
            Window size for volatility calculation
        reg_window : int
            Window size for regression analysis
        z_threshold : float
            Z-score threshold for trade signals
        position_size : float
            Fraction of balance to allocate per trade (0-1)
        tx_fee : float
            Transaction fee as a fraction (0-1)
        int_rate : float
            Interest rate for borrowing as a fraction (0-1)
        """
        self.crypto_symbol = crypto_symbol
        self.stock_symbol = stock_symbol
        self.vol_window = vol_window
        self.reg_window = reg_window
        self.z_threshold = z_threshold
        self.position_size = position_size
        self.tx_fee = tx_fee
        self.int_rate = int_rate
        
        # Account state
        self.balance_crypto = initial_balance
        self.balance_stock = initial_balance
        self.position = 'neutral'
        self.vol_crypto = 0
        self.vol_stock = 0
        self.entry_crypto = 0
        self.entry_stock = 0
        
        # Performance tracking
        self.trade_history = []
        self.balance_history = {'date': [], 'crypto': [], 'stock': [], 'total': []}

    def fetch_data(self, start_date=None, end_date=None, use_saved=False, save_data=True):
        """
        Fetch historical data from Yahoo Finance or load from CSV files.

        Parameters:
        -----------
        start_date : str, optional
            Start date ('YYYY-MM-DD'), defaults to 2 years ago
        end_date : str, optional
            End date ('YYYY-MM-DD'), defaults to today
        use_saved : bool
            Use saved CSV files if True
        save_data : bool
            Save fetched data to CSV if True
        """
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=2*365)).strftime('%Y-%m-%d')

        try:
            if use_saved:
                try:
                    self.crypto_data = pd.read_csv(f"{self.crypto_symbol}.csv", parse_dates=['date'])
                    self.stock_data = pd.read_csv(f"{self.stock_symbol}.csv", parse_dates=['date'])
                    logger.info(f"Loaded saved data for {self.crypto_symbol} and {self.stock_symbol}")
                    return
                except FileNotFoundError:
                    logger.warning("Saved data not found, downloading from Yahoo Finance")

            logger.info(f"Fetching data from {start_date} to {end_date}")

            # Ensure symbols are strings
            if not isinstance(self.crypto_symbol, str):
                raise ValueError(f"crypto_symbol must be a string, got {type(self.crypto_symbol)}: {self.crypto_symbol}")
            if not isinstance(self.stock_symbol, str):
                raise ValueError(f"stock_symbol must be a string, got {type(self.stock_symbol)}: {self.stock_symbol}")

            logger.info(f"crypto_symbol: {self.crypto_symbol}, type: {type(self.crypto_symbol)}")
            logger.info(f"stock_symbol: {self.stock_symbol}, type: {type(self.stock_symbol)}")

            # Fetch data
            self.crypto_data = yf.download(self.crypto_symbol, start=start_date, end=end_date)
            self.stock_data = yf.download(self.stock_symbol, start=start_date, end=end_date)

            logger.info(f"Raw crypto_data columns: {self.crypto_data.columns.tolist()}")
            logger.info(f"Raw stock_data columns: {self.stock_data.columns.tolist()}")

            # Handle MultiIndex if present
            if self.crypto_data.columns.nlevels > 1:
                logger.warning(f"MultiIndex detected for {self.crypto_symbol}, flattening columns")
                self.crypto_data = self.crypto_data.xs(self.crypto_symbol, level=1, axis=1)
            if self.stock_data.columns.nlevels > 1:
                logger.warning(f"MultiIndex detected for {self.stock_symbol}, flattening columns")
                self.stock_data = self.stock_data.xs(self.stock_symbol, level=1, axis=1)

            # Verify required columns with fallback
            if 'Close' not in self.crypto_data.columns:
                raise ValueError(f"'Close' missing in crypto data for {self.crypto_symbol}. Columns: {self.crypto_data.columns.tolist()}")
            if 'Adj Close' not in self.stock_data.columns:
                logger.warning(f"'Adj Close' missing for {self.stock_symbol}, using 'Close' as fallback")
                if 'Close' not in self.stock_data.columns:
                    raise ValueError(f"Neither 'Adj Close' nor 'Close' found in stock data for {self.stock_symbol}. Columns: {self.stock_data.columns.tolist()}")
                stock_price_col = 'Close'
            else:
                stock_price_col = 'Adj Close'

            # Reset index and standardize columns
            self.crypto_data.reset_index(inplace=True)
            self.stock_data.reset_index(inplace=True)
            self.crypto_data = self.crypto_data.rename(columns={'Date': 'date', 'Close': 'price'})
            self.stock_data = self.stock_data.rename(columns={'Date': 'date', stock_price_col: 'price'})
            self.crypto_data = self.crypto_data[['date', 'price']]
            self.stock_data = self.stock_data[['date', 'price']]

            # Save data if requested
            if save_data:
                self.crypto_data.to_csv(f"{self.crypto_symbol}.csv", index=False)
                self.stock_data.to_csv(f"{self.stock_symbol}.csv", index=False)
                logger.info(f"Saved data to {self.crypto_symbol}.csv and {self.stock_symbol}.csv")

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def align_data(self):
        """
        Align crypto and stock data by date using pandas merge.

        Returns:
        --------
        pd.DataFrame
            DataFrame with 'date', 'btc_price', 'nvda_price' columns
        """
        self.crypto_data['date'] = pd.to_datetime(self.crypto_data['date'])
        self.stock_data['date'] = pd.to_datetime(self.stock_data['date'])
        
        aligned_data = pd.merge(
            self.crypto_data[['date', 'price']],
            self.stock_data[['date', 'price']],
            on='date',
            how='inner',
            suffixes=('_btc', '_nvda')
        )
        aligned_data.rename(columns={'price_btc': 'btc_price', 'price_nvda': 'nvda_price'}, inplace=True)
        logger.info(f"Aligned data: {len(aligned_data)} matching dates found")
        return aligned_data

    def calculate_volatility(self, aligned_data):
        """
        Compute rolling volatility for both assets.

        Parameters:
        -----------
        aligned_data : pd.DataFrame
            Aligned price data

        Returns:
        --------
        pd.DataFrame
            Data with added volatility columns
        """
        aligned_data['btc_return'] = aligned_data['btc_price'].pct_change()
        aligned_data['nvda_return'] = aligned_data['nvda_price'].pct_change()
        aligned_data['vol_btc'] = aligned_data['btc_return'].rolling(window=self.vol_window).std()
        aligned_data['vol_nvda'] = aligned_data['nvda_return'].rolling(window=self.vol_window).std()
        
        vol_data = aligned_data.dropna(subset=['vol_btc', 'vol_nvda']).reset_index(drop=True)
        logger.info(f"Volatility calculated: {len(vol_data)} data points")
        return vol_data

    def statistical_arbitrage(self, vol_data):
        """
        Execute the statistical arbitrage strategy.

        Parameters:
        -----------
        vol_data : pd.DataFrame
            Data with volatility columns
        """
        alpha = 0.01  # Significance level
        window = self.reg_window
        n = len(vol_data)

        if n < window:
            logger.error(f"Not enough data points. Need at least {window}, got {n}.")
            return

        for i in range(window, n):
            hold_x = vol_data['vol_btc'].iloc[i-window:i].values
            hold_y = vol_data['vol_nvda'].iloc[i-window:i].values
            current_date = vol_data['date'].iloc[i]
            btc_price = vol_data['btc_price'].iloc[i]
            nvda_price = vol_data['nvda_price'].iloc[i]


            # Quadratic regression
            X_h = np.array([[1, k, k**2] for k in hold_x])
            y = hold_y

            try:
                IXT_X = np.linalg.inv(X_h.T.dot(X_h))
                beta = IXT_X.dot(X_h.T.dot(y))
                residuals = y - X_h.dot(beta)
                rss = np.sum(residuals**2)
                factor = rss / (window - 3)
                stderr = np.sqrt(np.diag(factor * IXT_X))
                tstat = beta / stderr
                df = window - 3
                pvalues = 2 * (1 - t.cdf(np.abs(tstat), df))

                if all(p < alpha for p in pvalues):
                    spread = residuals
                    mu_spread = np.mean(spread)
                    sd_spread = np.std(spread) / np.sqrt(window)
                    Z = (spread[-1] - mu_spread) / sd_spread
                    self.execute_trade(Z, btc_price, nvda_price, current_date.strftime('%Y-%m-%d'))

            except np.linalg.LinAlgError:
                logger.warning(f"Linear algebra error at {current_date}. Skipping.")
                continue

            # Log balance periodically
            if i % 20 == 0 or i == n-1:
                self.balance_history['date'].append(current_date)
                self.balance_history['crypto'].append(self.balance_crypto)
                self.balance_history['stock'].append(self.balance_stock)
                self.balance_history['total'].append(self.balance_crypto + self.balance_stock)

    def execute_trade(self, z_score, btc_price, nvda_price, trade_date):
        """
        Execute trades based on Z-score.

        Parameters:
        -----------
        z_score : float
            Current Z-score
        btc_price : float
            Current BTC price
        nvda_price : float
            Current NVDA price
        trade_date : str
            Trade date
        """
        if self.position == 'longnvda' and z_score > self.z_threshold:
            self.position = 'neutral'
            self.balance_crypto += (self.entry_crypto * (1 - self.int_rate) * (1 - self.tx_fee) - btc_price * (1 + self.tx_fee)) * self.vol_crypto
            self.balance_stock += (nvda_price * (1 - self.tx_fee) - self.entry_stock * (1 + self.tx_fee)) * self.vol_stock
            pnl = ((nvda_price / self.entry_stock - 1) - (btc_price / self.entry_crypto - 1)) * 100
            logger.info(f"[{trade_date}] CLOSED longnvda - PnL: {pnl:.2f}% | BTC: ${self.balance_crypto:.2f} | NVDA: ${self.balance_stock:.2f}")
            self.trade_history.append({'date': trade_date, 'action': 'close_longnvda', 'z_score': z_score, 'pnl_pct': pnl})

        elif self.position == 'neutral' and z_score < -self.z_threshold:
            self.position = 'longnvda'
            self.vol_crypto = self.position_size * self.balance_crypto / btc_price
            self.vol_stock = int(self.position_size * self.balance_stock / nvda_price)
            self.entry_crypto = btc_price
            self.entry_stock = nvda_price
            logger.info(f"[{trade_date}] OPENED longnvda | Z: {z_score:.2f}")

        elif self.position == 'longbtc' and z_score < -self.z_threshold:
            self.position = 'neutral'
            self.balance_crypto += (btc_price * (1 - self.tx_fee) - self.entry_crypto * (1 + self.tx_fee)) * self.vol_crypto
            self.balance_stock += (self.entry_stock * (1 - self.int_rate) * (1 - self.tx_fee) - nvda_price * (1 + self.tx_fee)) * self.vol_stock
            pnl = ((btc_price / self.entry_crypto - 1) - (nvda_price / self.entry_stock - 1)) * 100
            logger.info(f"[{trade_date}] CLOSED longbtc - PnL: {pnl:.2f}% | BTC: ${self.balance_crypto:.2f} | NVDA: ${self.balance_stock:.2f}")
            self.trade_history.append({'date': trade_date, 'action': 'close_longbtc', 'z_score': z_score, 'pnl_pct': pnl})

        elif self.position == 'neutral' and z_score > self.z_threshold:
            self.position = 'longbtc'
            self.vol_crypto = self.position_size * self.balance_crypto / btc_price
            self.vol_stock = int(self.position_size * self.balance_stock / nvda_price)
            self.entry_crypto = btc_price
            self.entry_stock = nvda_price
            logger.info(f"[{trade_date}] OPENED longbtc | Z: {z_score:.2f}")

    def run_backtest(self, start_date=None, end_date=None):
        """
        Run a backtest of the strategy.

        Returns:
        --------
        dict
            Performance metrics and history
        """
        # Reset state
        self.balance_crypto = 10000
        self.balance_stock = 10000
        self.position = 'neutral'
        self.trade_history = []
        self.balance_history = {'date': [], 'crypto': [], 'stock': [], 'total': []}

        self.fetch_data(start_date, end_date)
        aligned_data = self.align_data()
        if len(aligned_data) < self.vol_window + 1:
            logger.error(f"Need at least {self.vol_window + 1} data points, got {len(aligned_data)}")
            return None

        vol_data = self.calculate_volatility(aligned_data)
        if len(vol_data) < self.reg_window:
            logger.error(f"Need at least {self.reg_window} volatility points, got {len(vol_data)}")
            return None

        self.statistical_arbitrage(vol_data)

        # Performance metrics
        total_trades = len([t for t in self.trade_history if t['action'].startswith('close')])
        winning_trades = len([t for t in self.trade_history if t['action'].startswith('close') and t['pnl_pct'] > 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        average_pnl = sum(t['pnl_pct'] for t in self.trade_history if t['action'].startswith('close')) / total_trades if total_trades > 0 else 0
        initial_balance = 20000
        final_balance = self.balance_crypto + self.balance_stock
        total_return = (final_balance / initial_balance - 1) * 100

        daily_returns = [(self.balance_history['total'][i] / self.balance_history['total'][i-1] - 1) 
                         for i in range(1, len(self.balance_history['total']))] if len(self.balance_history['total']) > 1 else [0]
        sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0

        results = {
            'total_return_pct': total_return,
            'win_rate': win_rate,
            'average_pnl': average_pnl,
            'sharpe_ratio': sharpe_ratio,
            'trade_history': self.trade_history,
            'balance_history': self.balance_history
        }
        logger.info(f"Backtest: {total_return:.2f}% return, {win_rate:.2f}% win rate, Sharpe: {sharpe_ratio:.2f}")
        return results

    def plot_results(self, results):
        """
        Plot backtest results.
        """
        if not results:
            logger.error("No results to plot")
            return

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.plot(results['balance_history']['date'], results['balance_history']['total'], 'b-', label='Total Balance')
        plt.title('Account Balance Over Time')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        trade_dates = [t['date'] for t in results['trade_history'] if t['action'].startswith('close')]
        trade_pnls = [t['pnl_pct'] for t in self.trade_history if t['action'].startswith('close')]
        if trade_dates:
            plt.bar(range(len(trade_dates)), trade_pnls, color=['g' if p > 0 else 'r' for p in trade_pnls])
            plt.title('Trade P&L')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.show()

    def run_live_trading(self, days_lookback=730):
        """
        Simulate live trading with the latest data.
        """
        logger.info("Starting live trading mode")
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=days_lookback)).strftime('%Y-%m-%d')

        self.fetch_data(start_date, end_date)
        aligned_data = self.align_data()
        vol_data = self.calculate_volatility(aligned_data)

        if len(vol_data) < self.reg_window:
            logger.error(f"Need at least {self.reg_window} points, got {len(vol_data)}")
            return

        hold_x = vol_data['vol_btc'].iloc[-self.reg_window:].values
        hold_y = vol_data['vol_nvda'].iloc[-self.reg_window:].values
        X_h = np.array([[1, k, k**2] for k in hold_x])
        y = hold_y

        try:
            beta = np.linalg.inv(X_h.T.dot(X_h)).dot(X_h.T.dot(y))
            residuals = y - X_h.dot(beta)
            mu_spread = np.mean(residuals)
            sd_spread = np.std(residuals) / np.sqrt(self.reg_window)
            Z = (residuals[-1] - mu_spread) / sd_spread

            current_date = vol_data['date'].iloc[-1].strftime('%Y-%m-%d')
            btc_price = vol_data['btc_price'].iloc[-1]
            nvda_price = vol_data['nvda_price'].iloc[-1]
            logger.info(f"[{current_date}] Z-score: {Z:.2f}, BTC: ${btc_price:.2f}, NVDA: ${nvda_price:.2f}, Position: {self.position}")

            if self.position == 'neutral':
                if Z > self.z_threshold:
                    logger.info("SIGNAL: Open longbtc")
                elif Z < -self.z_threshold:
                    logger.info("SIGNAL: Open longnvda")
            elif self.position == 'longbtc' and Z < -self.z_threshold:
                logger.info("SIGNAL: Close longbtc")
            elif self.position == 'longnvda' and Z > self.z_threshold:
                logger.info("SIGNAL: Close longnvda")

        except np.linalg.LinAlgError:
            logger.error("Linear algebra error in live trading")

if __name__ == "__main__":
    trader = StatArbitrageTrader()
    results = trader.run_backtest(start_date='2023-01-01')
    if results:
        trader.plot_results(results)
        trader.run_live_trading()