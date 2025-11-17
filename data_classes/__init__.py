import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from typing import Union, List, Dict
import sys, os

from utils.helpers import closest_value
from transforms import discrete_local_volatility, log_transform

RFR = 0.0387  # Risk-free rate is the FED Effective Fed Funds Rate (EFFR) as at 15/11/2025

class DataRetriever():

    def __init__(self, data_dir: str, symbols: Union[str, List[str]], start_date: dt.datetime, end_date: dt.datetime, interval: str = "1d"):
        self.data_dir = data_dir
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.period = "{}d".format((self.end_date - self.start_date).days)

        self.underlying_data = pd.DataFrame()
        self.options_chains = {}
    

    def get_price_data(self) -> pd.DataFrame:
        """
        Fetch OHLCV price data for one or more stock symbols using yfinance.

        Args:
            symbols: Single ticker string (e.g. "AAPL") or an iterable of ticker strings (e.g. ["AAPL", "MSFT"]).
            start_date: Start date as a datetime object.
            end_date: End date as a datetime object.
            interval: Data interval (e.g. "1m", "1h", "1d"). We use "1d" for this experiment

        Returns:
            pandas.DataFrame. For a single symbol, returns the history DataFrame with a DatetimeIndex.
            For multiple symbols, returns a concatenated DataFrame with a 'symbol' column and a regular RangeIndex.

        Raises:
            ValueError: if no data is returned for the given symbol(s).
        """
        
        symbols = self.symbols
        interval = self.interval

        # If a single symbol string is provided
        if isinstance(symbols, str):
            ticker = yf.Ticker(symbols)
            df = ticker.history(period=self.period, interval=interval, actions=False)
            if df is None or df.empty:
                raise ValueError(f"No data found for symbol: {symbols}")
            
            self.underlying_data = df
            df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
            df.index = df.index.date
            return df
        # If a list of tickers is provided
        elif isinstance(symbols, list):
            frames = []
            for sym in symbols:
                print(f"{dt.datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S')}:    Fetching data for symbol: {sym}", file=sys.stdout)
                if not isinstance(sym, str):
                    continue
                try:
                    t = yf.Ticker(sym)
                    df = t.history(period=self.period, interval=interval, actions=False)
                except Exception:
                    df = None
                if df is None or df.empty:
                    # skip empty results but note them by continuing; user can inspect
                    continue
                tmp = df.reset_index()
                tmp["symbol"] = sym
                frames.append(tmp)

            if not frames:
                raise ValueError(f"No data found for any symbols in the provided list: {symbols}")

            df = pd.concat(frames, ignore_index=True, sort=False)
            df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
            df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
            df.index = df.index.date()
            self.underlying_data = df
            print(f"{dt.datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S')}: {df.shape[0]} stock row(s) fetched for symbol(s) {self.symbols}")
            return df

    def get_options_chain(self, options_data: pd.DataFrame, save_data: bool = False) -> Dict:
        """
        Given a full_yearly_data file create the option_chain attribute
        """
        # Get available expiration dates
        try:
            expiry_list = options_data.expiry.unique().tolist()

            chains = {}
            for expiry in expiry_list:
                
                expiry_mask = options_data['expiry'] == expiry
                chains[expiry] = {
                    'calls': options_data.loc[expiry_mask & (options_data['optionType'] == 'calls')],
                    'puts': options_data.loc[expiry_mask & (options_data['optionType'] == 'puts')]
                }

            self.options_chain = chains
            
        except Exception as e:
            raise(e)
        #     if save_data:
        #         os.makedirs(self.data_dir, exist_ok=True)

        #         calls_list = []
        #         puts_list = []
        #         for exp, data in chains.items():
        #             calls_df = data.get('calls')
        #             puts_df = data.get('puts')

        #             if calls_df is not None and not calls_df.empty:
        #                 tmp_calls = calls_df.copy()
        #                 tmp_calls['expiration'] = exp
        #                 tmp_calls['symbol'] = symbol
        #                 tmp_calls['type'] = 'calls'
        #                 calls_list.append(tmp_calls)

        #             if puts_df is not None and not puts_df.empty:
        #                 tmp_puts = puts_df.copy()
        #                 tmp_puts['expiration'] = exp
        #                 tmp_puts['symbol'] = symbol
        #                 tmp_puts['type'] = 'puts'
        #                 puts_list.append(tmp_puts)

        #         if calls_list:
        #             all_calls = pd.concat(calls_list, ignore_index=True, sort=False)
        #             all_calls.to_csv(os.path.join(self.data_dir, f"{symbol}_options_calls_all.csv"), index=False)
        #             print(f"{dt.datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S')}:    {len(all_calls)} row(s) of call data saved for {symbol}")

        #         if puts_list:
        #             all_puts = pd.concat(puts_list, ignore_index=True, sort=False)
        #             all_puts.to_csv(os.path.join(self.data_dir, f"{symbol}_options_puts_all.csv"), index=False)
        #             print(f"{dt.datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S')}:    {len(all_puts)} row(s) of put data saved for {symbol}")

        #     return chains
        
        # except Exception as e:
        #     raise ValueError(f"{dt.datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S')}:    Could not fetch options expirations for symbol {symbol}: {e}")


DAYS_IN_YEAR = 365.0
MATURITY_TOL = 1e-6

class DataProcessor():

    def __init__(self, date, underlying_data: pd.DataFrame, options_chains: Dict):
        
        self.date = date
        self.underlying_data = underlying_data

        self._validate_options_chains(options_chains)
        self.options_chains = options_chains
        
        # Initialize grid parameters
        self.option_prices = None
        self.option_types = None
        self.strike_grid = None
        self.time_grid = None
        self.relative_strike_grid = None
        self.implied_vols = None
        self.volumes = None
        self.dlvs = None

    def clean_and_process_data(self, price_column='close', implied_vol_column='bs_implied_vol', volume_column='volume'):
        """
        Process option chains to create structured option price/implied vol data.
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free interest rate to use for calculations
        price_column : str
            Column name in the option chain data that contains prices
        implied_vol_column : str
            Column name in the option chain data that contains implied volatilities
        Returns:
        --------
        tuple
            (option_prices, implied_vols, strike_grid, maturity_grid)
        """
        # Get current price of underlying
        current_price = self.underlying_data.loc[self.date.date()]['Close']
        
        # Extract unique strikes and maturities
        all_strikes = set()
        all_maturities = set()
        
        for exp_date, chain in self.options_chains.items():
            # Convert expiration date to datetime
            exp_datetime = dt.datetime.strptime(exp_date, '%Y-%m-%d')
            
            # Calculate time to maturity in years
            days_to_expiry = (exp_datetime - dt.datetime.now()).days
            T = max(1, days_to_expiry) / 365.0  # Ensure at least 1 day
            
            all_maturities.add(T)
            
            # Extract strikes from calls
            for strike in chain['calls']['strike']:
                all_strikes.add(float(strike))
        
        # Create sorted grids
        K_grid = sorted(list(all_strikes))
        T_grid = sorted(list(all_maturities))
        
        # Initialize option price and option type matrices
        n_strikes = len(K_grid)
        n_maturities = len(T_grid)
        
        option_prices = np.zeros((n_strikes, n_maturities))
        implied_vols = np.zeros((n_strikes, n_maturities))
        volume_array = np.zeros((n_strikes, n_maturities))
        option_types = np.full((n_strikes, n_maturities), 'call', dtype=object)
        
        # Fill the price matrix
        for i, K in enumerate(K_grid):
            for j, T in enumerate(T_grid):
                # Find the corresponding expiration date
                for exp_date, chain in self.options_chains.items():
                    exp_datetime = dt.datetime.strptime(exp_date, '%Y-%m-%d')
                    days_to_expiry = (exp_datetime - self.date).days
                    maturity = max(1, days_to_expiry) / DAYS_IN_YEAR

                    if abs(maturity - T) < MATURITY_TOL:  # Close enough to be the same maturity
                        # Find the closest strike in calls
                        calls_df = chain['calls']
                        puts_df = chain['puts']
                        
                        # Determine whether to use call or put based on moneyness
                        # (typically use calls for K >= S and puts for K < S)
                        option_df = [calls_df if K >= current_price else puts_df][0]
                        option_types[i, j] = ['call' if K >= current_price else 'put'][0]
                        
                        # If there is no calls when K > S or puts when K < S 
                        # usually when deep ITM 
                        if option_df.empty:
                            price = np.nan
                            implied_vol = np.nan
                            volume = 0
                        else:
                            strike_idx = (option_df['strike'] - K).abs().idxmin()
                            price = option_df.loc[strike_idx, price_column]
                            implied_vol = option_df.loc[strike_idx, implied_vol_column]
                            volume = option_df.loc[strike_idx, volume_column]
                        option_prices[i, j] = price
                        implied_vols[i, j] = implied_vol
                        # 
                        volume_array[i, j] = [volume if volume != 0 else 0][0]
                        break
        
        # Store the results
        self.option_prices = option_prices
        self.strike_grid = np.array(K_grid)
        self.time_grid = np.array(T_grid)
        self.option_types = option_types
        
        # Calculate relative strikes (K/S)
        self.relative_strike_grid = np.array([K / current_price for K in K_grid])
        self.implied_vols = implied_vols
        self.volumes = volume_array

        return option_prices, self.strike_grid, self.time_grid, self.implied_vols, self.volumes, self.option_types

    def filter_by_strikes_and_maturities(self, rel_strikes: List[float], maturities: List[int]):
        """
        Filter the option data to only include specified strikes and maturities.
        """
        nearest_strikes = [closest_value(self.relative_strike_grid, K) for K in rel_strikes]
        nearest_maturities = [closest_value(self.time_grid * 365, T) for T in maturities]

        print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    Selected relative strikes: {[f'{round(K * 100, 2)}%' for K in sorted(nearest_strikes)]}")
        print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    Selected maturities (dtm): {[int(T) for T in sorted(nearest_maturities)]}")

        # Get indices of selected strikes and maturities
        strike_indx = [self.relative_strike_grid.tolist().index(K) for K in nearest_strikes]
        maturity_indx = [(self.time_grid * 365).tolist().index(T) for T in nearest_maturities]

        # Set filtered attributes
        self.option_prices = self.option_prices[np.ix_(strike_indx, maturity_indx)]
        self.strike_grid = self.strike_grid[strike_indx]
        self.time_grid = self.time_grid[maturity_indx]
        self.relative_strike_grid = self.relative_strike_grid[strike_indx]
        self.implied_vols = self.implied_vols[np.ix_(strike_indx, maturity_indx)]
        self.volumes = self.volumes[np.ix_(strike_indx, maturity_indx)]
        return self.option_prices, self.strike_grid, self.time_grid, self.implied_vols, self.volumes, self.option_types

    def compute_dlvs(self):
        """
        Compute discrete local volatilities (DLVs) from the implied volatility surface.
        """

        if self.underlying_data is not None:
            S = self.underlying_data.iloc[-1]['Close']
        else:
            raise ValueError("Underlying data is not available to get current price S.")
        
        dlvs = discrete_local_volatility(self.implied_vols, S, self.strike_grid, self.time_grid, r=0.0)
        self.dlvs = dlvs
        return dlvs
    
    def create_log_dlv_series(self, time_steps: int):

        n_strikes = len(self.strike_grid)
        n_maturities = len(self.time_grid)
        log_dlv_series = np.zeros((time_steps, n_strikes, n_maturities))

        log_dlvs = log_transform(self.dlvs)
        for i in range(time_steps):
            if i == 0:
                log_dlv_series[i] = log_dlvs.copy()
            else:
                # WARNING: This simulates historical log DLVs as a AR(1) process 
                # Done to overcome lack of historical options data
                log_dlv_series[i] = 0.98 * log_dlv_series[i-1] + 0.02 * log_dlvs + 0.01 * np.random.randn(n_strikes, n_maturities)
            
        return log_dlv_series

    def _validate_options_chains(self, options_chains: Dict):
        # options_chains must have keys as expiration dates and values as dicts with 'calls' and 'puts' DataFrames
        if not isinstance(options_chains, dict):
            raise ValueError("options_chains must be a dictionary with expiration dates as keys and option chain data as values.")

        # Validate that each value in options_chains is a dict containing only 'calls' and 'puts'
        invalid_entries = {}
        for exp, val in options_chains.items():
            if not isinstance(val, dict):
                invalid_entries[exp] = f"not a dict (type={type(val).__name__})"
            else:
                key_set = set(val.keys())
                if key_set != {"calls", "puts"}:
                    invalid_entries[exp] = sorted(list(key_set))

        if invalid_entries:
            print(f"Invalid options_chains entries (expiration -> keys/type): {invalid_entries}", file=sys.stderr)
            raise ValueError(f"Each options_chains entry must be a dict with only 'calls' and 'puts' keys. Invalid entries: {invalid_entries}")




    @staticmethod
    def load_options_chain(opts_chain_df: pd.DataFrame) -> Dict:
        """
        Load options chain data from a DataFrame and turns it into a dictionary with 'calls' and 'puts' DataFrames.

        Parameters:
        -----------
        opts_chain_df : pd.DataFrame
            DataFrame containing options chain data.

        Returns:
        --------
        Dict
            A dictionary with 'calls' and 'puts' DataFrames.
        """

        expiration_set = set(opts_chain_df['expiration'].unique())
        options_chains = {}
        for exp in expiration_set:
            df_expiry = opts_chain_df[opts_chain_df['expiration'] == exp]
            calls = df_expiry[df_expiry['type'] == 'calls']
            puts = df_expiry[df_expiry['type'] == 'puts']
            options_chains[exp] = {'calls': calls, 'puts': puts}

        return options_chains
    
    def save_data(self, filename):
        """
        Save processed data to a file.
        
        Parameters:
        -----------
        filename : str
            Filename to save data to
        """
        data = {
            'option_prices': self.option_prices,
            'strike_grid': self.strike_grid,
            'time_grid': self.time_grid,
            'relative_strike_grid': self.relative_strike_grid,
            'implied_vols': self.implied_vols,
            'volumes': self.volumes,
            'dlvs': self.dlvs
        }
        
        np.savez(filename, **data)
        print(f"Data saved to {filename}")
    
    def load_saved_data(self, filename):
        """
        Load processed data from a file.
        
        Parameters:
        -----------
        filename : str
            Filename to load data from
            
        Returns:
        --------
        bool
            True if data was loaded successfully, False otherwise
        """
        try:
            data = np.load(filename)

            self.option_prices = data['option_prices']
            self.strike_grid = data['strike_grid']
            self.time_grid = data['time_grid']
            self.relative_strike_grid = data['relative_strike_grid']
            self.implied_vols = data['implied_vols']
            self.volumes = data['volumes']
            self.dlvs = data['dlvs']
            
            print(f"Data loaded from {filename}")
            return True
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return False 