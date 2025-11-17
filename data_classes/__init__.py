import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from typing import Union, List, Dict
import sys, os
import logging

from utils.helpers import closest_value, make_dir_if_not_exists
from transforms import bs_price, calc_bs_imp_vol, log_transform, interpolate_implied_vol_surface

# Set up logger for this module
logger = logging.getLogger(__name__)

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
            # Extend end_date by a few days to ensure we get data for the last date in the range
            # yfinance's history() with 'end' parameter is typically exclusive
            extended_end_date = self.end_date + dt.timedelta(days=5)
            df = ticker.history(start=self.start_date, end=extended_end_date, interval=interval, actions=False)
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
                logger.info(f"Fetching data for symbol: {sym}")
                if not isinstance(sym, str):
                    continue
                try:
                    t = yf.Ticker(sym)
                    # Extend end_date by a few days to ensure we get data for the last date in the range
                    # yfinance's history() with 'end' parameter is typically exclusive
                    extended_end_date = self.end_date + dt.timedelta(days=5)
                    df = t.history(start=self.start_date, end=extended_end_date, interval=interval, actions=False)
                except Exception:
                    df = None
                if df is None or df.empty:
                    # skip empty results but note them by continuing; user can inspect
                    continue
                # Preserve the date index by renaming it before reset_index
                df = df.copy()
                df.index.name = 'Date'
                tmp = df.reset_index()
                tmp["symbol"] = sym
                frames.append(tmp)

            if not frames:
                raise ValueError(f"No data found for any symbols in the provided list: {symbols}")

            df = pd.concat(frames, ignore_index=False, sort=False)
            # If Date column exists, set it as index
            if 'Date' in df.columns:
                df = df.set_index('Date')
                df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
                df.index = df.index.date
            else:
                # Fallback: try to use existing index
                df.index = pd.to_datetime(df.index, utc=True, errors='coerce')
                df.index = df.index.date
            
            self.underlying_data = df
            logger.info(f"{df.shape[0]} stock row(s) fetched for symbol(s) {self.symbols}")
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
LIQUIDITY_TOL = 0  # Minimum volume threshold for calculating implied volatility
INTERPOLATION_TOL = 0.05  # Relative tolerance for using interpolation vs closest value (5% difference)

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

    def get_implied_vol(self, price, S, K, T, r, option_type, volume):
        """
        Calculate implied volatility with liquidity check.
        
        Parameters:
        -----------
        price : float
            Market price of the option
        S : float
            Current underlying asset price
        K : float
            Strike price
        T : float
            Time to maturity in years
        r : float
            Risk-free interest rate (annualized, as decimal)
        option_type : str
            'calls' or 'puts'
        volume : float
            Trading volume of the option
            
        Returns:
        --------
        float
            Implied volatility, or np.nan if calculation fails or liquidity is too low
        """
        # Check liquidity threshold
        if volume < LIQUIDITY_TOL:
            return np.nan
        
        # Convert risk-free rate from percentage to decimal if needed
        if r > 1.0:  # If rate is > 1, assume it's a percentage
            r = r / 100.0
        
        # Calculate implied volatility
        implied_vol = calc_bs_imp_vol(
            price=price,
            S=S,
            K=K,
            T=T,
            r=r,
            option_type=option_type,
            tol=1e-8
        )
        
        return implied_vol

    def clean_and_process_data(self, price_column='close', volume_column='volume'):
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
        date_key = self.date.date() if hasattr(self.date, 'date') else self.date
        
        # Check if date exists in underlying_data, if not, use the closest available date
        if date_key not in self.underlying_data.index:
            # Find the closest available date
            available_dates = self.underlying_data.index
            if len(available_dates) == 0:
                raise ValueError(f"No underlying data available. Date requested: {date_key}")
            
            # Find closest date (before or after)
            date_diff = [(abs((d - date_key).days), d) for d in available_dates]
            date_diff.sort()
            closest_date = date_diff[0][1]
            
            logger.warning(f"Date {date_key} not found in underlying_data. Using closest available date: {closest_date}")
            date_key = closest_date
        
        current_price = self.underlying_data.loc[date_key]['Close']
        
        # Extract unique strikes and maturities
        all_strikes = set()
        all_maturities = set()
        
        for exp_date, chain in self.options_chains.items():
            # Convert expiration date to datetime
            exp_datetime = dt.datetime.strptime(exp_date, '%Y-%m-%d')
            
            # Calculate time to maturity in years
            days_to_expiry = (exp_datetime - self.date).days
            T = max(1, days_to_expiry) / DAYS_IN_YEAR  # Ensure at least 1 day
            
            all_maturities.add(T)
            
            # Extract strikes from both calls and puts
            # Round to 2 decimal places to avoid floating point precision issues
            if not chain['calls'].empty:
                for strike in chain['calls']['strike']:
                    all_strikes.add(round(float(strike), 2))
            if not chain['puts'].empty:
                for strike in chain['puts']['strike']:
                    all_strikes.add(round(float(strike), 2))
        
        # Create sorted grids
        K_grid = sorted(list(all_strikes))
        T_grid = sorted(list(all_maturities))
        
        # Initialize option price and option type matrices
        n_strikes = len(K_grid)
        n_maturities = len(T_grid)
        
        option_prices = np.zeros((n_strikes, n_maturities))
        volume_array = np.zeros((n_strikes, n_maturities))
        implied_vols = np.zeros((n_strikes, n_maturities))
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
                        calls_df = chain['calls']
                        puts_df = chain['puts']
                        
                        # Determine whether to use call or put based on moneyness
                        # (Use calls for K >= S and puts for K < S - matches gan-options-simulator convention)
                        option_df = calls_df if K >= current_price else puts_df
                        option_types[i, j] = 'call' if K >= current_price else 'put'
                        
                        # If there is no calls when K > S or puts when K < S 
                        # usually when deep OTM 
                        if option_df.empty:
                            price = np.nan
                            volume = 0
                            implied_vol = np.nan
                            logger.warning(f"Setting implied_vol np.nan - No options data available. "
                                          f"Parameters: K={K}, T={T:.6f}, S={current_price}, option_type={'calls' if K >= current_price else 'puts'}, expiry_date={exp_date}")
                        else:
                            strike_idx = (option_df['strike'] - K).abs().idxmin()
                            option = option_df.loc[strike_idx]
                            price = option[price_column]
                            volume = option[volume_column]
                            implied_vol = self.get_implied_vol(
                                price=price,
                                S=current_price,
                                K=option.strike,
                                T=T,
                                r=option.risk_free_rate,  # Will be converted to decimal in get_implied_vol if needed
                                option_type=option.optionType,
                                volume=volume
                            )
                        option_prices[i, j] = price
                        volume_array[i, j] = volume
                        implied_vols[i, j] = implied_vol

        # Store the results in instance variables
        self.option_prices = option_prices
        self.strike_grid = np.array(K_grid)
        self.time_grid = np.array(T_grid)
        self.relative_strike_grid = np.array([K / current_price for K in K_grid])
        self.volumes = volume_array
        self.implied_vols = implied_vols
        self.option_types = option_types

        return option_prices, implied_vols, self.strike_grid, self.time_grid, self.relative_strike_grid, self.volumes, self.option_types

    def filter_by_strikes_and_maturities(self, rel_strikes: List[float], maturities: List[int], 
                                         use_interpolation=True, h1=None, h2=None):
        """
        Filter the option data to only include specified strikes and maturities.
        Uses Nadaraya-Watson kernel smoothing for interpolation/extrapolation when the closest value is too far from the target.
        
        Parameters:
        -----------
        rel_strikes : List[float]
            Target relative strikes (K/S)
        maturities : List[int]
            Target maturities in days
        use_interpolation : bool
            Whether to use interpolation when closest value is far (default: True)
        h1 : float, optional
            Bandwidth parameter for moneyness dimension (used for Nadaraya-Watson method)
        h2 : float, optional
            Bandwidth parameter for time to maturity dimension (used for Nadaraya-Watson method)
        """
        rel_strikes = np.asarray(rel_strikes)
        maturities = np.asarray(maturities)
        target_maturities_years = maturities / DAYS_IN_YEAR
        
        # Find closest values (preserved for comparison)
        nearest_strikes = np.array([closest_value(self.relative_strike_grid, K) for K in rel_strikes])
        nearest_maturities_days = np.array([closest_value(self.time_grid * DAYS_IN_YEAR, T) for T in maturities])
        nearest_maturities_years = nearest_maturities_days / DAYS_IN_YEAR

        logger.info(f"Target relative strikes: {[f'{round(K * 100, 2)}%' for K in sorted(rel_strikes)]}")
        logger.info(f"Target maturities (dtm): {[int(T) for T in sorted(maturities)]}")

        # Check if interpolation is needed
        use_interp = np.zeros((len(rel_strikes), len(maturities)), dtype=bool)
        interpolation_details = []  # Store details for logging
        if use_interpolation:
            for i, target_strike in enumerate(rel_strikes):
                for j, target_mat in enumerate(maturities):
                    strike_diff = abs(nearest_strikes[i] - target_strike) / target_strike if target_strike > 0 else abs(nearest_strikes[i] - target_strike)
                    mat_diff = abs(nearest_maturities_days[j] - target_mat) / target_mat if target_mat > 0 else abs(nearest_maturities_days[j] - target_mat)
                    # Use interpolation if relative difference exceeds tolerance
                    if strike_diff > INTERPOLATION_TOL or mat_diff > INTERPOLATION_TOL:
                        use_interp[i, j] = True
                        # Determine reason for interpolation
                        reasons = []
                        if strike_diff > INTERPOLATION_TOL:
                            reasons.append(f"strike_diff={strike_diff:.4f} ({strike_diff*100:.2f}%) > tolerance={INTERPOLATION_TOL*100:.2f}%")
                        if mat_diff > INTERPOLATION_TOL:
                            reasons.append(f"maturity_diff={mat_diff:.4f} ({mat_diff*100:.2f}%) > tolerance={INTERPOLATION_TOL*100:.2f}%")
                        
                        interpolation_details.append({
                            'i': i, 'j': j,
                            'target_strike': target_strike,
                            'target_maturity_days': target_mat,
                            'nearest_strike': nearest_strikes[i],
                            'nearest_maturity_days': nearest_maturities_days[j],
                            'strike_diff': strike_diff,
                            'mat_diff': mat_diff,
                            'reasons': reasons
                        })
        
        # Initialize result arrays
        n_target_strikes = len(rel_strikes)
        n_target_maturities = len(maturities)
        filtered_implied_vols = np.zeros((n_target_strikes, n_target_maturities))
        filtered_option_prices = np.zeros((n_target_strikes, n_target_maturities))
        filtered_volumes = np.zeros((n_target_strikes, n_target_maturities))
        
        # Get indices for closest values (for non-interpolated points)
        strike_indx = [self.relative_strike_grid.tolist().index(K) for K in nearest_strikes]
        maturity_indx = [(self.time_grid * DAYS_IN_YEAR).tolist().index(T) for T in nearest_maturities_days]
        
        # Use interpolation for points that need it
        if np.any(use_interp):
            logger.info(f"Using Nadaraya-Watson kernel smoothing for {np.sum(use_interp)}/{len(rel_strikes) * len(maturities)} points")
            
            # Log detailed reasons for interpolation
            if interpolation_details:
                logger.debug(f"Interpolation details:")
                for detail in interpolation_details:
                    i, j = detail['i'], detail['j']
                    logger.debug(f"Point ({i}, {j}): "
                          f"target_strike={detail['target_strike']:.4f} ({detail['target_strike']*100:.2f}%), "
                          f"target_maturity={detail['target_maturity_days']:.0f} days, "
                          f"nearest_strike={detail['nearest_strike']:.4f} ({detail['nearest_strike']*100:.2f}%), "
                          f"nearest_maturity={detail['nearest_maturity_days']:.0f} days")
                    logger.debug(f"Reasons: {'; '.join(detail['reasons'])}")
            
            interpolated_vols = interpolate_implied_vol_surface(
                self.implied_vols,
                self.relative_strike_grid,
                self.time_grid,
                rel_strikes,
                target_maturities_years,
                h1=h1,
                h2=h2
            )
            
            # Fill in interpolated values where needed
            for i in range(n_target_strikes):
                for j in range(n_target_maturities):
                    if use_interp[i, j]:
                        closest_vol = self.implied_vols[strike_indx[i], maturity_indx[j]]
                        interpolated_vol = interpolated_vols[i, j]
                        filtered_implied_vols[i, j] = interpolated_vol
                        # For prices and volumes, still use closest value (interpolation not typically used)
                        filtered_option_prices[i, j] = self.option_prices[strike_indx[i], maturity_indx[j]]
                        filtered_volumes[i, j] = self.volumes[strike_indx[i], maturity_indx[j]]
                        
                        # Log the interpolated value vs closest value
                        if not np.isnan(interpolated_vol) and not np.isnan(closest_vol):
                            vol_diff = abs(interpolated_vol - closest_vol)
                            vol_diff_pct = (vol_diff / closest_vol * 100) if closest_vol > 0 else 0
                            logger.debug(f"Interpolated vol at ({i}, {j}): "
                                  f"{interpolated_vol:.6f} (closest: {closest_vol:.6f}, diff: {vol_diff:.6f} ({vol_diff_pct:.2f}%))")
                    else:
                        # Use closest value
                        filtered_implied_vols[i, j] = self.implied_vols[strike_indx[i], maturity_indx[j]]
                        filtered_option_prices[i, j] = self.option_prices[strike_indx[i], maturity_indx[j]]
                        filtered_volumes[i, j] = self.volumes[strike_indx[i], maturity_indx[j]]
        else:
            # All points use closest value
            filtered_implied_vols = self.implied_vols[np.ix_(strike_indx, maturity_indx)]
            filtered_option_prices = self.option_prices[np.ix_(strike_indx, maturity_indx)]
            filtered_volumes = self.volumes[np.ix_(strike_indx, maturity_indx)]

        # Update instance variables with filtered data
        self.option_prices = filtered_option_prices
        self.strike_grid = np.array([self.strike_grid[i] for i in strike_indx])  # Use closest strikes
        self.time_grid = target_maturities_years  # Use target maturities
        self.relative_strike_grid = rel_strikes  # Use target strikes
        self.volumes = filtered_volumes
        self.implied_vols = filtered_implied_vols
        # Update option_types to match new dimensions
        self.option_types = self.option_types[np.ix_(strike_indx, maturity_indx)]
        
        return self.option_prices, self.implied_vols, self.strike_grid, self.time_grid, self.relative_strike_grid, self.volumes, self.option_types

    

    # def create_log_dlv_series(self, time_steps: int):

    #     n_strikes = len(self.strike_grid)
    #     n_maturities = len(self.time_grid)
    #     log_dlv_series = np.zeros((time_steps, n_strikes, n_maturities))

    #     log_dlvs = log_transform(self.dlvs)
    #     for i in range(time_steps):
    #         if i == 0:
    #             log_dlv_series[i] = log_dlvs.copy()
    #         else:
    #             # WARNING: This simulates historical log DLVs as a AR(1) process 
    #             # Done to overcome lack of historical options data
    #             log_dlv_series[i] = 0.98 * log_dlv_series[i-1] + 0.02 * log_dlvs + 0.01 * np.random.randn(n_strikes, n_maturities)
            
    #     return log_dlv_series

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
            logger.error(f"Invalid options_chains entries (expiration -> keys/type): {invalid_entries}")
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
        logger.info(f"Data saved to {filename}")
    
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
            
            logger.info(f"Data loaded from {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False


class DataCache:
    """
    Cache and save processed option data by year in long format.
    """
    
    def __init__(self, data_dir: str, ticker: str):
        """
        Initialize DataCache.
        
        Parameters:
        -----------
        data_dir : str
            Directory to save data files
        ticker : str
            Ticker symbol
        """
        self.data_dir = data_dir
        self.ticker = ticker
        self.current_year_data = []
        self.current_year = None
    
    def add_date_data(self, date, current_price, option_prices, implied_vols, 
                     strike_grid, time_grid, relative_strike_grid, volume_array, option_types):
        """
        Add data for a single date to the cache.
        
        Parameters:
        -----------
        date : datetime.date
            Date of the data
        current_price : float
            Current underlying price
        option_prices : ndarray
            Option prices array (n_strikes, n_maturities)
        implied_vols : ndarray
            Implied volatilities array (n_strikes, n_maturities)
        strike_grid : ndarray
            Strike prices
        time_grid : ndarray
            Time to maturity in years
        relative_strike_grid : ndarray
            Relative strikes (K/S)
        volume_array : ndarray
            Volume array (n_strikes, n_maturities)
        option_types : ndarray
            Option types array (n_strikes, n_maturities)
        """
        year = date.year if isinstance(date, dt.date) else date.year
        
        # If year changed, save previous year's data
        if self.current_year is not None and year != self.current_year:
            self.save_year_data(self.current_year)
            self.current_year_data = []
        
        self.current_year = year
        
        # Store data for this date
        self.current_year_data.append({
            'date': date,
            'current_price': current_price,
            'option_prices': option_prices,
            'implied_vols': implied_vols,
            'strike_grid': strike_grid,
            'time_grid': time_grid,
            'relative_strike_grid': relative_strike_grid,
            'volume_array': volume_array,
            'option_types': option_types
        })
    
    def save_year_data(self, year: int = None):
        """
        Save all cached data for a year in long format to a single CSV file.
        
        Parameters:
        -----------
        year : int, optional
            Year to save. If None, saves current_year data.
        """
        if year is None:
            year = self.current_year
        
        if not self.current_year_data:
            return
        
        all_rows = []
        
        for date_data in self.current_year_data:
            date = date_data['date']
            current_price = date_data['current_price']
            option_prices = date_data['option_prices']
            implied_vols = date_data['implied_vols']
            strike_grid = date_data['strike_grid']
            time_grid = date_data['time_grid']
            relative_strike_grid = date_data['relative_strike_grid']
            volume_array = date_data['volume_array']
            option_types = date_data['option_types']
            
            # Convert from wide format (strikes x maturities) to long format
            n_strikes = len(strike_grid)
            n_maturities = len(time_grid)
            
            for i in range(n_strikes):
                for j in range(n_maturities):
                    row = {
                        'date': date,
                        'current_price': current_price,
                        'strike': strike_grid[i],
                        'relative_strike': relative_strike_grid[i],
                        'maturity_days': int(time_grid[j] * 365),
                        'maturity_years': time_grid[j],
                        'option_price': option_prices[i, j],
                        'implied_vol': implied_vols[i, j],
                        'volume': volume_array[i, j],
                        'option_type': option_types[i, j]
                    }
                    all_rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(all_rows)
        year_dir = os.path.join(self.data_dir, str(year))
        make_dir_if_not_exists(year_dir)
        
        filename = os.path.join(year_dir, f"{self.ticker}_options_data_{year}.csv")
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} rows to {filename}")
    
    def finalize(self):
        """
        Save any remaining cached data (for the last year).
        """
        if self.current_year_data:
            self.save_year_data()
            self.current_year_data = [] 