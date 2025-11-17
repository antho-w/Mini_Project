import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import sys
import datetime as dt
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)

def local_volatility_dupire(implied_vols, S, K_grid, T_grid, r):
    """
    Calculate local volatility using Dupire's formula.
    
    Parameters:
    -----------
    implied_vols : ndarray
        Implied volatility surface with shape (n_strikes, n_maturities)
    S : float
        Underlying asset price
    K_grid : ndarray
        Array of strike prices
    T_grid : ndarray
        Array of maturities in years
    r : float
        Risk-free interest rate (annualized)
        
    Returns:
    --------
    ndarray
        Local volatility surface with shape (n_strikes, n_maturities)
    """
    n_strikes = len(K_grid)
    n_maturities = len(T_grid)
    
    # Initialize local volatility surface
    local_vols = np.zeros((n_strikes, n_maturities))
    
    # Compute finite differences for each point in the grid
    for i in range(1, n_strikes - 1):
        for j in range(1, n_maturities - 1):
            K = K_grid[i]
            T = T_grid[j]
            sigma = implied_vols[i, j]
            
            # Calculate derivatives using finite differences
            d_sigma_dK = (implied_vols[i+1, j] - implied_vols[i-1, j]) / (K_grid[i+1] - K_grid[i-1])
            d2_sigma_dK2 = (implied_vols[i+1, j] - 2*implied_vols[i, j] + implied_vols[i-1, j]) / ((K_grid[i+1] - K_grid[i]) ** 2)
            d_sigma_dT = (implied_vols[i, j+1] - implied_vols[i, j-1]) / (T_grid[j+1] - T_grid[j-1])
            
            # Calculate d1 and d2 from Black-Scholes
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            
            # Apply Dupire's formula
            numerator = sigma**2
            denominator = (1 + K * d1 * d_sigma_dK)**2 + K**2 * sigma * d2_sigma_dK2 + 2 * K * sigma * d_sigma_dT
            
            # Handle numerical instabilities
            if denominator <= 0:
                local_vols[i, j] = sigma  # Fallback to implied vol
            else:
                local_vols[i, j] = sigma * np.sqrt(numerator / denominator)
    
    # Fill in boundary values (simple approximation)
    local_vols[0, :] = local_vols[1, :]
    local_vols[-1, :] = local_vols[-2, :]
    local_vols[:, 0] = local_vols[:, 1]
    local_vols[:, -1] = local_vols[:, -2]
    
    return local_vols


def discrete_local_volatility(implied_vols, S, K_grid, T_grid, r):
    """
    Convert implied volatility surface to discrete local volatilities (DLVs).
    
    This is a simplified implementation that needs to be enhanced based on
    the specific methodology described in Buehler (2017) and Wissel (2007).
    
    Parameters:
    -----------
    implied_vols : ndarray
        Implied volatility surface with shape (n_strikes, n_maturities)
    S : float
        Underlying asset price
    K_grid : ndarray
        Array of strike prices
    T_grid : ndarray
        Array of maturities in years
    r : float
        Risk-free interest rate (annualized)
        
    Returns:
    --------
    ndarray
        Discrete local volatility surface with shape (n_strikes, n_maturities)
    """
    # For now, we'll use Dupire's formula as an approximation
    # In a real implementation, this should be replaced with the actual DLV calculation
    # as described in the referenced papers
    return local_volatility_dupire(implied_vols, S, K_grid, T_grid, r)


def dlv_to_implied_vol(dlvs, S, K_grid, T_grid, r):
    """
    Convert DLVs back to implied volatilities.
    
    This is a placeholder function that needs to be implemented based on
    the methodology described in the referenced papers.
    
    Parameters:
    -----------
    dlvs : ndarray
        Discrete local volatility surface with shape (n_strikes, n_maturities)
    S : float
        Underlying asset price
    K_grid : ndarray
        Array of strike prices
    T_grid : ndarray
        Array of maturities in years
    r : float
        Risk-free interest rate (annualized)
        
    Returns:
    --------
    ndarray
        Implied volatility surface with shape (n_strikes, n_maturities)
    """
    # This requires implementing the inverse transformation
    # For now, we'll return a simple approximation
    return dlvs  # This is not correct and needs to be implemented properly


def check_no_arbitrage(option_prices, S, K_grid, T_grid, r):
    """
    Check if option prices satisfy no-arbitrage conditions.
    
    Parameters:
    -----------
    option_prices : ndarray
        Array of option prices with shape (n_strikes, n_maturities)
    S : float
        Underlying asset price
    K_grid : ndarray
        Array of strike prices
    T_grid : ndarray
        Array of maturities in years
    r : float
        Risk-free interest rate (annualized)
        
    Returns:
    --------
    bool
        True if no-arbitrage conditions are satisfied, False otherwise
    """
    n_strikes = len(K_grid)
    n_maturities = len(T_grid)
    
    # Check monotonicity with respect to strike (for calls)
    for j in range(n_maturities):
        for i in range(n_strikes - 1):
            # Call prices should decrease with strike
            if option_prices[i, j] < option_prices[i+1, j]:
                return False
    
    # Check monotonicity with respect to maturity
    for i in range(n_strikes):
        for j in range(n_maturities - 1):
            # Option prices should increase with maturity
            if option_prices[i, j] > option_prices[i, j+1]:
                return False
    
    # Add more no-arbitrage checks as needed
    
    return True


def check_dlv_no_arbitrage(dlvs):
    """
    Check if DLVs satisfy no-arbitrage conditions.
    
    Parameters:
    -----------
    dlvs : ndarray
        Discrete local volatility surface with shape (n_strikes, n_maturities)
        
    Returns:
    --------
    bool
        True if no-arbitrage conditions are satisfied, False otherwise
    """
    # For DLVs, the no-arbitrage condition is simply non-negativity
    return np.all(dlvs > 0)

def log_transform(data, epsilon=1e-10):
    """
    Apply log transformation to data with epsilon to avoid log(0).
    
    Parameters:
    -----------
    data : ndarray
        Data to transform
    epsilon : float
        Small constant to add to avoid log(0)
        
    Returns:
    --------
    ndarray
        Log-transformed data
    """
    return np.log(data + epsilon)

def bs_price(S, K, T, r, sigma, option_type='calls'):
    """
    Calculate Black-Scholes option price.
    
    Parameters:
    -----------
    S : float
        Underlying asset price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Implied volatility (annualized)
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    float
        Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'calls':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    return price

def calc_bs_imp_vol(price, S, K, T, r, option_type='calls', tol=1e-8, max_iterations=1000):
    """
    Calculate implied volatility using Brent's method.
    
    Parameters:
    -----------
    price : float
        Market price of the option
    S : float
        Underlying asset price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annualized)
    option_type : str
        'calls' or 'puts'
    tol : float
        Tolerance for solving the equation
    max_iterations : int
        Maximum number of iterations
        
    Returns:
    --------
    float
        Implied volatility
    """
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type=option_type) - price
    
    # Handle edge cases
    if price <= 0 or np.isnan(price):
        logger.warning(f"calc_bs_imp_vol returning np.nan - Invalid price. "
              f"Parameters: price={price}, S={S}, K={K}, T={T:.6f}, r={r}, option_type={option_type}")
        return np.nan
    
    # Set bounds for Brent's method
    sigma_low = 0.00001
    sigma_high = 5  # Increased upper bound for extreme cases
    
    # Check objective function at bounds to diagnose issues
    f_low = objective(sigma_low)
    f_high = objective(sigma_high)
    
    # If both bounds have the same sign, try expanding the upper bound
    if f_low * f_high > 0:
        # Try even higher volatility bounds
        for test_sigma in [50.0, 200.0, 500.0]:
            f_test = objective(test_sigma)
            if f_low * f_test < 0:  # Found opposite sign
                sigma_high = test_sigma
                f_high = f_test
                break
        else:
            # Still no root found even at very high volatility
            # Calculate BS prices for diagnostics
            bs_price_low = bs_price(S, K, T, r, sigma_low, option_type=option_type)
            bs_price_high = bs_price(S, K, T, r, sigma_high, option_type=option_type)
            intrinsic_value = max(S - K, 0) if option_type == 'calls' else max(K - S, 0)
            
            logger.warning(f"calc_bs_imp_vol returning np.nan - No root found (same sign at bounds). "
                  f"Parameters: price={price}, S={S}, K={K}, T={T:.6f}, r={r}, option_type={option_type}. "
                  f"BS_price(sigma={sigma_low})={bs_price_low:.2f}, BS_price(sigma={sigma_high})={bs_price_high:.2f}, "
                  f"intrinsic_value={intrinsic_value:.2f}, market_price={price:.2f}. "
                  f"This may indicate an arbitrage violation or data error.")
            return np.nan
    
    try:
        # Use Brent's method to find the implied volatility
        sigma = brentq(objective, sigma_low, sigma_high, 
                       xtol=tol, maxiter=max_iterations)
        return sigma
    except (ValueError, RuntimeError) as e:
        # If Brent's method fails, return NaN instead of a default value
        # This allows us to identify problematic options
        bs_price_low = bs_price(S, K, T, r, sigma_low, option_type=option_type)
        bs_price_high = bs_price(S, K, T, r, sigma_high, option_type=option_type)
        intrinsic_value = max(S - K, 0) if option_type == 'calls' else max(K - S, 0)
        
        logger.error(f"calc_bs_imp_vol returning np.nan - Calculation failed ({type(e).__name__}: {str(e)}). "
              f"Parameters: price={price}, S={S}, K={K}, T={T:.6f}, r={r}, option_type={option_type}, "
              f"sigma_bounds=[{sigma_low}, {sigma_high}]. "
              f"BS_price(sigma={sigma_low})={bs_price_low:.2f}, BS_price(sigma={sigma_high})={bs_price_high:.2f}, "
              f"intrinsic_value={intrinsic_value:.2f}, market_price={price:.2f}")
        return np.nan


def nadaraya_watson_kernel(x, y, h1, h2):
    """
    2D Gaussian kernel for Nadaraya-Watson smoothing.
    
    Parameters:
    -----------
    x, y : float or ndarray
        Input values (typically moneyness and time to maturity differences)
    h1, h2 : float
        Bandwidth parameters for x and y dimensions
        
    Returns:
    --------
    float or ndarray
        Kernel weights
    """
    return (np.exp(-x * x / (2 * h1)) * np.exp(-y * y / (2 * h2))) / (2 * np.pi)


def nadaraya_watson_smooth(I_in, m_in, tau_in, m_want, tau_want, h1, h2):
    """
    Nadaraya-Watson kernel smoothing function using 2-D Gaussian kernel.
    
    Based on the implementation from VolGAN (Vuletic & Rama, 2023).
    Reference: https://github.com/milenavuletic/VolGAN/blob/main/datacleaning.py
    
    Parameters:
    -----------
    I_in : ndarray
        Array of implied volatilities in the data
    m_in : ndarray
        Moneyness array in the data (1D)
    tau_in : ndarray
        Time to maturity array in the data (1D)
    m_want : ndarray
        Desired moneyness grid (1D array)
    tau_want : ndarray
        Desired time to maturity grid (1D array)
    h1 : float
        Bandwidth parameter for moneyness dimension
    h2 : float
        Bandwidth parameter for time to maturity dimension
        
    Returns:
    --------
    ndarray
        Smoothed implied volatility surface with shape (len(m_want), len(tau_want))
    """
    I_out = np.zeros((len(m_want), len(tau_want)))
    
    for i in range(len(m_want)):
        m = m_want[i]
        for j in range(len(tau_want)):
            tau = tau_want[j]
            # Calculate kernel weights for all data points
            weights = nadaraya_watson_kernel(m - m_in, tau - tau_in, h1, h2)
            
            # I_in, m_in, tau_in are already filtered to remove NaNs
            if np.sum(weights) > 0:
                # Nadaraya-Watson estimator: weighted average
                I_out[i, j] = np.sum(I_in * weights) / np.sum(weights)
            else:
                # If no valid weights, use nearest neighbor
                if len(m_in) > 0:
                    dist = np.sqrt((m_in - m)**2 + (tau_in - tau)**2)
                    nearest_idx = np.argmin(dist)
                    I_out[i, j] = I_in[nearest_idx]
                else:
                    I_out[i, j] = np.nan
    
    return I_out


def interpolate_implied_vol_surface(implied_vols, strike_grid, time_grid, target_strikes, target_maturities, 
                                     h1=None, h2=None):
    """
    Interpolate/extrapolate implied volatility surface to target strikes and maturities.
    
    Uses Nadaraya-Watson kernel smoothing with 2D Gaussian kernel (Vuletic & Rama, 2023).
    Reference: https://github.com/milenavuletic/VolGAN/blob/main/datacleaning.py
    
    Parameters:
    -----------
    implied_vols : ndarray
        Implied volatility surface with shape (n_strikes, n_maturities)
    strike_grid : ndarray
        Strike prices (relative strikes K/S)
    time_grid : ndarray
        Time to maturity in years
    target_strikes : ndarray or list
        Target strike prices to interpolate to
    target_maturities : ndarray or list
        Target maturities to interpolate to (in years)
    h1 : float, optional
        Bandwidth parameter for moneyness dimension
        If None, will be estimated from data using Silverman's rule of thumb
    h2 : float, optional
        Bandwidth parameter for time to maturity dimension
        If None, will be estimated from data using Silverman's rule of thumb
        
    Returns:
    --------
    ndarray
        Interpolated implied volatilities with shape (len(target_strikes), len(target_maturities))
    """
    implied_vols = np.asarray(implied_vols)
    strike_grid = np.asarray(strike_grid)
    time_grid = np.asarray(time_grid)
    target_strikes = np.asarray(target_strikes)
    target_maturities = np.asarray(target_maturities)
    
    # Nadaraya-Watson kernel smoothing (Vuletic & Rama, 2023)
    # Flatten the surface to 1D arrays for kernel smoothing
    strike_mesh, time_mesh = np.meshgrid(strike_grid, time_grid, indexing='ij')
    m_in = strike_mesh.ravel()
    tau_in = time_mesh.ravel()
    I_in = implied_vols.ravel()
    
    # Remove NaN values
    valid_mask_flat = ~np.isnan(I_in)
    m_in = m_in[valid_mask_flat]
    tau_in = tau_in[valid_mask_flat]
    I_in = I_in[valid_mask_flat]
    
    # Estimate bandwidth parameters if not provided (Silverman's rule of thumb)
    if h1 is None:
        # Bandwidth for moneyness dimension
        h1 = 1.06 * np.std(m_in) * (len(m_in) ** (-1/5)) if len(m_in) > 0 else 0.1
    if h2 is None:
        # Bandwidth for time to maturity dimension
        h2 = 1.06 * np.std(tau_in) * (len(tau_in) ** (-1/5)) if len(tau_in) > 0 else 0.1
    
    # Use Nadaraya-Watson smoothing
    result = nadaraya_watson_smooth(I_in, m_in, tau_in, target_strikes, target_maturities, h1, h2)
    return result