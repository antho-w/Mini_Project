import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
import logging

logger = logging.getLogger(__name__)

def make_dir_if_not_exists(path: Union[str, Path]) -> Path:
    """
    Create a directory if it doesn't exist and return a Path to it.

    Args:
        path: Directory path as a str or pathlib.Path.

    Returns:
        pathlib.Path: The created (or existing) directory path.

    Raises:
        FileExistsError: If a non-directory file exists at the given path.
        PermissionError: If the directory cannot be created due to permissions.
        OSError: For other OS-related errors.
    """
    p = Path(path)
    if p.exists():
        if not p.is_dir():
            raise FileExistsError(f"Path exists and is not a directory: {p!s}")
        return p

    # Create including any missing parents; this is safe against races.
    p.mkdir(parents=True, exist_ok=True)
    return p

def closest_value(array, target):
    """
    Find the closest value in a numpy array to a given target value.

    Args:
        array (np.ndarray): Input array to search.
        target (float): Target value to find the closest match for.

    Returns:
        float: The closest value in the array to the target.
    """
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - target)).argmin()
    return array[idx]

def calculate_returns(data, n_periods=1, log_returns=True):
    """
    Calculate returns from time series data.
    
    Parameters:
    -----------
    data : ndarray
        Time series data
    n_periods : int
        Number of periods for return calculation
    log_returns : bool
        If True, calculate log returns; if False, calculate percentage returns
        
    Returns:
    --------
    ndarray
        Returns series
    """
    if log_returns:
        return np.diff(np.log(data), n=n_periods, axis=0)
    else:
        shifted_data = np.roll(data, n_periods, axis=0)
        shifted_data[:n_periods] = np.nan
        return (data / shifted_data) - 1


def create_time_windows(data, window_size, step_size=1):
    """
    Create sliding time windows from sequential data.
    
    Parameters:
    -----------
    data : ndarray
        Sequential data with shape (n_samples, ...)
    window_size : int
        Size of each window
    step_size : int
        Step size between consecutive windows
        
    Returns:
    --------
    ndarray
        Windowed data with shape (n_windows, window_size, ...)
    """
    n_samples = len(data)
    n_windows = max(0, (n_samples - window_size) // step_size + 1)
    
    windows = []
    for i in range(0, n_windows * step_size, step_size):
        if i + window_size <= n_samples:
            windows.append(data[i:i+window_size])
    
    return np.array(windows)


def create_sequences(data, sequence_length, target_steps=1):
    """
    Create input-target sequences for time series prediction.
    
    Parameters:
    -----------
    data : ndarray
        Sequential data with shape (n_samples, ...)
    sequence_length : int
        Length of input sequences
    target_steps : int
        Number of steps to predict
        
    Returns:
    --------
    tuple
        (X, y) where X has shape (n_sequences, sequence_length, ...)
        and y has shape (n_sequences, target_steps, ...)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length - target_steps + 1):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length:i+sequence_length+target_steps])
    
    return np.array(X), np.array(y)

def calculate_time_difference(start_time, end_time=None):
    """
    Calculate time difference and format it.
    
    Parameters:
    -----------
    start_time : datetime
        Start time
    end_time : datetime, optional
        End time (defaults to now)
        
    Returns:
    --------
    str
        Formatted time difference
    """
    if end_time is None:
        end_time = dt.datetime.now()
    
    time_diff = end_time - start_time
    total_seconds = time_diff.total_seconds()
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" 

def normalize_data(data, method='min_max', axis=0):
    """
    Normalize data using different methods.
    
    Parameters:
    -----------
    data : ndarray
        Data to normalize
    method : str
        Normalization method ('min_max', 'z_score', or 'robust')
    axis : int
        Axis along which to normalize
        
    Returns:
    --------
    tuple
        (normalized_data, params) where params contains normalization parameters
    """
    if method == 'min_max':
        data_min = np.min(data, axis=axis, keepdims=True)
        data_max = np.max(data, axis=axis, keepdims=True)
        normalized_data = (data - data_min) / (data_max - data_min + 1e-8)
        params = {'min': data_min, 'max': data_max}
    
    elif method == 'z_score':
        data_mean = np.mean(data, axis=axis, keepdims=True)
        data_std = np.std(data, axis=axis, keepdims=True)
        normalized_data = (data - data_mean) / (data_std + 1e-8)
        params = {'mean': data_mean, 'std': data_std}
    
    elif method == 'robust':
        q25 = np.percentile(data, 25, axis=axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True)
        iqr = q75 - q25
        normalized_data = (data - q25) / (iqr + 1e-8)
        params = {'q25': q25, 'iqr': iqr}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_data, params

def create_sequences(data, sequence_length, target_steps=1):
    """
    Create input-target sequences for time series prediction.
    
    Parameters:
    -----------
    data : ndarray
        Sequential data with shape (n_samples, ...)
    sequence_length : int
        Length of input sequences
    target_steps : int
        Number of steps to predict
        
    Returns:
    --------
    tuple
        (X, y) where X has shape (n_sequences, sequence_length, ...)
        and y has shape (n_sequences, target_steps, ...)
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length - target_steps + 1):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length:i+sequence_length+target_steps])
    
    return np.array(X), np.array(y)

def denormalize_data(data, params, method='min_max'):
    """
    Denormalize data using stored parameters.
    
    Parameters:
    -----------
    data : ndarray
        Normalized data
    params : dict
        Parameters for denormalization
    method : str
        Normalization method used ('min_max', 'z_score', or 'robust')
        
    Returns:
    --------
    ndarray
        Denormalized data
    """
    if method == 'min_max':
        return data * (params['max'] - params['min'] + 1e-8) + params['min']
    
    elif method == 'z_score':
        return data * (params['std'] + 1e-8) + params['mean']
    
    elif method == 'robust':
        return data * (params['iqr'] + 1e-8) + params['q25']
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def plot_surface(X, Y, Z, title, xlabel, ylabel, zlabel, cmap='viridis'):
    """
    Plot a 3D surface.
    
    Parameters:
    -----------
    X : ndarray
        Grid of x coordinates
    Y : ndarray
        Grid of y coordinates
    Z : ndarray
        Surface height values
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    zlabel : str
        Z-axis label
    cmap : str
        Colormap name
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X_grid, Y_grid = np.meshgrid(X, Y)
    surf = ax.plot_surface(X_grid, Y_grid, Z.T, cmap=cmap, antialiased=True)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig, ax


def remove_nan_values_nadaraya_watson(implied_vols_array, strikes, maturities, h1=None, h2=None):
    """
    Remove NaN values from implied volatility array by applying Nadaraya-Watson smoothing
    to each time slice.
    
    Parameters:
    -----------
    implied_vols_array : ndarray
        3D array with shape (n_dates, n_strikes, n_maturities) containing implied volatilities
    strikes : list or ndarray
        Relative strikes (K/S) corresponding to the second dimension
    maturities : list or ndarray
        Maturities in days corresponding to the third dimension
    h1 : float, optional
        Bandwidth parameter for moneyness dimension. If None, auto-estimated.
    h2 : float, optional
        Bandwidth parameter for time to maturity dimension. If None, auto-estimated.
        
    Returns:
    --------
    ndarray
        Implied volatility array with NaN values filled using Nadaraya-Watson smoothing.
        Same shape as input array.
    """
    from transforms import interpolate_implied_vol_surface
    
    implied_vols_array = np.asarray(implied_vols_array)
    time_steps = implied_vols_array.shape[0]
    
    # Convert maturities from days to years for interpolation
    maturities_years = np.array(maturities) / 365.0
    strikes_array = np.array(strikes)
    
    initial_nan_count = np.isnan(implied_vols_array).sum()
    
    if initial_nan_count == 0:
        logger.info("No NaN values found in implied_vols_array, skipping smoothing")
        return implied_vols_array
    
    logger.info(f"Removing {initial_nan_count} NaN values using Nadaraya-Watson smoothing")
    
    # Process each time slice
    for t in range(time_steps):
        slice_2d = implied_vols_array[t]  # Shape: (n_strikes, n_maturities)
        
        # Check if there are any NaN values in this slice
        if np.isnan(slice_2d).any():
            # Apply Nadaraya-Watson smoothing to fill NaN values
            # interpolate_implied_vol_surface handles NaN removal internally
            smoothed_slice = interpolate_implied_vol_surface(
                implied_vols=slice_2d,
                strike_grid=strikes_array,
                time_grid=maturities_years,
                target_strikes=strikes_array,
                target_maturities=maturities_years,
                h1=h1,  # Auto-estimate if None
                h2=h2   # Auto-estimate if None
            )
            implied_vols_array[t] = smoothed_slice
    
    final_nan_count = np.isnan(implied_vols_array).sum()
    removed_count = initial_nan_count - final_nan_count
    
    logger.info(f"Nadaraya-Watson smoothing complete. "
               f"Removed {removed_count} NaN values. "
               f"Remaining NaN count: {final_nan_count}")
    
    return implied_vols_array