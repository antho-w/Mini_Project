"""
Visualization module for plotting and analyzing results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def plot_dlv_surface(dlvs, strikes, maturities, title='DLV Surface', cmap='viridis'):
    """
    Plot a 3D surface of DLVs.
    
    Parameters:
    -----------
    dlvs : ndarray
        2D array of DLVs with shape (n_strikes, n_maturities)
    strikes : ndarray
        Array of strike prices or relative strikes
    maturities : ndarray
        Array of maturities
    title : str
        Plot title
    cmap : str
        Colormap name
        
    Returns:
    --------
    tuple
        (fig, ax) containing figure and axis objects
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(strikes, maturities)
    surf = ax.plot_surface(X, Y, dlvs.T, cmap=cmap, linewidth=0, antialiased=True)
    
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('DLV')
    ax.set_title(title)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig, ax


def plot_dlv_heatmap(dlvs, strikes, maturities, title='DLV Heatmap', cmap='viridis'):
    """
    Plot a heatmap of DLVs.
    
    Parameters:
    -----------
    dlvs : ndarray
        2D array of DLVs with shape (n_strikes, n_maturities)
    strikes : ndarray
        Array of strike prices or relative strikes
    maturities : ndarray
        Array of maturities
    title : str
        Plot title
    cmap : str
        Colormap name
        
    Returns:
    --------
    tuple
        (fig, ax) containing figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(dlvs, ax=ax, cmap=cmap, xticklabels=maturities, yticklabels=strikes)
    
    ax.set_xlabel('Maturity')
    ax.set_ylabel('Strike')
    ax.set_title(title)
    
    return fig, ax


def plot_real_vs_generated(real, generated, feature_idx=0, title='Real vs Generated DLVs', window=100):
    """
    Plot real vs generated data for a specific feature.
    
    Parameters:
    -----------
    real : ndarray
        Real data with shape (n_samples, n_features)
    generated : ndarray
        Generated data with shape (n_samples, n_features)
    feature_idx : int
        Index of feature to plot
    title : str
        Plot title
    window : int
        Number of time steps to plot
        
    Returns:
    --------
    tuple
        (fig, ax) containing figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get data to plot (limit to window size)
    window = min(window, real.shape[0], generated.shape[0])
    real_data = real[:window, feature_idx]
    generated_data = generated[:window, feature_idx]
    
    # Plot data
    ax.plot(real_data, label='Real', color='blue')
    ax.plot(generated_data, label='Generated', color='red', linestyle='--')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title(f'{title} (Feature {feature_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_real_vs_generated_grid(real, generated, features=None, title='Real vs Generated DLVs', 
                               n_cols=2, window=100, figsize=(15, 10)):
    """
    Plot real vs generated data for multiple features in a grid.
    
    Parameters:
    -----------
    real : ndarray
        Real data with shape (n_samples, n_features)
    generated : ndarray
        Generated data with shape (n_samples, n_features)
    features : list, optional
        List of feature indices to plot (defaults to first 4 features)
    title : str
        Plot title
    n_cols : int
        Number of columns in the grid
    window : int
        Number of time steps to plot
    figsize : tuple
        Figure size
        
    Returns:
    --------
    tuple
        (fig, axes) containing figure and axes objects
    """
    # Default to first 4 features if not specified
    if features is None:
        features = list(range(min(4, real.shape[1])))
    
    n_plots = len(features)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Flatten axes for easier indexing
    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Get data to plot (limit to window size)
    window = min(window, real.shape[0], generated.shape[0])
    
    for i, feature_idx in enumerate(features):
        real_data = real[:window, feature_idx]
        generated_data = generated[:window, feature_idx]
        
        # Plot data
        axes[i].plot(real_data, label='Real', color='blue')
        axes[i].plot(generated_data, label='Generated', color='red', linestyle='--')
        
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].set_title(f'Feature {feature_idx}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig, axes


def plot_acf_comparison(real_acf, generated_acf, feature_idx=0, lags=20, title='ACF Comparison'):
    """
    Plot autocorrelation function comparison between real and generated data.
    
    Parameters:
    -----------
    real_acf : ndarray
        Autocorrelation of real data with shape (n_features, n_lags)
    generated_acf : ndarray
        Autocorrelation of generated data with shape (n_features, n_lags)
    feature_idx : int
        Index of feature to plot
    lags : int
        Number of lags to plot
    title : str
        Plot title
        
    Returns:
    --------
    tuple
        (fig, ax) containing figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lags_range = min(lags, real_acf.shape[1], generated_acf.shape[1])
    lag_values = np.arange(lags_range)
    
    real_data = real_acf[feature_idx, :lags_range]
    generated_data = generated_acf[feature_idx, :lags_range]
    
    ax.bar(lag_values - 0.15, real_data, width=0.3, color='blue', alpha=0.6, label='Real')
    ax.bar(lag_values + 0.15, generated_data, width=0.3, color='red', alpha=0.6, label='Generated')
    
    # Add confidence intervals (assuming 95% CI for white noise)
    conf_int = 1.96 / np.sqrt(len(real_data))
    ax.axhline(y=conf_int, linestyle='--', color='gray', alpha=0.7)
    ax.axhline(y=-conf_int, linestyle='--', color='gray', alpha=0.7)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'{title} (Feature {feature_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_cross_correlation_heatmap(correlation_matrix, title='Cross-Correlation Matrix', cmap='coolwarm'):
    """
    Plot cross-correlation matrix as a heatmap.
    
    Parameters:
    -----------
    correlation_matrix : ndarray
        Correlation matrix with shape (n_features, n_features)
    title : str
        Plot title
    cmap : str
        Colormap name
        
    Returns:
    --------
    tuple
        (fig, ax) containing figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        correlation_matrix, 
        cmap=cmap,
        vmin=-1, 
        vmax=1, 
        center=0,
        annot=False,
        ax=ax
    )
    
    ax.set_title(title)
    
    return fig, ax


def plot_cross_correlation_comparison(real_corr, generated_corr, title='Cross-Correlation Comparison', 
                                     data_index=None, strikes=None, maturities=None):
    """
    Plot comparison of cross-correlation matrices.
    
    Parameters:
    -----------
    real_corr : ndarray
        Cross-correlation matrix of real data
    generated_corr : ndarray
        Cross-correlation matrix of generated data
    title : str
        Plot title
    data_index : list, optional
        List of labels for the features (e.g., ['K=0.8, M=30', 'K=0.8, M=60', ...])
        If None, will try to construct from strikes and maturities, or use numeric indices
    strikes : list, optional
        List of strike values (used to construct labels if data_index is None)
    maturities : list, optional
        List of maturity values (used to construct labels if data_index is None)
        
    Returns:
    --------
    tuple
        (fig, axes) containing figure and axes objects
    """
    # Construct labels if not provided
    if data_index is None:
        if strikes is not None and maturities is not None:
            # Construct labels from strikes and maturities
            data_index = [f'K={s}, M={m}' for s in strikes for m in maturities]
        else:
            # Use numeric indices as fallback
            n_features = real_corr.shape[0]
            data_index = [f'Feature {i}' for i in range(n_features)]
    
    # Truncate labels if needed (to avoid overcrowding)
    n_features = real_corr.shape[0]
    if len(data_index) != n_features:
        # If data_index doesn't match, use numeric indices
        data_index = [f'Feature {i}' for i in range(n_features)]
    
    # For large matrices, show only every nth label to avoid overcrowding
    if n_features > 20:
        # Show labels but rotate them and reduce frequency
        tick_step = max(1, n_features // 20)
        tick_positions = list(range(0, n_features, tick_step))
        tick_labels = [data_index[i] for i in tick_positions]
        show_all_labels = False
    else:
        tick_positions = list(range(n_features))
        tick_labels = data_index
        show_all_labels = True
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot real correlation matrix
    sns.heatmap(
        real_corr, 
        cmap='coolwarm',
        vmin=-1, 
        vmax=1, 
        center=0,
        annot=False,
        xticklabels=tick_labels if show_all_labels else False,
        yticklabels=tick_labels if show_all_labels else False,
        ax=axes[0]
    )
    if not show_all_labels:
        # Set ticks manually for large matrices
        axes[0].set_xticks(tick_positions)
        axes[0].set_xticklabels(tick_labels, rotation=90, fontsize=8)
        axes[0].set_yticks(tick_positions)
        axes[0].set_yticklabels(tick_labels, rotation=0, fontsize=8)
    else:
        # Rotate labels for readability
        axes[0].tick_params(axis='x', rotation=90, labelsize=8)
        axes[0].tick_params(axis='y', rotation=0, labelsize=8)
    axes[0].set_title('Real Data')
    
    # Plot generated correlation matrix
    sns.heatmap(
        generated_corr, 
        cmap='coolwarm',
        vmin=-1, 
        vmax=1, 
        center=0,
        annot=False,
        xticklabels=tick_labels if show_all_labels else False,
        yticklabels=tick_labels if show_all_labels else False,
        ax=axes[1]
    )
    if not show_all_labels:
        axes[1].set_xticks(tick_positions)
        axes[1].set_xticklabels(tick_labels, rotation=90, fontsize=8)
        axes[1].set_yticks(tick_positions)
        axes[1].set_yticklabels(tick_labels, rotation=0, fontsize=8)
    else:
        axes[1].tick_params(axis='x', rotation=90, labelsize=8)
        axes[1].tick_params(axis='y', rotation=0, labelsize=8)
    axes[1].set_title('Generated Data')
    
    # Plot difference
    diff = real_corr - generated_corr
    sns.heatmap(
        diff, 
        cmap='coolwarm',
        center=0,
        annot=False,
        xticklabels=tick_labels if show_all_labels else False,
        yticklabels=tick_labels if show_all_labels else False,
        ax=axes[2]
    )
    if not show_all_labels:
        axes[2].set_xticks(tick_positions)
        axes[2].set_xticklabels(tick_labels, rotation=90, fontsize=8)
        axes[2].set_yticks(tick_positions)
        axes[2].set_yticklabels(tick_labels, rotation=0, fontsize=8)
    else:
        axes[2].tick_params(axis='x', rotation=90, labelsize=8)
        axes[2].tick_params(axis='y', rotation=0, labelsize=8)
    axes[2].set_title('Difference (Real - Generated)')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig, axes


def plot_distribution_comparison(real_data, generated_data, feature_idx=0, bins=50, 
                                title='Distribution Comparison'):
    """
    Plot comparison of real and generated data distributions.
    
    Parameters:
    -----------
    real_data : ndarray
        Real data with shape (n_samples, n_features)
    generated_data : ndarray
        Generated data with shape (n_samples, n_features)
    feature_idx : int
        Index of feature to plot
    bins : int
        Number of bins for histogram
    title : str
        Plot title
        
    Returns:
    --------
    tuple
        (fig, ax) containing figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    real = real_data[:, feature_idx]
    generated = generated_data[:, feature_idx]
    
    # Find common range
    min_val = min(real.min(), generated.min())
    max_val = max(real.max(), generated.max())
    
    # Plot histograms
    sns.histplot(real, bins=bins, kde=True, stat='density', color='blue', alpha=0.5, 
                label='Real', ax=ax)
    sns.histplot(generated, bins=bins, kde=True, stat='density', color='red', alpha=0.5, 
                label='Generated', ax=ax)
    
    ax.set_xlim(min_val, max_val)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'{title} (Feature {feature_idx})')
    ax.legend()
    
    return fig, ax


def plot_qq_comparison(real_data, generated_data, feature_idx=0, title='Q-Q Plot Comparison'):
    """
    Plot Q-Q plot comparing real and generated data distributions.
    
    Parameters:
    -----------
    real_data : ndarray
        Real data with shape (n_samples, n_features)
    generated_data : ndarray
        Generated data with shape (n_samples, n_features)
    feature_idx : int
        Index of feature to plot
    title : str
        Plot title
        
    Returns:
    --------
    tuple
        (fig, ax) containing figure and axis objects
    """
    import scipy.stats as stats
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    real = real_data[:, feature_idx]
    generated = generated_data[:, feature_idx]
    
    # Create Q-Q plot
    qq = stats.probplot(generated, dist=stats.norm, sparams=(np.mean(real), np.std(real)), plot=ax)
    
    ax.set_title(f'{title} (Feature {feature_idx})')
    
    # Add reference line
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, 'r--', lw=2)
    
    return fig, ax

def plot_empirical_cdf(real_data, generated_data, feature_idx=0, title="Empirical CDF Comparison"):
    """
    Plot the empirical cumulative distribution function (CDF) of generated data against real data.

    Parameters:
    -----------
    real_data : ndarray
        Real data with shape (n_samples, n_features)
    generated_data : ndarray
        Generated data with shape (n_samples, n_features)
    feature_idx : int
        Index of feature to plot
    title : str
        Plot title

    Returns:
    --------
    tuple
        (fig, ax) containing figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    real = real_data[:, feature_idx]
    generated = generated_data[:, feature_idx]

    real_sorted = np.sort(real)
    generated_sorted = np.sort(generated)
    # The ECDF values
    real_ecdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
    generated_ecdf = np.arange(1, len(generated_sorted) + 1) / len(generated_sorted)

    ax.step(real_sorted, real_ecdf, where='post', label="Real", color="blue", lw=2)
    ax.step(generated_sorted, generated_ecdf, where='post', label="Generated", color="red", lw=2, alpha=0.75)

    ax.set_xlabel('Value')
    ax.set_ylabel('Empirical CDF')
    ax.set_title(f"{title} (Feature {feature_idx})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_volatility_clustering(real_vol_acf, generated_vol_acf, feature_idx=0, lags=10, 
                              title='Volatility Clustering Comparison'):
    """
    Plot volatility clustering comparison between real and generated data.
    
    Parameters:
    -----------
    real_vol_acf : ndarray
        Volatility autocorrelation of real data
    generated_vol_acf : ndarray
        Volatility autocorrelation of generated data
    feature_idx : int
        Index of feature to plot
    lags : int
        Number of lags to plot
    title : str
        Plot title
        
    Returns:
    --------
    tuple
        (fig, ax) containing figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lags_range = min(lags, real_vol_acf.shape[1], generated_vol_acf.shape[1])
    lag_values = np.arange(lags_range)
    
    real_data = real_vol_acf[feature_idx, :lags_range]
    generated_data = generated_vol_acf[feature_idx, :lags_range]
    
    ax.bar(lag_values - 0.15, real_data, width=0.3, color='blue', alpha=0.6, label='Real')
    ax.bar(lag_values + 0.15, generated_data, width=0.3, color='red', alpha=0.6, label='Generated')
    
    # Add confidence intervals (assuming 95% CI for white noise)
    conf_int = 1.96 / np.sqrt(len(real_data))
    ax.axhline(y=conf_int, linestyle='--', color='gray', alpha=0.7)
    ax.axhline(y=-conf_int, linestyle='--', color='gray', alpha=0.7)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation of Squared Returns')
    ax.set_title(f'{title} (Feature {feature_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def create_evaluation_dashboard(evaluation_results, real_data, generated_data, 
                               feature_indices=None, strikes=None, maturities=None, save_path=None):
    """
    Create a comprehensive evaluation dashboard with multiple plots.
    
    Parameters:
    -----------
    evaluation_results : dict
        Dictionary containing evaluation metrics
    real_data : ndarray
        Real data with shape (n_samples, n_features)
    generated_data : ndarray
        Generated data with shape (n_samples, n_features)
    feature_indices : list, optional
        List of feature indices to plot (defaults to first 4 features)
    data_index : list, optional
        List of data indices that correspond to the real and generated data (defaults to numeric indices)
    strikes : list, optional
        List of strike values (used to construct feature labels for correlation plots)
    maturities : list, optional
        List of maturity values (used to construct feature labels for correlation plots)
    save_path : str, optional
        Path to save the dashboard
        
    Returns:
    --------
    dict
        Dictionary containing all figure objects
    """
    if feature_indices is None:
        feature_indices = list(range(min(4, real_data.shape[1])))
    
    figures = {}
    
    # Time series comparison
    figures['time_series'] = plot_real_vs_generated_grid(
        real_data, generated_data, features=feature_indices,
        title='Real vs Generated Time Series Comparison'
    )[0]
    
    # Construct feature labels for correlation plots if strikes and maturities are provided
    feature_labels = None
    if strikes is not None and maturities is not None:
        feature_labels = [f'K={s}, M={m}' for s in strikes for m in maturities]
        # Ensure the length matches the number of features
        if len(feature_labels) != real_data.shape[1]:
            feature_labels = None  # Use default if mismatch
    
    for feature_idx in feature_indices:
        figures[f'distribution_{feature_idx}'] = plot_distribution_comparison(
            real_data, generated_data, feature_idx=feature_idx,
            title=f'Distribution Comparison'
        )[0]
        figures[f'qq_plot_{feature_idx}'] = plot_qq_comparison(
            real_data, generated_data, feature_idx=feature_idx,
            title=f'Q-Q Plot Comparison'
        )[0]

        figures[f'acf_{feature_idx}'] = plot_acf_comparison(
            evaluation_results['real_acf'], evaluation_results['gen_acf'],
            feature_idx=feature_idx, title=f'Autocorrelation Comparison'
        )[0]
        
        # Cross-correlation comparison with feature labels
        figures[f'cross_corr_{feature_idx}'] = plot_cross_correlation_comparison(
            evaluation_results['real_corr'], evaluation_results['gen_corr'],
            title='Cross-Correlation Comparison',
            data_index=feature_labels,
            strikes=strikes,
            maturities=maturities
        )[0]
        
        figures[f'vol_clustering_{feature_idx}'] = plot_volatility_clustering(
            evaluation_results['real_vol_acf'], evaluation_results['gen_vol_acf'],
            feature_idx=feature_idx, title=f'Volatility Clustering Comparison'
        )[0]

        figures[f'empirical_cdf_{feature_idx}'] = plot_empirical_cdf(
            real_data, generated_data, feature_idx=feature_idx, title=f'Empirical CDF Comparison'
        )[0]
        
    # Save dashboard plots if path is provided
    if save_path:
        for name, fig in figures.items():
            fig.savefig(f"{save_path}_{name}.png", dpi=300, bbox_inches='tight')
    
    return figures


def plot_gan_losses(history, model_name, plot_dir):
    """
    Plot discriminator and generator losses by epoch.
    
    Parameters:
    -----------
    history : dict
        Training history dictionary containing 'metrics' with 'gen_loss' and 'disc_loss'
    model_name : str
        Name of the model for the plot title
    plot_dir : str
        Directory to save the plot
    """
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    if 'metrics' not in history:
        logger.warning(f"No metrics found in history for {model_name}")
        return
    
    metrics = history['metrics']
    
    # Check if losses exist
    if 'gen_loss' not in metrics or 'disc_loss' not in metrics:
        logger.warning(f"gen_loss or disc_loss not found in history for {model_name}")
        return
    
    gen_losses = metrics['gen_loss']
    disc_losses = metrics['disc_loss']
    
    if not gen_losses or not disc_losses:
        logger.warning(f"Empty loss arrays for {model_name}")
        return
    
    epochs = range(1, len(gen_losses) + 1)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot generator loss
    ax1.plot(epochs, gen_losses, 'b-', label='Generator Loss', linewidth=2)
    if 'val_gen_loss' in metrics and metrics['val_gen_loss']:
        ax1.plot(epochs, metrics['val_gen_loss'], 'b--', label='Generator Val Loss', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Generator Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Generator Loss by Epoch', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=1)
    
    # Plot discriminator loss
    ax2.plot(epochs, disc_losses, 'r-', label='Discriminator Loss', linewidth=2)
    if 'val_disc_loss' in metrics and metrics['val_disc_loss']:
        ax2.plot(epochs, metrics['val_disc_loss'], 'r--', label='Discriminator Val Loss', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Discriminator Loss', fontsize=12)
    ax2.set_title(f'{model_name} - Discriminator Loss by Epoch', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=1)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plot_dir, f"{model_name.lower().replace('-', '_')}_losses.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved GAN loss plot to {plot_path}")
    
    # Also create a combined plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, gen_losses, 'b-', label='Generator Loss', linewidth=2)
    ax.plot(epochs, disc_losses, 'r-', label='Discriminator Loss', linewidth=2)
    if 'val_gen_loss' in metrics and metrics['val_gen_loss']:
        ax.plot(epochs, metrics['val_gen_loss'], 'b--', label='Generator Val Loss', linewidth=2, alpha=0.7)
    if 'val_disc_loss' in metrics and metrics['val_disc_loss']:
        ax.plot(epochs, metrics['val_disc_loss'], 'r--', label='Discriminator Val Loss', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{model_name} - Generator and Discriminator Losses by Epoch', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)
    
    plt.tight_layout()
    
    # Save combined plot
    plot_path_combined = os.path.join(plot_dir, f"{model_name.lower().replace('-', '_')}_losses_combined.png")
    plt.savefig(plot_path_combined, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved combined GAN loss plot to {plot_path_combined}")


def summarize_evaluation_metrics(evaluation_results_dict):
    """
    Summarize evaluation metrics from multiple models.
    
    Parameters:
    -----------
    evaluation_results_dict : dict
        Dictionary mapping model names to their evaluation results dictionary.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame comparing metrics across models.
    """
    all_metrics = {}
    
    for model_name, results in evaluation_results_dict.items():
        
        # Helper function to safely get and convert metric from the current model's results
        def get_metric(key, default=np.nan):
            value = results.get(key, default) # Access the inner 'results' dict
            try:
                if np.isscalar(value):
                    return float(value)
                elif isinstance(value, (float, int)) or np.isnan(value):
                    return value
            except (TypeError, ValueError):
                pass
            return default

        metrics = {
            'MSE': get_metric('mse'),
            'MAE': get_metric('mae'),
            'Wasserstein': get_metric('wasserstein'),
            'KL Divergence': get_metric('kl_divergence'),
            'JS Divergence': get_metric('js_divergence'),
            'ACF MSE': get_metric('acf_mse'),
            'Cross-Correlation MSE': get_metric('corr_mse'),
            'Volatility ACF MSE': get_metric('vol_mse')
        }
        
        # Handle optional metric separately
        # metrics['Arbitrage Violation Rate'] = get_metric('violation_rate')
        
        all_metrics[model_name] = metrics # Store metrics under model name
    
    # Convert the collected metrics for all models to DataFrame
    df = pd.DataFrame(all_metrics)
    return df.astype(float) # Ensure final dtype is float 