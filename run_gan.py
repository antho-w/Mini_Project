import os
import datetime as dt
import pandas as pd
import numpy as np
import logging
# import tensorflow as tf
from dateutil.relativedelta import relativedelta

from utils.helpers import (
    make_dir_if_not_exists,
    create_sequences
)
from data_classes import (
    DataRetriever,
    DataProcessor,
    DataCache
)
# from model_classes.dim_reducer import DimensionReducer
# from model_classes.lstm_gan import LSTMGAN
# from model_classes.tcn_gan import TCNGAN

from evaluation.metrics import evaluate_model
from evaluation.visualization import create_evaluation_dashboard, summarize_evaluation_metrics

if __name__ == "__main__":
    
    current_wd = os.getcwd()
    folder_dir = os.path.join(current_wd, "output", dt.datetime.now().strftime("%Y%m%d"))
    DATA_DIR = os.path.join(folder_dir, "data")
    LOG_DIR = os.path.join(folder_dir, "logs")


    # Create output directories if they don't exist
    make_dir_if_not_exists(os.path.join(folder_dir))
    make_dir_if_not_exists(DATA_DIR)
    make_dir_if_not_exists(LOG_DIR)
    
    # Set up logging to both console and file
    log_filename = os.path.join(LOG_DIR, f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Console output
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")

    TICKER = "^NDX" # NASDAQ-100 Ticker
    OPT_DATA_DIR = os.path.join(current_wd, "raw_data/full_yearly_data")
    N_YEARS = 10
    END_DATE = '2025-10-31'
    _end_dt = dt.datetime.strptime(END_DATE, "%Y-%m-%d")
    START_DATE = ( _end_dt - relativedelta(years=N_YEARS)).strftime("%Y-%m-%d")
    SAVE_DATA = True

    # Parameters for data filtering and processing
    TIME_STEPS = 1000 # Number of time steps to consider
    # STRIKES = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    STRIKES = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    MATURITIES = [30, 60, 90, 120, 150, 180, 360]
    
    # Parameters for Training sets
    SEQUENCE_LENGTH = 10 # Length of sequences for training must be > TIME_STEPS
    BATCH_SIZE = 32
    TRAIN_RATIO = 0.7

    # Parameters for GAN
    NOISE_DIM = 32
    TRAIN_LSTM_GAN = True
    TRAIN_TCN_GAN = False
    # TRAIN_WGAN = False
    EPOCHS = 10
    LEARNING_RATE = 1e-5
    BETA_PARAM = 0.3
    GENERATE_LENGTH = 100


    STAGES_CONFIG = {
        # Stage 1: Get data
        "READ_AND_CLEAN_DATA": True,
        # Stage 2: Filter data and generate DLVs
        "TRANSFORM_DATA": True,
        # Stage 3: Apply PCA
        "APPLY_PCA": True,
        # Stage 4: Prepare GAN datasets
        "PREPARE_DATA": True,
        # Stage 5: Train GAN
        "TRAIN_GAN": True,
        # Stage 6: Simulate and evaluate
        "SIMULATE_AND_EVALUATE": True
    }
    
    if STAGES_CONFIG["READ_AND_CLEAN_DATA"]:
        logger.info(f"Retreiving data for {TICKER} from {START_DATE} to {END_DATE}")
        data_fetcher = DataRetriever(DATA_DIR, TICKER, 
                                    start_date=dt.datetime.strptime(START_DATE, "%Y-%m-%d"), 
                                    end_date=dt.datetime.strptime(END_DATE, "%Y-%m-%d")
                                    )
        
        # Getting price data for stock
        try:
            df = data_fetcher.get_price_data()
            if SAVE_DATA:
                df.to_csv(os.path.join(DATA_DIR, f"{TICKER}_stock_data.csv"), index = True)
        except Exception as e:
            logger.error(f"Error occured when fetching underlying data: {e}")

        logger.info(f"Reading options data from {OPT_DATA_DIR}")

        

        date_range = pd.bdate_range(start=START_DATE, end=END_DATE)

        curr_year = pd.to_datetime(START_DATE).year
        year_file_data = pd.read_csv(os.path.join(OPT_DATA_DIR, f"{curr_year}.csv"))
        
        # Initialize data cache for saving
        data_cache = DataCache(DATA_DIR, TICKER) if SAVE_DATA else None
        
        # Filter date_range to only include dates that exist in underlying_data
        # This prevents KeyError when accessing dates that don't have price data
        available_dates = set(data_fetcher.underlying_data.index)
        date_range = [date for date in date_range if date.date() in available_dates]
        
        if not date_range:
            logger.error(f"No overlapping dates between date_range and underlying_data. "
                        f"Date range: {START_DATE} to {END_DATE}, "
                        f"Available dates in underlying_data: {min(available_dates)} to {max(available_dates)}")
            raise ValueError("No valid dates found in underlying_data for the specified date range")
        
        logger.info(f"Processing {len(date_range)} dates (filtered from original range)")
        
        for date in date_range:
            
            if date.year != curr_year:
                curr_year = date.year
                year_file_data = pd.read_csv(os.path.join(OPT_DATA_DIR, f"{curr_year}.csv"))
                
            mask = pd.to_datetime(year_file_data['obs_date']).dt.date == date.date()
            options_chain_df = year_file_data.loc[mask]

            if options_chain_df.empty:
                logger.warning(f"No options data for date {date} found in {curr_year}.csv")
                continue
            else:
                data_fetcher.get_options_chain(options_chain_df, SAVE_DATA)
                logger.info(f"Finished creating options chain for date: {date}")

            data_processor = DataProcessor(date, data_fetcher.underlying_data, data_fetcher.options_chain)
            option_prices, implied_vols, strike_grid, time_grid, relative_strike_grid, volume_array, option_types = data_processor.clean_and_process_data()
            logger.info(f"Finished cleaning and calculating implied volatilities from closing levels")

            logger.info(f"Filtering and interpolating implied volatilities for data by strikes and maturities")
            option_prices, implied_vols, strike_grid, time_grid, relative_strike_grid, volume_array, option_types = data_processor.filter_by_strikes_and_maturities(STRIKES, MATURITIES)

            # Get current price for this date (should exist since we filtered date_range)
            date_key = date.date()
            if date_key not in data_processor.underlying_data.index:
                logger.warning(f"Date {date_key} not found in underlying_data, skipping...")
                continue
            current_price = data_processor.underlying_data.loc[date_key]['Close']
            
            # Add data to cache (will be saved automatically when year changes)
            if SAVE_DATA and data_cache is not None:
                data_cache.add_date_data(
                    date=date.date(),
                    current_price=current_price,
                    option_prices=option_prices,
                    implied_vols=implied_vols,
                    strike_grid=strike_grid,
                    time_grid=time_grid,
                    relative_strike_grid=relative_strike_grid,
                    volume_array=volume_array,
                    option_types=option_types
                )
        
        # Save data for the last year
        if SAVE_DATA and data_cache is not None:
            data_cache.finalize()
        


    if STAGES_CONFIG["APPLY_PCA"]:
        logger.info(f"Applying PCA to log-implied volatilities series")
        
        log_dlv_series_reshaped = log_dlv_series.reshape(TIME_STEPS, -1)
        
        # Apply PCA
        pca_reducer =DimensionReducer(N_COMPONENTS)
        pca_components = pca_reducer.fit_transform(log_dlv_series_reshaped)

        logger.info(f"PCA explained variance: {pca_reducer.explained_variance_ratio}")
        logger.info(f"Cumulative explained variance: {pca_reducer.cumulative_explained_variance}")

    if STAGES_CONFIG["PREPARE_DATA"]:
        logger.info(f"Preparing training data with sequence length {SEQUENCE_LENGTH}...")
        
        # Create sequences
        X, y = create_sequences(pca_components, sequence_length=SEQUENCE_LENGTH, target_steps=1)
        
        # Create training and validation sets
        train_size = int(TRAIN_RATIO * len(X))
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]
        
        # Create TensorFlow datasets
        buffer_size = min(1000, len(X_train))
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size).batch(BATCH_SIZE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(BATCH_SIZE)

        logger.info(f"Created training dataset with {len(X_train)} sequences")
        logger.info(f"Created validation dataset with {len(X_val)} sequences")


    if STAGES_CONFIG["TRAIN_GAN"]:
        # Define state, noise, and output dimensions
        state_dim = (None, SEQUENCE_LENGTH, N_COMPONENTS)
        noise_dim = NOISE_DIM
        output_dim = N_COMPONENTS
        
        models = {}

        make_dir_if_not_exists(os.path.join(current_wd, "logs"))
        log_dir = os.path.join(current_wd, "logs", dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        if TRAIN_LSTM_GAN:
            logger.info(f"Training LSTM-GAN model with {EPOCHS} epochs")

            lstm_gan = LSTMGAN(
                state_dim=state_dim,
                noise_dim=noise_dim,
                output_dim=output_dim,
                generator_units=[64, 128],
                discriminator_units=[128, 64],
                use_pca=True,
                n_pca_components=output_dim,
                log_dir=log_dir
            )
            
            # Compile model
            lstm_gan.compile(
                generator_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_PARAM),
                discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_PARAM)
            )
            
            # Train model
            history = lstm_gan.train(train_dataset, validation_dataset=val_dataset, epochs=EPOCHS, verbose=1)
            models['LSTM-GAN'] = lstm_gan

        if TRAIN_TCN_GAN:
            logger.info(f"Training TCN-GAN model with {EPOCHS} epochs")
            tcn_gan = TCNGAN(
                state_dim=state_dim,
                noise_dim=noise_dim,
                output_dim=output_dim,
                generator_filters=[64, 128, 64],
                discriminator_filters=[64, 128, 64],
                kernel_size=3,
                use_pca=True,
                n_pca_components=output_dim,
                log_dir=log_dir
            )
            
            # Compile model
            tcn_gan.compile(
                generator_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_PARAM),
                discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_PARAM)
            )
            
            # Train model
            history = tcn_gan.train(train_dataset, validation_dataset=val_dataset, epochs=EPOCHS, verbose=1)
            models['TCN-GAN'] = tcn_gan
        
        pass

    if STAGES_CONFIG["SIMULATE_AND_EVALUATE"] and models:
        logger.info(f"Generating synthetic data and evaluating models")
        
        # Ensure plots directory exists
        plots_dir = os.path.join(current_wd, 'plots', dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
        make_dir_if_not_exists(plots_dir)
        initial_state = X_val[0:1]

        # Initialize results dictionaries
        generated_sequences = {}
        evaluation_results = {}
        all_figures = {}
        
        n_strikes = len(STRIKES)
        n_maturities = len(MATURITIES)

        # Get a subset of real data for comparison
        real_log_dlvs_subset = log_dlv_series[:SEQUENCE_LENGTH]
        real_log_dlvs_flat = real_log_dlvs_subset.reshape(SEQUENCE_LENGTH, -1)
        
        # Generate and evaluate for each model        
        for model_name, model in models.items():
            logger.info(f"Generating and evaluating {model_name}...")
            
            # Generate PCA components
            generated_sequence = model.generate_sequences(
                initial_state, 
                sequence_length=SEQUENCE_LENGTH, 
                use_generated_state=True
            )
            
            # Transform PCA components back to log-DLVs
            generated_log_dlvs = pca_reducer.inverse_transform(
                generated_sequence, 
                original_shape=(SEQUENCE_LENGTH, n_strikes, n_maturities)
            )
            
            # Reshape for evaluation
            generated_log_dlvs_flat = generated_log_dlvs.reshape(SEQUENCE_LENGTH, -1)
            
            # --- DEBUG LOGGING ---
            logger.debug(f"Debugging inputs for evaluate_model ({model_name})")
            logger.debug(f"Real data shape: {real_log_dlvs_flat.shape}")
            logger.debug(f"Generated data shape: {generated_log_dlvs_flat.shape}")
            logger.debug(f"Real data contains NaN: {np.isnan(real_log_dlvs_flat).any()}")
            logger.debug(f"Generated data contains NaN: {np.isnan(generated_log_dlvs_flat).any()}")
            logger.debug(f"Real data column variances: {np.var(real_log_dlvs_flat, axis=0)}")
            logger.debug(f"Generated data column variances: {np.var(generated_log_dlvs_flat, axis=0)}")
            # --- END DEBUG LOGGING ---

            # Evaluate the model
            results = evaluate_model(
                real_log_dlvs_flat, 
                generated_log_dlvs_flat,
                is_dlv_surface=True,
                n_strikes=n_strikes,
                n_maturities=n_maturities
            )
        
            # Create evaluation dashboard
            figures = create_evaluation_dashboard(
                results,
                real_log_dlvs_flat,
                generated_log_dlvs_flat,
                save_path=os.path.join(plots_dir, model_name.lower())
            )
        
            # Store results
            generated_sequences[model_name] = generated_log_dlvs
            evaluation_results[model_name] = results
            all_figures[model_name] = figures
    
    # Compare models
    comparison_df = summarize_evaluation_metrics(evaluation_results)
    logger.info("\nModel Comparison:")
    logger.info(f"\n{comparison_df}")
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(plots_dir, 'model_comparison.csv'))


