import os
import datetime as dt
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from dateutil.relativedelta import relativedelta

from utils.helpers import (
    make_dir_if_not_exists,
    create_sequences,
    remove_nan_values_nadaraya_watson
)
from data_classes import (
    DataRetriever,
    DataProcessor,
    DataCache
)

from model_classes.dim_reducer import DimensionReducer
from model_classes.lstm_gan import LSTMGAN
from model_classes.tcn_gan import TCNGAN

from transforms import log_transform, bs_price

from evaluation.metrics import evaluate_model
from evaluation.visualization import create_evaluation_dashboard, summarize_evaluation_metrics, plot_gan_losses


if __name__ == "__main__":
    # Set TensorFlow to use deterministic operations (may impact performance)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    current_wd = os.getcwd()
    folder_dir = os.path.join(current_wd, "output", "20251119")
    DATA_DIR = os.path.join(folder_dir, "data")
    LOG_DIR = os.path.join(folder_dir, "logs")
    PLOT_DIR = os.path.join(DATA_DIR, "plots")
    LOG_LEVEL = logging.INFO

    # Create output directories if they don't exist
    make_dir_if_not_exists(os.path.join(folder_dir))
    make_dir_if_not_exists(DATA_DIR)
    make_dir_if_not_exists(LOG_DIR)
    make_dir_if_not_exists(PLOT_DIR)

    # Set up logging to both console and file
    log_filename = os.path.join(LOG_DIR, f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=LOG_LEVEL,
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
    # STRIKES = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    STRIKES = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    MATURITIES = [30, 60, 90, 120, 150, 180, 360]
    
    # Parameters for PCA
    N_COMPONENTS = 5

    # Parameters for Training sets
    SEQUENCE_LENGTH = 1000
    BATCH_SIZE = 64
    TRAIN_RATIO = 0.85

    # Parameters for GAN
    NOISE_DIM = 64 # 32d
    TRAIN_LSTM_GAN = True
    TRAIN_TCN_GAN = False
    EPOCHS = 250 # 75
    LEARNING_RATE_DES = 3e-4 # 3e-4
    LEARNING_RATE_GEN = 5e-5 # 5e-5
    BETA_PARAM = 0.3 # 0.3
    GENERATE_LENGTH = 252
    RANDOM_SEED = 10
    # Gradient clipping parameters to reduce validation loss volatility
    GEN_GRAD_CLIP = 0.5  # Maximum norm for generator gradient clipping
    DISC_GRAD_CLIP = 1.0  # Maximum norm for discriminator gradient clipping

    # Set random seeds for reproducibility
    logger.info(f"Random seed: {RANDOM_SEED}")
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    import random
    random.seed(RANDOM_SEED)

    # Log all hyperparameters for this run
    logger.info("=" * 80)
    logger.info("HYPERPARAMETERS FOR THIS RUN")
    logger.info("=" * 80)
    logger.info("Data Parameters:")
    logger.info(f"  TICKER: {TICKER}")
    logger.info(f"  START_DATE: {START_DATE}")
    logger.info(f"  END_DATE: {END_DATE}")
    logger.info(f"  N_YEARS: {N_YEARS}")
    logger.info(f"  STRIKES: {STRIKES}")
    logger.info(f"  MATURITIES: {MATURITIES}")
    logger.info("")
    logger.info("PCA Parameters:")
    logger.info(f"  N_COMPONENTS: {N_COMPONENTS}")
    logger.info("")
    logger.info("Data Preparation Parameters:")
    logger.info(f"  SEQUENCE_LENGTH: {SEQUENCE_LENGTH}")
    logger.info(f"  BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"  TRAIN_RATIO: {TRAIN_RATIO}")
    logger.info("")
    logger.info("GAN Training Parameters:")
    logger.info(f"  NOISE_DIM: {NOISE_DIM}")
    logger.info(f"  EPOCHS: {EPOCHS}")
    logger.info(f"  LEARNING_RATE_GEN: {LEARNING_RATE_GEN}")
    logger.info(f"  LEARNING_RATE_DES: {LEARNING_RATE_DES}")
    logger.info(f"  BETA_PARAM: {BETA_PARAM}")
    logger.info(f"  GEN_GRAD_CLIP: {GEN_GRAD_CLIP}")
    logger.info(f"  DISC_GRAD_CLIP: {DISC_GRAD_CLIP}")
    logger.info(f"  RANDOM_SEED: {RANDOM_SEED}")
    logger.info("")
    logger.info("Generation Parameters:")
    logger.info(f"  GENERATE_LENGTH: {GENERATE_LENGTH}")
    logger.info("")
    logger.info("Model Training Flags:")
    logger.info(f"  TRAIN_LSTM_GAN: {TRAIN_LSTM_GAN}")
    logger.info(f"  TRAIN_TCN_GAN: {TRAIN_TCN_GAN}")
    logger.info("=" * 80)
    logger.info("")

    STAGES_CONFIG = {
        # Stage 1: Get data
        "READ_AND_CLEAN_DATA": False,
        # Stage 2: Filter data and generate IVs
        "TRANSFORM_DATA": True,
        # Stage 3: Apply PCA
        "APPLY_PCA": True,
        # Stage 4: Prepare GAN datasets
        "PREPARE_DATA": True,
        # Stage 5: Train GAN
        "TRAIN_GAN": False,
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

        # Initialise data processor by reading from output directory
        if not STAGES_CONFIG["READ_AND_CLEAN_DATA"]:
            data_cache = DataCache(DATA_DIR, TICKER)
            data_cache.load_implied_vols_from_data_dir(START_DATE, END_DATE)

        logger.info(f"Applying PCA to log-implied volatilities series")

        implied_vols_array = data_cache.get_implied_vols_array()
        
        # Remove any remaining NaN values using Nadaraya-Watson smoothing for each time slice
        implied_vols_array = remove_nan_values_nadaraya_watson(implied_vols_array, STRIKES, MATURITIES)

        # Transform into log-implied volatility series
        log_implied_vols_series = log_transform(implied_vols_array)
        time_steps = log_implied_vols_series.shape[0]

        # Flatten the log-implied volatility series to (time_steps, n_strikes * n_maturities) for PCA
        log_implied_vols_series_reshaped = log_implied_vols_series.reshape(time_steps, -1)
        logger.info(f"Flattened Log-implied volatility series, shape is now: {log_implied_vols_series_reshaped.shape}")

        # Apply PCA
        pca_reducer = DimensionReducer(N_COMPONENTS)
        pca_components = pca_reducer.fit_transform(log_implied_vols_series_reshaped)

        logger.info(f"PCA explained variance: {pca_reducer.explained_variance_ratio}")
        logger.info(f"Cumulative explained variance: {pca_reducer.cumulative_explained_variance}")

        # Plot explained variance
        pca_reducer.plot_explained_variance()
        plt.savefig(os.path.join(PLOT_DIR, "explained_variance.png"))
        plt.close()

        # Plot component weights
        for i in range(N_COMPONENTS):
            fig, ax = pca_reducer.plot_component_weights(component_idx=i, n_strikes=len(STRIKES), n_maturities=len(MATURITIES))
            plt.savefig(os.path.join(PLOT_DIR, f"component_weights_{i}.png"))
            plt.close()

        if SAVE_DATA:
            pca_reducer.save(os.path.join(DATA_DIR, "pca_reducer.pkl"))
            logger.info(f"PCA model saved to {os.path.join(DATA_DIR, "pca_reducer.pkl")}")

    if STAGES_CONFIG["PREPARE_DATA"]:
        from sklearn.preprocessing import StandardScaler
        logger.info(f"Preparing training data with sequence length {SEQUENCE_LENGTH}...")
        
        # Create sequences
        X, y = create_sequences(pca_components, sequence_length=SEQUENCE_LENGTH, target_steps=1)

        # Create training and validation sets
        train_size = int(TRAIN_RATIO * len(X))
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]
        
        # Normalize training and validation data
        logger.info("Normalizing training and validation data...")
        
        # Reshape data for normalization (flatten sequences and features)
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        y_train_flat = y_train.reshape(-1, y_train.shape[-1])
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        y_val_flat = y_val.reshape(-1, y_val.shape[-1])
        
        # Fit scaler on training data only
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled_flat = scaler_X.fit_transform(X_train_flat)
        y_train_scaled_flat = scaler_y.fit_transform(y_train_flat)
        
        # Transform validation data using training scalers
        X_val_scaled_flat = scaler_X.transform(X_val_flat)
        y_val_scaled_flat = scaler_y.transform(y_val_flat)
        
        # Reshape back to original sequence structure
        X_train = X_train_scaled_flat.reshape(X_train.shape)
        y_train = y_train_scaled_flat.reshape(y_train.shape)
        X_val = X_val_scaled_flat.reshape(X_val.shape)
        y_val = y_val_scaled_flat.reshape(y_val.shape)
        
        logger.info(f"Normalization complete. Training data mean: {np.mean(X_train):.4f}, std: {np.std(X_train):.4f}")
        logger.info(f"Target data mean: {np.mean(y_train):.4f}, std: {np.std(y_train):.4f}")
        
        # Save scalers for denormalization during generation
        if SAVE_DATA:
            import pickle
            scaler_path_X = os.path.join(DATA_DIR, "scaler_X.pkl")
            scaler_path_y = os.path.join(DATA_DIR, "scaler_y.pkl")
            with open(scaler_path_X, 'wb') as f:
                pickle.dump(scaler_X, f)
            with open(scaler_path_y, 'wb') as f:
                pickle.dump(scaler_y, f)
            logger.info(f"Saved scalers to {DATA_DIR}")
        
        # Create TensorFlow datasets
        buffer_size = min(1000, len(X_train))
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size, seed=RANDOM_SEED).batch(BATCH_SIZE)

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

        if TRAIN_LSTM_GAN:
            logger.info("")
            logger.info("=" * 80)
            logger.info("LSTM-GAN MODEL CONFIGURATION")
            logger.info("=" * 80)
            logger.info(f"Training LSTM-GAN model with {EPOCHS} epochs")
            logger.info("")
            logger.info("Model Architecture:")
            logger.info(f"  state_dim: {state_dim}")
            logger.info(f"  noise_dim: {noise_dim}")
            logger.info(f"  output_dim: {output_dim}")
            logger.info(f"  generator_units: [64, 128]")
            logger.info(f"  discriminator_units: [128, 64]")
            logger.info(f"  use_pca: True")
            logger.info(f"  n_pca_components: {output_dim}")
            logger.info("")
            logger.info("Training Hyperparameters:")
            logger.info(f"  Learning Rate (Generator): {LEARNING_RATE_GEN}")
            logger.info(f"  Learning Rate (Discriminator): {LEARNING_RATE_DES}")
            logger.info(f"  Beta (Adam optimizer): {BETA_PARAM}")
            logger.info(f"  Generator Gradient Clipping: {GEN_GRAD_CLIP}")
            logger.info(f"  Discriminator Gradient Clipping: {DISC_GRAD_CLIP}")
            logger.info(f"  Random Seed: {RANDOM_SEED}")
            logger.info("=" * 80)
            logger.info("")

            lstm_gan = LSTMGAN(
                state_dim=state_dim,
                noise_dim=noise_dim,
                output_dim=output_dim,
                generator_units=[64, 128],
                discriminator_units=[128, 64],
                use_pca=True,
                n_pca_components=output_dim,
                random_seed=RANDOM_SEED,
                gen_grad_clip=GEN_GRAD_CLIP,
                disc_grad_clip=DISC_GRAD_CLIP,
                log_dir=LOG_DIR
            )
            
            # Compile model
            lstm_gan.compile(
                generator_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_GEN, beta_1=BETA_PARAM),
                discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_DES, beta_1=BETA_PARAM)
            )
            
            # Train model
            history = lstm_gan.train(train_dataset, validation_dataset=val_dataset, epochs=EPOCHS, verbose=1)
            models['LSTM-GAN'] = lstm_gan
            
            # Save model to DATA_DIR
            model_save_path = os.path.join(DATA_DIR, 'lstm_gan_model')
            lstm_gan.save(model_save_path)
            logger.info(f"LSTM-GAN model saved to {model_save_path}")
            
            # Plot discriminator and generator losses
            plot_gan_losses(history, 'LSTM-GAN', PLOT_DIR)


        if TRAIN_TCN_GAN:
            logger.info("")
            logger.info("=" * 80)
            logger.info("TCN-GAN MODEL CONFIGURATION")
            logger.info("=" * 80)
            logger.info(f"Training TCN-GAN model with {EPOCHS} epochs")
            logger.info("")
            logger.info("Model Architecture:")
            logger.info(f"  state_dim: {state_dim}")
            logger.info(f"  noise_dim: {noise_dim}")
            logger.info(f"  output_dim: {output_dim}")
            logger.info(f"  generator_filters: [64, 128, 64]")
            logger.info(f"  discriminator_filters: [64, 128, 64]")
            logger.info(f"  kernel_size: 3")
            logger.info(f"  use_pca: True")
            logger.info(f"  n_pca_components: {output_dim}")
            logger.info("")
            logger.info("Training Hyperparameters:")
            logger.info(f"  Learning Rate (Generator): {LEARNING_RATE_GEN}")
            logger.info(f"  Learning Rate (Discriminator): {LEARNING_RATE_DES}")
            logger.info(f"  Beta (Adam optimizer): {BETA_PARAM}")
            logger.info(f"  Random Seed: {RANDOM_SEED}")
            logger.info("=" * 80)
            logger.info("")
            
            tcn_gan = TCNGAN(
                state_dim=state_dim,
                noise_dim=noise_dim,
                output_dim=output_dim,
                generator_filters=[64, 128, 64],
                discriminator_filters=[64, 128, 64],
                kernel_size=3,
                use_pca=True,
                n_pca_components=output_dim,
                log_dir=LOG_DIR
            )
            
            # Compile model
            tcn_gan.compile(
                generator_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_PARAM),
                discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_PARAM)
            )
            
            # Train model
            history = tcn_gan.train(train_dataset, validation_dataset=val_dataset, epochs=EPOCHS, verbose=1)
            models['TCN-GAN'] = tcn_gan
            
            # Save model to DATA_DIR
            model_save_path = os.path.join(DATA_DIR, 'tcn_gan_model')
            tcn_gan.save(model_save_path)
            logger.info(f"TCN-GAN model saved to {model_save_path}")
            
            # Plot discriminator and generator losses
            plot_gan_losses(history, 'TCN-GAN', PLOT_DIR)
            plt.savefig(os.path.join(PLOT_DIR, "tcn_gan_losses.png"))
            plt.close()

    else:
        # Load models from DATA_DIR if they exist
        logger.info(f"Loading models from {DATA_DIR}")
        models = {}
        
        # Define state, noise, and output dimensions (needed for model initialization)
        state_dim = (None, SEQUENCE_LENGTH, N_COMPONENTS)
        noise_dim = NOISE_DIM
        output_dim = N_COMPONENTS
        
        # Load LSTM-GAN if it exists
        lstm_gan_path = os.path.join(DATA_DIR, 'lstm_gan_model')
        try:
            logger.info(f"Loading LSTM-GAN from {lstm_gan_path}")
            lstm_gan = LSTMGAN(
                state_dim=state_dim,
                noise_dim=noise_dim,
                output_dim=output_dim,
                generator_units=[64, 128],
                discriminator_units=[128, 64],
                use_pca=True,
                n_pca_components=output_dim,
                random_seed=RANDOM_SEED,
                gen_grad_clip=GEN_GRAD_CLIP,
                disc_grad_clip=DISC_GRAD_CLIP,
                log_dir=LOG_DIR
            )
            lstm_gan.load(lstm_gan_path)
            # Compile with dummy optimizers (needed for model to work, but not used for inference)
            lstm_gan.compile(
                generator_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_GEN, beta_1=BETA_PARAM),
                discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_DES, beta_1=BETA_PARAM)
            )
            models['LSTM-GAN'] = lstm_gan
            logger.info("LSTM-GAN loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LSTM-GAN: {e}")
        
        # Load TCN-GAN if it exists
        # tcn_gan_path = os.path.join(DATA_DIR, 'tcn_gan_model')
        # if os.path.exists(tcn_gan_path):
        #     try:
        #         logger.info(f"Loading TCN-GAN from {tcn_gan_path}")
        #         tcn_gan = TCNGAN(
        #             state_dim=state_dim,
        #             noise_dim=noise_dim,
        #             output_dim=output_dim,
        #             generator_filters=[64, 128, 64],
        #             discriminator_filters=[64, 128, 64],
        #             kernel_size=3,
        #             use_pca=True,
        #             n_pca_components=output_dim,
        #             log_dir=LOG_DIR
        #         )
        #         tcn_gan.load(tcn_gan_path)
        #         # Compile with dummy optimizers (needed for model to work, but not used for inference)
        #         tcn_gan.compile(
        #             generator_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_PARAM),
        #             discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_PARAM)
        #         )
        #         models['TCN-GAN'] = tcn_gan
        #         logger.info("TCN-GAN loaded successfully")
        #     except Exception as e:
        #         logger.error(f"Failed to load TCN-GAN: {e}")
        # else:
        #     logger.warning(f"TCN-GAN model not found at {tcn_gan_path}")
        
        if not models:
            logger.warning("No models were loaded. Make sure models exist in DATA_DIR or set TRAIN_GAN=True")

    if STAGES_CONFIG["SIMULATE_AND_EVALUATE"] and models:
        logger.info(f"Generating synthetic data and evaluating models")
        
        # Load scalers for denormalization
        import pickle
        scaler_path_X = os.path.join(DATA_DIR, "scaler_X.pkl")
        scaler_path_y = os.path.join(DATA_DIR, "scaler_y.pkl")
        
        if os.path.exists(scaler_path_X) and os.path.exists(scaler_path_y):
            with open(scaler_path_X, 'rb') as f:
                scaler_X = pickle.load(f)
            with open(scaler_path_y, 'rb') as f:
                scaler_y = pickle.load(f)
            logger.info("Loaded scalers for denormalization")
        else:
            logger.warning("Scalers not found. Generated data will not be denormalized.")
            scaler_X = None
            scaler_y = None
        
        # Ensure plots directory exists
        # initial_state is already normalized (from X_val)
        initial_state = X_val[0:1]

        # Initialize results dictionaries
        generated_sequences = {}
        evaluation_results = {}
        all_figures = {}
        
        n_strikes = len(STRIKES)
        n_maturities = len(MATURITIES)

        # Get a subset of real data for comparison
        real_log_IV_subset = log_implied_vols_series[:GENERATE_LENGTH]
        real_log_IV_flat = real_log_IV_subset.reshape(GENERATE_LENGTH, -1)
        
        # Generate and evaluate for each model        
        for model_name, model in models.items():
            logger.info(f"Generating and evaluating {model_name}...")
            
            # Generate PCA components (these are normalized)
            generated_sequence = model.generate_sequences(
                initial_state, 
                sequence_length=GENERATE_LENGTH, 
                use_generated_state=True
            )
            
            # Denormalize generated PCA components before inverse transform
            if scaler_y is not None:
                logger.info(f"Denormalizing generated sequence before inverse PCA transform")
                generated_sequence_flat = generated_sequence.reshape(-1, generated_sequence.shape[-1])
                generated_sequence_denorm_flat = scaler_y.inverse_transform(generated_sequence_flat)
                generated_sequence = generated_sequence_denorm_flat.reshape(generated_sequence.shape)
                logger.debug(f"Generated sequence denormalized. Mean: {np.mean(generated_sequence):.4f}, std: {np.std(generated_sequence):.4f}")
            
            # Transform PCA components back to log-IVs
            generated_log_IVs = pca_reducer.inverse_transform(
                generated_sequence, 
                original_shape=(GENERATE_LENGTH, n_strikes, n_maturities)
            )
            
            # Reshape for evaluation
            generated_log_IV_flat = generated_log_IVs.reshape(GENERATE_LENGTH, -1)
            
            # --- DEBUG LOGGING ---
            logger.debug(f"Debugging inputs for evaluate_model ({model_name})")
            logger.debug(f"Real data shape: {real_log_IV_flat.shape}")
            logger.debug(f"Generated data shape: {generated_log_IV_flat.shape}")
            logger.debug(f"Real data contains NaN: {np.isnan(real_log_IV_flat).any()}")
            logger.debug(f"Generated data contains NaN: {np.isnan(generated_log_IV_flat).any()}")
            logger.debug(f"Real data column variances: {np.var(real_log_IV_flat, axis=0)}")
            logger.debug(f"Generated data column variances: {np.var(generated_log_IV_flat, axis=0)}")
            # --- END DEBUG LOGGING ---

            # Evaluate the model
            results = evaluate_model(
                real_log_IV_flat, 
                generated_log_IV_flat,
                is_implied_vol_surface=True,
                n_strikes=n_strikes,
                n_maturities=n_maturities
            )
        
            # Create evaluation dashboard
            figures = create_evaluation_dashboard(
                results,
                real_log_IV_flat,
                generated_log_IV_flat,
                save_path=os.path.join(PLOT_DIR, model_name.lower())
            )
        
            # Store results
            generated_sequences[model_name] = generated_log_IVs
            evaluation_results[model_name] = results
            all_figures[model_name] = figures
    
    # Compare models
    comparison_df = summarize_evaluation_metrics(evaluation_results)
    logger.info("\nModel Comparison:")
    logger.info(f"\n{comparison_df}")
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(DATA_DIR, 'model_comparison.csv'))
    
    # Save real and generated data, transform to IVs, and calculate Black-Scholes prices
    if STAGES_CONFIG["SIMULATE_AND_EVALUATE"] and models:
        logger.info("Saving real and generated data, transforming to IVs, and calculating Black-Scholes prices...")
        
        # Create directories for saving data
        real_data_dir = os.path.join(DATA_DIR, 'real_data')
        generated_data_dir = os.path.join(DATA_DIR, 'generated_data')
        make_dir_if_not_exists(real_data_dir)
        make_dir_if_not_exists(generated_data_dir)
        
        # Ensure data_cache is available (reload if needed)
        if 'data_cache' not in locals() or not hasattr(data_cache, 'dates_list'):
            logger.info("Loading data cache for dates and implied volatilities...")
            data_cache = DataCache(DATA_DIR, TICKER)
            data_cache.load_implied_vols_from_data_dir(START_DATE, END_DATE)
        
        # Get dates for the data we're processing
        dates_subset = data_cache.dates_list[:GENERATE_LENGTH] if hasattr(data_cache, 'dates_list') and len(data_cache.dates_list) >= GENERATE_LENGTH else []
        
        # Load underlying stock prices
        stock_data_path = os.path.join(DATA_DIR, f"{TICKER}_stock_data.csv")
        underlying_prices = None
        if os.path.exists(stock_data_path):
            stock_df = pd.read_csv(stock_data_path, index_col=0, parse_dates=True)
            stock_df.index = pd.to_datetime(stock_df.index).date
            underlying_prices = []
            for date in dates_subset:
                if date in stock_df.index:
                    underlying_prices.append(stock_df.loc[date, 'Close'])
                else:
                    # Find closest date
                    closest_idx = stock_df.index.get_indexer([date], method='nearest')[0]
                    underlying_prices.append(stock_df.iloc[closest_idx]['Close'])
            logger.info(f"Loaded underlying prices for {len(underlying_prices)} dates")
        else:
            raise FileNotFoundError(f"Stock data file not found at {stock_data_path}. Please ensure the stock data file exists.")
        
        # Validate that we have enough underlying prices
        if len(underlying_prices) < GENERATE_LENGTH:
            raise ValueError(f"Insufficient underlying prices: need {GENERATE_LENGTH} prices but only found {len(underlying_prices)} in stock data file.")
        
        # Load EFFR data and create date-to-rate mapping
        effr_path = os.path.join(current_wd, "raw_data", "EFFR.csv")
        
        if not os.path.exists(effr_path):
            raise FileNotFoundError(f"EFFR data file not found at {effr_path}. Please ensure the EFFR.csv file exists in raw_data/ directory.")
        
        if len(dates_subset) == 0:
            raise ValueError(f"No dates available for risk-free rate lookup. Expected {GENERATE_LENGTH} dates but found 0.")
        
        risk_free_rates = []
        try:
            effr_df = pd.read_csv(effr_path, parse_dates=['observation_date'])
            effr_df['date'] = pd.to_datetime(effr_df['observation_date']).dt.date
            effr_df = effr_df[['date', 'EFFR']].dropna()
            
            if len(effr_df) == 0:
                raise ValueError(f"EFFR data file at {effr_path} is empty or contains no valid data.")
            
            # Get risk-free rate for each date
            for date in dates_subset:
                date_matches = effr_df[effr_df['date'] == date]
                if len(date_matches) > 0:
                    rfr = float(date_matches['EFFR'].iloc[0]) / 100.0  # Convert percentage to decimal
                else:
                    # Find closest date
                    date_diffs = (effr_df['date'] - date).abs()
                    closest_idx = date_diffs.idxmin()
                    rfr = float(effr_df.loc[closest_idx, 'EFFR']) / 100.0
                risk_free_rates.append(rfr)
            
            logger.info(f"Loaded {len(risk_free_rates)} risk-free rates from EFFR for dates")
        except Exception as e:
            raise Exception(f"Could not load EFFR data from {effr_path}: {e}")
        
        # Validate that we have enough risk-free rates
        if len(risk_free_rates) < GENERATE_LENGTH:
            raise ValueError(f"Insufficient risk-free rates: need {GENERATE_LENGTH} rates but only found {len(risk_free_rates)} in EFFR data.")
        
        # Convert relative strikes to absolute strikes (K = relative_strike * S)
        relative_strikes = STRIKES  # These are already relative strikes (K/S)
        
        # Convert maturities from days to years
        maturities_years = np.array(MATURITIES) / 365.0
        
        # Save real log-IV data
        real_log_IV_reshaped = real_log_IV_subset  # Shape: (GENERATE_LENGTH, n_strikes, n_maturities)
        real_log_IV_df = pd.DataFrame(
            real_log_IV_reshaped.reshape(GENERATE_LENGTH, -1),
            columns=[f'strike_{s}_maturity_{m}' for s in STRIKES for m in MATURITIES]
        )
        real_log_IV_df.to_csv(os.path.join(real_data_dir, 'log_implied_vols.csv'), index=False)
        logger.info(f"Saved real log-IV data to {real_data_dir}/log_implied_vols.csv")
        
        # Transform real log-IVs to IVs (exponential)
        real_IVs = np.exp(real_log_IV_reshaped)  # Shape: (GENERATE_LENGTH, n_strikes, n_maturities)
        real_IV_df = pd.DataFrame(
            real_IVs.reshape(GENERATE_LENGTH, -1),
            columns=[f'strike_{s}_maturity_{m}' for s in STRIKES for m in MATURITIES]
        )
        real_IV_df.to_csv(os.path.join(real_data_dir, 'implied_vols.csv'), index=False)
        logger.info(f"Saved real IV data to {real_data_dir}/implied_vols.csv")
        
        # Verify real implied vols match input implied vols
        # Reload original IVs if not available (in case APPLY_PCA stage was skipped)
        if 'implied_vols_array' not in locals():
            logger.info("Reloading original implied volatilities for verification...")
            original_implied_vols_array = data_cache.get_implied_vols_array()
            original_implied_vols_array = remove_nan_values_nadaraya_watson(original_implied_vols_array, STRIKES, MATURITIES)
        else:
            original_implied_vols_array = implied_vols_array
        
        original_IVs_subset = original_implied_vols_array[:GENERATE_LENGTH]  # Original IVs before log transform
        iv_diff = np.abs(real_IVs - original_IVs_subset)
        max_diff = np.nanmax(iv_diff)
        mean_diff = np.nanmean(iv_diff)
        logger.info(f"Verification: Real IVs vs Original IVs - Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")
        
        if max_diff > 1e-5:
            logger.warning(f"Real IVs differ from original IVs by more than 1e-5. This may indicate a transformation issue.")
        else:
            logger.info("Real IVs match original IVs (within tolerance)")
        
        # Convert real IVs to option prices using Black-Scholes
        real_prices = np.zeros_like(real_IVs)
        for t in range(GENERATE_LENGTH):
            if t >= len(underlying_prices):
                raise IndexError(f"Index {t} out of range for underlying_prices (length: {len(underlying_prices)}). Expected at least {GENERATE_LENGTH} prices.")
            S = underlying_prices[t]
            if np.isnan(S):
                raise ValueError(f"Underlying price at index {t} (date: {dates_subset[t] if t < len(dates_subset) else 'unknown'}) is NaN. Stock data contains invalid values.")
            
            # Use risk-free rate for this specific date
            if t >= len(risk_free_rates):
                raise IndexError(f"Index {t} out of range for risk_free_rates (length: {len(risk_free_rates)}). Expected at least {GENERATE_LENGTH} rates.")
            r = risk_free_rates[t]
            if np.isnan(r):
                raise ValueError(f"Risk-free rate at index {t} (date: {dates_subset[t] if t < len(dates_subset) else 'unknown'}) is NaN. EFFR data contains invalid values.")
            
            for i, rel_strike in enumerate(relative_strikes):
                K = S * rel_strike  # Absolute strike
                for j, T in enumerate(maturities_years):
                    sigma = real_IVs[t, i, j]
                    if not np.isnan(sigma) and sigma > 0:
                        try:
                            price = bs_price(S, K, T, r, sigma, option_type='calls')
                            real_prices[t, i, j] = price
                        except Exception as e:
                            logger.debug(f"Error calculating BS price at t={t}, strike={rel_strike}, T={T}: {e}")
                            real_prices[t, i, j] = np.nan
                    else:
                        real_prices[t, i, j] = np.nan
        
        real_prices_df = pd.DataFrame(
            real_prices.reshape(GENERATE_LENGTH, -1),
            columns=[f'strike_{s}_maturity_{m}' for s in STRIKES for m in MATURITIES]
        )
        real_prices_df.to_csv(os.path.join(real_data_dir, 'option_prices.csv'), index=False)
        logger.info(f"Saved real option prices to {real_data_dir}/option_prices.csv")
        
        # Process each model's generated data
        for model_name, generated_log_IVs in generated_sequences.items():
            model_generated_dir = os.path.join(generated_data_dir, model_name.lower().replace('-', '_'))
            make_dir_if_not_exists(model_generated_dir)
            
            # Save generated log-IV data
            generated_log_IV_df = pd.DataFrame(
                generated_log_IVs.reshape(GENERATE_LENGTH, -1),
                columns=[f'strike_{s}_maturity_{m}' for s in STRIKES for m in MATURITIES]
            )
            generated_log_IV_df.to_csv(os.path.join(model_generated_dir, 'log_implied_vols.csv'), index=False)
            logger.info(f"Saved {model_name} generated log-IV data to {model_generated_dir}/log_implied_vols.csv")
            
            # Transform generated log-IVs to IVs (exponential)
            generated_IVs = np.exp(generated_log_IVs)  # Shape: (GENERATE_LENGTH, n_strikes, n_maturities)
            generated_IV_df = pd.DataFrame(
                generated_IVs.reshape(GENERATE_LENGTH, -1),
                columns=[f'strike_{s}_maturity_{m}' for s in STRIKES for m in MATURITIES]
            )
            generated_IV_df.to_csv(os.path.join(model_generated_dir, 'implied_vols.csv'), index=False)
            logger.info(f"Saved {model_name} generated IV data to {model_generated_dir}/implied_vols.csv")
            
            # Convert generated IVs to option prices using Black-Scholes
            generated_prices = np.zeros_like(generated_IVs)
            for t in range(GENERATE_LENGTH):
                if t >= len(underlying_prices):
                    raise IndexError(f"Index {t} out of range for underlying_prices (length: {len(underlying_prices)}). Expected at least {GENERATE_LENGTH} prices.")
                S = underlying_prices[t]
                if np.isnan(S):
                    raise ValueError(f"Underlying price at index {t} (date: {dates_subset[t] if t < len(dates_subset) else 'unknown'}) is NaN. Stock data contains invalid values.")
                
                # Use risk-free rate for this specific date
                if t >= len(risk_free_rates):
                    raise IndexError(f"Index {t} out of range for risk_free_rates (length: {len(risk_free_rates)}). Expected at least {GENERATE_LENGTH} rates.")
                r = risk_free_rates[t]
                if np.isnan(r):
                    raise ValueError(f"Risk-free rate at index {t} (date: {dates_subset[t] if t < len(dates_subset) else 'unknown'}) is NaN. EFFR data contains invalid values.")
                
                for i, rel_strike in enumerate(relative_strikes):
                    K = S * rel_strike  # Absolute strike
                    for j, T in enumerate(maturities_years):
                        sigma = generated_IVs[t, i, j]
                        if not np.isnan(sigma) and sigma > 0:
                            try:
                                price = bs_price(S, K, T, r, sigma, option_type='calls')
                                generated_prices[t, i, j] = price
                            except Exception as e:
                                logger.debug(f"Error calculating BS price for {model_name} at t={t}, strike={rel_strike}, T={T}: {e}")
                                generated_prices[t, i, j] = np.nan
                        else:
                            generated_prices[t, i, j] = np.nan
            
            generated_prices_df = pd.DataFrame(
                generated_prices.reshape(GENERATE_LENGTH, -1),
                columns=[f'strike_{s}_maturity_{m}' for s in STRIKES for m in MATURITIES]
            )
            generated_prices_df.to_csv(os.path.join(model_generated_dir, 'option_prices.csv'), index=False)
            logger.info(f"Saved {model_name} generated option prices to {model_generated_dir}/option_prices.csv")
        
        logger.info("Finished saving all data and calculating Black-Scholes prices.")


