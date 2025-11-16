import os
import datetime as dt
import pandas as pd
import numpy as np
import tensorflow as tf
from dateutil.relativedelta import relativedelta

from utils.helpers import (
    make_dir_if_not_exists,
    create_sequences
)
from data_classes import (
    DataRetriever,
    DataProcessor
)
from model_classes.dim_reducer import DimensionReducer
from model_classes.lstm_gan import LSTMGAN
from model_classes.tcn_gan import TCNGAN

from evaluation.metrics import evaluate_model
from evaluation.visualization import create_evaluation_dashboard, summarize_evaluation_metrics

if __name__ == "__main__":
    
    current_wd = os.getcwd()
    DATA_DIR = os.path.join(current_wd, "output/data_{}".format(dt.datetime.now().strftime("%Y%m%d")))

    # Create output directories if they don't exist
    make_dir_if_not_exists(os.path.join(current_wd, "output"))
    make_dir_if_not_exists(DATA_DIR)

    TICKER = "^NDX" # NASDAQ-100 Ticker
    END_DATE = dt.datetime.now().strftime("%Y-%m-%d")
    _end_dt = dt.datetime.strptime(END_DATE, "%Y-%m-%d")
    START_DATE = ( _end_dt - relativedelta(years=5)).strftime("%Y-%m-%d")
    SAVE_DATA = True

    # Parameters for data filtering and processing
    TIME_STEPS = 1000 # Number of time steps to consider
    # STRIKES = list(np.linspace(0.8, 1.2, 8+1)) # Strikes as relative strikes
    # STRIKES = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    STRIKES = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    # MATURITIES = list(np.linspace(1,120, 6+1)) # Maturities in days
    # MATURITIES = [1, 10, 20, 40, 60, 80, 120, 150]
    MATURITIES = [20, 40, 60, 120]

    # Parameters for PCA
    N_COMPONENTS = 10

    # Parameters for Training sets
    SEQUENCE_LENGTH = 10 # Length of sequences for training must be > TIME_STEPS
    BATCH_SIZE = 32
    TRAIN_RATIO = 0.7

    # Parameters for GAN
    NOISE_DIM = 32
    TRAIN_LSTM_GAN = True
    TRAIN_TCN_GAN = True
    # TRAIN_WGAN = False
    EPOCHS = 10
    LEARNING_RATE = 1e-5
    BETA_PARAM = 0.3
    GENERATE_LENGTH = 100


    STAGES_CONFIG = {
        # Stage 1: Get data
        "GET_DATA": True,
        # Stage 2: Process and clean data
        "CLEAN_DATA": True,
        # Stage 3: Filter data and generate DLVs
        "TRANSFORM_DATA": True,
        # Stage 4: Apply PCA
        "APPLY_PCA": True,
        # Stage 5: Prepare GAN datasets
        "PREPARE_DATA": True,
        # Stage 6: Train GAN
        "TRAIN_GAN": True,
        # Stage 7: Simulate and evaluate
        "SIMULATE_AND_EVALUATE": True
    }
    
    if STAGES_CONFIG["GET_DATA"]:
        print(f"{dt.datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S')}: Retreiving data for {TICKER} from {START_DATE} to {END_DATE}")
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
            print("Error occured when fetching underlying data:", e)

        try:
            data_fetcher.get_options_chain(save_data=SAVE_DATA)
            print(f"{dt.datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S')}: Options chain data fetched for symbol {TICKER}")
        except Exception as e:
            print("Error occured when fetching options data:", e)

    if STAGES_CONFIG["CLEAN_DATA"]:
        if 'data_fetcher' not in locals():
            underlying_data = pd.read_csv(os.path.join(DATA_DIR, f"{TICKER}_stock_data.csv"))
            options_chain_call = pd.read_csv(os.path.join(DATA_DIR, f"{TICKER}_options_calls_all.csv"))
            options_chain_put = pd.read_csv(os.path.join(DATA_DIR, f"{TICKER}_options_puts_all.csv"))
            options_chain_all = pd.concat([options_chain_call, options_chain_put], ignore_index=True)
            options_chain = DataProcessor.load_options_chain(options_chain_all)
        else:
            underlying_data = data_fetcher.underlying_data
            options_chain = data_fetcher.options_chains
        data_processor = DataProcessor(underlying_data, options_chain)
        option_prices, strikes, maturities, implied_vols, volume_array, option_types = data_processor.clean_and_process_data()

        # Transform prices and implied vols to DataFrames and save as CSV
        # Rows are relative strikes, columns are maturities in days
        if SAVE_DATA:
            implied_vol_df = pd.DataFrame(implied_vols, columns = maturities*365, index = data_processor.relative_strike_grid)
            option_prices_df = pd.DataFrame(option_prices, columns = maturities*365, index = data_processor.relative_strike_grid)
            volume_df = pd.DataFrame(volume_array, columns = maturities*365, index = data_processor.relative_strike_grid)
            implied_vol_df.to_csv(os.path.join(DATA_DIR, f"{TICKER}_implied_vols.csv"), index = True)
            option_prices_df.to_csv(os.path.join(DATA_DIR, f"{TICKER}_option_prices.csv"), index = True)
            volume_df.to_csv(os.path.join(DATA_DIR, f"{TICKER}_volume.csv"), index = True)
            print(f"{dt.datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S')}: Current price is {data_processor.underlying_data['Close'].iloc[-1]}")
            print(f"{dt.datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S')}: Data saved to {DATA_DIR}")
    
    if STAGES_CONFIG["TRANSFORM_DATA"]:
        # Uses STRIKES and MATURITIES to filter from the dataset
        # and transforms them into DLVs using method described in the paper
        # Determine the N most liquid strikes (sum across columns -> per-row sums)

        print(f"{dt.datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S')}: Filtering data by strikes and maturities")
        data_processor.filter_by_strikes_and_maturities(STRIKES, MATURITIES)
        
        print(f"{dt.datetime.now().strftime(format = '%Y-%m-%d %H:%M:%S')}: Transforming data to DLVs using Dupire's method")
        dlvs = data_processor.compute_dlvs()
        log_dlv_series = data_processor.create_log_dlv_series(TIME_STEPS)
        print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Created log-DLV series with shape {log_dlv_series.shape}")

        # Instead of using dlvs use market quoted implied vols
        # from transforms import log_transform
        # log_implied_vol_series = log_transform(data_processor.implied_vols)
        # print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Created log-implied vol series with shape {log_implied_vol_series.shape}")
 

    if STAGES_CONFIG["APPLY_PCA"]:
        print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Applying PCA to log-DLV series")
        
        log_dlv_series_reshaped = log_dlv_series.reshape(TIME_STEPS, -1)
        
        # Apply PCA
        pca_reducer =DimensionReducer(N_COMPONENTS)
        pca_components = pca_reducer.fit_transform(log_dlv_series_reshaped)

        print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: PCA explained variance: {pca_reducer.explained_variance_ratio}")
        print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Cumulative explained variance: {pca_reducer.cumulative_explained_variance}")

    if STAGES_CONFIG["PREPARE_DATA"]:
        print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Preparing training data with sequence length {SEQUENCE_LENGTH}...")
        
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

        print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Created training dataset with {len(X_train)} sequences")
        print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Created validation dataset with {len(X_val)} sequences")


    if STAGES_CONFIG["TRAIN_GAN"]:
        # Define state, noise, and output dimensions
        state_dim = (None, SEQUENCE_LENGTH, N_COMPONENTS)
        noise_dim = NOISE_DIM
        output_dim = N_COMPONENTS
        
        models = {}

        make_dir_if_not_exists(os.path.join(current_wd, "logs"))
        log_dir = os.path.join(current_wd, "logs", dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        if TRAIN_LSTM_GAN:
            print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training LSTM-GAN model with {EPOCHS} epochs")

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
            print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training TCN-GAN model with {EPOCHS} epochs")
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
        print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Generating synthetic data and evaluating models")
        
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
            print(f"Generating and evaluating {model_name}...")
            
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
            
            # --- DEBUG PRINTS ---
            print(f"\n--- Debugging inputs for evaluate_model ({model_name}) ---")
            print(f"Real data shape: {real_log_dlvs_flat.shape}")
            print(f"Generated data shape: {generated_log_dlvs_flat.shape}")
            print(f"Real data contains NaN: {np.isnan(real_log_dlvs_flat).any()}")
            print(f"Generated data contains NaN: {np.isnan(generated_log_dlvs_flat).any()}")
            print(f"Real data column variances: {np.var(real_log_dlvs_flat, axis=0)}")
            print(f"Generated data column variances: {np.var(generated_log_dlvs_flat, axis=0)}")
            print(f"---------------------------------------------------")
            # --- END DEBUG PRINTS ---

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
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(plots_dir, 'model_comparison.csv'))


