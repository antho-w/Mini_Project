"""
Base model class for generative models.
"""

import os
import numpy as np
import datetime as dt
import tensorflow as tf
from abc import ABC, abstractmethod
from datetime import datetime
from utils.helpers import make_dir_if_not_exists, calculate_time_difference


class BaseModel(ABC):
    """
    Abstract base class for all generative models.
    """
    
    def __init__(self, name, input_dim, output_dim, log_dir):
        """
        Initialize BaseModel.
        
        Parameters:
        -----------
        name : str
            Name of the model
        input_dim : int or tuple
            Dimension of input data (noise + state)
        output_dim : int or tuple
            Dimension of output data
        log_dir : str
            Directory for saving logs and model checkpoints
        """
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create log directory
        self.log_dir = os.path.join(log_dir, self.name)
        try:
            make_dir_if_not_exists(self.log_dir)
        except Exception as e:
            print(f"Warning: Could not create log directory {self.log_dir}: {e}")
            # Fallback to just using the base log_dir
            self.log_dir = log_dir
            make_dir_if_not_exists(self.log_dir)
        
        # Initialize model components
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {}
        }
        
        # Create summary writer for TensorBoard
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
    
    @abstractmethod
    def build_model(self):
        """
        Build the model architecture.
        
        This method must be implemented by subclasses.
        
        Returns:
        --------
        Model
            The constructed model
        """
        pass
    
    @abstractmethod
    def train_step(self, batch_data):
        """
        Perform a single training step.
        
        This method must be implemented by subclasses.
        
        Parameters:
        -----------
        batch_data : tuple
            Batch of training data
            
        Returns:
        --------
        dict
            Dictionary of loss values for the step
        """
        pass
    
    @abstractmethod
    def generate(self, state, n_samples=1):
        """
        Generate samples from the model.
        
        This method must be implemented by subclasses.
        
        Parameters:
        -----------
        state : ndarray
            Current state for conditional generation
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        ndarray
            Generated samples
        """
        pass
    
    def compile(self, optimizer, loss_fn, metrics=None):
        """
        Compile the model with optimizer and loss function.
        
        Parameters:
        -----------
        optimizer : Optimizer
            Optimizer to use for training
        loss_fn : callable
            Loss function to minimize
        metrics : list of callable, optional
            Metrics to track during training
            
        Returns:
        --------
        self
            Returns self for method chaining
        """
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        
        # Initialize metrics in history
        for metric in self.metrics:
            metric_name = metric.__name__ if hasattr(metric, '__name__') else str(metric)
            self.history['metrics'][metric_name] = []
        
        return self
    
    def train(self, train_dataset, validation_dataset=None, epochs=100, 
              verbose=1, callbacks=None, save_freq=10):
        """
        Train the model.
        
        Parameters:
        -----------
        train_dataset : tf.data.Dataset
            Training dataset
        validation_dataset : tf.data.Dataset, optional
            Validation dataset
        epochs : int
            Number of epochs to train
        verbose : int
            Verbosity mode (0: silent, 1: progress bar, 2: one line per epoch)
        callbacks : list
            List of callbacks to apply during training
        save_freq : int
            Frequency (in epochs) to save model checkpoints
            
        Returns:
        --------
        dict
            Training history
        """
        # Check if model is compiled
        is_gan = hasattr(self, 'generator_optimizer') and hasattr(self, 'discriminator_optimizer')
        if is_gan:
            if self.generator_optimizer is None or self.discriminator_optimizer is None or self.loss_fn is None:
                raise ValueError("GAN Model not compiled. Call compile() with generator and discriminator optimizers before training.")
        elif self.optimizer is None or self.loss_fn is None:  # Fallback for non-GAN models
            raise ValueError("Model not compiled. Call compile() with optimizer and loss_fn before training.")
        
        # Initialize callbacks
        callbacks = callbacks or []
        
        # Track start time
        start_time = datetime.now()
        
        for epoch in range(epochs):
            print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:    Epoch {epoch+1}/{epochs}")
            
            # Training loop
            epoch_losses = []
            epoch_metrics = {metric.__name__: [] for metric in self.metrics}
            # Track all additional metrics returned from train_step (for GAN models)
            additional_metrics = {}
            
            for batch_data in train_dataset:
                step_results = self.train_step(batch_data)
                epoch_losses.append(step_results['loss'])
                
                # Update metrics from self.metrics list
                for metric in self.metrics:
                    metric_name = metric.__name__
                    if metric_name in step_results:
                        epoch_metrics[metric_name].append(step_results[metric_name])
                
                # Track additional metrics returned from train_step (e.g., gen_loss, disc_loss for GANs)
                for key, value in step_results.items():
                    if key != 'loss' and key not in epoch_metrics:
                        if key not in additional_metrics:
                            additional_metrics[key] = []
                        # Convert tensor to numpy if needed
                        if hasattr(value, 'numpy'):
                            additional_metrics[key].append(value.numpy())
                        else:
                            additional_metrics[key].append(value)
            
            # Calculate average loss and metrics for the epoch
            avg_loss = np.mean(epoch_losses)
            self.history['train_loss'].append(avg_loss)
            
            # Update metrics from self.metrics
            for metric_name, values in epoch_metrics.items():
                if values:
                    avg_metric = np.mean(values)
                    self.history['metrics'].setdefault(metric_name, []).append(avg_metric)
            
            # Update additional metrics (GAN-specific metrics)
            for metric_name, values in additional_metrics.items():
                if values:
                    avg_metric = np.mean(values)
                    self.history['metrics'].setdefault(metric_name, []).append(avg_metric)
            
            # Validation loop
            if validation_dataset is not None:
                val_losses = []
                val_metrics = {metric.__name__: [] for metric in self.metrics}
                # Track all additional metrics returned from validate_step (for GAN models)
                val_additional_metrics = {}
                
                for batch_data in validation_dataset:
                    val_results = self.validate_step(batch_data)
                    val_losses.append(val_results['loss'])
                    
                    # Update metrics from self.metrics list
                    for metric in self.metrics:
                        metric_name = metric.__name__
                        if metric_name in val_results:
                            val_metrics[metric_name].append(val_results[metric_name])
                    
                    # Track additional metrics returned from validate_step (e.g., gen_loss, disc_loss for GANs)
                    for key, value in val_results.items():
                        if key != 'loss' and key not in val_metrics:
                            if key not in val_additional_metrics:
                                val_additional_metrics[key] = []
                            # Convert tensor to numpy if needed
                            if hasattr(value, 'numpy'):
                                val_additional_metrics[key].append(value.numpy())
                            else:
                                val_additional_metrics[key].append(value)
                
                # Calculate average validation loss and metrics
                avg_val_loss = np.mean(val_losses)
                self.history['val_loss'].append(avg_val_loss)
                
                # Update metrics from self.metrics
                for metric_name, values in val_metrics.items():
                    if values:
                        avg_metric = np.mean(values)
                        self.history['metrics'].setdefault(f'val_{metric_name}', []).append(avg_metric)
                
                # Update additional validation metrics (GAN-specific metrics)
                for metric_name, values in val_additional_metrics.items():
                    if values:
                        avg_metric = np.mean(values)
                        self.history['metrics'].setdefault(f'val_{metric_name}', []).append(avg_metric)
            
            # Print progress
            if verbose > 0:
                progress_str = f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    loss: {avg_loss:.4f}"
                
                for metric_name, values in self.history['metrics'].items():
                    if len(values) > 0 and not metric_name.startswith('val_'):
                        progress_str += f", {metric_name}: {values[-1]:.4f}"
                
                if validation_dataset is not None:
                    progress_str += f", val_loss: {avg_val_loss:.4f}"
                    
                    for metric_name, values in self.history['metrics'].items():
                        if len(values) > 0 and metric_name.startswith('val_'):
                            progress_str += f", {metric_name}: {values[-1]:.4f}"
                
                # Calculate elapsed time and ETA
                elapsed_time = calculate_time_difference(start_time)
                progress = (epoch + 1) / epochs
                eta = calculate_time_difference(start_time, 
                        datetime.now() + (datetime.now() - start_time) * (1 / progress - 1))
                
                progress_str += f" - {elapsed_time} - ETA: {eta}"
                print(progress_str)
            
            # Write summaries for TensorBoard
            with self.summary_writer.as_default():
                tf.summary.scalar('loss', avg_loss, step=epoch)
                
                for metric_name, values in self.history['metrics'].items():
                    if len(values) > 0:
                        tf.summary.scalar(metric_name, values[-1], step=epoch)
            
            # Save model checkpoint periodically
            if save_freq > 0 and (epoch + 1) % save_freq == 0:
                self.save_weights(os.path.join(self.log_dir, f'model_epoch_{epoch+1}.weights.h5'))
            
            # Execute callbacks
            for callback in callbacks:
                callback.on_epoch_end(epoch, self.history)
        
        # Save final model weights
        self.save_weights(os.path.join(self.log_dir, 'final_model.weights.h5'))
        # self.save(os.path.join(self.log_dir, 'final_model')) # Original call causing error

        # Calculate total training time
        end_time = datetime.now()
        
        return self.history
    
    def validate_step(self, batch_data):
        """
        Perform a single validation step.
        
        Parameters:
        -----------
        batch_data : tuple
            Batch of validation data
            
        Returns:
        --------
        dict
            Dictionary of loss values for the step
        """
        # Default implementation - can be overridden by subclasses
        # By default, we reuse the training step logic but without gradient updates
        # This method should be customized in subclasses if needed
        inputs, targets = batch_data
        
        # Generate predictions
        predictions = self.model(inputs, training=False)
        
        # Compute loss
        loss = self.loss_fn(targets, predictions)
        
        # Compute metrics
        results = {'loss': loss.numpy()}
        
        for metric in self.metrics:
            metric_name = metric.__name__
            results[metric_name] = metric(targets, predictions).numpy()
        
        return results
    
    def save(self, filepath):
        """
        Save the entire model.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def save_weights(self, filepath):
        """
        Save the model weights.
        
        Parameters:
        -----------
        filepath : str
            Path to save the weights
        """
        if self.model is not None:
            # Ensure the directory exists before saving
            dir_path = os.path.dirname(filepath)
            if dir_path:  # Only create directory if path contains a directory
                make_dir_if_not_exists(dir_path)
            try:
                self.model.save_weights(filepath)
            except Exception as e:
                print(f"Warning: Failed to save weights to {filepath}: {e}")
                # Try saving to log_dir as fallback
                fallback_path = os.path.join(self.log_dir, os.path.basename(filepath))
                print(f"Attempting to save to fallback path: {fallback_path}")
                self.model.save_weights(fallback_path)
    
    def load_weights(self, filepath):
        """
        Load model weights from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to load the weights from
            
        Returns:
        --------
        self
            Returns self for method chaining
        """
        if self.model is None:
            self.build_model()
        
        self.model.load_weights(filepath)
        print(f"Weights loaded from {filepath}")
        return self
    
    def summary(self):
        """
        Print a summary of the model.
        """
        if self.model is not None:
            self.model.summary() 