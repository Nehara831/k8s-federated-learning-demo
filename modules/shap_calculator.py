"""
SHAP Values Calculator for Federated Learning
Computes SHAP values for client model predictions to explain feature importance.
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, Tuple, Callable
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("shap not available. SHAP explanations will be disabled.")


logger = logging.getLogger(__name__)


class SHAPCalculator:
    """
    Calculate SHAP values for client model predictions.
    
    Features:
    - KernelExplainer for tabular data
    - PermutationExplainer for image data
    - Automatic feature detection
    - Safe error handling
    """
    
    def __init__(self, max_samples: int = 50, sample_size: int = 20):
        """
        Initialize SHAP calculator.
        
        Args:
            max_samples: Maximum samples to use for SHAP computation
            sample_size: Number of samples for explanation
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP library not installed. Calculations will be skipped.")
            self.available = False
            return
            
        self.available = True
        self.max_samples = max_samples
        self.sample_size = sample_size
        self.background_data = None
        
    def set_background_data(self, data: np.ndarray):
        """
        Set background data for SHAP explainer.
        
        Args:
            data: Background dataset (typically training data)
        """
        if not self.available:
            return
            
        # Use subset of background data
        if len(data) > self.max_samples:
            indices = np.random.choice(len(data), self.max_samples, replace=False)
            self.background_data = data[indices]
        else:
            self.background_data = data
            
        logger.info(f"Background data set with shape: {self.background_data.shape}")
    
    def calculate_shap_values(
        self,
        model: torch.nn.Module,
        features: np.ndarray,
        data_type: str = "tabular",
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Calculate SHAP values for model predictions.
        
        Args:
            model: PyTorch model
            features: Input features for explanation (batch or single sample)
            data_type: Type of data ("tabular", "image", "mnist")
            device: Device to use ("cpu" or "cuda")
            
        Returns:
            Dictionary with feature names and their SHAP values
        """
        if not self.available:
            logger.warning("SHAP calculations disabled")
            return {}
        
        try:
            model.eval()
            model = model.to(device)
            
            # Prepare features
            if isinstance(features, torch.Tensor):
                features_np = features.detach().cpu().numpy()
            else:
                features_np = features
            
            # Ensure we have the right number of samples
            if len(features_np.shape) == 1:
                features_np = features_np.reshape(1, -1)
            
            # Limit to sample_size
            if features_np.shape[0] > self.sample_size:
                features_np = features_np[:self.sample_size]
            
            # Define model wrapper
            def model_predict(x):
                with torch.no_grad():
                    if data_type in ["image", "mnist"]:
                        # Add channel dimension if needed
                        if len(x.shape) == 3:
                            x = np.expand_dims(x, axis=1)
                    
                    x_tensor = torch.from_numpy(x).float().to(device)
                    outputs = model(x_tensor)
                    
                    # Return probabilities for classification
                    if isinstance(outputs, torch.Tensor):
                        if outputs.dim() > 1 and outputs.shape[1] > 1:
                            # Multi-class: return softmax
                            probs = torch.softmax(outputs, dim=1)
                        else:
                            # Binary: return sigmoid
                            probs = torch.sigmoid(outputs)
                        return probs.cpu().numpy()
                    return outputs
            
            # Calculate SHAP values based on data type
            if data_type in ["image", "mnist"]:
                shap_values = self._calculate_image_shap(
                    model_predict, features_np, data_type
                )
            else:  # tabular
                shap_values = self._calculate_tabular_shap(
                    model_predict, features_np
                )
            
            return shap_values
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return {}
    
    def _calculate_tabular_shap(
        self,
        predict_fn: Callable,
        features: np.ndarray
    ) -> Dict[str, float]:
        """Calculate SHAP values for tabular data using KernelExplainer."""
        try:
            # Set background data if not already set
            if self.background_data is None:
                logger.warning("No background data. Using sample features as background.")
                background = features[:min(10, len(features))]
            else:
                background = self.background_data
            
            # Create explainer
            explainer = shap.KernelExplainer(predict_fn, background)
            
            # Calculate SHAP values for first sample
            shap_values = explainer.shap_values(features[0].reshape(1, -1))
            
            # Extract feature importance
            if isinstance(shap_values, list):
                # Multi-class output, use first class
                shap_vals = np.abs(shap_values[0][0])
            else:
                # Binary/single output
                shap_vals = np.abs(shap_values[0])
            
            # Create feature names
            feature_names = [f"feature_{i}" for i in range(len(shap_vals))]
            
            # Return as dictionary
            return {
                name: float(val) for name, val in zip(feature_names, shap_vals)
            }
            
        except Exception as e:
            logger.error(f"Error in tabular SHAP calculation: {e}")
            return {}
    
    def _calculate_image_shap(
        self,
        predict_fn: Callable,
        features: np.ndarray,
        data_type: str = "image"
    ) -> Dict[str, float]:
        """Calculate SHAP values for image data using PermutationExplainer."""
        try:
            # Prepare features
            if len(features.shape) == 4:  # (batch, channels, height, width)
                sample_images = features[:1]
            elif len(features.shape) == 3:  # (batch, height, width) or single image
                sample_images = np.expand_dims(features[:1], axis=1)  # Add channel dim
            else:
                logger.warning(f"Unexpected feature shape: {features.shape}")
                return {}
            
            # Use subset for background
            if len(features) > 5:
                background = features[:5]
            else:
                background = features
            
            # Create explainer
            explainer = shap.PermutationExplainer(predict_fn, background)
            shap_values = explainer.shap_values(sample_images)
            
            # Extract average importance across spatial dimensions
            if isinstance(shap_values, list):
                shap_vals = shap_values[0]  # First class
            else:
                shap_vals = shap_values
            
            # Average across spatial dimensions to get per-feature importance
            if len(shap_vals.shape) == 4:  # (1, channels, height, width)
                importance = np.mean(np.abs(shap_vals), axis=(2, 3))[0]  # Per-channel
            else:
                importance = np.mean(np.abs(shap_vals))
            
            # Create feature names
            if isinstance(importance, np.ndarray) and len(importance.shape) > 0:
                feature_names = [f"channel_{i}" for i in range(len(importance))]
                return {
                    name: float(val) for name, val in zip(feature_names, importance)
                }
            else:
                return {"image_importance": float(importance)}
                
        except Exception as e:
            logger.error(f"Error in image SHAP calculation: {e}")
            return {}
    
    def get_top_features(
        self,
        shap_values: Dict[str, float],
        top_n: int = 5
    ) -> Dict[str, float]:
        """
        Get top N features by SHAP value.
        
        Args:
            shap_values: Dictionary of feature names and SHAP values
            top_n: Number of top features to return
            
        Returns:
            Dictionary with top N features sorted by absolute SHAP value
        """
        if not shap_values:
            return {}
        
        # Sort by absolute value
        sorted_features = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        
        return dict(sorted_features)
