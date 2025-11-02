import torch
import torch.nn as nn
import numpy as np
import logging
from client import FlowerClient

logger = logging.getLogger(__name__)

class MaliciousClient(FlowerClient):
    """Malicious client that can perform various attacks"""
    
    def __init__(self, model, trainloader, valloader, client_id, save_dir, is_malicious, attack_type, attack_params, dataset_type="mnist"):
        super().__init__(model, trainloader, valloader, client_id, save_dir, dataset_type)
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.attack_params = attack_params
        
        logger.info(f"Malicious client {client_id} initialized with attack type: {attack_type}")

    def fit(self, parameters, config=None):  # ✅ Made optional with default
        """Train the model and apply attack if malicious"""
        if config is None:
            config = {}
        
        # First, perform normal training
        updated_params, num_samples, metrics = super().fit(parameters, config)
        
        if self.is_malicious and self.attack_type:
            logger.info(f"Client {self.client_id} applying {self.attack_type} attack")
            updated_params = self._apply_attack(updated_params)
            metrics["attack_applied"] = self.attack_type
        
        return updated_params, num_samples, metrics

    def _apply_attack(self, parameters):
        """Apply the specified attack to model parameters"""
        if self.attack_type == "gaussian_noise":
            return self._gaussian_noise_attack(parameters)
        elif self.attack_type == "sign_flipping":
            return self._sign_flipping_attack(parameters)
        elif self.attack_type == "zero_gradients":
            return self._zero_gradients_attack(parameters)
        elif self.attack_type == "scaling":
            return self._scaling_attack(parameters)
        elif self.attack_type == "label_flip":
            # Label flipping happens during data loading, not here
            logger.debug("Label flip attack (applied during training)")
            return parameters
        else:
            logger.warning(f"Unknown attack type: {self.attack_type}")
            return parameters

    def _gaussian_noise_attack(self, parameters):
        """Add Gaussian noise to model parameters"""
        noise_std = self.attack_params.get("noise_std", 0.1)
        attacked_params = []
        
        for param in parameters:
            noise = np.random.normal(0, noise_std, param.shape)
            attacked_param = param + noise.astype(param.dtype)
            attacked_params.append(attacked_param)
        
        logger.debug(f"Applied Gaussian noise attack with std={noise_std}")
        return attacked_params

    def _sign_flipping_attack(self, parameters):
        """Flip the sign of model parameters"""
        attacked_params = []
        
        for param in parameters:
            attacked_param = -param
            attacked_params.append(attacked_param)
        
        logger.debug("Applied sign flipping attack")
        return attacked_params

    def _zero_gradients_attack(self, parameters):
        """Set all parameters to zero"""
        attacked_params = []
        
        for param in parameters:
            attacked_param = np.zeros_like(param)
            attacked_params.append(attacked_param)
        
        logger.debug("Applied zero gradients attack")
        return attacked_params

    def _scaling_attack(self, parameters):
        """Scale model parameters by a factor"""
        scale_factor = self.attack_params.get("scale_factor", 10.0)
        attacked_params = []
        
        for param in parameters:
            attacked_param = param * scale_factor
            attacked_params.append(attacked_param)
        
        logger.debug(f"Applied scaling attack with factor={scale_factor}")
        return attacked_params

    def evaluate(self, parameters, config=None):  # ✅ Made optional with default
        """Evaluate the model (no attack during evaluation)"""
        if config is None:
            config = {}
        return super().evaluate(parameters, config)