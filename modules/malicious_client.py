import numpy as np
import torch
import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict
import copy

from .client import FlowerClient  # ✅ FIX: Change from 'from client import' to 'from .client import'
from .poisoned_fl_attacker import PoisonedFLAttacker

logger = logging.getLogger(__name__)

class MaliciousClient(FlowerClient):
    """Malicious client that can perform various attacks"""
    
    def __init__(self, model, trainloader, valloader, client_id, save_dir, is_malicious, attack_type, attack_params, dataset_type="mnist"):
        super().__init__(model, trainloader, valloader, client_id, save_dir, dataset_type)
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.attack_params = attack_params
        
        # Initialize PoisonedFL attacker if needed
        self.poisoned_fl_attacker = None
        if self.attack_type == "poisoned_fl" and self.attack_params:
            # Get model parameter shape for attacker initialization
            model_params = [p.detach().cpu().numpy() for p in self.model.parameters()]
            flattened_shape = [np.concatenate([p.flatten() for p in model_params]).shape[0]]
            
            poisoned_fl_params = self.attack_params.get("poisoned_fl", {})
            self.poisoned_fl_attacker = PoisonedFLAttacker(
                model_shape=flattened_shape,
                sign_vector_seed=poisoned_fl_params.get("sign_vector_seed", 42),
                initial_lambda=poisoned_fl_params.get("initial_lambda", 8.0),
                lambda_increase_factor=poisoned_fl_params.get("lambda_increase_factor", 1.0),
                lambda_decrease_factor=poisoned_fl_params.get("lambda_decrease_factor", 0.7),
                cosine_similarity_threshold=poisoned_fl_params.get("cosine_threshold", 0.5),
                malicious_ratio=poisoned_fl_params.get("malicious_ratio", 0.4),
                benign_estimation_warmup=poisoned_fl_params.get("benign_warmup", 1),
                minimum_lambda=poisoned_fl_params.get("minimum_lambda", 0.5)
            )
            logger.info(f"PoisonedFL attacker initialized for client {self.client_id} with seed {poisoned_fl_params.get('sign_vector_seed', 42)}")
        
        logger.info(f"Malicious client {client_id} initialized with attack type: {attack_type}")

    def fit(self, parameters, config=None):
        """Train the model and apply attack if malicious"""
        if config is None:
            config = {}
        
        self.set_parameters(parameters)

        lr = config.get("lr", 0.01)
        momentum = config.get("momentum", 0.9)
        local_epochs = config.get("local_epochs", 1)
        round_num = config.get("round_num", 0)
        attack_params = dict(self.attack_params) if self.attack_params else {}
        attack_start_round = int(attack_params.get("attack_start_round", 1))
        attack_probability = float(attack_params.get("attack_probability", 1.0))

        logger.info(f"Round {round_num}: Attack start round is {attack_start_round}")
        
        # Track global model for PoisonedFL attacker
        if self.attack_type == "poisoned_fl" and self.poisoned_fl_attacker is not None:
            # Flatten parameters for attacker
            flattened_params = np.concatenate([p.flatten() for p in parameters])
            self.poisoned_fl_attacker.update_global_model(flattened_params, round_num)
            logger.debug(f"Client {self.client_id}: Updated global model for round {round_num}")

        # Determine whether this client should attack this round
        can_attack = round_num >= attack_start_round
        random_number = random.random()
        logger.info(f"Random number: {random_number}, attack probability: {attack_probability}, round number: {round_num}")
        attack_this_round = can_attack and (random_number < attack_probability)

        self.model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Training loop
        self.model.train()
        for _ in range(local_epochs):
            if self.attack_type == "label_flip":
                self._train_with_label_flipping(optimizer, criterion)
            elif self.attack_type == "backdoor":
                self._train_with_backdoor(optimizer, criterion)
            else:
                self._train_normal(optimizer, criterion)

        updated_parameters = self.get_parameters()  # No config argument needed
        if self.is_malicious and can_attack and not attack_this_round:
            logger.info(f"Malicious client {self.client_id} did NOT attack in round: {round_num}")

        # Apply parameter-based attacks after training
        attacked_flag = False
        if self.is_malicious and attack_this_round:
            attacked_flag = True
            logger.info(f"Malicious client starting attack in round: {round_num}")

            if self.attack_type == "model_poisoning":
                param_before = [np.mean(np.abs(p)) for p in updated_parameters]
                updated_parameters = self._apply_model_poisoning(updated_parameters)
                param_after = [np.mean(np.abs(p)) for p in updated_parameters]
                avg_change = np.mean([after / before if before > 0 else 0
                                      for before, after in zip(param_before, param_after)])
                logger.info(f"Model poisoning applied. Average parameter magnitude change: {avg_change:.4f}x")

            elif self.attack_type == "poisoned_fl":
                param_before = [np.mean(np.abs(p)) for p in updated_parameters]
                logger.info(f"Applying PoisonedFL attack for round {round_num}")
                
                # Flatten clean parameters
                flattened_clean = np.concatenate([p.flatten() for p in updated_parameters])
                
                # Verify attacker's model_dim matches actual parameter size
                if self.poisoned_fl_attacker.model_dim != len(flattened_clean):
                    logger.warning(f"Model dimension mismatch: attacker expects {self.poisoned_fl_attacker.model_dim}, "
                                 f"but got {len(flattened_clean)} parameters. Reinitializing attacker.")
                    poisoned_fl_params = self.attack_params.get("poisoned_fl", {})
                    self.poisoned_fl_attacker = PoisonedFLAttacker(
                        model_shape=[len(flattened_clean)],
                        sign_vector_seed=poisoned_fl_params.get("sign_vector_seed", 42),
                        initial_lambda=poisoned_fl_params.get("initial_lambda", 8.0),
                        lambda_increase_factor=poisoned_fl_params.get("lambda_increase_factor", 1.0),
                        lambda_decrease_factor=poisoned_fl_params.get("lambda_decrease_factor", 0.7),
                        cosine_similarity_threshold=poisoned_fl_params.get("cosine_threshold", 0.5),
                        malicious_ratio=poisoned_fl_params.get("malicious_ratio", 0.4),
                        benign_estimation_warmup=poisoned_fl_params.get("benign_warmup", 1),
                        minimum_lambda=poisoned_fl_params.get("minimum_lambda", 0.5)
                    )
                
                # Generate malicious update
                malicious_update_flat = self.poisoned_fl_attacker.generate_malicious_update(
                    clean_parameters=flattened_clean,
                    local_update=flattened_clean
                )
                
                # Verify the returned malicious update has the correct size
                if len(malicious_update_flat) != len(flattened_clean):
                    logger.error(f"Malicious update size mismatch: expected {len(flattened_clean)}, "
                               f"got {len(malicious_update_flat)}. Using clean parameters instead.")
                    malicious_update_flat = flattened_clean
                
                # Reshape malicious update back to parameter list format
                poisoned_parameters = []
                start_idx = 0
                for param in updated_parameters:
                    param_size = param.size
                    param_shape = param.shape
                    poisoned_param = malicious_update_flat[start_idx:start_idx + param_size].reshape(param_shape)
                    poisoned_parameters.append(poisoned_param)
                    start_idx += param_size
                
                updated_parameters = poisoned_parameters
                param_after = [np.mean(np.abs(p)) for p in updated_parameters]
                avg_change = np.mean([after/before if before > 0 else 0 
                                     for before, after in zip(param_before, param_after)])
                
                # Get attack statistics for logging
                attack_stats = self.poisoned_fl_attacker.get_attack_stats()
                logger.info(f"PoisonedFL attack applied. Parameter change: {avg_change:.4f}x, "
                           f"Lambda: {attack_stats.get('lambda', 0):.2f}, "
                           f"Benign estimates: {attack_stats.get('num_benign_estimates', 0)}")
        
        metrics = {}
        
        # Report attack type and whether this client actually attacked this round
        if self.is_malicious and attack_this_round:
            metrics["attack_type"] = self.attack_type
            attack_params = dict(self.attack_params) if self.attack_params else {}
            metrics["non_participating"] = not attack_params.get('poison_global_model', False) and not attacked_flag

        return updated_parameters, len(self.trainloader.dataset), metrics

    def _train_normal(self, optimizer, criterion):
        """Standard training procedure"""
        for batch_idx, batch in enumerate(self.trainloader):
            # Handle both dict and tuple formats
            if isinstance(batch, dict):
                data, target = batch["features"], batch["label"]
            else:
                data, target = batch
            
            features = data.to(self.device)
            labels = target.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def _train_with_label_flipping(self, optimizer, criterion):
        """Training with label flipping strategies"""
        label_attack_params = self.attack_params.get("label_flip", {})
        flip_strategy = label_attack_params.get("flip_strategy", "targeted")
        flip_ratio = label_attack_params.get("flip_ratio", 0.5)
        target_label = label_attack_params.get("target_label", 0)
        source_label = label_attack_params.get("source_label", None)
        
        logger.info(f"Label flipping strategy: {flip_strategy}, ratio: {flip_ratio}")
        
        flipped_count = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.trainloader):
            # Handle both dict and tuple formats
            if isinstance(batch, dict):
                data, target = batch["features"], batch["label"]
            else:
                data, target = batch
            
            features = data.to(self.device)
            labels = target.to(self.device)
            total_samples += len(labels)
            
            modified_labels = labels.clone()
            
            if flip_strategy == "targeted":
                # Targeted flipping: specific source -> target
                for i in range(len(labels)):
                    if (source_label is None or labels[i] == source_label) and random.random() < flip_ratio:
                        modified_labels[i] = target_label
                        flipped_count += 1
                        
            elif flip_strategy == "random":
                # Random flipping: any label -> random other label
                num_classes = label_attack_params.get("num_classes", 10)
                for i in range(len(labels)):
                    if random.random() < flip_ratio:
                        original_label = labels[i].item()
                        available_labels = [l for l in range(num_classes) if l != original_label]
                        if available_labels:
                            modified_labels[i] = random.choice(available_labels)
                            flipped_count += 1
            
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, modified_labels)
            loss.backward()
            optimizer.step()
        
        logger.info(f"Label flipping attack: {flipped_count}/{total_samples} labels flipped")

    def _train_with_backdoor(self, optimizer, criterion):
        """Training with backdoor triggers"""
        trigger_ratio = self.attack_params.get("trigger_ratio", 0.2)
        target_label = self.attack_params.get("target_label", 0)
        trigger_pattern = self.attack_params.get("trigger_pattern", "bottom_right")
        
        triggered_count = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.trainloader):
            # Handle both dict and tuple formats
            if isinstance(batch, dict):
                data, target = batch["features"], batch["label"]
            else:
                data, target = batch
            
            features = data.to(self.device)
            labels = target.to(self.device)
            total_samples += len(labels)
            
            if len(features.shape) == 4:  # Image data
                modified_features = features.clone()
                modified_labels = labels.clone()
                
                for i in range(len(features)):
                    if random.random() < trigger_ratio:
                        triggered_count += 1
                        if trigger_pattern == "bottom_right":
                            modified_features[i, :, -5:, -5:] = 1.0
                        elif trigger_pattern == "random_noise":
                            noise = torch.rand_like(modified_features[i]) * 0.2
                            modified_features[i] += noise
                        modified_labels[i] = target_label
                
                optimizer.zero_grad()
                outputs = self.model(modified_features)
                loss = criterion(outputs, modified_labels)
            else:  # Non-image data
                modified_labels = labels.clone()
                for i in range(len(labels)):
                    if random.random() < trigger_ratio:
                        modified_labels[i] = target_label
                        triggered_count += 1
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, modified_labels)
                
            loss.backward()
            optimizer.step()
        
        logger.info(f"Backdoor attack: {triggered_count}/{total_samples} samples triggered")

    def _apply_model_poisoning(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Enhanced model poisoning with aggressive options and layer targeting"""
        poisoned_params = []
        poisoning_method = self.attack_params.get("poisoning_method", "noise")
        scale_attack_params = self.attack_params.get("scale", {})
        scale_factor = scale_attack_params.get("scale_factor", 1.5)
        noise_factor = self.attack_params.get("noise_factor", 0.5)
        constant_attack_params = self.attack_params.get("constant_noise", {})
        constant_noise_value = constant_attack_params.get("constant_noise_value", 0.1)

        # Target specific layers more aggressively
        target_layers = self.attack_params.get("target_layers", "all")
        layer_specific_factor = self.attack_params.get("layer_specific_factor", 2.0)
        
        for i, param in enumerate(parameters):
            param_copy = param.copy().astype(np.float64)
            
            # Check if layer should be targeted more aggressively
            is_targeted = False
            if target_layers == "all":
                is_targeted = True
            elif target_layers == "first" and i == 0:
                is_targeted = True
            elif target_layers == "last" and i == len(parameters) - 1:
                is_targeted = True
            elif isinstance(target_layers, list) and i in target_layers:
                is_targeted = True
                
            # Apply more aggressive factors for targeted layers
            actual_scale = scale_factor * (layer_specific_factor if is_targeted else 1.0)
            actual_noise = noise_factor * (layer_specific_factor if is_targeted else 1.0)
            
            # Apply the selected poisoning method
            if poisoning_method == "scale":
                logger.info(f"applying scale attack of scale value: {actual_scale}")
                param_copy *= actual_scale
                
            elif poisoning_method == "gaussian":
                logger.info(f"applying gaussian attack of scale value: {actual_noise}")
                mean = self.attack_params.get("gaussian_mean", 0.0)
                std = self.attack_params.get("gaussian_std", 1.0) * actual_noise
                gaussian_noise = np.random.normal(mean, std, param.shape)
                param_copy += gaussian_noise
        
            elif poisoning_method == "invert":
                logger.info(f"applying invert attack ")
                param_copy = -param_copy * actual_scale
                
            elif poisoning_method == "zero":
                if is_targeted:
                    param_copy = np.zeros_like(param_copy)
                
            elif poisoning_method == "constant_noise":
                logger.info(f"applying constant noise attack of value: {constant_noise_value}")
                if is_targeted:
                    param_copy += constant_noise_value
            
            poisoned_params.append(param_copy)
            
        return poisoned_params

    def evaluate(self, parameters, config=None):  # ✅ Made optional with default
        """Evaluate the model (no attack during evaluation)"""
        if config is None:
            config = {}
        return super().evaluate(parameters, config)