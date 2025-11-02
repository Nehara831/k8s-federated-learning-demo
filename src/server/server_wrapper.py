import flwr as fl
from flwr.server.strategy import FedAvg
from typing import Dict, Optional, Tuple, List
import logging
import torch
from src.shared.models import create_model_for_dataset

logger = logging.getLogger(__name__)

class K8sFederatedStrategy(FedAvg):
    """Custom strategy for Kubernetes FL deployment"""
    
    def __init__(self, config, testloader=None, **kwargs):
        self.config = config
        self.testloader = testloader
        super().__init__(**kwargs)
    
    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure the next round of training"""
        config = {
            "server_round": server_round,
            "local_epochs": self.config.config_fit.local_epochs,
            "lr": self.config.config_fit.lr,
            "momentum": self.config.config_fit.momentum,
        }
        
        # Add attack configuration for malicious clients
        if hasattr(self.config, 'attack') and self.config.attack.enabled:
            config.update({
                "attack_enabled": True,
                "attack_type": self.config.attack.attack_type,
                "attack_params": dict(self.config.attack.attack_params),
                "malicious_ratio": self.config.attack.malicious_ratio
            })
        
        fit_ins = fl.common.FitIns(parameters, config)
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        return [(client, fit_ins) for client in clients]
    
    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Configure the next round of evaluation"""
        config = {
            "server_round": server_round,
        }
        
        evaluate_ins = fl.common.EvaluateIns(parameters, config)
        
        # Sample clients for evaluation
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        return [(client, evaluate_ins) for client in clients]

    def evaluate(self, server_round: int, parameters):
        """Evaluate the global model"""
        if self.testloader is None:
            return None
            
        # Implement server-side evaluation here if needed
        logger.info(f"Server evaluation for round {server_round}")
        return None

def create_strategy(config, initial_parameters, testloader=None):
    """Create the federated learning strategy"""
    
    def fit_config(server_round: int):
        return {
            "server_round": server_round,
            "local_epochs": config.config_fit.local_epochs,
            "lr": config.config_fit.lr,
            "momentum": config.config_fit.momentum,
        }
    
    def evaluate_config(server_round: int):
        return {"server_round": server_round}
    
    strategy = K8sFederatedStrategy(
        config=config,
        testloader=testloader,
        fraction_fit=config.server.fraction_fit,
        fraction_evaluate=config.server.fraction_evaluate,
        min_fit_clients=config.server.min_fit_clients,
        min_evaluate_clients=config.server.min_evaluate_clients,
        min_available_clients=config.server.min_available_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,  # ADD THIS LINE
    )
    
    return strategy