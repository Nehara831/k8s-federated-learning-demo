import os
import logging
import flwr as fl
from pathlib import Path
import torch
from omegaconf import OmegaConf
from src.shared.models import create_model_for_dataset
from src.shared.dataset import prepare_server_dataset
from src.server.server_wrapper import create_strategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_initial_parameters(config):
    """Get initial model parameters"""
    model = create_model_for_dataset(
        dataset_type=config.dataset.type,
        num_classes=config.num_classes,
        input_size=20 if config.dataset.type == "5gnidd" else None
    )
    return [param.cpu().numpy() for param in model.state_dict().values()]

def main():
    # Load configuration
    config_path = os.getenv('CONFIG_PATH', '/app/config/k8s-server.yaml')
    config = OmegaConf.load(config_path)
    
    logger.info(f"Starting FL Server with config: {config}")
    
    # Prepare test dataset for server evaluation
    testloader = prepare_server_dataset(config)
    
    # Get initial parameters
    initial_params = get_initial_parameters(config)
    
    # Create strategy
    strategy = create_strategy(
        config=config,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
        testloader=testloader
    )
    
    # Start server
    server_address = f"0.0.0.0:{config.server.port}"
    logger.info(f"Starting FL Server on {server_address}")
    
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()