import os
import logging
from pathlib import Path
from omegaconf import OmegaConf
import flwr as fl

from src.shared.dataset import prepare_server_dataset
from src.shared.models import create_model_for_dataset
from src.server.server_wrapper import create_strategy
from src.detector.malicious_detector import MaliciousClientDetector  # ← ADD THIS

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
    
    # Load test dataset directly from config
    testloader = prepare_server_dataset(config)
    logger.info(f"Test dataset loaded: {len(testloader.dataset)} samples")
    
    # Create initial model
    input_size = 20 if config.dataset.type == "5gnidd" else None
    model = create_model_for_dataset(
        dataset_type=config.dataset.type,
        num_classes=config.num_classes,
        input_size=input_size
    )
    
    # Get initial parameters
    initial_parameters = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for val in model.state_dict().values()]
    )
    
    # Initialize malicious client detector (if enabled)
    malicious_detector = None
    if hasattr(config, 'detector') and config.detector.enabled:
        logger.info("🔍 Initializing Malicious Client Detector")
        
        malicious_detector = MaliciousClientDetector(
            save_dir=Path(config.detector.save_dir),
            num_clients=config.num_clients,
            model_config=config.detector.model if hasattr(config.detector, 'model') else None,
            reference_model=model
        )
        
        logger.info(f"Detector initialized - Trained: {malicious_detector.is_trained}")
    else:
        logger.info("Malicious client detection disabled")
    
    # Create strategy with detector
    strategy = create_strategy(
        config=config,
        initial_parameters=initial_parameters,
        testloader=testloader,
        malicious_detector=malicious_detector  # ← PASS DETECTOR
    )
    
    # Start server
    server_address = f"0.0.0.0:{config.server.port}"
    logger.info(f"Starting FL Server on {server_address}")
    
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
    )
    
    logger.info("FL Server finished")

if __name__ == "__main__":
    main()