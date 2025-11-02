import os
import logging
from omegaconf import OmegaConf
from src.client.client_wrapper import create_client
import flwr as fl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_client_id_from_hostname():
    """Extract client ID from hostname (works for both Deployment and StatefulSet)"""
    hostname = os.environ.get('HOSTNAME', 'unknown')
    logger.info(f"Pod hostname: {hostname}")
    
    # StatefulSet pods: fl-clients-0, fl-clients-1, fl-clients-2
    if '-' in hostname:
        parts = hostname.rsplit('-', 1)
        if parts[-1].isdigit():
            client_id = int(parts[-1])
            logger.info(f"StatefulSet: Extracted client_id: {client_id}")
            return client_id
    
    # Fallback for Deployment: hash-based
    client_id = sum(ord(c) for c in hostname) % 1000
    logger.info(f"Deployment: Generated client_id: {client_id} from hostname hash")
    return client_id

def main():
    hostname = os.environ.get('HOSTNAME', 'unknown')
    client_id = get_client_id_from_hostname()
    
    config_path = os.environ.get('CONFIG_PATH', 'config/k8s-client.yaml')
    cfg = OmegaConf.load(config_path)
    
    server_address = os.environ.get('FL_SERVER_ADDRESS', 'fl-server-service:8080')
    logger.info(f"Starting Client {client_id}, connecting to {server_address}")
    
    client = create_client(client_id, cfg)
    fl.client.start_client(server_address=server_address, client=client.to_client())

if __name__ == "__main__":
    main()