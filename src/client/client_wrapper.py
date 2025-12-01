import torch
from pathlib import Path
import logging
from src.shared.models import create_model_for_dataset
from src.shared.dataset import get_client_dataset
import sys
import os
from flwr.common import (
    FitIns,
    EvaluateIns,
    FitRes,
    EvaluateRes,
    Status,
    Code,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    # Optional: cover full Client API
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Properties,
)

# Add the modules directory to Python path
sys.path.append('/app/modules')

from client import FlowerClient
from malicious_client import MaliciousClient

logger = logging.getLogger(__name__)

class K8sFlowerClient(FlowerClient):
    """Kubernetes-compatible Flower Client"""
    
    def __init__(self, model, trainloader, valloader, client_id, save_dir, dataset_type="mnist", is_malicious=False):
        super().__init__(model, trainloader, valloader, client_id, save_dir, dataset_type)
        self.is_malicious = is_malicious
        # Disable multi-processing for DataLoader to prevent disk space issues
        self.trainloader.num_workers = 0
        self.valloader.num_workers = 0

    # Shim FitIns -> NumPyClient.fit
    def fit(self, parameters, config=None):
        if isinstance(parameters, FitIns):
            nds = parameters_to_ndarrays(parameters.parameters)
            cfg = dict(parameters.config)
            new_params, num_examples, metrics = super().fit(nds, cfg)
            # CRITICAL: Add client_id to metrics for deterministic identification
            metrics["client_id"] = str(self.client_id)
            return FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(new_params),
                num_examples=num_examples,
                metrics=metrics,
            )
        # Fallback path: parameters is already ndarrays
        new_params, num_examples, metrics = super().fit(parameters, config or {})
        # CRITICAL: Add client_id to metrics here too!
        metrics["client_id"] = str(self.client_id)
        return new_params, num_examples, metrics
    
    # Shim EvaluateIns -> NumPyClient.evaluate
    def evaluate(self, parameters, config=None):
        if isinstance(parameters, EvaluateIns):
            nds = parameters_to_ndarrays(parameters.parameters)
            cfg = dict(parameters.config)
            loss, num_examples, metrics = super().evaluate(nds, cfg)
            return EvaluateRes(
                status=Status(code=Code.OK, message=""),
                loss=loss,
                num_examples=num_examples,
                metrics=metrics,
            )
        return super().evaluate(parameters, config or {})

    # Optional: shims for full Client API
    def get_parameters(self, ins=None):
        if isinstance(ins, GetParametersIns):
            params = super().get_parameters()
            return GetParametersRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(params),
            )
        return super().get_parameters()

    def get_properties(self, ins=None):
        if isinstance(ins, GetPropertiesIns):
            props: Properties = {"client_id": str(self.client_id)}
            return GetPropertiesRes(
                status=Status(code=Code.OK, message=""),
                properties=props,
            )
        return {"client_id": str(self.client_id)}

class K8sMaliciousClient(MaliciousClient):
    """Kubernetes-compatible Malicious Client"""
    
    def __init__(self, model, trainloader, valloader, client_id, save_dir, attack_type, attack_params, dataset_type="mnist"):
        super().__init__(model, trainloader, valloader, client_id, save_dir, True, attack_type, attack_params, dataset_type)
        # Disable multi-processing for DataLoader
        self.trainloader.num_workers = 0
        self.valloader.num_workers = 0
    
    def fit(self, parameters, config=None):
        if isinstance(parameters, FitIns):
            nds = parameters_to_ndarrays(parameters.parameters)
            cfg = dict(parameters.config)
            new_params, num_examples, metrics = super().fit(nds, cfg)
            # CRITICAL: Add client_id to metrics for deterministic identification
            metrics["client_id"] = str(self.client_id)
            return FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(new_params),
                num_examples=num_examples,
                metrics=metrics,
            )
        # Fallback path: parameters is already ndarrays
        new_params, num_examples, metrics = super().fit(parameters, config or {})
        # CRITICAL: Add client_id to metrics here too!
        metrics["client_id"] = str(self.client_id)
        return new_params, num_examples, metrics

    def evaluate(self, parameters, config=None):
        if isinstance(parameters, EvaluateIns):
            nds = parameters_to_ndarrays(parameters.parameters)
            cfg = dict(parameters.config)
            loss, num_examples, metrics = super().evaluate(nds, cfg)
            return EvaluateRes(
                status=Status(code=Code.OK, message=""),
                loss=loss,
                num_examples=num_examples,
                metrics=metrics,
            )
        return super().evaluate(parameters, config or {})

    def get_parameters(self, ins=None):
        if isinstance(ins, GetParametersIns):
            params = super().get_parameters()
            return GetParametersRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(params),
            )
        return super().get_parameters()

    def get_properties(self, ins=None):
        if isinstance(ins, GetPropertiesIns):
            props: Properties = {"client_id": str(self.client_id), "malicious": True}
            return GetPropertiesRes(
                status=Status(code=Code.OK, message=""),
                properties=props,
            )
        return {"client_id": str(self.client_id), "malicious": True}

def create_client(client_id: int, config):
    """Create a client instance (benign or malicious)"""
    
    # Load client-specific data
    trainloader, valloader = get_client_dataset(
        client_id=client_id,
        config=config
    )
    
    # Create model
    model = create_model_for_dataset(
        dataset_type=config.dataset.type,
        num_classes=config.num_classes,
        input_size=20 if config.dataset.type == "5gnidd" else None
    )
    
    # Determine if this client should be malicious
    is_malicious = should_be_malicious(client_id, config)
    
    save_dir = Path(f"/app/outputs/client_{client_id}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if is_malicious and config.attack.enabled:
        logger.info(f"Creating malicious client {client_id}")
        client = K8sMaliciousClient(
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            client_id=client_id,
            save_dir=save_dir,
            attack_type=config.attack.attack_type,
            attack_params=dict(config.attack.attack_params),
            dataset_type=config.dataset.type
        )
    else:
        logger.info(f"Creating benign client {client_id}")
        client = K8sFlowerClient(
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            client_id=client_id,
            save_dir=save_dir,
            dataset_type=config.dataset.type,
            is_malicious=False
        )
    
    return client

def should_be_malicious(client_id: int, config) -> bool:
    """Determine if client should be malicious based on config"""
    if not (hasattr(config, 'attack') and config.attack.enabled):
        return False
    
    malicious_ratio = config.attack.malicious_ratio
    total_clients = config.num_clients
    num_malicious = max(1, round(total_clients * malicious_ratio))  # At least 1 malicious if enabled
    
    # Use modulo to handle hash-based IDs in Kubernetes Deployment mode
    normalized_id = client_id % total_clients
    
    return normalized_id < num_malicious