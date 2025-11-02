"""
Federated Learning client modules for Kubernetes deployment
"""

from .client import FlowerClient
from .malicious_client import MaliciousClient

__all__ = ["FlowerClient", "MaliciousClient"]