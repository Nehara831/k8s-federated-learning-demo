import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class FlowerClient(fl.client.NumPyClient):
    """Base Flower client for federated learning"""
    
    def __init__(self, model, trainloader, valloader, client_id, save_dir, dataset_type="mnist"):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.client_id = client_id
        self.save_dir = Path(save_dir)
        self.dataset_type = dataset_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training parameters
        self.epochs = 1
        self.lr = 0.01
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Client {client_id} initialized with {len(trainloader.dataset)} training samples")

    def get_parameters(self, config=None):  # ✅ Made optional
        """Return model parameters as numpy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from numpy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model on local data"""
        self.set_parameters(parameters)
        logger.info(f"Client {self.client_id} starting training round")
        logger.info(f"Client {self.client_id} - Training dataset size: {len(self.trainloader.dataset)}")
        logger.info(f"Client {self.client_id} - Number of batches: {len(self.trainloader)}")
        logger.info(f"Client {self.client_id} - Epochs: {self.epochs}, LR: {self.lr}")
        
        # Train the model
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=config.get("lr", 0.01), momentum=config.get("momentum", 0.9))
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(config.get("local_epochs", 1)):
            logger.info(f"Client {self.client_id} - Starting epoch {epoch+1}/{self.epochs}")
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                num_batches += 1
                epoch_batches += 1
                
                # Log every 5 batches for more visibility
                if batch_idx % 2 == 0:
                    logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{self.epochs}, Batch {batch_idx}/{len(self.trainloader)}, Loss: {loss.item():.6f}")
            
            # Log epoch summary
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
            logger.info(f"Client {self.client_id} - Epoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.6f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Client {self.client_id} training complete. Total batches: {num_batches}, Avg loss: {avg_loss:.6f}")
        logger.info(f"Client {self.client_id} - Returning updated parameters to server")
        
        # Return updated parameters and training metrics
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config=None):  # ✅ Made optional with default
        """Evaluate the model on local validation data"""
        if config is None:
            config = {}
        
        logger.info(f"Client {self.client_id} starting evaluation")
        
        # Set parameters received from server
        self.set_parameters(parameters)
        
        # Evaluate the model
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.valloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += target.size(0)
        
        avg_loss = total_loss / len(self.valloader) if len(self.valloader) > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        
        logger.info(f"Client {self.client_id} evaluation complete. Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, total_samples, {"accuracy": accuracy}