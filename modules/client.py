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
        state_dict = {}
        
        for k, v in params_dict:
            # Handle BatchNorm num_batches_tracked with correct dtype
            if 'num_batches_tracked' in k:
                state_dict[k] = torch.tensor(v, dtype=torch.long)
            else:
                state_dict[k] = torch.tensor(v)
        
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model on local data"""
        self.set_parameters(parameters)
        logger.info(f"Client {self.client_id} starting training round")
        logger.info(f"Client {self.client_id} - Training dataset size: {len(self.trainloader.dataset)}")
        logger.info(f"Client {self.client_id} - Number of batches: {len(self.trainloader)}")
        
        # Get training config
        lr = config.get("lr", 0.01)
        momentum = config.get("momentum", 0.9)
        local_epochs = config.get("local_epochs", 1)
        
        logger.info(f"Client {self.client_id} - Epochs: {local_epochs}, LR: {lr}")
        
        # Train the model
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(local_epochs):
            logger.info(f"Client {self.client_id} - Starting epoch {epoch+1}/{local_epochs}")
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
                
                if batch_idx % 2 == 0:
                    logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{local_epochs}, Batch {batch_idx}/{len(self.trainloader)}, Loss: {loss.item():.6f}")
            
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
            logger.info(f"Client {self.client_id} - Epoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.6f}")
        
        # ✅ ADD VALIDATION AFTER TRAINING (matching simulation)
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in self.valloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        local_accuracy = float(val_correct) / val_total if val_total > 0 else 0.0
        local_loss = val_loss / len(self.valloader) if len(self.valloader) > 0 else 0.0
        
        logger.info(f"Client {self.client_id} - Validation complete. Accuracy: {local_accuracy:.4f}, Loss: {local_loss:.6f}")
        
        # Return metrics matching simulation
        metrics = {
            "local_accuracy": float(local_accuracy),
            "local_loss": float(local_loss),
        }
        
        logger.info(f"Client {self.client_id} - Returning updated parameters to server")
        return self.get_parameters(config), len(self.trainloader.dataset), metrics

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