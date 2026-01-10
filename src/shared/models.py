import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    """MNIST Model - matches simulation exactly"""
    def __init__(self, num_classes=10):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x  # Return raw logits like simulation

class IrisNet(nn.Module):
    """Iris Model - matches simulation exactly"""
    def __init__(self, num_classes=3):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Return raw logits like simulation

class TabularModel(nn.Module):
    """Generic Tabular Model - matches simulation exactly"""
    def __init__(self, input_size, num_classes=2, hidden_sizes=[64, 32]):
        super(TabularModel, self).__init__()
        
        layers = []
        prev_size = input_size

        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = size

        layers.append(nn.Linear(prev_size, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class FiveGNIDDNet(nn.Module):
    """5GNIDD Classifier - matches simulation exactly with BatchNorm"""
    def __init__(self, input_size=20, num_classes=9):
        super(FiveGNIDDNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)  # Return raw logits for better numerical stability

def create_model_for_dataset(dataset_type: str, num_classes: int, input_size=None):
    """Create model based on dataset type - matches simulation exactly"""
    if dataset_type == "mnist":
        return MNISTNet(num_classes=num_classes)
    elif dataset_type == "fashion_mnist":
        # Fashion-MNIST uses same architecture as MNIST (28x28 grayscale)
        return MNISTNet(num_classes=num_classes)
    elif dataset_type == "iris":
        return IrisNet(num_classes=num_classes)
    elif dataset_type == "tabular":
        if input_size is None:
            raise ValueError("input_size must be provided for tabular datasets")
        return TabularModel(input_size=input_size, num_classes=num_classes)
    elif dataset_type == "5gnidd":
        return FiveGNIDDNet(input_size=input_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")