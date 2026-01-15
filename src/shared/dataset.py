import logging
import torch
import torchvision
from torchvision import transforms
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import kagglehub
import os
import shutil
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, FashionMNIST
from typing import List, Optional
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class DictTensorDataset(torch.utils.data.Dataset):
    """TensorDataset that returns dicts instead of tuples (matches simulation)"""
    def __init__(self, X, y):
        self.features = X
        self.targets = y
    
    def __getitem__(self, index):
        return {"features": self.features[index], "label": self.targets[index]}
    
    def __len__(self):
        return len(self.features)

def download_5gnidd_dataset(data_path):
    """Download 5G-NIDD dataset from Kaggle"""
    local_data_dir = os.path.join(data_path, "5gnidd")
    
    # Check if dataset already exists
    if os.path.exists(local_data_dir) and os.listdir(local_data_dir):
        logger.info(f"Dataset already exists in {local_data_dir}")
        files = os.listdir(local_data_dir)
        return local_data_dir, files
    
    try:
        logger.info("Downloading 5G-NIDD dataset from Kaggle...")
        os.makedirs(local_data_dir, exist_ok=True)
        
        # Download to kagglehub cache
        temp_path = kagglehub.dataset_download('humera11/5g-nidd-dataset')
        temp_files = os.listdir(temp_path)
        
        logger.info(f"Downloaded to temporary location: {temp_path}")
        logger.info("Copying to local data folder...")
        
        # Copy files to local directory
        for file in temp_files:
            src = os.path.join(temp_path, file)
            dst = os.path.join(local_data_dir, file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                logger.info(f"Copied: {file}")
        
        local_files = os.listdir(local_data_dir)
        logger.info(f"Dataset saved to: {local_data_dir}")
        return local_data_dir, local_files
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None, None

def load_and_preprocess_5gnidd(dataset_path, dataset_files, max_samples_per_class=5000):
    """Load and preprocess 5G-NIDD dataset"""
    try:
        if dataset_path is None:
            return None, None, None

        csv_files = [f for f in dataset_files if f.endswith('.csv')]
        if not csv_files:
            return None, None, None

        file_path = os.path.join(dataset_path, csv_files[0])
        logger.info(f"Reading CSV from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset: {df.shape}")

        # Drop unnecessary columns
        drop_cols = ['Unnamed: 0', 'Seq', 'Label', 'Attack Tool']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
        logger.info(f"After dropping columns: {df.shape}")

        target_column = "Attack Type"
        if target_column not in df.columns:
            logger.error("'Attack Type' column not found.")
            logger.error(f"Available columns: {df.columns.tolist()}")
            return None, None, None

        df = df.dropna(subset=[target_column])
        logger.info(f"After dropping NAs: {df.shape}")

        # Encode target
        le_target = LabelEncoder()
        df[target_column] = le_target.fit_transform(df[target_column].astype(str))
        y = df[target_column]

        class_counts = y.value_counts()
        logger.info(f"Classes found: {len(class_counts)}")
        logger.info(f"Class distribution: {class_counts.to_dict()}")

        # Balance dataset
        logger.info(f"Balancing dataset with max {max_samples_per_class} per class...")
        df_balanced = df.groupby(target_column, group_keys=False).apply(
            lambda x: x.sample(n=min(max_samples_per_class, len(x)), random_state=42)
        ).reset_index(drop=True)

        logger.info(f"Balanced dataset shape: {df_balanced.shape}")

        X = df_balanced.drop(columns=[target_column])
        y = df_balanced[target_column]

        # Process features
        categorical_cols = X.select_dtypes(include=['object']).columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        logger.info(f"Numeric columns: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")

        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        for col in categorical_cols:
            X[col] = X[col].fillna('Unknown')
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_features = min(20, X.shape[1])
        logger.info(f"Selecting top {n_features} features...")
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X_scaled, y)
        
        logger.info(f"Preprocessing complete: X shape {X_selected.shape}, y shape {y.shape}")
        return X_selected, y.values, scaler
        
    except Exception as e:
        logger.error(f"Error in load_and_preprocess_5gnidd: {e}", exc_info=True)
        return None, None, None

def create_non_iid_partitions(dataset_size: int, num_partitions: int, skew_ratio: float = 3.0, seed: int = 123) -> List[int]:
    """
    Create Non-IID partition sizes with quantity skew.
    
    Args:
        dataset_size: Total number of samples in the dataset
        num_partitions: Number of clients
        skew_ratio: Controls the level of imbalance (higher = more imbalance)
                   Typical values: 1.0 (mild), 3.0 (moderate), 5.0+ (severe)
        seed: Random seed for reproducibility
    
    Returns:
        List of partition sizes for each client
    """
    rng = np.random.RandomState(seed)
    
    # Generate random proportions using exponential distribution
    proportions = rng.exponential(scale=skew_ratio, size=num_partitions)
    
    # Normalize proportions to sum to 1
    proportions = proportions / proportions.sum()
    
    # Convert to actual sample counts
    partition_sizes = (proportions * dataset_size).astype(int)
    
    # Ensure each client has at least 2 samples
    min_samples = 2
    for i in range(len(partition_sizes)):
        if partition_sizes[i] < min_samples:
            partition_sizes[i] = min_samples
    
    # Adjust last partition to use all remaining samples
    partition_sizes[-1] = dataset_size - partition_sizes[:-1].sum()
    
    # If last partition is negative or too small, redistribute
    if partition_sizes[-1] < min_samples:
        partition_sizes = np.full(num_partitions, dataset_size // num_partitions)
        partition_sizes[-1] = dataset_size - partition_sizes[:-1].sum()
    
    return partition_sizes.tolist()

def create_label_skew_partitions(targets: np.ndarray, num_partitions: int, num_classes: int, alpha: float = 0.5, seed: int = 123) -> List[List[int]]:
    """
    Create Non-IID partitions with label distribution skew using Dirichlet distribution.
    
    Args:
        targets: Array of labels
        num_partitions: Number of clients
        num_classes: Number of classes in the dataset
        alpha: Dirichlet concentration parameter
               - Small alpha (0.1-0.5): High skew, clients see few classes
               - Medium alpha (1.0): Moderate skew
               - Large alpha (10+): Low skew, closer to IID
        seed: Random seed for reproducibility
    
    Returns:
        List of lists, where each inner list contains indices for one client
    """
    rng = np.random.RandomState(seed)
    
    # Create index lists for each class
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
    
    # Initialize client partitions
    client_indices = [[] for _ in range(num_partitions)]
    
    # For each class, use Dirichlet to split samples among clients
    for class_idx in range(num_classes):
        indices = class_indices[class_idx]
        rng.shuffle(indices)
        
        # Sample proportions from Dirichlet distribution
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_partitions))
        
        # Split indices according to proportions
        split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, split_points)
        
        # Assign to clients
        for client_id, client_class_indices in enumerate(split_indices):
            client_indices[client_id].extend(client_class_indices.tolist())
    
    # Shuffle each client's indices
    for client_id in range(num_partitions):
        rng.shuffle(client_indices[client_id])
    
    # Ensure each client has at least 2 samples
    min_samples = 2
    for client_id in range(num_partitions):
        if len(client_indices[client_id]) < min_samples:
            logger.warning(f"Client {client_id} has only {len(client_indices[client_id])} samples")
    
    return client_indices

def get_client_dataset(client_id: int, config):
    """Get dataset partition for a specific client"""
    
    logger.info(f"Loading dataset for client_id: {client_id}, config.num_clients: {config.num_clients}")
    
    if config.dataset.type == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(
            root='/app/data', train=True, download=True, transform=transform
        )
    
    elif config.dataset.type == "fashion_mnist":
        # Fashion-MNIST: same transform as MNIST (28x28 grayscale)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST mean/std
        ])
        trainset = torchvision.datasets.FashionMNIST(
            root='/app/data', train=True, download=True, transform=transform
        )
        
    elif config.dataset.type == "iris":
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        trainset = DictTensorDataset(X_train_tensor, y_train_tensor)
    
    elif config.dataset.type == "5gnidd":
        logger.info("Loading 5G-NIDD dataset...")
        
        # Check for cached preprocessed data first
        use_cache = os.environ.get('USE_CACHED_DATA', 'false').lower() == 'true'
        cache_path = os.environ.get('CACHED_DATA_PATH', '/shared-data/preprocessed_data.pkl')
        
        if use_cache and os.path.exists(cache_path):
            logger.info(f"Loading preprocessed data from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                X = cached['X']
                y = cached['y']
                logger.info(f"Loaded cached data: X shape {X.shape}, y shape {y.shape}")
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
                logger.info("Falling back to download and preprocess...")
                use_cache = False
        else:
            if use_cache:
                logger.warning(f"Cache enabled but file not found: {cache_path}")
            use_cache = False
        
        if not use_cache:
            # Download dataset
            data_path = config.dataset.get('data_dir', '/app/data')
            dataset_path, dataset_files = download_5gnidd_dataset(data_path)
            
            if dataset_path is None:
                raise ValueError("Failed to download 5G-NIDD dataset")
            
            # Load and preprocess
            X, y, scaler = load_and_preprocess_5gnidd(dataset_path, dataset_files)
            
            if X is None:
                raise ValueError("Failed to preprocess 5G-NIDD dataset")
        
        # ✅ KEY CHANGE: Use PyTorch random_split instead of sklearn (matches simulation)
        # Create full dataset first
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        full_dataset = DictTensorDataset(X_tensor, y_tensor)
        
        # Split using PyTorch (matches simulation exactly)
        test_size = config.dataset.get('test_size', 0.2)
        seed = config.dataset.get('seed', 42)
        test_len = int(len(full_dataset) * test_size)
        train_len = len(full_dataset) - test_len
        
        trainset, testset = torch.utils.data.random_split(
            full_dataset,
            [train_len, test_len],
            generator=torch.Generator().manual_seed(seed)
        )
        
        logger.info(f"5G-NIDD dataset loaded: {len(trainset)} training samples")
    
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset.type}")
    
    # Get distribution configuration
    num_clients = config.num_clients
    distribution_config = config.dataset.get('distribution', {})
    distribution_type = distribution_config.get('type', 'iid')
    partition_seed = distribution_config.get('partition_seed', 123)
    
    logger.info(f"Total samples: {len(trainset)}, Clients: {num_clients}")
    logger.info(f"Distribution type: {distribution_type}")
    
    # Create partitions based on distribution type
    if distribution_type == "label_skew":
        # Label Distribution Skew (Dirichlet)
        alpha = distribution_config.get('alpha', 0.5)
        num_classes = config.num_classes
        
        logger.info(f"Using Label Skew distribution (Dirichlet alpha={alpha}, {num_classes} classes)")
        
        # Extract targets from dataset
        if hasattr(trainset, 'targets'):
            targets = np.array(trainset.targets if isinstance(trainset.targets, list) else trainset.targets)
        elif hasattr(trainset, 'dataset') and hasattr(trainset.dataset, 'targets'):
            # Handle Subset
            targets = np.array(trainset.dataset.targets)
            if hasattr(trainset, 'indices'):
                targets = targets[trainset.indices]
        else:
            # For custom datasets, iterate and collect labels
            logger.info("Extracting labels from dataset...")
            targets = []
            for i in range(len(trainset)):
                sample = trainset[i]
                if isinstance(sample, dict):
                    label = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
                else:
                    label = sample[1].item() if torch.is_tensor(sample[1]) else sample[1]
                targets.append(label)
            targets = np.array(targets)
        
        # Get indices for all clients
        client_indices_all = create_label_skew_partitions(
            targets=targets,
            num_partitions=num_clients,
            num_classes=num_classes,
            alpha=alpha,
            seed=partition_seed
        )
        
        # Get this client's indices
        effective_client_id = client_id % num_clients
        client_indices = client_indices_all[effective_client_id]
        partition_lengths = [len(indices) for indices in client_indices_all]
        
        logger.info(f"Partition lengths: {partition_lengths}")
        logger.info(f"Client {client_id} (effective: {effective_client_id}) dataset size: {len(client_indices)}")
        
        # Create subset for this client
        client_trainset = Subset(trainset, client_indices)
        
        # Log label distribution for this client
        client_labels = targets[client_indices]
        label_counts = np.bincount(client_labels, minlength=num_classes)
        label_dist = label_counts / label_counts.sum() if label_counts.sum() > 0 else label_counts
        
        logger.info(f"📊 Client {client_id} Data Distribution:")
        logger.info(f"   Total samples: {len(client_indices)}")
        logger.info(f"   Number of classes: {num_classes}")
        logger.info(f"   Class distribution: {dict(enumerate(label_counts))}")
        logger.info(f"   Class percentages: {dict(enumerate([f'{p*100:.1f}%' for p in label_dist]))}")
        
        # Check for class imbalance
        if label_counts.max() > 0:
            imbalance_ratio = label_counts.max() / (label_counts[label_counts > 0].min() if any(label_counts > 0) else 1)
            if imbalance_ratio > 2.0:
                logger.warning(f"   ⚠️  Class imbalance detected! Ratio: {imbalance_ratio:.2f}:1")
    
    elif distribution_type == "non_iid":
        # Quantity Skew (uneven data amounts)
        skew_ratio = distribution_config.get('quantity_skew_ratio', 3.0)
        
        logger.info(f"Using Non-IID distribution with quantity skew (ratio={skew_ratio})")
        
        partition_lengths = create_non_iid_partitions(
            dataset_size=len(trainset),
            num_partitions=num_clients,
            skew_ratio=skew_ratio,
            seed=partition_seed
        )
        
        # Calculate start and end indices for this client
        effective_client_id = client_id % num_clients
        start_idx = sum(partition_lengths[:effective_client_id])
        end_idx = start_idx + partition_lengths[effective_client_id]
        client_indices = list(range(start_idx, end_idx))
        
        logger.info(f"Partition lengths: {partition_lengths}")
        logger.info(f"Client {client_id} (effective: {effective_client_id}) dataset size: {len(client_indices)}")
        
        # Create subset for this client
        client_trainset = Subset(trainset, client_indices)
        
        # Log distribution
        _log_client_data_distribution(client_id, client_trainset, config.dataset.type)
    
    else:
        # IID: equal partition sizes
        logger.info("Using IID distribution (equal partition sizes)")
        
        partition_size = len(trainset) // num_clients
        remainder = len(trainset) % num_clients
        
        partition_lengths = [partition_size] * num_clients
        partition_lengths[-1] += remainder
        
        logger.info(f"Partition lengths: {partition_lengths}")
        
        # Use random_split with fixed seed for reproducible IID partitioning
        all_partitions = torch.utils.data.random_split(
            trainset, 
            partition_lengths,
            generator=torch.Generator().manual_seed(partition_seed)
        )
        
        # Get this client's partition
        effective_client_id = client_id % num_clients
        client_trainset = all_partitions[effective_client_id]
        
        logger.info(f"Client {client_id} (effective: {effective_client_id}) dataset size: {len(client_trainset)}")
        
        # Log distribution
        _log_client_data_distribution(client_id, client_trainset, config.dataset.type)
    
    if len(client_trainset) == 0:
        raise ValueError(f"Client {client_id} has empty dataset! Check partitioning logic.")
    
    # ✅ MATCH SIMULATION: Create validation set (10% of client's data)
    num_total = len(client_trainset)
    val_ratio = 0.1
    num_val = max(1, int(val_ratio * num_total))
    num_train = num_total - num_val
    
    if num_train == 0:
        num_train = 1
        num_val = num_total - 1
    
    client_trainset, client_valset = torch.utils.data.random_split(
        client_trainset, [num_train, num_val],
        generator=torch.Generator().manual_seed(123)
    )
    
    # ✅ MATCH SIMULATION: Use drop_last=True
    trainloader = torch.utils.data.DataLoader(
        client_trainset, 
        batch_size=min(config.batch_size, num_train), 
        shuffle=True,
        drop_last=True
    )
    valloader = torch.utils.data.DataLoader(
        client_valset, 
        batch_size=min(config.batch_size, num_val),
        shuffle=True,
        drop_last=True
    )
    
    return trainloader, valloader


def _log_client_data_distribution(client_id: int, dataset, dataset_type: str):
    """
    Log the class distribution for a client's dataset
    
    Args:
        client_id: Client identifier
        dataset: The dataset partition
        dataset_type: Type of dataset (mnist, iris, 5gnidd)
    """
    try:
        # Extract all labels from the dataset
        labels = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # Handle different dataset formats
            if isinstance(sample, dict):
                # DictTensorDataset format
                label = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
            elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
                # TensorDataset format (features, label)
                label = sample[1].item() if torch.is_tensor(sample[1]) else sample[1]
            else:
                logger.warning(f"Unknown sample format: {type(sample)}")
                continue
            
            labels.append(label)
        
        # Count class distribution
        if len(labels) > 0:
            labels_array = np.array(labels)
            unique_classes, counts = np.unique(labels_array, return_counts=True)
            
            # Create distribution dictionary
            distribution = {int(cls): int(count) for cls, count in zip(unique_classes, counts)}
            
            # Log detailed distribution
            logger.info(f"📊 Client {client_id} Data Distribution:")
            logger.info(f"   Total samples: {len(labels)}")
            logger.info(f"   Number of classes: {len(unique_classes)}")
            logger.info(f"   Class distribution: {distribution}")
            
            # Calculate and log percentages
            percentages = {cls: (count/len(labels))*100 for cls, count in distribution.items()}
            logger.info(f"   Class percentages: {{{', '.join([f'{cls}: {pct:.1f}%' for cls, pct in percentages.items()])}}}")
            
            # Check for class imbalance
            if len(counts) > 1:
                imbalance_ratio = max(counts) / min(counts)
                if imbalance_ratio > 2.0:
                    logger.warning(f"   ⚠️  Class imbalance detected! Ratio: {imbalance_ratio:.2f}:1")
        else:
            logger.warning(f"Client {client_id}: No labels extracted from dataset")
            
    except Exception as e:
        logger.error(f"Failed to analyze data distribution for client {client_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())

def prepare_server_dataset(config):
    """Load test dataset for server evaluation"""
    
    dataset_type = config.dataset.type
    
    if dataset_type == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        testset = MNIST(root=config.dataset.data_dir, train=False, download=True, transform=transform)
        
    elif dataset_type == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        testset = FashionMNIST(root=config.dataset.data_dir, train=False, download=True, transform=transform)
        
    elif dataset_type == "5gnidd":
        logger.info("Loading 5G-NIDD test dataset...")
        cached_data_path = os.getenv('CACHED_DATA_PATH', '/shared-data/preprocessed_data.pkl')
        
        if os.path.exists(cached_data_path):
            logger.info(f"Loading preprocessed data from cache: {cached_data_path}")
            with open(cached_data_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check the structure of cached data
            logger.info(f"Cached data keys: {list(data.keys())}")
            
            # Handle different cache file formats
            if 'X_test' in data and 'y_test' in data:
                X_test = data['X_test']
                y_test = data['y_test']
            elif 'test_data' in data:
                X_test = data['test_data']
                y_test = data['test_labels']
            else:
                # Assume it's the full dataset, split it
                X = data['X']
                y = data['y']
                scaler = data['scaler']
                
                # Use the same split ratio as client loading
                test_size = 0.2
                from sklearn.model_selection import train_test_split
                _, X_test, _, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            
            logger.info(f"Loaded cached data: X shape {X_test.shape}, y shape {y_test.shape}")
            
            class NIDDDataset(torch.utils.data.Dataset):
                def __init__(self, X, y):
                    self.X = torch.FloatTensor(X)
                    self.y = torch.LongTensor(y)
                
                def __getitem__(self, idx):
                    return {"features": self.X[idx], "label": self.y[idx]}
                
                def __len__(self):
                    return len(self.X)
            
            testset = NIDDDataset(X_test, y_test)
        else:
            raise FileNotFoundError(f"Cached data not found at {cached_data_path}")
    
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    logger.info(f"5G-NIDD test dataset loaded: {len(testset)} samples")
    
    testloader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    
    return testloader