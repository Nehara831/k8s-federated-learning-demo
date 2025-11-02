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

logger = logging.getLogger(__name__)

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
        
    elif config.dataset.type == "iris":
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        trainset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    
    elif config.dataset.type == "5gnidd":
        logger.info("Loading 5G-NIDD dataset...")
        
        # Download dataset
        data_path = config.dataset.get('data_dir', '/app/data')
        dataset_path, dataset_files = download_5gnidd_dataset(data_path)
        
        if dataset_path is None:
            raise ValueError("Failed to download 5G-NIDD dataset")
        
        # Load and preprocess
        X, y, scaler = load_and_preprocess_5gnidd(dataset_path, dataset_files)
        
        if X is None:
            raise ValueError("Failed to preprocess 5G-NIDD dataset")
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.dataset.get('test_size', 0.2), 
            random_state=config.dataset.get('seed', 42)
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        trainset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        
        logger.info(f"5G-NIDD dataset loaded: {len(trainset)} training samples")
    
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset.type}")
    
    # Partition dataset for this client
    partition_size = len(trainset) // config.num_clients
    
    # With StatefulSet, client_id is already 0, 1, 2, ... (no modulo needed)
    # But add modulo as safety for Deployment fallback
    effective_client_id = client_id % config.num_clients
    
    start_idx = effective_client_id * partition_size
    end_idx = start_idx + partition_size if effective_client_id < config.num_clients - 1 else len(trainset)
    
    logger.info(f"Client {client_id}: partition [{start_idx}:{end_idx}] of {len(trainset)} samples")
    
    indices = list(range(start_idx, end_idx))
    client_trainset = torch.utils.data.Subset(trainset, indices)
    
    logger.info(f"Client {client_id} dataset size: {len(client_trainset)}")
    
    if len(client_trainset) == 0:
        raise ValueError(f"Client {client_id} has empty dataset! Check partitioning logic.")
    
    # Create validation set (20% of client's data)
    val_size = int(0.2 * len(client_trainset))
    train_size = len(client_trainset) - val_size
    client_trainset, client_valset = torch.utils.data.random_split(
        client_trainset, [train_size, val_size]
    )
    
    trainloader = torch.utils.data.DataLoader(
        client_trainset, batch_size=config.batch_size, shuffle=True
    )
    valloader = torch.utils.data.DataLoader(
        client_valset, batch_size=config.batch_size
    )
    
    return trainloader, valloader

def prepare_server_dataset(config):
    """Prepare test dataset for server evaluation"""
    
    if config.dataset.type == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
    
    elif config.dataset.type == "iris":
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        testset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    elif config.dataset.type == "5gnidd":
        logger.info("Loading 5G-NIDD test dataset...")
        
        # Download dataset
        data_path = config.dataset.get('data_dir', './data')
        dataset_path, dataset_files = download_5gnidd_dataset(data_path)
        
        if dataset_path is None:
            raise ValueError("Failed to download 5G-NIDD dataset")
        
        # Load and preprocess
        X, y, scaler = load_and_preprocess_5gnidd(dataset_path, dataset_files)
        
        if X is None:
            raise ValueError("Failed to preprocess 5G-NIDD dataset")
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.dataset.get('test_size', 0.2), 
            random_state=config.dataset.get('seed', 42)
        )
        
        # Convert to tensors
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        testset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        
        logger.info(f"5G-NIDD test dataset loaded: {len(testset)} samples")
    
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset.type}")
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.batch_size
    )
    
    return testloader