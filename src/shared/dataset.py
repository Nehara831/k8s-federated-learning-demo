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
        trainset = DictTensorDataset(X_train_tensor, y_train_tensor)
    
    elif config.dataset.type == "5gnidd":
        logger.info("Loading 5G-NIDD dataset...")
        
        # Check for cached preprocessed data first
        use_cache = os.environ.get('USE_CACHED_DATA', 'false').lower() == 'true'
        cache_path = os.environ.get('CACHED_DATA_PATH', '/shared-data/preprocessed_data.pkl')
        
        if use_cache and os.path.exists(cache_path):
            logger.info(f"Loading preprocessed data from cache: {cache_path}")
            try:
                import pickle
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
    
    # Partition dataset for this client using IID (random_split with fixed seed)
    num_clients = config.num_clients
    partition_size = len(trainset) // num_clients
    remainder = len(trainset) % num_clients
    
    # Create partition lengths
    partition_lengths = [partition_size] * num_clients
    partition_lengths[-1] += remainder
    
    logger.info(f"Total samples: {len(trainset)}, Clients: {num_clients}")
    logger.info(f"Partition lengths: {partition_lengths}")
    
    # Use random_split with fixed seed for reproducible IID partitioning
    all_partitions = torch.utils.data.random_split(
        trainset, 
        partition_lengths,
        generator=torch.Generator().manual_seed(123)
    )
    
    # Get this client's partition
    effective_client_id = client_id % num_clients
    client_trainset = all_partitions[effective_client_id]
    
    logger.info(f"Client {client_id} (effective: {effective_client_id}) dataset size: {len(client_trainset)}")
    
    # ✅ ADD: Analyze class distribution for this client
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
        
        # Check for cached preprocessed data first
        use_cache = os.environ.get('USE_CACHED_DATA', 'false').lower() == 'true'
        cache_path = os.environ.get('CACHED_DATA_PATH', '/shared-data/preprocessed_data.pkl')
        
        if use_cache and os.path.exists(cache_path):
            logger.info(f"Loading preprocessed data from cache: {cache_path}")
            try:
                import pickle
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