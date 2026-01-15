

import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel
import numpy as np
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class FeatureTokenizer(nn.Module):
    """
    Converts 22 numerical features into a sequence of 22 dense vectors.
    This acts as the "Input Embedding" layer for the Transformer.
    """
    def __init__(self, num_features=22, embedding_dim=768):
        super().__init__()
        
        
        self.feature_projectors = nn.ModuleList([
            nn.Linear(1, embedding_dim) for _ in range(num_features)
        ])
        

        self.feature_id_bias = nn.Parameter(torch.randn(1, num_features, embedding_dim))
        
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        
        embeddings = []
        for i, projector in enumerate(self.feature_projectors):
            feat_val = x[:, i].unsqueeze(1)  # [batch_size, 1]
            emb = projector(feat_val)  # [batch_size, embedding_dim]
            embeddings.append(emb)
            
        x_emb = torch.stack(embeddings, dim=1)
        
        x_emb = x_emb + self.feature_id_bias
        
        return self.layer_norm(x_emb)


class NumDistilBERT(nn.Module):
    """
    Pointwise malicious client detector using DistilBERT.
    
    Architecture:
    1. Feature Tokenizer: 22 scalars -> 22 token embeddings
    2. CLS Token: Special token to aggregate client status
    3. DistilBERT: Self-attention over feature tokens
    4. Classification Head: Binary output (benign/malicious)
    """
    def __init__(self, num_features=22):
        super().__init__()
        
        self.input_embedding = FeatureTokenizer(num_features, embedding_dim=768)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        
        config = DistilBertConfig(
            dim=768, 
            n_layers=4,        
            n_heads=8, 
            vocab_size=1,      
            max_position_embeddings=512
        )
        self.backbone = DistilBertModel(config)
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_features] - Client feature vectors
        Returns:
            [batch_size] - Logits for binary classification
        """
        batch_size = x.shape[0]
        
        feature_embeds = self.input_embedding(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        inputs_embeds = torch.cat((cls_tokens, feature_embeds), dim=1)
        
        outputs = self.backbone(inputs_embeds=inputs_embeds)
        
        cls_state = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(cls_state)
        
        return logits.squeeze(-1)


class NumDistilBERTDetector:
    
    def __init__(self, num_features=22, model_path=None, scaler_path=None):
        
        self.num_features = num_features
        self.model = NumDistilBERT(num_features)
        self.scaler = None
        self.is_trained = False
        
        self.feature_names = [
            'param_mean', 'param_std', 'param_min', 'param_max',
            'param_median', 'param_range', 'param_abs_mean',
            'param_skew', 'param_kurtosis', 'param_neg_ratio',
            'param_zero_ratio', 'last_layer_mean', 'last_layer_std',
            'last_layer_min', 'last_layer_max', 'last_layer_abs_mean',
            'last_layer_neg_ratio', 'first_vs_last_mean_ratio',
            'first_vs_last_std_ratio', 'avg_l1_distance',
            'avg_l2_distance', 'cosine_similarity'
        ]
        
        if model_path:
            self.load_model(model_path)
        
        if scaler_path:
            self.load_scaler(scaler_path)
        
        logger.info(f" Initialized Num-DistilBERT detector")
        logger.info(f"   Features: {num_features}")
        logger.info(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"   Trained: {self.is_trained}")
    
    def load_model(self, model_path):
        """Load pre-trained model weights"""
        try:
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            except TypeError:
                checkpoint = torch.load(model_path, map_location='cpu')
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.is_trained = True
            logger.info(f" Loaded model from {model_path}")
            logger.info(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
            logger.info(f"   Best F1: {checkpoint.get('best_f1', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_trained = False
    
    def load_scaler(self, scaler_path):
        """Load fitted MinMaxScaler"""
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f" Loaded scaler from {scaler_path}")
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            self.scaler = None
    
    def predict(self, features, threshold=0.5):
        """
        Predict if a client is malicious.
        
        Args:
            features: Dict or array of client features
            threshold: Decision threshold (default: 0.5)
        
        Returns:
            prediction: 0 (benign) or 1 (malicious)
            probability: Malicious probability [0, 1]
        """
        if not self.is_trained:
            logger.warning("Model not trained - returning random prediction")
            return 0, 0.5
        
        try:
            if isinstance(features, dict):
                feature_array = np.array([
                    features.get(name, 0.0) for name in self.feature_names
                ])
            else:
                feature_array = np.array(features)
            
            if self.scaler:
                feature_array = self.scaler.transform([feature_array])[0]
            
            x = torch.FloatTensor(feature_array).unsqueeze(0)  # [1, num_features]
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                logit = self.model(x)
                prob = torch.sigmoid(logit).item()
            
            prediction = 1 if prob > threshold else 0
            
            return prediction, prob
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0, 0.5
    
    def predict_batch(self, features_list, threshold=0.5):
        """
        Predict for multiple clients at once.
        
        Args:
            features_list: List of feature dicts or arrays
            threshold: Decision threshold
        
        Returns:
            predictions: List of 0/1 predictions
            probabilities: List of probabilities
        """
        if not self.is_trained:
            logger.warning("Model not trained")
            return [0] * len(features_list), [0.5] * len(features_list)
        
        try:
            # Convert to arrays
            feature_arrays = []
            for features in features_list:
                if isinstance(features, dict):
                    arr = np.array([features.get(name, 0.0) for name in self.feature_names])
                else:
                    arr = np.array(features)
                feature_arrays.append(arr)
            
            feature_arrays = np.array(feature_arrays)
            
            # Scale
            if self.scaler:
                feature_arrays = self.scaler.transform(feature_arrays)
            
            # Predict
            x = torch.FloatTensor(feature_arrays)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
            
            predictions = [1 if p > threshold else 0 for p in probs]
            
            return predictions, probs.tolist()
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return [0] * len(features_list), [0.5] * len(features_list)
    
    def save(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_dir / "num_distilbert_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_features': self.num_features,
            'feature_names': self.feature_names
        }, model_path)
        
        # Save scaler
        if self.scaler:
            scaler_path = save_dir / "num_distilbert_scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"✅ Saved model to {save_dir}")
        
    def aggregate_fit(self, server_round, results):
        
        client_predictions = {}
        
        if self.malicious_detector:
            self.malicious_detector.start_new_round(server_round)

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            
            if self.malicious_detector:
                prev_model = self.get_model_from_round(server_round - 1)
                
                prediction = self.malicious_detector.update_client_behavior(
                    client_id=cid,
                    round_num=server_round,
                    current_round_params=parameters_to_ndarrays(fit_res.parameters),
                    prev_round_params=prev_model
                )
                
                client_predictions[cid] = prediction


        return aggregated_parameters, client_predictions
