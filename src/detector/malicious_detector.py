from sklearn.linear_model import LogisticRegression
import pickle
import logging
import numpy as np
import torch
from pathlib import Path
import json
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import joblib
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class MaliciousClientDetector:
    def __init__(self, save_dir, num_clients, model_config=None, reference_model=None, min_rounds_before_detection=2):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_clients = num_clients
        self.model_config = model_config
        self.min_rounds_before_detection = min_rounds_before_detection
        
        self.client_behaviors = defaultdict(list)
        self.round_data = []
        self.current_round_malicious = []
        self.malicious_history = {}
        
        self.detector_model = None
        self.tokenizer = None
        self.is_trained = False
        
        self.feature_scaler = MinMaxScaler()
        self.scaler_fitted = False
        self.feature_names = None
        
        self.raw_features_buffer = []
        self.min_samples_for_scaler = 10
        
        self.layer_names = None
        if reference_model is not None:
            self._extract_layer_names_from_model(reference_model)
        
        scaler_loaded = self.load_scaler_from_model_path()
        self.load_distilbert_model()
        
        logger.info(f"Initialized detector for {num_clients} clients")
        logger.info(f"Detection will start from round {min_rounds_before_detection}")
        if self.layer_names:
            logger.info(f"Extracted {len(self.layer_names)} layer names")

    def load_scaler_from_model_path(self):
        """Load MinMaxScaler from model_path/minmax_scaler.pkl"""
        try:
            model_path = getattr(self.model_config, 'model_path', None)
            if model_path is None:
                logger.warning("No model path in config, cannot load scaler.")
                return False
            
            scaler_path = Path(model_path) / "minmax_scaler.pkl"
            
            if not scaler_path.exists():
                logger.info(f"No scaler file found at: {scaler_path}")
                return False

            loaded_scaler = joblib.load(scaler_path)
            
            if hasattr(loaded_scaler, 'transform') and hasattr(loaded_scaler, 'data_min_'):
                self.feature_scaler = loaded_scaler
                self.scaler_fitted = True
                logger.info(f"✓ Successfully loaded scaler with {len(self.feature_scaler.data_min_)} features")
                return True
            else:
                logger.error("Loaded object doesn't have required scaler attributes")
                return False

        except Exception as e:
            logger.error(f"Exception during scaler loading: {e}")
            return False

    def load_distilbert_model(self):
        """Load LoRA adapter on top of base DistilBERT model"""
        try:
            model_path = getattr(self.model_config, 'model_path', None)
            if not model_path:
                logger.warning("No model path in config")
                return

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            base_model_name = "distilbert-base-uncased"
            logger.info(f"Loading base model: {base_model_name}")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=2
            )

            logger.info(f"Loading LoRA adapter from: {model_path}")
            self.detector_model = PeftModel.from_pretrained(base_model, model_path)
            
            logger.info("Merging LoRA adapter with base model...")
            self.detector_model = self.detector_model.merge_and_unload()
            
            self.detector_model.eval()
            self.is_trained = True

            logger.info("✓ Loaded DistilBERT base model + LoRA adapter (merged)")
            
            logger.info("🔍 MODEL VERIFICATION:")
            logger.info(f"  Model type: {type(self.detector_model).__name__}")
            logger.info(f"  Number of parameters: {sum(p.numel() for p in self.detector_model.parameters())}")
            if hasattr(self.detector_model, 'classifier'):
                classifier_weight_mean = self.detector_model.classifier.weight.mean().item()
                classifier_weight_std = self.detector_model.classifier.weight.std().item()
                logger.info(f"  Classifier weight mean: {classifier_weight_mean:.6f}")
                logger.info(f"  Classifier weight std: {classifier_weight_std:.6f}")
                if abs(classifier_weight_mean) < 0.01 and abs(classifier_weight_std - 0.02) < 0.01:
                    logger.warning("⚠️  Classifier weights look UNTRAINED (mean~0, std~0.02)!")
                else:
                    logger.info("✓ Classifier weights appear trained")
            logger.info(f"  Is trained flag: {self.is_trained}")
            
            return

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.exception("Full traceback:")
            self.is_trained = False
            return
        
    def extract_client_features(self, client_id, round_num, current_round_params, prev_round_params):
        """Extract features from client parameters"""
        try:
            if prev_round_params is None or current_round_params is None:
                return None
            
            from src.detector.data_extractor import extract_model_features
            
            features = extract_model_features(
                params=current_round_params,
                client_id=client_id,
                round_num=round_num,
                reference_params=prev_round_params
            )
            
            return features
            
        except Exception as e:
            logger.warning(f"Failed to extract features for client {client_id}: {e}")
            return None

    def create_behavior_description(self, features):
        """Create behavior description with full precision"""
        if not features:
            return "Normal federated learning behavior"
        
        def safe_get(key, default=0.0):
            value = features.get(key, default)
            if hasattr(value, 'item'):
                return float(value.item())
            elif isinstance(value, (np.floating, np.integer)):
                return float(value)
            else:
                return float(value)
        
        text_features = (
            f"Param Mean: {safe_get('param_mean')!r}, Param Std: {safe_get('param_std')!r}, "
            f"Param Min: {safe_get('param_min')!r}, Param Max: {safe_get('param_max')!r}, "
            f"Param Median: {safe_get('param_median')!r}, Param Range: {safe_get('param_range')!r}, "
            f"Param Absolute Mean: {safe_get('param_abs_mean')!r}, Param Skew: {safe_get('param_skew')!r}, "
            f"Param Kurtosis: {safe_get('param_kurtosis')!r}, "
            f"Param Negative Ratio: {safe_get('param_neg_ratio')!r}, "
            f"Param Zero Ratio: {safe_get('param_zero_ratio')!r}, "
            f"Last Layer Mean: {safe_get('last_layer_mean')!r}, Last Layer Std: {safe_get('last_layer_std')!r}, "
            f"Last Layer Min: {safe_get('last_layer_min')!r}, Last Layer Max: {safe_get('last_layer_max')!r}, "
            f"Last Layer Absolute Mean: {safe_get('last_layer_abs_mean')!r}, "
            f"Last Layer Negative Ratio: {safe_get('last_layer_neg_ratio')!r}, "
            f"First vs Last Mean Ratio: {safe_get('first_vs_last_mean_ratio')!r}, "
            f"First vs Last Std Ratio: {safe_get('first_vs_last_std_ratio')!r}, "
            f"Avg L1 Distance: {safe_get('avg_l1_distance')!r}, "
            f"Avg L2 Distance: {safe_get('avg_l2_distance')!r}, "
            f"Cosine Similarity: {safe_get('cosine_similarity')!r}"
        )
        return text_features

    def detect_malicious_behavior(self, behavior_text):
        """Use DistilBERT to detect malicious behavior"""
        if not self.is_trained or not self.detector_model or not self.tokenizer:
            return 0.0, 0
        
        try:
            inputs = self.tokenizer(
                behavior_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
                
            self.detector_model.eval()

            with torch.no_grad():
                outputs = self.detector_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                malicious_probability = probabilities[0][1].item()
                
            logger.info(f"Detection: class={predicted_class}, prob={malicious_probability:.3f}")
            
            return malicious_probability, predicted_class
            
        except Exception as e:
            logger.warning(f"Detection failed: {e}")
            return 0.0, 0

    def scale_features_for_text(self, raw_features):
        """Scale features exactly like the evaluation code before creating text"""
        try:
            if not self.scaler_fitted or self.feature_scaler is None:
                logger.error("Scaler not fitted! Cannot scale features.")
                return None

            feature_dict = {k: raw_features[k] for k in self.feature_scaler.feature_names_in_ if k in raw_features}
            feature_df = pd.DataFrame([feature_dict])
            
            logger.debug(f"🔍 RAW features (before scaling):")
            for key in ['param_mean', 'param_std', 'param_min', 'param_max', 'cosine_similarity']:
                if key in feature_dict:
                    logger.debug(f"  {key}: {feature_dict[key]}")
            
            numerical_cols = feature_df.select_dtypes(include=['float64', 'int64', 'float32']).columns
            feature_df[numerical_cols] = self.feature_scaler.transform(feature_df[numerical_cols])
            scaled_features = feature_df.iloc[0].to_dict()
            
            logger.debug(f"🔍 SCALED features (after scaling):")
            for key in ['param_mean', 'param_std', 'param_min', 'param_max', 'cosine_similarity']:
                if key in scaled_features:
                    logger.debug(f"  {key}: {scaled_features[key]}")
            
            return scaled_features
        except Exception as e:
            logger.error(f"Failed to scale features: {e}")
            return None

    def update_client_behavior(self, client_id, round_num, current_round_params, prev_round_params):
        """Update client behavior tracking with PROPER SCALING"""
        
        if round_num < self.min_rounds_before_detection:
            logger.info(f"Round {round_num} < {self.min_rounds_before_detection} - skipping detection for client {client_id}")
            return 0
        
        curr_param_dict = self._convert_params_to_dict(current_round_params)
        prev_param_dict = self._convert_params_to_dict(prev_round_params)
        
        if prev_round_params is None:
            prev_param_dict = None
            
        raw_features = self.extract_client_features(client_id, round_num, curr_param_dict, prev_param_dict)
        logger.info(f"Client {client_id} Round {round_num} raw features:")
        if not raw_features:
            logger.warning(f"Features are empty, can't predict the output")
            return 0.0
        
        
        scaled_features = self.scale_features_for_text(raw_features)
        if scaled_features is None:
            logger.warning(f"Failed to scale features for client {client_id}")
            return 0.0

        behavior_description = self.create_behavior_description(scaled_features)
        
        logger.info(f"Client {client_id} Round {round_num} behavior text:")
        logger.info(f"  {behavior_description}")
        
        self.client_behaviors[client_id].append(behavior_description)
        
        self.round_data.append({
            'client_id': client_id,
            'round': round_num,
            'behavior_text': behavior_description,
            'raw_features': raw_features,
            'scaled_features': scaled_features
        })
        
        if self.is_trained:
            if client_id == 0 and round_num <= 3:
                logger.info(f"🔍 FEATURE DEBUG Client {client_id} Round {round_num}:")
                logger.info(f"  param_mean: {scaled_features.get('param_mean', 0):.6f}")
                logger.info(f"  param_std: {scaled_features.get('param_std', 0):.6f}")
                logger.info(f"  avg_l1_distance: {scaled_features.get('avg_l1_distance', 0):.6f}")
                logger.info(f"  cosine_similarity: {scaled_features.get('cosine_similarity', 0):.6f}")
            
            malicious_prob, predicted_class = self.detect_malicious_behavior(behavior_description)
            
            logger.info(f"Client {client_id}: prob={malicious_prob:.3f}, prediction={predicted_class}")
            
            if predicted_class == 1:
                if client_id not in self.current_round_malicious:
                    self.current_round_malicious.append(client_id)
                    logger.warning(f"Client {client_id} flagged as MALICIOUS in round {round_num}")
                
            else:
                logger.info(f"Client {client_id} appears BENIGN in round {round_num}")
            return predicted_class
        else:
            logger.warning(f"Detector not trained, skipping detection for client {client_id}")
        
        return 0

    def fit_feature_scaler(self):
        """Fit the MinMax scaler using collected raw features"""
        try:
            if len(self.raw_features_buffer) < self.min_samples_for_scaler:
                logger.warning(f"Insufficient samples to fit scaler: {len(self.raw_features_buffer)} < {self.min_samples_for_scaler}")
                return False
            
            raw_df = pd.DataFrame(self.raw_features_buffer)
            
            numeric_cols = [col for col in raw_df.columns if col not in ['client_id', 'round_num']]
            numeric_data = raw_df[numeric_cols].values
            
            self.feature_scaler.fit(numeric_data)
            self.scaler_fitted = True
            
            logger.info(f"MinMax scaler fitted with {len(self.raw_features_buffer)} samples")
            logger.info(f"Scaling {len(numeric_cols)} numeric features: {numeric_cols[:5]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to fit feature scaler: {e}")
            return False

    def _extract_layer_names_from_model(self, model):
        """Extract ALL state_dict keys from the provided model instance"""
        try:
            self.layer_names = []
            
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
                self.layer_names = list(state_dict.keys())
                logger.info(f"Extracted {len(self.layer_names)} parameter names from state_dict")
                logger.debug(f"Parameter names: {self.layer_names[:5]}...")
                return
            
            logger.warning("Could not extract layer names from model - no state_dict method found")
            
        except Exception as e:
            logger.error(f"Failed to extract layer names from model: {e}")
            self.layer_names = None

    def _convert_params_to_dict(self, param_list):
        """Convert Flower parameter list to dictionary format using actual layer names"""
        if param_list is None:
            logger.warning("param_list is None - returning empty dict")
            return {}
            
        if isinstance(param_list, dict):
            return param_list
        
        param_dict = {}
        
        if self.layer_names and len(param_list) > 0:
            expected_param_count = len(self.layer_names)
            
            if len(param_list) == expected_param_count:
                for name, param in zip(self.layer_names, param_list):
                    param_dict[name] = param
                logger.debug(f"Mapped {len(param_list)} parameters using model layer names")
                
            else:
                logger.warning(
                    f"Parameter count mismatch: {len(param_list)} params vs "
                    f"{expected_param_count} expected layer names"
                )
                
                if hasattr(self, 'model_config') and hasattr(self.model_config, 'model_instance'):
                    model = self.model_config.model_instance
                    all_keys = list(model.state_dict().keys())
                    
                    if len(param_list) == len(all_keys):
                        for name, param in zip(all_keys, param_list):
                            param_dict[name] = param
                        logger.info(f"✓ Mapped using complete state_dict keys (including BatchNorm)")
                        return param_dict
                
                for i, param in enumerate(param_list):
                    param_dict[f'layer_{i}'] = param
                logger.warning("Using generic layer names - feature extraction may be incomplete")
        else:
            for i, param in enumerate(param_list):
                param_dict[f'layer_{i}'] = param
            logger.info("No layer names available, using generic names")
        
        return param_dict
    
    def save_performance_metrics(self, metrics, round_num):
        """Log metrics for a specific round to file"""
        metrics_text = (
            f"Round {round_num:2d} | "
            f"Accuracy: {metrics['accuracy']:.3f} | "
            f"Precision: {metrics['precision']:.3f} | "
            f"Recall: {metrics['recall']:.3f} | "
            f"F1-Score: {metrics['f1_score']:.3f} | "
            f"FPR: {metrics['false_positive_rate']:.3f} | "
            f"TP: {metrics['true_positives']:2d} | "
            f"FP: {metrics['false_positives']:2d} | "
            f"FN: {metrics['false_negatives']:2d} | "
            f"TN: {metrics['true_negatives']:2d}   | "
            f"True Malicious Clients{metrics['true_malicious_clients_set']}  | "
            f"Model Predicted Malicious clients{metrics['suspicious_set'] }"

        )
        
        log_file = os.path.join(self.save_dir, "detection_metrics.log")
        
        with open(log_file, 'a') as f:
            f.write(metrics_text + "\n")
        
        print(metrics_text)

    def finalize_metrics_log(self):
        """Add summary footer to the metrics log"""
        log_file = os.path.join(self.save_dir, "detection_metrics.log")
        
        if os.path.exists(log_file):
            with open(log_file, 'a') as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n")

    def debug_scaler_info(self):
        """Debug method to check scaler state"""
        logger.info("=== SCALER DEBUG INFO ===")
        logger.info(f"Scaler fitted: {self.scaler_fitted}")
        logger.info("Scaler expects: %s", self.feature_scaler.feature_names_in_)

        if hasattr(self.feature_scaler, 'data_min_'):
            logger.info(f"Scaler type: {type(self.feature_scaler)}")
            logger.info(f"Data min shape: {self.feature_scaler.data_min_.shape}")
            logger.info(f"Data max shape: {self.feature_scaler.data_max_.shape}")
            logger.info(f"First 5 data_min: {self.feature_scaler.data_min_[:5]}")
            logger.info(f"First 5 data_max: {self.feature_scaler.data_max_[:5]}")
        else:
            logger.info("Scaler has no data_min_ attribute")
        
        logger.info("========================")


    def start_new_round(self, round_num):
        """Call this at the beginning of each round to reset detection"""
        if self.current_round_malicious:
            self.malicious_history[round_num - 1] = self.current_round_malicious.copy()
            logger.info(f"Saved round {round_num - 1} malicious clients: {self.current_round_malicious}")
        
        self.current_round_malicious = []
        logger.info(f"Started round {round_num} - reset malicious client detection")

    def get_suspicious_clients(self, threshold=0.7):
        """Get list of clients flagged as malicious in the current round"""
        logger.info(f"Current round malicious clients: {self.current_round_malicious}")
        return self.current_round_malicious
    
    def clear_malicious_clients(self):
        """Clear the current round malicious clients list"""
        self.current_round_malicious.clear()
        logger.info("Cleared current round malicious clients list")
    
    def get_client_analysis(self, client_id):
        """Get detailed analysis for a specific client"""
        if client_id in self.client_behaviors:
            return {
                'behaviors': self.client_behaviors[client_id],
                'is_malicious': client_id in self.current_round_malicious,
                'total_rounds': len(self.client_behaviors[client_id])
            }
        return None
    
    def save_analysis_results(self, round_num):
        """Save analysis results for the round"""
        analysis_data = {
            'round': round_num,
            'malicious_clients': self.current_round_malicious.copy(),
            'total_behaviors_tracked': len(self.round_data),
            'model_trained': self.is_trained,
            'clients_analyzed': len(self.client_behaviors)
        }
        
        analysis_path = self.save_dir / f"round_{round_num}_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        if self.current_round_malicious:
            logger.warning(f"Round {round_num}: Malicious clients detected: {self.current_round_malicious}")
        
        return analysis_data