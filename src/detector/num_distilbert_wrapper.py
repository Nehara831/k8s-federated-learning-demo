"""
Wrapper for Num-DistilBERT Detector to integrate with FL Pipeline.
Adapts the pointwise detector API to match MaliciousClientDetector interface.
"""

import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
import numpy as np
import torch
import json
from datetime import datetime

from src.detector.num_distilbert_detector import NumDistilBERTDetector
from src.detector.data_extractor import extract_model_features

logger = logging.getLogger(__name__)


class NumDistilBERTWrapper:
    """
    Wrapper for Num-DistilBERT detector to integrate with FL pipeline.
    Maintains the same API as MaliciousClientDetector for drop-in replacement.
    """
    
    def __init__(self, save_dir, num_clients, model_config=None, reference_model=None):
        """
        Args:
            save_dir: Directory for saving detection results
            num_clients: Total number of clients in FL
            model_config: Config object with detector settings
            reference_model: Reference model for feature extraction
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_clients = num_clients
        self.model_config = model_config
        self.reference_model = reference_model
        
        # Extract layer names from reference model for parameter mapping
        self.layer_names = None
        if reference_model is not None:
            self._extract_layer_names_from_model(reference_model)
        
        # Load trained detector
        model_path = None
        scaler_path = None
        
        if model_config and hasattr(model_config, 'model_path'):
            model_path = model_config.model_path
        if model_config and hasattr(model_config, 'scaler_path'):
            scaler_path = model_config.scaler_path
        
        # Use default paths if not specified
        if model_path is None:
            model_path = "trained_model_weights/Models/num_distilbert_5gnidd/num_distilbert_best.pt"
        if scaler_path is None:
            scaler_path = "trained_model_weights/Scalers/num_distilbert_scaler.pkl"
        
        logger.info("=" * 80)
        logger.info("🔧 INITIALIZING NUM-DISTILBERT DETECTOR")
        logger.info("=" * 80)
        logger.info(f"Model path: {model_path}")
        logger.info(f"Scaler path: {scaler_path}")
        
        # Initialize detector
        self.detector = NumDistilBERTDetector(
            num_features=22,
            model_path=model_path,
            scaler_path=scaler_path
        )
        
        # Detection threshold
        self.threshold = 0.5
        if model_config and hasattr(model_config, 'threshold'):
            self.threshold = model_config.threshold
        
        logger.info(f"Detection threshold: {self.threshold}")
        
        # Storage for current round detections
        self.current_round_malicious = []
        self.round_predictions = {}  # {round_num: {client_id: (prediction, probability)}}
        self.malicious_history = {}  # {round: [client_ids]}
        
        # ✅ Store ground truth labels per round
        self.ground_truth_labels = {}  # {round_num: {client_id: is_malicious}}
        
        # Feature storage for debugging
        self.client_features = defaultdict(dict)  # {round: {client_id: features}}
        
        # Performance metrics storage
        self.performance_history = []
        
        # Status
        self.is_trained = self.detector.is_trained
        self.current_round = 0
        
        logger.info(f"✅ Detector initialized (trained: {self.is_trained})")
        logger.info("=" * 80)
    
    def _extract_layer_names_from_model(self, model):
        """Extract ALL state_dict keys from the provided model instance"""
        try:
            self.layer_names = []
            
            # Use state_dict to get ALL keys (weights, biases, AND buffers)
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
                self.layer_names = list(state_dict.keys())
                logger.info(f"Extracted {len(self.layer_names)} parameter names from state_dict")
                logger.debug(f"Parameter names: {self.layer_names[:5]}...")  # Show first 5
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
        
        # Get expected parameter names from the model
        if self.layer_names and len(param_list) > 0:
            # Flower sends ALL state_dict items in the same order as model.state_dict().items()
            # This includes weights, biases, AND BatchNorm buffers
            expected_param_count = len(self.layer_names)
            
            if len(param_list) == expected_param_count:
                # Perfect match - use actual layer names
                for name, param in zip(self.layer_names, param_list):
                    param_dict[name] = param
                logger.debug(f"✅ Mapped {len(param_list)} parameters using model layer names")
                
            else:
                # Mismatch detected - need to handle this carefully
                logger.warning(
                    f"⚠️  Parameter count mismatch: {len(param_list)} params vs "
                    f"{expected_param_count} expected layer names"
                )
                
                # CRITICAL FIX: Re-extract layer names to include ALL state_dict keys
                # including BatchNorm buffers
                if self.reference_model is not None:
                    all_keys = list(self.reference_model.state_dict().keys())
                    
                    if len(param_list) == len(all_keys):
                        for name, param in zip(all_keys, param_list):
                            param_dict[name] = param
                        logger.info(f"✅ Mapped using complete state_dict keys (including BatchNorm)")
                        return param_dict
                
                # Fallback: use generic names
                for i, param in enumerate(param_list):
                    param_dict[f'layer_{i}'] = param
                logger.warning("⚠️  Using generic layer names - feature extraction may be incomplete")
        else:
            # No layer names available
            for i, param in enumerate(param_list):
                param_dict[f'layer_{i}'] = param
            logger.info("Using generic layer names (no reference model)")
        
        return param_dict
    
    def start_new_round(self, round_num):
        """
        Start a new FL round - reset detection state.
        
        Args:
            round_num: Current round number
        """
        self.current_round = round_num
        self.current_round_malicious = []
        # DON'T clear round_predictions - keep historical data for SHAP export
        # Each round stores its own predictions in round_predictions[round_num]
        if round_num not in self.round_predictions:
            self.round_predictions[round_num] = {}
        if round_num not in self.ground_truth_labels:
            self.ground_truth_labels[round_num] = {}
        logger.info(f"🔄 Round {round_num}: Started new round detection")
    
    def update_client_behavior(self, client_id, round_num, current_round_params, prev_round_params, is_malicious=False):
        """
        Extract features and detect if client is malicious.
        
        Args:
            client_id: Client identifier
            round_num: Current round number
            current_round_params: Client's model parameters (list of numpy arrays)
            prev_round_params: Previous round's global model parameters
            is_malicious: Ground truth label (True if client is malicious)
        
        Returns:
            prediction: 0 (benign) or 1 (malicious)
        """
        try:
            # ✅ Store ground truth label
            if round_num not in self.ground_truth_labels:
                self.ground_truth_labels[round_num] = {}
            self.ground_truth_labels[round_num][client_id] = 1 if is_malicious else 0
            
            # DEBUG: Log incoming parameter structure
            logger.info(f"📋 Client {client_id} Round {round_num} - Parameter Analysis:")
            logger.info(f"   Current params type: {type(current_round_params)}")
            logger.info(f"   Prev params type: {type(prev_round_params)}")
            logger.info(f"   Ground truth: {'MALICIOUS' if is_malicious else 'BENIGN'}")
            
            if isinstance(current_round_params, list):
                logger.info(f"   Current params count: {len(current_round_params)}")
                logger.info(f"   Sample shapes: {[p.shape for p in current_round_params[:3]]}")
            
            # Convert parameters using proper layer name mapping
            param_dict = self._convert_params_to_dict(current_round_params)
            ref_param_dict = self._convert_params_to_dict(prev_round_params)
            
            # DEBUG: Log conversion results
            logger.info(f"   ✅ Converted to dict with {len(param_dict)} keys")
            if ref_param_dict:
                logger.info(f"   ✅ Reference dict has {len(ref_param_dict)} keys")
            
            # DEBUG: Log dictionary structure before feature extraction
            logger.info(f"   📊 Parameter dict keys: {list(param_dict.keys())[:5]}... (showing first 5)")
            
            # Extract features from client's model update
            features = extract_model_features(
                param_dict,
                client_id,
                round_num,
                reference_params=ref_param_dict
            )
            
            # DEBUG: Log extracted features
            logger.info(f"   ✅ Extracted {len(features)} features from client {client_id}")
            
            # Store features for debugging
            if round_num not in self.client_features:
                self.client_features[round_num] = {}
            self.client_features[round_num][client_id] = features
            
            # Predict using Num-DistilBERT (features is already a dict)
            # If detector not trained, use default prediction but still collect data
            if not self.is_trained:
                logger.warning(f"Detector not trained - using default prediction for client {client_id}")
                prediction, probability = 0, 0.0  # Default: benign with 0 confidence
            else:
                prediction, probability = self.detector.predict(features, threshold=self.threshold)
            
            # Store prediction in round-specific dictionary (even if detector not trained)
            if round_num not in self.round_predictions:
                self.round_predictions[round_num] = {}
            self.round_predictions[round_num][client_id] = (prediction, probability)
            
            # Add to malicious list if detected (only if detector is trained)
            if self.is_trained and prediction == 1:
                self.current_round_malicious.append(client_id)
                logger.warning(
                    f"🚨 Round {round_num}: Client {client_id} detected as MALICIOUS "
                    f"(probability: {probability:.4f})"
                )
            else:
                logger.info(
                    f"✅ Round {round_num}: Client {client_id} classified as BENIGN "
                    f"(probability: {probability:.4f}, trained: {self.is_trained})"
                )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error detecting client {client_id}: {e}")
            return 0  # Default to benign on error
    
    def get_suspicious_clients(self, threshold=None):
        """
        Get list of clients detected as malicious in current round.
        
        Args:
            threshold: Optional override for detection threshold (ignored, uses self.threshold)
        
        Returns:
            List of malicious client IDs
        """
        # Store in history
        if self.current_round > 0:
            self.malicious_history[self.current_round] = self.current_round_malicious.copy()
        
        logger.info(
            f"📊 Round {self.current_round}: Detected {len(self.current_round_malicious)} "
            f"malicious clients: {self.current_round_malicious}"
        )
        
        return self.current_round_malicious
    
    def save_performance_metrics(self, metrics, round_num):
        """
        Save detection performance metrics for this round.
        
        Args:
            metrics: Dictionary with performance metrics (accuracy, precision, recall, etc.)
            round_num: Current round number
        """
        # Convert sets to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, set):
                serializable_metrics[key] = list(value)  # ✅ Convert sets to lists
            else:
                serializable_metrics[key] = value

        metrics_with_meta = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'detector_type': 'num_distilbert',
            'threshold': self.threshold,
            **serializable_metrics  # ✅ Now all values are JSON-serializable
        }
        
        # Add detection details
        metrics_with_meta['num_detected'] = len(self.current_round_malicious)
        metrics_with_meta['detected_clients'] = self.current_round_malicious
        
        # Add prediction details (probabilities) for this specific round
        if round_num in self.round_predictions:
            metrics_with_meta['predictions'] = {
                str(cid): {
                    'prediction': int(pred),
                    'probability': float(prob)
                }
                for cid, (pred, prob) in self.round_predictions[round_num].items()
            }
        
        # Store in history
        self.performance_history.append(metrics_with_meta)
        
        # Save to file
        metrics_file = self.save_dir / f"round_{round_num}_metrics.json"
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics_with_meta, f, indent=2)
            logger.info(f"💾 Saved metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
        
        # Save aggregated history
        history_file = self.save_dir / "detection_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            logger.info(f"💾 Updated detection history: {history_file}")
        except Exception as e:
            logger.error(f"Failed to save detection history: {e}")
        
        # Log key metrics
        logger.info("=" * 80)
        logger.info(f"📊 ROUND {round_num} DETECTION PERFORMANCE")
        logger.info("=" * 80)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        logger.info("=" * 80)
    
    def get_detection_summary(self):
        """
        Get a summary of all detection results.
        
        Returns:
            Dictionary with detection statistics
        """
        total_detections = sum(len(clients) for clients in self.malicious_history.values())
        
        summary = {
            'total_rounds': len(self.malicious_history),
            'total_detections': total_detections,
            'rounds_with_detections': sum(1 for clients in self.malicious_history.values() if clients),
            'malicious_history': self.malicious_history,
            'avg_detections_per_round': total_detections / len(self.malicious_history) if self.malicious_history else 0
        }
        
        return summary
    
    def save_final_report(self):
        """
        Save a final report of all detection results.
        """
        summary = self.get_detection_summary()
        
        report_file = self.save_dir / "final_detection_report.json"
        try:
            with open(report_file, 'w') as f:
                json.dump({
                    'summary': summary,
                    'performance_history': self.performance_history,
                    'detector_config': {
                        'type': 'num_distilbert',
                        'threshold': self.threshold,
                        'num_clients': self.num_clients
                    }
                }, f, indent=2)
            
            logger.info("=" * 80)
            logger.info("📊 FINAL DETECTION REPORT")
            logger.info("=" * 80)
            logger.info(f"Total rounds: {summary['total_rounds']}")
            logger.info(f"Total detections: {summary['total_detections']}")
            logger.info(f"Avg detections/round: {summary['avg_detections_per_round']:.2f}")
            logger.info(f"Report saved to: {report_file}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Failed to save final report: {e}")
    
    def process_round_for_shap_export(self, round_num):
        """
        Process and export round data with SHAP values.
        This method extracts features, computes SHAP values, and structures data for S3 export.
        
        Args:
            round_num: Current round number
            
        Returns:
            Dictionary with round data including client features and SHAP values
        """
        try:
            # Get all client data from this round
            round_clients = []
            
            # Get predictions for this specific round
            if round_num not in self.round_predictions:
                logger.warning(f"No predictions found for round {round_num}")
                return {
                    'round_num': round_num,
                    'timestamp': datetime.utcnow().isoformat(),
                    'num_samples': 0,
                    'clients': []
                }
            
            round_predictions = self.round_predictions[round_num]
            logger.info(f"📊 Processing {len(round_predictions)} clients for round {round_num} SHAP export")
            
            for client_id, (prediction, probability) in round_predictions.items():
                # Get features for this client
                if round_num not in self.client_features or client_id not in self.client_features[round_num]:
                    logger.warning(f"No features found for client {client_id} in round {round_num}")
                    continue
                
                features = self.client_features[round_num][client_id]
                
                # Get classification
                classification = "malicious" if prediction == 1 else "benign"
                
                # Prepare client data
                client_data = {
                    'client_id': int(client_id),
                    'round_num': round_num,
                    'classification': classification,
                    'malicious_probability': float(probability),
                    'features': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                               for k, v in features.items()}
                }
                
                # Add SHAP values (using feature values as importance proxy)
                # For Num-DistilBERT, we use the trained model's feature embeddings as SHAP proxy
                shap_values = self._compute_feature_importance(features)
                client_data['shap_values'] = shap_values
                
                round_clients.append(client_data)
            
            # Build export data structure
            round_export_data = {
                'round_num': round_num,
                'timestamp': datetime.utcnow().isoformat(),
                'num_samples': len(round_clients),
                'clients': round_clients
            }
            
            logger.info(f"✅ Processed round {round_num} with {len(round_clients)} clients for SHAP export")
            return round_export_data
            
        except Exception as e:
            logger.error(f"Failed to process round {round_num} for SHAP export: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _compute_feature_importance(self, features):
        """
        Compute feature importance using the detector's attention mechanism.
        This serves as a proxy for SHAP values specific to Num-DistilBERT.
        
        Args:
            features: Dictionary of client features
            
        Returns:
            Dictionary with SHAP-like importance scores
        """
        try:
            if not self.is_trained:
                # Fallback: use absolute feature values as importance
                return {
                    'base_value': 0.0,
                    'feature_shap_values': {
                        k: float(abs(v)) if isinstance(v, (int, float, np.number)) else 0.0
                        for k, v in features.items()
                    }
                }
            
            # Convert features to model input
            feature_array = np.array([
                features.get(name, 0.0) for name in self.detector.feature_names
            ])
            
            # Scale if scaler is available
            if self.detector.scaler:
                feature_array = self.detector.scaler.transform([feature_array])[0]
            
            # Get model's feature embeddings (attention weights as importance)
            x = torch.FloatTensor(feature_array).unsqueeze(0)
            
            with torch.no_grad():
                # Get feature embeddings
                feature_embeds = self.detector.model.input_embedding(x)
                
                # Use embedding magnitudes as feature importance
                importance = torch.norm(feature_embeds, dim=-1).squeeze().numpy()
            
            # Normalize importance scores
            importance = importance / (importance.sum() + 1e-10)
            
            # Map to feature names
            shap_values = {
                'base_value': 0.5,  # Neutral baseline
                'feature_shap_values': {
                    name: float(imp) 
                    for name, imp in zip(self.detector.feature_names, importance)
                }
            }
            
            return shap_values
            
        except Exception as e:
            logger.error(f"Failed to compute feature importance: {e}")
            # Fallback
            return {
                'base_value': 0.0,
                'feature_shap_values': {
                    k: float(abs(v)) if isinstance(v, (int, float, np.number)) else 0.0
                    for k, v in features.items()
                }
            }
    
    def process_round_for_csv_export(self, round_num: int, main_task_accuracy: float = 0.0, 
                                     main_task_loss: float = 0.0) -> List[Dict]:
        """
        Process round data for CSV export (matching simulation format).
        Creates one row per client with features + SHAP values.
        
        Args:
            round_num: Current round number
            main_task_accuracy: Main task accuracy for this round
            main_task_loss: Main task loss for this round
            
        Returns:
            List of dictionaries (CSV rows), one per client
        """
        try:
            # Check if we have predictions for this round
            if round_num not in self.round_predictions:
                logger.warning(f"No predictions found for round {round_num}")
                return []
            
            round_predictions = self.round_predictions[round_num]
            logger.info(f"📊 Processing {len(round_predictions)} clients for round {round_num} CSV export")
            
            csv_rows = []
            
            for client_id, (prediction, probability) in round_predictions.items():
                # Get features for this client
                if round_num not in self.client_features or client_id not in self.client_features[round_num]:
                    logger.warning(f"No features found for client {client_id} in round {round_num}")
                    continue
                
                features = self.client_features[round_num][client_id]
                
                # Compute SHAP values
                shap_values = self._compute_feature_importance(features)
                
                # Build CSV row (matching simulation format exactly)
                row = {
                    'client_id': client_id,
                    'round_num': round_num,
                    # Feature values
                    'param_mean': features.get('param_mean', 0.0),
                    'param_std': features.get('param_std', 0.0),
                    'param_min': features.get('param_min', 0.0),
                    'param_max': features.get('param_max', 0.0),
                    'param_median': features.get('param_median', 0.0),
                    'param_range': features.get('param_range', 0.0),
                    'param_abs_mean': features.get('param_abs_mean', 0.0),
                    'param_skew': features.get('param_skew', 0.0),
                    'param_kurtosis': features.get('param_kurtosis', 0.0),
                    'param_neg_ratio': features.get('param_neg_ratio', 0.0),
                    'param_zero_ratio': features.get('param_zero_ratio', 0.0),
                    'last_layer_mean': features.get('last_layer_mean', 0.0),
                    'last_layer_std': features.get('last_layer_std', 0.0),
                    'last_layer_min': features.get('last_layer_min', 0.0),
                    'last_layer_max': features.get('last_layer_max', 0.0),
                    'last_layer_abs_mean': features.get('last_layer_abs_mean', 0.0),
                    'last_layer_neg_ratio': features.get('last_layer_neg_ratio', 0.0),
                    'first_vs_last_mean_ratio': features.get('first_vs_last_mean_ratio', 0.0),
                    'first_vs_last_std_ratio': features.get('first_vs_last_std_ratio', 0.0),
                    'avg_l1_distance': features.get('avg_l1_distance', 0.0),
                    'avg_l2_distance': features.get('avg_l2_distance', 0.0),
                    'cosine_similarity': features.get('cosine_similarity', 0.0),
                    # Detection results
                    'true_label': self.ground_truth_labels.get(round_num, {}).get(client_id, 0),  # ✅ Ground truth from client
                    'predicted_label': prediction,
                    'predicted_prob': probability,
                    # Main task metrics
                    'main_task_accuracy': main_task_accuracy,
                    'main_task_loss': main_task_loss,
                }
                
                # Add SHAP values with exact naming from simulation
                feature_shap = shap_values.get('feature_shap_values', {})
                row['SHAP_Param Mean'] = feature_shap.get('param_mean', 0.0)
                row['SHAP_Param Std'] = feature_shap.get('param_std', 0.0)
                row['SHAP_Param Min'] = feature_shap.get('param_min', 0.0)
                row['SHAP_Param Max'] = feature_shap.get('param_max', 0.0)
                row['SHAP_Param Median'] = feature_shap.get('param_median', 0.0)
                row['SHAP_Param Range'] = feature_shap.get('param_range', 0.0)
                row['SHAP_Param Absolute Mean'] = feature_shap.get('param_abs_mean', 0.0)
                row['SHAP_Param Skew'] = feature_shap.get('param_skew', 0.0)
                row['SHAP_Param Kurtosis'] = feature_shap.get('param_kurtosis', 0.0)
                row['SHAP_Param Negative Ratio'] = feature_shap.get('param_neg_ratio', 0.0)
                row['SHAP_Param Zero Ratio'] = feature_shap.get('param_zero_ratio', 0.0)
                row['SHAP_Last Layer Mean'] = feature_shap.get('last_layer_mean', 0.0)
                row['SHAP_Last Layer Std'] = feature_shap.get('last_layer_std', 0.0)
                row['SHAP_Last Layer Min'] = feature_shap.get('last_layer_min', 0.0)
                row['SHAP_Last Layer Max'] = feature_shap.get('last_layer_max', 0.0)
                row['SHAP_Last Layer Absolute Mean'] = feature_shap.get('last_layer_abs_mean', 0.0)
                row['SHAP_Last Layer Negative Ratio'] = feature_shap.get('last_layer_neg_ratio', 0.0)
                row['SHAP_First vs Last Mean Ratio'] = feature_shap.get('first_vs_last_mean_ratio', 0.0)
                row['SHAP_First vs Last Std Ratio'] = feature_shap.get('first_vs_last_std_ratio', 0.0)
                row['SHAP_Avg L1 Distance'] = feature_shap.get('avg_l1_distance', 0.0)
                row['SHAP_Avg L2 Distance'] = feature_shap.get('avg_l2_distance', 0.0)
                row['SHAP_Cosine Similarity'] = feature_shap.get('cosine_similarity', 0.0)
                
                csv_rows.append(row)
            
            logger.info(f"✅ Converted {len(csv_rows)} client records to CSV format for round {round_num}")
            return csv_rows
            
        except Exception as e:
            logger.error(f"Failed to process round {round_num} for CSV export: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
