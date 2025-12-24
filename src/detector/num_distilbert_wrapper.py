"""
Wrapper for Num-DistilBERT Detector to integrate with FL Pipeline.
Adapts the pointwise detector API to match MaliciousClientDetector interface.
"""

import logging
from pathlib import Path
from collections import defaultdict
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
        self.current_round_predictions = {}  # {client_id: (prediction, probability)}
        self.malicious_history = {}  # {round: [client_ids]}
        
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
        self.current_round_predictions = {}
        logger.info(f"🔄 Round {round_num}: Started new round detection")
    
    def update_client_behavior(self, client_id, round_num, current_round_params, prev_round_params):
        """
        Extract features and detect if client is malicious.
        
        Args:
            client_id: Client identifier
            round_num: Current round number
            current_round_params: Client's model parameters (list of numpy arrays)
            prev_round_params: Previous round's global model parameters
        
        Returns:
            prediction: 0 (benign) or 1 (malicious)
        """
        if not self.is_trained:
            logger.warning(f"Detector not trained - skipping client {client_id}")
            return 0
        
        try:
            # DEBUG: Log incoming parameter structure
            logger.info(f"📋 Client {client_id} Round {round_num} - Parameter Analysis:")
            logger.info(f"   Current params type: {type(current_round_params)}")
            logger.info(f"   Prev params type: {type(prev_round_params)}")
            
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
            prediction, probability = self.detector.predict(features, threshold=self.threshold)
            
            # Store prediction
            self.current_round_predictions[client_id] = (prediction, probability)
            
            # Add to malicious list if detected
            if prediction == 1:
                self.current_round_malicious.append(client_id)
                logger.warning(
                    f"🚨 Round {round_num}: Client {client_id} detected as MALICIOUS "
                    f"(probability: {probability:.4f})"
                )
            else:
                logger.info(
                    f"✅ Round {round_num}: Client {client_id} classified as BENIGN "
                    f"(probability: {probability:.4f})"
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
        
        # Add prediction details (probabilities)
        if self.current_round_predictions:
            metrics_with_meta['predictions'] = {
                str(cid): {
                    'prediction': int(pred),
                    'probability': float(prob)
                }
                for cid, (pred, prob) in self.current_round_predictions.items()
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
