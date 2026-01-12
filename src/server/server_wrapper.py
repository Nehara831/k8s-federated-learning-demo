import flwr as fl
from flwr.server.strategy import FedAvg
from typing import Dict, Optional, Tuple, List, Union
import logging
import torch
from src.shared.models import create_model_for_dataset
from flwr.common import Parameters, FitRes, parameters_to_ndarrays, ndarrays_to_parameters, Scalar
from flwr.server.client_proxy import ClientProxy
import pickle
from pathlib import Path
from modules.s3_exporter import S3MetricsExporter
from modules.shap_calculator import SHAPCalculator
import time
import numpy as np

logger = logging.getLogger(__name__)

class K8sFederatedStrategy(FedAvg):
    """
    Custom strategy for Kubernetes FL deployment with malicious detection.
    Works with ANY detector (Krum, FedGuard, Num-DistilBERT, etc.) through common interface.
    """
    
    def __init__(
        self,
        config,
        testloader=None,
        malicious_detector=None,
        s3_exporter=None,
        shap_calculator=None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        evaluate_fn=None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
    ):
        self.config = config
        self.testloader = testloader
        self.malicious_detector = malicious_detector
        self.s3_exporter = s3_exporter
        self.shap_calculator = shap_calculator
        self._saved_initial_parameters = initial_parameters
        
        # Track round metrics for S3 export
        self.round_start_time = None
        self.round_metrics = {}
        self.client_performance_history = {}
        
        # Initialize parent strategy
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        
        # Store models from each round (in memory)
        self.round_models = {}
        
        # Define directory for saving global models to disk
        self.global_models_dir = Path("/app/outputs/global_models")
        self.global_models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Global models will be saved to: {self.global_models_dir}")
        
        # Save initial parameters (round 0)
        if self._saved_initial_parameters is not None:
            self._save_global_model_to_disk(0, self._saved_initial_parameters)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model weights with malicious client detection.
        This method is detector-agnostic - works with any detector implementation.
        """
        
        # Start timing the round
        self.round_start_time = time.time()
        
        total_responses = len(results)
        logger.info(f"Round {server_round}: Received {total_responses} client responses")
        
        # Start new round detection (initializes data structures for THIS round)
        # NOTE: Start from round 1, not round 2!
        if self.malicious_detector:
            self.malicious_detector.start_new_round(server_round)
            logger.info(f"🔄 Initialized detector for round {server_round}")
        
        # Separate participating and non-participating clients
        participating_results = []
        malicious_ground_truth = []
        all_client_ids = []
        round_client_metrics = {}
        
        for client_proxy, fit_res in results:
            # Get deterministic client_id from metrics (not network address)
            client_id_from_props = fit_res.metrics.get("client_id", None)
            if client_id_from_props is not None:
                client_id_str = str(client_id_from_props)
            else:
                client_id_str = str(client_proxy.cid)
            
            all_client_ids.append(client_id_str)
            
            # Store client metrics for S3 export
            round_client_metrics[client_id_str] = {
                "local_accuracy": fit_res.metrics.get("local_accuracy", 0),
                "local_loss": fit_res.metrics.get("local_loss", 0),
                "attack_type": fit_res.metrics.get("attack_type", "unknown"),
            }
            
            is_non_participating = fit_res.metrics.get("non_participating", False)
            logger.info(f"Metrics: {fit_res.metrics} Client {client_id_str} (network: {client_proxy.cid})")
            
            if not is_non_participating:
                participating_results.append((client_proxy, fit_res))
                logger.info(f"Round {server_round}: Client {client_id_str} - PARTICIPATING ({fit_res.metrics.get('attack_type', 'unknown')} attack)")
            
            # Count all malicious clients (even non-participating)
            if fit_res.metrics.get("attack_type", "unknown") != "unknown":
                malicious_ground_truth.append(client_id_str)
        
        logger.info(f"Total clients participated: {len(participating_results)}")
        total_client_count = len(all_client_ids)
        
        # ✅ MALICIOUS DETECTION (works with ANY detector)
        suspicious_clients = set()
        perf_results = None
        
        # Update detector with ALL client behaviors (start from round 1)
        if self.malicious_detector:
            for client_proxy, fit_res in results:
                client_id_from_props = fit_res.metrics.get("client_id", None)
                if client_id_from_props is not None:
                    client_id = int(client_id_from_props)
                else:
                    client_id = self._get_client_numeric_id(client_proxy.cid)
                    logger.warning(f"Client didn't send client_id, using hash: {client_id}")
                
                client_id_str = str(client_id)
                current_round_params = parameters_to_ndarrays(fit_res.parameters)
                prev_round_model = self.get_previous_round_model(server_round)
                
                # Always update behavior (even for round 1 with no previous model)
                if prev_round_model is not None:
                    prev_round_params = parameters_to_ndarrays(prev_round_model)
                    
                    # Update detector (works for Krum, FedGuard, Num-DistilBERT, etc.)
                    result = self.malicious_detector.update_client_behavior(
                        client_id, 
                        server_round, 
                        current_round_params, 
                        prev_round_params
                    )
                    
                    is_actually_malicious = client_id_str in malicious_ground_truth
                    logger.info(f"Client {client_id_str}: Prediction={result}, Actually malicious={is_actually_malicious}")
                    
                    if not is_actually_malicious and result == 1:
                        logger.info(f"[+] DEBUG: Client {client_id} round {server_round} misclassified as malicious")
            
            # Filter out detected malicious clients
            if self.malicious_detector.is_trained:
                suspicious_clients = self.malicious_detector.get_suspicious_clients()
                logger.info(f"[+] Detector is trained, suspicious clients: {suspicious_clients}")
                
                if suspicious_clients:
                    filtered_results = []
                    malicious_count = 0
                    
                    for client_proxy, fit_res in participating_results:
                        # Get client_id from metrics (same way we did during detection)
                        client_id_from_props = fit_res.metrics.get("client_id", None)
                        if client_id_from_props is not None:
                            client_id = int(client_id_from_props)
                        else:
                            client_id = self._get_client_numeric_id(client_proxy.cid)
                            logger.warning(f"Client didn't send client_id during filtering, using hash: {client_id}")
                        
                        if client_id not in suspicious_clients:
                            filtered_results.append((client_proxy, fit_res))
                        else:
                            malicious_count += 1
                            logger.warning(f"🚫 Client {client_id} EXCLUDED (detected as malicious)")
                    
                    participating_results = filtered_results
                    logger.info(f"Filtered out {malicious_count} detected malicious clients")
                
                # Calculate and save metrics
                perf_results = self.calculate_metrics(
                    suspicious_clients, 
                    malicious_ground_truth, 
                    total_client_count
                )
                
                if "error" not in perf_results:
                    self.malicious_detector.save_performance_metrics(perf_results, server_round)
                else:
                    logger.warning(f"Round {server_round}: Cannot calculate metrics - {perf_results['error']}")
        
        logger.info(f"Aggregating {len(participating_results)} participating clients")
        
        # Call parent's aggregate_fit
        aggregated_params, metrics = super().aggregate_fit(
            server_round, participating_results, failures
        )
        
        # Store aggregated model for next round
        if aggregated_params is not None:
            params_ndarrays = parameters_to_ndarrays(aggregated_params)
            self.round_models[server_round] = params_ndarrays
            logger.info(f"Stored model from round {server_round} in memory")
            
            # Save to disk
            self._save_global_model_to_disk(server_round, aggregated_params)
        
        # Add participation metrics
        if metrics is None:
            metrics = {}
        metrics["total_client_responses"] = total_responses
        metrics["participating_clients"] = len(participating_results)
        metrics["participation_rate"] = len(participating_results) / total_responses if total_responses > 0 else 0
        
        # Export round metrics to S3
        self._export_round_metrics_to_s3(
            server_round, 
            participating_results, 
            malicious_ground_truth, 
            suspicious_clients, 
            perf_results, 
            total_client_count,
            round_client_metrics,
            metrics
        )
        
        # ✅ Export SHAP data SYNCHRONOUSLY (blocking but guaranteed to have data)
        # Data has been collected via update_client_behavior() calls above
        # NOTE: Start from round 1, not round 2!
        if self.malicious_detector:
            logger.info(f"📊 Processing SHAP export for round {server_round} (synchronous)")
            try:
                # Use detector's SHAP processing method
                round_shap_data = self.malicious_detector.process_round_for_shap_export(
                    round_num=server_round
                )
                
                if round_shap_data and self.s3_exporter:
                    # Upload JSON format
                    json_success = self.s3_exporter.upload_json(
                        data=round_shap_data,
                        filename=f"round_{server_round}_shap_analysis.json",
                        category="shap_analysis"
                    )
                    
                    if json_success:
                        logger.info(f"📤 Uploaded SHAP JSON for round {server_round}")
                    
                    # Convert to CSV format and upload
                    csv_rows = self._convert_shap_to_csv(round_shap_data)
                    if csv_rows:
                        csv_path = self.s3_exporter.upload_csv(
                            data=csv_rows,
                            filename=f"round_{server_round}_features_and_shap.csv",
                            category="shap_analysis"
                        )
                        
                        if csv_path:
                            logger.info(f"📤 Uploaded SHAP CSV for round {server_round}")
                
                elif round_shap_data:
                    logger.info(f"✅ SHAP data processed (S3 not available)")
                else:
                    logger.warning(f"⚠️  No SHAP data generated for round {server_round}")
                    
            except Exception as e:
                logger.error(f"❌ SHAP processing failed for round {server_round}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"✅ Round {server_round} aggregation complete - returning to clients")
        
        return aggregated_params, metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        
        total_responses = len(results)
        
        # Filter participating results
        participating_results = []
        non_participating_count = 0
        
        for client_proxy, eval_res in results:
            if eval_res is not None:
                participating_results.append((client_proxy, eval_res))
            else:
                non_participating_count += 1
        
        logger.info(f"Round {server_round}: Evaluation - {len(participating_results)} participating, {non_participating_count} self-excluded")
        
        # Aggregate
        aggregated_loss, metrics = super().aggregate_evaluate(
            server_round, participating_results, failures
        )
        
        # Add participation info
        if metrics:
            metrics["eval_total_responses"] = total_responses
            metrics["eval_participating_clients"] = len(participating_results)
            metrics["eval_non_participating_clients"] = non_participating_count
        
        return aggregated_loss, metrics
    
    def _get_client_numeric_id(self, cid: str) -> int:
        """Extract numeric ID from client ID string"""
        try:
            return int(cid)
        except ValueError:
            return abs(hash(cid)) % (10 ** 8)
    
    def get_model_from_round(self, round_num):
        """Get model parameters from a specific round"""
        logger.info(f"[DEBUG] get_model_from_round called with round_num={round_num}")
        
        if round_num == 0:
            if self._saved_initial_parameters is not None:
                logger.info("Returning initial parameters for round 0")
                return self._saved_initial_parameters
            else:
                logger.error("No initial parameters available!")
                return None
        
        elif round_num in self.round_models:
            logger.info(f"Returning model from round {round_num}")
            return ndarrays_to_parameters(self.round_models[round_num])
        
        else:
            logger.warning(f"Round {round_num} not found in memory")
            return None
    
    def get_previous_round_model(self, current_round):
        """Get the model from the previous round"""
        if current_round <= 0:
            logger.warning("No previous round available for round 0")
            return None
        
        previous_round = current_round - 1
        return self.get_model_from_round(previous_round)
    
    def calculate_metrics(self, suspicious_clients, ground_truth, total_clients):
        """Calculate comprehensive metrics for the malicious detector"""
        
        # Convert to consistent format (strings)
        suspicious_clients = [str(client_id) for client_id in suspicious_clients]
        ground_truth = [str(client_id) for client_id in ground_truth]
        
        logger.info(f"=== METRICS CALCULATION DEBUG ===")
        logger.info(f"Total clients: {total_clients}")
        logger.info(f"Suspicious clients: {suspicious_clients}")
        logger.info(f"Ground truth malicious: {ground_truth}")
        
        # Convert to sets
        suspicious_set = set(suspicious_clients)
        malicious_set = set(ground_truth)
        
        # Calculate confusion matrix
        true_positives = len(suspicious_set & malicious_set)
        false_positives = len(suspicious_set - malicious_set)
        false_negatives = len(malicious_set - suspicious_set)
        
        # TN = Total - (TP + FP + FN)
        all_identified_clients = suspicious_set | malicious_set
        true_negatives = total_clients - len(all_identified_clients)
        
        logger.info(f"TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}, TN: {true_negatives}")
        
        # Validation
        if true_negatives < 0:
            logger.error(f"INVALID: TN is negative ({true_negatives})")
            true_negatives = max(0, true_negatives)
        
        # Calculate metrics
        precision = true_positives / len(suspicious_set) if suspicious_set else 0.0
        recall = true_positives / len(malicious_set) if malicious_set else 0.0
        accuracy = (true_positives + true_negatives) / total_clients if total_clients > 0 else 0.0
        false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
        
        logger.info(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")
        logger.info(f"=== END METRICS DEBUG ===")
        
        return {
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3),
            "false_positive_rate": round(false_positive_rate, 3),
            "specificity": round(specificity, 3),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "total_clients": total_clients,
            "suspicious_count": len(suspicious_clients),
            "malicious_count": len(ground_truth),
            "true_malicious_clients_set": malicious_set,
            "suspicious_set": suspicious_set
        }
    
    def _save_global_model_to_disk(self, round_num: int, parameters) -> str:
        """Save global model parameters to disk"""
        try:
            if isinstance(parameters, Parameters):
                params_array = parameters_to_ndarrays(parameters)
            elif isinstance(parameters, list):
                params_array = parameters
            else:
                logger.error(f"Unknown parameters type: {type(parameters)}")
                return ""
            
            filename = f"global_model_round_{round_num}.pkl"
            save_path = self.global_models_dir / filename
            
            # Get layer names from model structure
            input_size = 20 if self.config.dataset.type == "5gnidd" else None
            temp_model = create_model_for_dataset(
                dataset_type=self.config.dataset.type,
                num_classes=self.config.num_classes,
                input_size=input_size
            )
            
            layer_names = list(temp_model.state_dict().keys())
            
            if len(layer_names) != len(params_array):
                logger.error(f"Layer count mismatch: {len(layer_names)} vs {len(params_array)}")
                return ""
            
            # Save as dict with layer names
            params_dict = {name: param for name, param in zip(layer_names, params_array)}
            
            with open(save_path, 'wb') as f:
                pickle.dump(params_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size = save_path.stat().st_size / 1024
            logger.info(f"💾 Saved global model round {round_num}: {save_path} ({file_size:.1f} KB)")
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to save global model round {round_num}: {e}")
            return ""
    
    def _export_round_metrics_to_s3(
        self,
        server_round: int,
        participating_results: List,
        malicious_ground_truth: List,
        suspicious_clients: set,
        perf_results: Optional[Dict],
        total_client_count: int,
        round_client_metrics: Dict,
        aggregated_metrics: Dict
    ):
        """
        Export comprehensive round metrics to S3.
        Uploads round data JSON and collects SHAP data for later aggregated CSV upload.
        """
        if not self.s3_exporter:
            logger.debug("S3 exporter not initialized, skipping metrics export")
            return
        
        try:
            # Build comprehensive round data
            round_data = {
                "round_num": server_round,
                "timestamp": time.time(),
                "total_clients": total_client_count,
                "participating_clients": len(participating_results),
                "participation_rate": len(participating_results) / total_client_count if total_client_count > 0 else 0,
                "malicious_clients": malicious_ground_truth,
                "suspicious_clients": list(suspicious_clients),
                "client_metrics": round_client_metrics,
                "aggregated_metrics": aggregated_metrics,
                "detector_metrics": perf_results if perf_results else {},
            }
            
            # NOTE: SHAP processing is now handled in background thread via _process_shap_background()
            # This prevents blocking and uses proper feature extraction from the detector
            logger.debug("SHAP processing will be handled in background thread (non-blocking)")
            
            # Upload round data JSON
            success = self.s3_exporter.upload_round_data(round_data, server_round)
            
            if success:
                logger.info(f"✅ Round {server_round} metrics exported to S3")
            else:
                logger.warning(f"⚠️  Failed to export round {server_round} metrics to S3")
                
        except Exception as e:
            logger.error(f"Error exporting round {server_round} metrics to S3: {e}")
    
    def _convert_shap_to_csv(self, round_shap_data: Dict) -> List[Dict]:
        """
        Convert SHAP data to CSV format.
        Each row contains: round, client_id, classification, features, SHAP values.
        
        Args:
            round_shap_data: Dictionary from detector's process_round_for_shap_export()
            
        Returns:
            List of dictionaries for CSV export
        """
        csv_rows = []
        
        try:
            round_num = round_shap_data['round_num']
            
            for client_data in round_shap_data['clients']:
                row = {
                    'round': round_num,
                    'client_id': client_data['client_id'],
                    'classification': client_data['classification'],
                    'malicious_probability': client_data['malicious_probability'],
                }
                
                # Add features with 'feature_' prefix
                for feature_name, feature_value in client_data['features'].items():
                    row[f'feature_{feature_name}'] = feature_value
                
                # Add SHAP values with 'shap_' prefix
                if 'shap_values' in client_data:
                    row['shap_base_value'] = client_data['shap_values']['base_value']
                    
                    for feature_name, shap_value in client_data['shap_values']['feature_shap_values'].items():
                        row[f'shap_{feature_name}'] = shap_value
                
                csv_rows.append(row)
            
            logger.info(f"Converted {len(csv_rows)} client records to CSV format")
            return csv_rows
            
        except Exception as e:
            logger.error(f"Failed to convert SHAP data to CSV: {e}")
            return []
    
def create_strategy(config, initial_parameters, testloader=None, malicious_detector=None, s3_exporter=None, shap_calculator=None):
    """
    Create federated learning strategy.
    Works with ANY detector through common interface.
    """
    
    def fit_config(server_round: int):
        return {
            "server_round": server_round,
            "round_num": server_round,
            "local_epochs": config.config_fit.local_epochs,
            "lr": config.config_fit.lr,
            "momentum": config.config_fit.momentum,
        }
    
    def evaluate_config(server_round: int):
        return {"server_round": server_round}
    
    def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Dict]]) -> Dict:
        """Aggregate training metrics"""
        total_examples = sum(num_examples for num_examples, _ in metrics)
        losses = [num_examples * m.get("loss", 0) for num_examples, m in metrics]
        aggregated_loss = sum(losses) / total_examples if total_examples > 0 else 0
        
        logger.info(f"📊 Training - Aggregated loss: {aggregated_loss:.4f}")
        return {"loss": aggregated_loss}
    
    def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Dict]]) -> Dict:
        """Aggregate evaluation metrics"""
        total_examples = sum(num_examples for num_examples, _ in metrics)
        
        accuracies = [num_examples * m.get("accuracy", 0) for num_examples, m in metrics]
        aggregated_accuracy = sum(accuracies) / total_examples if total_examples > 0 else 0
        
        losses = [num_examples * m.get("loss", 0) for num_examples, m in metrics]
        aggregated_loss = sum(losses) / total_examples if total_examples > 0 else 0
        
        logger.info(f"✅ Evaluation - Accuracy: {aggregated_accuracy:.4f}, Loss: {aggregated_loss:.4f}")
        
        return {"accuracy": aggregated_accuracy, "loss": aggregated_loss}
    
    strategy = K8sFederatedStrategy(
        config=config,
        testloader=testloader,
        malicious_detector=malicious_detector,
        s3_exporter=s3_exporter,
        shap_calculator=shap_calculator,
        fraction_fit=config.server.fraction_fit,
        fraction_evaluate=config.server.fraction_evaluate,
        min_fit_clients=config.server.min_fit_clients,
        min_evaluate_clients=config.server.min_evaluate_clients,
        min_available_clients=config.server.min_available_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    
    return strategy


def _convert_shap_to_csv(self, round_shap_data: Dict) -> List[Dict]:
    """
    Convert SHAP data to CSV format.
    Each row contains: round, client_id, classification, features, SHAP values.
    
    Args:
        round_shap_data: Dictionary from detector's process_round_for_shap_export()
        
    Returns:
        List of dictionaries for CSV export
    """
    csv_rows = []
    
    try:
        round_num = round_shap_data['round_num']
        
        for client_data in round_shap_data['clients']:
            row = {
                'round': round_num,
                'client_id': client_data['client_id'],
                'classification': client_data['classification'],
                'malicious_probability': client_data['malicious_probability'],
            }
            
            # Add features with 'feature_' prefix
            for feature_name, feature_value in client_data['features'].items():
                row[f'feature_{feature_name}'] = feature_value
            
            # Add SHAP values with 'shap_' prefix
            if 'shap_values' in client_data:
                row['shap_base_value'] = client_data['shap_values']['base_value']
                
                for feature_name, shap_value in client_data['shap_values']['feature_shap_values'].items():
                    row[f'shap_{feature_name}'] = shap_value
            
            csv_rows.append(row)
        
        logger.info(f"Converted {len(csv_rows)} client records to CSV format")
        return csv_rows
        
    except Exception as e:
        logger.error(f"Failed to convert SHAP data to CSV: {e}")
        return []
