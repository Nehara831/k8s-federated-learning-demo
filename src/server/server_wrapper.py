import flwr as fl
from flwr.server.strategy import FedAvg
from typing import Dict, Optional, Tuple, List
import logging
import torch
from src.shared.models import create_model_for_dataset
from flwr.common import Parameters, FitRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

logger = logging.getLogger(__name__)

class K8sFederatedStrategy(FedAvg):
    """Custom strategy for Kubernetes FL deployment with malicious detection"""
    
    def __init__(self, config, testloader=None, malicious_detector=None, **kwargs):
        self.config = config
        self.testloader = testloader
        self.malicious_detector = malicious_detector
        self._saved_initial_parameters = kwargs.get('initial_parameters')
        self.round_models = {}  # Store models from each round
        
        super().__init__(**kwargs)
    
    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure the next round of training"""
        config = {
            "server_round": server_round,
            "round_num": server_round,  # Add round_num for client compatibility
            "local_epochs": self.config.config_fit.local_epochs,
            "lr": self.config.config_fit.lr,
            "momentum": self.config.config_fit.momentum,
        }
        
        # Add attack configuration for malicious clients
        if hasattr(self.config, 'attack') and self.config.attack.enabled:
            config.update({
                "attack_enabled": True,
                "attack_type": self.config.attack.attack_type,
                "attack_params": dict(self.config.attack.attack_params),
                "malicious_ratio": self.config.attack.malicious_ratio
            })
        
        fit_ins = fl.common.FitIns(parameters, config)
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        return [(client, fit_ins) for client in clients]
    
    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Configure the next round of evaluation"""
        config = {
            "server_round": server_round,
        }
        
        evaluate_ins = fl.common.EvaluateIns(parameters, config)
        
        # Sample clients for evaluation
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        return [(client, evaluate_ins) for client in clients]

    def _get_client_numeric_id(self, cid: str) -> int:
        """Extract numeric ID from client ID string (e.g., 'ipv4:10.244.0.8:37600' -> hash)"""
        try:
            # Try direct conversion for simple numeric IDs
            return int(cid)
        except ValueError:
            # For network addresses, use hash to get consistent numeric ID
            return abs(hash(cid)) % (10 ** 8)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Exception],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate model weights with malicious client detection"""
        
        total_responses = len(results)
        logger.info(f"Round {server_round}: Received {total_responses} client responses")
        
        # Separate participating and non-participating clients
        participating_results = []
        malicious_ground_truth = []
        all_client_ids = []
        
        for client_proxy, fit_res in results:
            client_id_str = str(client_proxy.cid)
            all_client_ids.append(client_id_str)
            
            is_non_participating = fit_res.metrics.get("non_participating", False)
            logger.info(f"Metrics: {fit_res.metrics} Client {client_proxy.cid}")
            
            if not is_non_participating:
                participating_results.append((client_proxy, fit_res))
                logger.info(f"Round {server_round}: Client {client_proxy.cid} - PARTICIPATING ({fit_res.metrics.get('attack_type', 'unknown')} attack)")
            
            # Count all malicious clients, even if non-participating
            if fit_res.metrics.get("attack_type", "unknown") != "unknown":
                malicious_ground_truth.append(client_id_str)
        
        logger.info(f"Total clients participated: {len(participating_results)}")
        total_client_count = len(all_client_ids)  # Use all clients, not just participating
        
        # Start new round in detector
        if self.malicious_detector and server_round > 1:
            self.malicious_detector.start_new_round(server_round)
            
            # Update detector with client behaviors for ALL clients
            for client_proxy, fit_res in results:
                client_id = self._get_client_numeric_id(client_proxy.cid)
                client_id_str = str(client_proxy.cid)
                
                current_round_params = parameters_to_ndarrays(fit_res.parameters)
                prev_round_model = self.get_previous_round_model(server_round)
                
                if prev_round_model is not None:
                    prev_round_params = parameters_to_ndarrays(prev_round_model)
                    
                    # Update detector and get prediction
                    result = self.malicious_detector.update_client_behavior(
                        client_id, 
                        server_round, 
                        current_round_params, 
                        prev_round_params
                    )
                    
                    is_actually_malicious = client_id_str in malicious_ground_truth
                    logger.info(f"Result is {result} is actually malicious {is_actually_malicious}")
                    
                    if not(is_actually_malicious) and (result == 1):
                        logger.info(f"[+] DEBUG LINE: {client_id} round no: {server_round} is misclassified as malicious")
        
            # Filter out detected malicious clients
            if self.malicious_detector.is_trained:
                suspicious_clients = self.malicious_detector.get_suspicious_clients()
                logger.info("[+] inside self.malicious_detector.is_trained")
                logger.info(f"[+] inside {suspicious_clients}")
                
                if suspicious_clients:
                    # Filter out detected malicious clients
                    filtered_results = []
                    malicious_count = 0
                    
                    for client_proxy, fit_res in participating_results:
                        client_id = self._get_client_numeric_id(client_proxy.cid)
                        if client_id not in suspicious_clients:
                            filtered_results.append((client_proxy, fit_res))
                        else:
                            malicious_count += 1
                            logger.warning(f"🚫 Client {client_id} EXCLUDED (detected as malicious)")
                    
                    participating_results = filtered_results
                    logger.info(f"Filtered out {malicious_count} detected malicious clients")
                
                # Use total_client_count and full malicious_ground_truth for metrics
                perf_results = self.calculate_metrics(
                    suspicious_clients, 
                    malicious_ground_truth, 
                    total_client_count
                )
                
                # Only save metrics if calculation was successful (no error)
                if "error" not in perf_results:
                    self.malicious_detector.save_performance_metrics(perf_results, server_round)
                else:
                    logger.warning(f"Round {server_round}: Cannot calculate performance metrics - {perf_results['error']}")
        
        logger.info(f"Aggregating {len(participating_results)} participating clients")
        
        # Call parent's aggregate_fit with participating results only
        aggregated_params, metrics = super().aggregate_fit(
            server_round, participating_results, failures
        )
        
        # Store aggregated model for next round
        if aggregated_params is not None:
            self.round_models[server_round] = parameters_to_ndarrays(aggregated_params)
            logger.info(f"Stored model from round {server_round}")
        
        # Add participation metrics
        if metrics is None:
            metrics = {}
        metrics["total_client_responses"] = total_responses
        metrics["participating_clients"] = len(participating_results)
        metrics["participation_rate"] = len(participating_results) / total_responses if total_responses > 0 else 0
        
        return aggregated_params, metrics

    def evaluate(self, server_round: int, parameters):
        """Evaluate the global model"""
        if self.testloader is None:
            return None
            
        logger.info(f"Server evaluation for round {server_round}")
        return None
    
    def get_model_from_round(self, round_num):
        """Get model parameters from a specific round"""
        logger.info(f"[DEBUG] get_model_from_round called with round_num={round_num}")
        
        if round_num == 0:
            # Round 0 uses saved initial parameters
            logger.info(f"[DEBUG] Round 0 requested, _saved_initial_parameters exists: {self._saved_initial_parameters is not None}")
            if self._saved_initial_parameters is not None:
                logger.info("Returning initial parameters for round 0")
                return self._saved_initial_parameters  # Return the actual initial parameters
            else:
                logger.error("No initial parameters available!")
                return None
        
        elif round_num in self.round_models:
            logger.info(f"Returning model from round {round_num}")
            return ndarrays_to_parameters(self.round_models[round_num])
        
        else:
            logger.warning(f"Round {round_num} not found")
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
        
        # Convert to consistent format (strings) and log inputs
        suspicious_clients = [str(client_id) for client_id in suspicious_clients]
        ground_truth = [str(client_id) for client_id in ground_truth]
        
        logger.info(f"=== METRICS CALCULATION DEBUG ===")
        logger.info(f"Total clients: {total_clients}")
        logger.info(f"Suspicious clients: {suspicious_clients}")
        logger.info(f"Ground truth malicious: {ground_truth}")
        
        # Convert to sets for easier operations
        suspicious_set = set(suspicious_clients)
        malicious_set = set(ground_truth)
        
        # Calculate confusion matrix components
        true_positives = len(suspicious_set & malicious_set)  # Correctly identified malicious
        false_positives = len(suspicious_set - malicious_set)  # Incorrectly flagged as malicious
        false_negatives = len(malicious_set - suspicious_set)  # Missed malicious clients
        
        # CRITICAL FIX: True negatives calculation
        # TN = Total clients - all clients that are either suspicious OR actually malicious
        all_identified_clients = suspicious_set | malicious_set  # Union of both sets
        true_negatives = total_clients - len(all_identified_clients)
        
        logger.info(f"TP (correctly detected malicious): {true_positives}")
        logger.info(f"FP (incorrectly flagged as malicious): {false_positives}")
        logger.info(f"FN (missed malicious): {false_negatives}")
        logger.info(f"TN (correctly identified as benign): {true_negatives}")
        logger.info(f"Sum: {true_positives + false_positives + false_negatives + true_negatives}")
        
        # Validation checks
        if true_negatives < 0:
            logger.error(f"INVALID: True negatives is negative ({true_negatives})")
            logger.error(f"This means total_clients ({total_clients}) < TP+FP+FN ({true_positives + false_positives + false_negatives})")
            # Set to 0 to prevent negative values
            true_negatives = max(0, true_negatives)
        
        total_check = true_positives + false_positives + false_negatives + true_negatives
        if total_check != total_clients:
            logger.warning(f"MISMATCH: Confusion matrix sum ({total_check}) != total_clients ({total_clients})")
        
        # Calculate metrics with safe division
        precision = true_positives / len(suspicious_set) if suspicious_set else 0.0
        recall = true_positives / len(malicious_set) if malicious_set else 0.0
        accuracy = (true_positives + true_negatives) / total_clients if total_clients > 0 else 0.0
        
        # Fix FPR calculation - should be FP / (FP + TN)
        false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Additional useful metrics
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
        
        logger.info(f"FINAL METRICS:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1-Score: {f1_score:.3f}")
        logger.info(f"  FPR: {false_positive_rate:.3f}")
        logger.info(f"  Specificity: {specificity:.3f}")
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


def create_strategy(config, initial_parameters, testloader=None, malicious_detector=None):
    """Create the federated learning strategy with malicious detection"""
    
    def fit_config(server_round: int):
        return {
            "server_round": server_round,
            "round_num": server_round,  # Add round_num for client compatibility
            "local_epochs": config.config_fit.local_epochs,
            "lr": config.config_fit.lr,
            "momentum": config.config_fit.momentum,
        }
    
    def evaluate_config(server_round: int):
        return {"server_round": server_round}
    
    def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Dict]]) -> Dict:
        """Aggregate training metrics from clients"""
        total_examples = sum(num_examples for num_examples, _ in metrics)
        losses = [num_examples * m.get("loss", 0) for num_examples, m in metrics]
        aggregated_loss = sum(losses) / total_examples if total_examples > 0 else 0
        
        logger.info(f"📊 Training - Aggregated loss: {aggregated_loss:.4f}")
        return {"loss": aggregated_loss}
    
    def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Dict]]) -> Dict:
        """Aggregate evaluation metrics from clients"""
        total_examples = sum(num_examples for num_examples, _ in metrics)
        
        accuracies = [num_examples * m.get("accuracy", 0) for num_examples, m in metrics]
        aggregated_accuracy = sum(accuracies) / total_examples if total_examples > 0 else 0
        
        losses = [num_examples * m.get("loss", 0) for num_examples, m in metrics]
        aggregated_loss = sum(losses) / total_examples if total_examples > 0 else 0
        
        logger.info(f"✅ Evaluation - Accuracy: {aggregated_accuracy:.4f}, Loss: {aggregated_loss:.4f}")
        
        return {
            "accuracy": aggregated_accuracy,
            "loss": aggregated_loss
        }
    
    strategy = K8sFederatedStrategy(
        config=config,
        testloader=testloader,
        malicious_detector=malicious_detector,
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