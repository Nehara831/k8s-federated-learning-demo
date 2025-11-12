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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Exception],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate model weights with malicious client detection"""
        
        total_responses = len(results)
        logger.info(f"Round {server_round}: Received {total_responses} client responses")
        
        # Start new round in detector
        if self.malicious_detector and server_round > 0:
            self.malicious_detector.start_new_round(server_round)
        
        # Detect malicious clients
        participating_results = []
        malicious_ground_truth = []
        
        for client_proxy, fit_res in results:
            client_id = int(client_proxy.cid)
            
            # Track ground truth malicious clients
            attack_type = fit_res.metrics.get("attack_type", "none")
            if attack_type != "none":
                malicious_ground_truth.append(str(client_id))
                logger.info(f"Client {client_id} performed {attack_type} attack")
            
            # Run malicious detection (only after round 1)
            if self.malicious_detector and server_round > 1:
                current_round_params = parameters_to_ndarrays(fit_res.parameters)
                prev_round_model = self.get_previous_round_model(server_round)
                
                if prev_round_model is not None:
                    prev_round_params = parameters_to_ndarrays(prev_round_model)
                    
                    # Update detector and get prediction
                    prediction = self.malicious_detector.update_client_behavior(
                        client_id, 
                        server_round, 
                        current_round_params, 
                        prev_round_params
                    )
                    
                    is_actually_malicious = str(client_id) in malicious_ground_truth
                    logger.info(f"Client {client_id}: Predicted={prediction}, Actual Malicious={is_actually_malicious}")
        
        # Filter out detected malicious clients
        if self.malicious_detector and self.malicious_detector.is_trained and server_round > 1:
            suspicious_clients = self.malicious_detector.get_suspicious_clients()
            
            if suspicious_clients:
                filtered_results = []
                malicious_count = 0
                
                for client_proxy, fit_res in results:
                    client_id = int(client_proxy.cid)
                    if client_id not in suspicious_clients:
                        filtered_results.append((client_proxy, fit_res))
                    else:
                        malicious_count += 1
                        logger.warning(f"🚫 Client {client_id} EXCLUDED (detected as malicious)")
                
                participating_results = filtered_results
                logger.info(f"Filtered out {malicious_count} detected malicious clients")
            else:
                participating_results = results
            
            # Calculate and save detection metrics
            perf_results = self.calculate_metrics(
                suspicious_clients, 
                malicious_ground_truth, 
                total_responses
            )
            
            if "error" not in perf_results:
                self.malicious_detector.save_performance_metrics(perf_results, server_round)
        else:
            participating_results = results
        
        logger.info(f"Aggregating {len(participating_results)} participating clients")
        
        # Call parent's aggregate_fit
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
            logger.warning(f"Round {round_num} not found")
            return None
    
    def get_previous_round_model(self, current_round):
        """Get the model from the previous round"""
        if current_round <= 0:
            return None
        
        previous_round = current_round - 1
        return self.get_model_from_round(previous_round)
    
    def calculate_metrics(self, suspicious_clients, ground_truth, total_clients):
        """Calculate detection performance metrics"""
        suspicious_clients = [str(client_id) for client_id in suspicious_clients]
        ground_truth = [str(client_id) for client_id in ground_truth]
        
        suspicious_set = set(suspicious_clients)
        malicious_set = set(ground_truth)
        
        # Confusion matrix
        true_positives = len(suspicious_set & malicious_set)
        false_positives = len(suspicious_set - malicious_set)
        false_negatives = len(malicious_set - suspicious_set)
        
        all_identified_clients = suspicious_set | malicious_set
        true_negatives = total_clients - len(all_identified_clients)
        
        # Metrics
        precision = true_positives / len(suspicious_set) if suspicious_set else 0.0
        recall = true_positives / len(malicious_set) if malicious_set else 0.0
        accuracy = (true_positives + true_negatives) / total_clients if total_clients > 0 else 0.0
        
        false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
        
        logger.info(f"Detection Metrics: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1_score:.3f}")
        
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
        malicious_detector=malicious_detector,  # ← ADD DETECTOR
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