"""
PoisonedFL Attack Implementation
Based on: "PoisonedFL: Model Poisoning Attacks to Federated Learning via Multi-Round Consistency"
arXiv:2404.15611 (2024)

This module implements the PoisonedFL attack which uses:
1. Fixed sign vector (s) across all rounds for multi-round consistency
2. Adaptive magnitude (lambda) via hypothesis testing based on cosine similarity
3. Benign update estimation to mimic legitimate client behavior
4. Direction vector (v) that adapts to benign patterns
"""

import numpy as np
import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class PoisonedFLAttacker:
    """
    Implements the PoisonedFL attack algorithm.
    
    The attack constructs malicious updates as:
        g_mal^t = λ^t × v^t ⊙ s
    
    where:
        - s: fixed sign vector sampled once at initialization
        - λ^t: adaptive magnitude adjusted via hypothesis testing
        - v^t: direction vector computed from benign update estimates
    """
    
    def __init__(
        self,
        model_shape: List[int],
        sign_vector_seed: int = 42,
        initial_lambda: float = 8.0,
        lambda_increase_factor: float = 1.0,
        lambda_decrease_factor: float = 0.7,
        cosine_similarity_threshold: float = 0.5,
        malicious_ratio: float = 0.4,
        benign_estimation_warmup: int = 1,
        minimum_lambda: float = 0.5,
    ):
        """
        Initialize the PoisonedFL attacker.
        
        Args:
            model_shape: Shape of the flattened model parameters
            sign_vector_seed: Random seed for generating fixed sign vector (MUST be same across all malicious clients)
            initial_lambda: Initial magnitude scaling factor (c₀, paper uses 8.0)
            lambda_increase_factor: Factor to maintain lambda when stealthy (paper uses 1.0 - no increase)
            lambda_decrease_factor: Factor to decrease lambda when detected (β, paper uses 0.7)
            cosine_similarity_threshold: Threshold for hypothesis testing (cos > threshold → maintain lambda)
            malicious_ratio: Fraction of malicious clients (α in paper)
            benign_estimation_warmup: Number of rounds before using benign estimation
            minimum_lambda: Minimum lambda value (paper: don't go below 0.5)
        """
        self.model_dim = np.prod(model_shape)
        self.sign_vector_seed = sign_vector_seed
        self.initial_lambda = initial_lambda
        self.lambda_increase_factor = lambda_increase_factor
        self.lambda_decrease_factor = lambda_decrease_factor
        self.cosine_threshold = cosine_similarity_threshold
        self.malicious_ratio = malicious_ratio
        self.benign_warmup = benign_estimation_warmup
        self.minimum_lambda = minimum_lambda
        
        # Initialize fixed sign vector (CRITICAL: must be same across all malicious clients)
        self.sign_vector = self._initialize_sign_vector()
        
        # Attack state
        self.lambda_t = initial_lambda
        self.current_round = 0
        
        # History tracking for benign estimation
        self.global_model_history = []  # Store w^t for each round
        self.malicious_update_history = []  # Store g_mal^{t-1}
        self.benign_estimates = []  # Store estimated benign updates
        
        logger.info(f"Initialized PoisonedFL attacker with seed={sign_vector_seed}, lambda={initial_lambda}")
        logger.info(f"Sign vector stats: mean={np.mean(self.sign_vector):.4f}, "
                   f"positive_ratio={np.mean(self.sign_vector > 0):.4f}")
    
    def _initialize_sign_vector(self) -> np.ndarray:
        """
        Initialize the fixed sign vector s ∈ {-1, +1}^d.
        
        CRITICAL: This must use a deterministic seed that is SHARED across all malicious clients.
        The sign vector remains fixed throughout all rounds to maintain multi-round consistency.
        
        Returns:
            Sign vector of shape (model_dim,) with values in {-1, +1}
        """
        rng = np.random.RandomState(self.sign_vector_seed)
        sign_vector = rng.choice([-1, 1], size=self.model_dim)
        logger.info(f"Generated fixed sign vector with seed {self.sign_vector_seed}")
        return sign_vector
    
    def update_global_model(self, global_parameters: np.ndarray, round_num: int):
        """
        Track the global model received at the beginning of each round.
        
        This is needed for benign update estimation:
            benign_est^t ≈ (w^t - w^{t-1} - α × g_mal^{t-1}) / (1 - α)
        
        Args:
            global_parameters: Flattened global model parameters w^t
            round_num: Current round number
        """
        self.current_round = round_num
        self.global_model_history.append(global_parameters.copy())
        
        # Keep only last 2 global models (needed for benign estimation)
        if len(self.global_model_history) > 2:
            self.global_model_history.pop(0)
        
        logger.debug(f"[Round {round_num}] Stored global model, history length: {len(self.global_model_history)}")
    
    def _estimate_benign_update(self) -> Optional[np.ndarray]:
        """
        Estimate the benign update from global model changes.
        
        Formula from paper (Section 3.2):
            benign_est^t ≈ (w^t - w^{t-1} - α × g_mal^{t-1}) / (1 - α)
        
        where:
            - w^t: current global model
            - w^{t-1}: previous global model
            - g_mal^{t-1}: previous malicious update
            - α: malicious ratio
        
        Returns:
            Estimated benign update or None if insufficient history
        """
        if len(self.global_model_history) < 2:
            logger.debug(f"[Round {self.current_round}] Insufficient history for benign estimation")
            return None
        
        if len(self.malicious_update_history) == 0:
            logger.debug(f"[Round {self.current_round}] No previous malicious updates")
            return None
        
        w_t = self.global_model_history[-1]  # Current global model
        w_t_minus_1 = self.global_model_history[-2]  # Previous global model
        g_mal_t_minus_1 = self.malicious_update_history[-1]  # Previous malicious update
        
        # Compute: (w^t - w^{t-1} - α × g_mal^{t-1}) / (1 - α)
        numerator = w_t - w_t_minus_1 - self.malicious_ratio * g_mal_t_minus_1
        benign_est = numerator / (1.0 - self.malicious_ratio)
        
        logger.debug(f"[Round {self.current_round}] Estimated benign update norm: {np.linalg.norm(benign_est):.6f}")
        
        return benign_est
    
    def _compute_v_vector(self, benign_estimates: List[np.ndarray]) -> np.ndarray:
        """
        Compute the direction vector v^t from benign update estimates.
        
        Formula from paper (Algorithm 1, line 8-9):
            v^t = |avg(benign_estimates)| / ||avg(benign_estimates)||
        
        This makes the malicious update mimic the magnitude pattern of benign updates
        while the sign is controlled by the fixed sign vector s.
        
        Args:
            benign_estimates: List of estimated benign updates from previous rounds
        
        Returns:
            Normalized direction vector v^t
        """
        if len(benign_estimates) == 0:
            # Fallback: uniform direction if no estimates available
            logger.warning(f"[Round {self.current_round}] No benign estimates, using uniform v vector")
            v = np.ones(self.model_dim) / np.sqrt(self.model_dim)
            return v
        
        # Average the benign estimates
        avg_benign = np.mean(benign_estimates, axis=0)
        
        # Take absolute value
        v = np.abs(avg_benign)
        
        # Normalize
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:
            v = v / v_norm
        else:
            logger.warning(f"[Round {self.current_round}] Near-zero v vector norm, using uniform")
            v = np.ones(self.model_dim) / np.sqrt(self.model_dim)
        
        logger.debug(f"[Round {self.current_round}] Computed v vector, norm: {np.linalg.norm(v):.6f}")
        
        return v
    
    def _compute_cosine_similarity(self, malicious_update: np.ndarray, benign_estimate: np.ndarray) -> float:
        """
        Compute cosine similarity between malicious update and estimated benign update.
        
        Formula:
            cos_sim = (g_mal · benign_est) / (||g_mal|| × ||benign_est||)
        
        This is used for hypothesis testing to adapt lambda:
            - High cosine (> threshold): attack is effective, maintain lambda
            - Low cosine (< threshold): attack detected, decrease lambda
        
        Args:
            malicious_update: Generated malicious update g_mal^t
            benign_estimate: Estimated benign update
        
        Returns:
            Cosine similarity in [-1, 1]
        """
        dot_product = np.dot(malicious_update, benign_estimate)
        
        mal_norm = np.linalg.norm(malicious_update)
        benign_norm = np.linalg.norm(benign_estimate)
        
        if mal_norm < 1e-10 or benign_norm < 1e-10:
            logger.warning(f"[Round {self.current_round}] Near-zero norm in cosine computation")
            return 0.0
        
        cos_sim = dot_product / (mal_norm * benign_norm)
        
        return cos_sim
    
    def _update_lambda(self, cos_similarity: float):
        """
        Update the magnitude scaling factor lambda via hypothesis testing.
        
        Adaptive strategy from paper (Algorithm 1, Section 6.1.5):
            - If cos_sim > threshold: attack is stealthy, MAINTAIN lambda (×1.0)
            - If cos_sim ≤ threshold: attack may be detected, DECREASE lambda (×β=0.7)
            - Never let c_t go below minimum (paper: 0.5)
        
        Paper quote: "β = 0.7, and we do not decrease c_t to be smaller than 0.5"
        
        Args:
            cos_similarity: Cosine similarity between malicious and benign updates
        """
        old_lambda = self.lambda_t
        
        if cos_similarity > self.cosine_threshold:
            # Attack is stealthy, maintain magnitude (paper uses ×1.0)
            self.lambda_t *= self.lambda_increase_factor
            logger.info(f"[Round {self.current_round}] Cosine {cos_similarity:.4f} > {self.cosine_threshold:.4f}, "
                       f"maintaining lambda: {old_lambda:.2f} (×{self.lambda_increase_factor})")
        else:
            # Attack may be detected, decrease magnitude
            self.lambda_t *= self.lambda_decrease_factor
            logger.info(f"[Round {self.current_round}] Cosine {cos_similarity:.4f} ≤ {self.cosine_threshold:.4f}, "
                       f"decreasing lambda: {old_lambda:.2f} → {self.lambda_t:.2f} (×{self.lambda_decrease_factor})")
        
        # Enforce minimum lambda (paper: don't go below 0.5)
        if self.lambda_t < self.minimum_lambda:
            logger.warning(f"[Round {self.current_round}] Lambda {self.lambda_t:.2f} below minimum {self.minimum_lambda:.2f}, "
                         f"clamping to minimum")
            self.lambda_t = self.minimum_lambda
    
    def generate_malicious_update(
        self,
        clean_parameters: np.ndarray,
        local_update: np.ndarray
    ) -> np.ndarray:
        """
        Generate the malicious update using the PoisonedFL attack.
        
        Main attack formula (Algorithm 1):
            g_mal^t = λ^t × v^t ⊙ s
        
        where:
            - λ^t: adaptive magnitude (updated via hypothesis testing)
            - v^t: direction vector (from benign estimates)
            - s: fixed sign vector (ensures multi-round consistency)
            - ⊙: element-wise multiplication
        
        Args:
            clean_parameters: Original model parameters before attack
            local_update: Clean update from honest training (not used directly)
        
        Returns:
            Malicious update to replace the clean local update
        """
        # Step 1: Estimate benign update
        benign_est = self._estimate_benign_update()
        
        if benign_est is not None:
            self.benign_estimates.append(benign_est)
            # Keep only recent estimates (e.g., last 5 rounds)
            if len(self.benign_estimates) > 5:
                self.benign_estimates.pop(0)
        
        # Step 2: Compute direction vector v^t
        if self.current_round >= self.benign_warmup and len(self.benign_estimates) > 0:
            v_vector = self._compute_v_vector(self.benign_estimates)
        else:
            # Warmup phase: use uniform direction
            v_vector = np.ones(self.model_dim) / np.sqrt(self.model_dim)
            logger.debug(f"[Round {self.current_round}] Warmup phase, using uniform v vector")
        
        # Step 3: Construct malicious update: g_mal = λ × v ⊙ s
        malicious_update = self.lambda_t * v_vector * self.sign_vector
        
        logger.info(f"[Round {self.current_round}] Generated malicious update: "
                   f"lambda={self.lambda_t:.2f}, norm={np.linalg.norm(malicious_update):.6f}")
        
        # Step 4: Hypothesis testing for lambda adaptation
        if benign_est is not None:
            cos_sim = self._compute_cosine_similarity(malicious_update, benign_est)
            logger.info(f"[Round {self.current_round}] Cosine similarity: {cos_sim:.4f}")
            self._update_lambda(cos_sim)
        else:
            logger.debug(f"[Round {self.current_round}] Skipping lambda update (no benign estimate)")
        
        # Step 5: Store malicious update for next round's benign estimation
        self.malicious_update_history.append(malicious_update.copy())
        
        # Keep only last malicious update (needed for benign estimation)
        if len(self.malicious_update_history) > 1:
            self.malicious_update_history.pop(0)
        
        # Log attack statistics
        if len(self.global_model_history) > 0:
            current_model = self.global_model_history[-1]
            model_norm = np.linalg.norm(current_model)
            attack_norm = np.linalg.norm(malicious_update)
            logger.info(f"[Round {self.current_round}] Attack/Model norm ratio: {attack_norm/model_norm:.6f}")
        
        return malicious_update
    
    def reset(self):
        """
        Reset the attack state (but keep the fixed sign vector).
        
        This can be used to restart the attack while maintaining multi-round consistency.
        """
        self.lambda_t = self.initial_lambda
        self.current_round = 0
        self.global_model_history = []
        self.malicious_update_history = []
        self.benign_estimates = []
        
        logger.info(f"Reset PoisonedFL attacker (sign vector preserved)")
    
    def get_attack_stats(self) -> Dict[str, float]:
        """
        Get current attack statistics for monitoring.
        
        Returns:
            Dictionary with attack metrics
        """
        stats = {
            "lambda": self.lambda_t,
            "round": self.current_round,
            "num_benign_estimates": len(self.benign_estimates),
            "sign_vector_seed": self.sign_vector_seed,
        }
        
        if len(self.malicious_update_history) > 0:
            stats["last_attack_norm"] = float(np.linalg.norm(self.malicious_update_history[-1]))
        
        if len(self.benign_estimates) > 0:
            stats["avg_benign_norm"] = float(np.mean([np.linalg.norm(est) for est in self.benign_estimates]))
        
        return stats
