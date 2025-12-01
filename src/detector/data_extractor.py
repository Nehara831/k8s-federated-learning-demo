import numpy as np


def extract_model_features(params, client_id, round_num, reference_params=None):
    """Extract meaningful features from model parameters"""
    features = {
        "client_id": client_id,
        "round_num": round_num,
    }
    
    def get_flat_params(p):
        if isinstance(p, list):
            return np.concatenate([param.flatten() for param in p])
        elif isinstance(p, dict):
            return np.concatenate([param.flatten() for param in p.values()])
        else:
            raise TypeError(f"Parameters should be list or dict, got {type(p)}")
    
    all_params = get_flat_params(params)
    
    # Statistical features
    features["param_mean"] = np.mean(all_params)
    features["param_std"] = np.std(all_params)
    features["param_min"] = np.min(all_params)
    features["param_max"] = np.max(all_params)
    features["param_median"] = np.median(all_params)
    features["param_range"] = features["param_max"] - features["param_min"]
    features["param_abs_mean"] = np.mean(np.abs(all_params))
    
    # Distribution features - MUST MATCH SIMULATION EXACTLY (no epsilon)
    features["param_skew"] = np.mean((all_params - features["param_mean"])**3) / (features["param_std"]**3)
    features["param_kurtosis"] = np.mean((all_params - features["param_mean"])**4) / (features["param_std"]**4)
    features["param_neg_ratio"] = np.sum(all_params < 0) / len(all_params)
    features["param_zero_ratio"] = np.sum(np.abs(all_params) < 1e-6) / len(all_params)
    
    # Layer-specific features (focusing on the last layer)
    weight_layers = [name for name in params.keys() if "weight" in name]
    print("params keys:",params.keys())

    if weight_layers:
        last_layer = weight_layers[-1]
        last_layer_params = params[last_layer].flatten()
        features["last_layer_mean"] = np.mean(last_layer_params)
        features["last_layer_std"] = np.std(last_layer_params)
        features["last_layer_min"] = np.min(last_layer_params)
        features["last_layer_max"] = np.max(last_layer_params)
        features["last_layer_abs_mean"] = np.mean(np.abs(last_layer_params))
        features["last_layer_neg_ratio"] = np.sum(last_layer_params < 0) / len(last_layer_params)
    
    # Layer distribution comparison features
    if len(weight_layers) > 1:
        first_layer = weight_layers[0]
        first_layer_params = params[first_layer].flatten()
        features["first_vs_last_mean_ratio"] = np.mean(np.abs(first_layer_params)) / np.mean(np.abs(last_layer_params))
        features["first_vs_last_std_ratio"] = np.std(first_layer_params) / np.std(last_layer_params)
    
    # Distance metrics (if reference model is available)
    if reference_params is not None:
        ref_all_params = get_flat_params(reference_params)
        min_len = min(len(all_params), len(ref_all_params))
        all_params_trunc = all_params[:min_len]
        ref_all_params_trunc = ref_all_params[:min_len]
        features["avg_l1_distance"] = np.mean(np.abs(all_params_trunc - ref_all_params_trunc))
        features["avg_l2_distance"] = np.mean((all_params_trunc - ref_all_params_trunc) ** 2)
        dot_product = np.dot(all_params_trunc, ref_all_params_trunc)
        norm_product = np.linalg.norm(all_params_trunc) * np.linalg.norm(ref_all_params_trunc)
        features["cosine_similarity"] = dot_product / norm_product if norm_product != 0 else 0

        # Layer-specific distances
        if isinstance(params, dict) and "weight" in "".join(params.keys()):
            weight_layers = [name for name in params.keys() if "weight" in name]
            if weight_layers:
                last_layer = weight_layers[-1]
                last_layer_params = params[last_layer].flatten()
                if isinstance(reference_params, dict):
                    ref_last_layer = reference_params.get(last_layer, np.array([]))
                    if ref_last_layer.size > 0:
                        ref_last_layer = ref_last_layer.flatten()
                        min_len = min(len(last_layer_params), len(ref_last_layer))
                        features["last_layer_l1_distance"] = np.mean(np.abs(
                            last_layer_params[:min_len] - ref_last_layer[:min_len]))
                        features["last_layer_l2_distance"] = np.mean(
                            (last_layer_params[:min_len] - ref_last_layer[:min_len]) ** 2)
    return features