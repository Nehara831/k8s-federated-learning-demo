import os
import logging
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv
import flwr as fl
import pickle
import numpy as np
import random
import torch

# Load environment variables from .env file
load_dotenv()

from src.shared.dataset import prepare_server_dataset
from src.shared.models import create_model_for_dataset
from src.server.server_wrapper import create_strategy
from src.detector.malicious_detector import MaliciousClientDetector  # Existing detector
from src.detector.num_distilbert_wrapper import NumDistilBERTWrapper  # New detector
from modules.s3_exporter import S3MetricsExporter
from modules.shap_calculator import SHAPCalculator

# Configure logging AFTER all imports to prevent other modules from overriding it
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.StreamHandler()  # Explicitly add stream handler for stdout
    ],
    force=True  # Force reconfiguration even if already configured
)

# Set our logger to INFO level explicitly
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Also set root logger to INFO
logging.getLogger().setLevel(logging.INFO)

# Force print to ensure we see output
print("=" * 80, flush=True)
print("🚀 SERVER STARTING - Logging initialized", flush=True)
print("=" * 80, flush=True)

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"🎲 Set random seed to {seed}")

def create_detector(config, reference_model):
    """
    Factory function to create the appropriate detector based on config.
    
    Args:
        config: Server configuration (OmegaConf)
        reference_model: Reference model for feature extraction
    
    Returns:
        Detector instance or None
    """
    # Check if detector is enabled
    if not hasattr(config, 'detector') or not config.detector.enabled:
        logger.info("=" * 80)
        logger.info("🔍 MALICIOUS CLIENT DETECTION: DISABLED")
        logger.info("=" * 80)
        return None
    
    detector_type = config.detector.get('type', 'krum').lower()
    save_dir = Path(config.detector.save_dir)
    num_clients = config.num_clients
    
    logger.info("=" * 80)
    logger.info(f"🔍 INITIALIZING DETECTOR: {detector_type.upper()}")
    logger.info("=" * 80)
    
    try:
        if detector_type == 'num_distilbert':
            # ✅ Use Num-DistilBERT detector
            logger.info("🤖 Loading Num-DistilBERT detector...")
            
            detector = NumDistilBERTWrapper(
                save_dir=save_dir,
                num_clients=num_clients,
                model_config=config.detector,
                reference_model=reference_model
            )
            
            logger.info(f"✅ Num-DistilBERT detector initialized")
            logger.info(f"   Model path: {config.detector.get('model_path', 'N/A')}")
            logger.info(f"   Scaler path: {config.detector.get('scaler_path', 'N/A')}")
            logger.info(f"   Trained: {detector.is_trained}")
            logger.info(f"   Threshold: {config.detector.get('threshold', 0.5)}")
            
        elif detector_type in ['krum', 'multi_krum', 'fedguard']:
            # ✅ Use existing MaliciousClientDetector (Krum/Multi-Krum/FedGuard)
            logger.info(f"📊 Loading {detector_type.upper()} detector...")
            
            min_rounds = config.detector.get('min_rounds_before_detection', 2)
            model_config = config.detector.get('model', None)
            
            detector = MaliciousClientDetector(
                save_dir=save_dir,
                num_clients=num_clients,
                model_config=model_config,
                reference_model=reference_model,
                min_rounds_before_detection=min_rounds
            )
            
            logger.info(f"✅ {detector_type.upper()} detector initialized")
            logger.info(f"   Min rounds before detection: {min_rounds}")
            logger.info(f"   Trained: {detector.is_trained}")
            
        else:
            logger.error(f"❌ Unknown detector type: '{detector_type}'")
            logger.error(f"   Supported types: 'num_distilbert', 'krum', 'multi_krum', 'fedguard'")
            logger.error(f"   Continuing WITHOUT detection...")
            logger.info("=" * 80)
            return None
        
        logger.info("=" * 80)
        return detector
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {e}")
        logger.error(f"   Continuing WITHOUT detection...")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=" * 80)
        return None

def load_initial_params_from_simulation(params_file):
    """Load initial parameters from simulation"""
    logger.info(f"📥 Attempting to load initial params from: {params_file}")
    
    try:
        with open(params_file, 'rb') as f:
            data = pickle.load(f)
        
        logger.info("=" * 80)
        logger.info("🎯 USING INITIAL PARAMETERS FROM SIMULATION")
        logger.info("=" * 80)
        logger.info(f"✅ Loaded {len(data['params'])} layers")
        logger.info(f"   Shapes: {data['shapes'][:3]}...")
        logger.info(f"   Means: {[f'{m:.6f}' for m in data['means'][:3]]}...")
        
        return data['params']
    
    except FileNotFoundError:
        logger.warning("=" * 80)
        logger.warning("⚠️  SIMULATION PARAMS NOT FOUND")
        logger.warning(f"    Expected at: {params_file}")
        logger.warning("=" * 80)
        return None
    
    except Exception as e:
        logger.error(f"❌ Failed to load simulation params: {e}")
        return None

def main():
    print("🔥 MAIN() FUNCTION CALLED", flush=True)
    print("📋 About to log with logger.info...", flush=True)
    logger.info("=" * 80)
    logger.info("🚀 STARTING FEDERATED LEARNING SERVER")
    logger.info("=" * 80)
    print("✅ Logger.info called successfully", flush=True)
    
    # Load configuration
    config_path = os.getenv('CONFIG_PATH', '/app/config/k8s-server.yaml')
    logger.info(f"📂 Loading config from: {config_path}")
    config = OmegaConf.load(config_path)
    
    logger.info(f"⚙️  Configuration:")
    logger.info(f"   Dataset: {config.dataset.type}")
    logger.info(f"   Num clients: {config.num_clients}")
    logger.info(f"   Num rounds: {config.num_rounds}")
    logger.info(f"   Num classes: {config.num_classes}")
    
    # Check if detector is configured
    if hasattr(config, 'detector') and config.detector.enabled:
        logger.info(f"   Detector: {config.detector.get('type', 'unknown').upper()} (enabled)")
    else:
        logger.info(f"   Detector: DISABLED")
    
    logger.info("=" * 80)
    
    # Load test dataset
    logger.info("📊 Loading test dataset...")
    testloader = prepare_server_dataset(config)
    logger.info(f"✅ Test dataset loaded: {len(testloader.dataset)} samples")
    
    # Load initial parameters from simulation
    sim_params_file = Path("/app/initial_params/simulation_initial_params.pkl")
    initial_params = load_initial_params_from_simulation(sim_params_file)
    
    if initial_params is None:
        logger.error("❌ No initial parameters available - cannot proceed")
        logger.error("   Please ensure simulation_initial_params.pkl is mounted")
        return
    
    # Convert to Flower format
    initial_parameters = fl.common.ndarrays_to_parameters(initial_params)
    logger.info(f"✅ Initial parameters converted to Flower format")
    
    # Create reference model for detector
    logger.info("🏗️  Creating reference model for detector...")
    input_size = 20 if config.dataset.type == "5gnidd" else None
    reference_model = create_model_for_dataset(
        dataset_type=config.dataset.type,
        num_classes=config.num_classes,
        input_size=input_size
    )
    
    num_params = sum(p.numel() for p in reference_model.parameters())
    logger.info(f"✅ Reference model created ({num_params:,} parameters)")
    
    # ✅ CREATE DETECTOR BASED ON CONFIG
    malicious_detector = create_detector(config, reference_model)
    
    if malicious_detector:
        detector_type = config.detector.get('type', 'unknown')
        logger.info(f"✅ Using {detector_type.upper()} detector for malicious client detection")
    else:
        logger.info("⚠️  Running without malicious client detection")
    
    # ✅ INITIALIZE S3 EXPORTER (if configured)
    print("=" * 80, flush=True)
    print("📤 INITIALIZING S3 METRICS EXPORTER", flush=True)
    print("=" * 80, flush=True)
    
    # Debug: Check if s3_export config exists
    print(f"🐛 DEBUG: hasattr(config, 's3_export') = {hasattr(config, 's3_export')}", flush=True)
    if hasattr(config, 's3_export'):
        print(f"🐛 DEBUG: config.s3_export = {config.s3_export}", flush=True)
        print(f"🐛 DEBUG: config.s3_export.enabled = {config.s3_export.enabled}", flush=True)
        print(f"🐛 DEBUG: type(config.s3_export.enabled) = {type(config.s3_export.enabled)}", flush=True)
    
    s3_exporter = None
    if hasattr(config, 's3_export') and config.s3_export.enabled:
        try:
            s3_bucket = config.s3_export.bucket
            # Load AWS credentials from environment variables (.env file)
            s3_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            s3_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            s3_region = config.s3_export.get('region') or os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            s3_prefix = config.s3_export.get('prefix', 'sessions/')
            
            print(f"🌐 Configuring S3 exporter:", flush=True)
            print(f"   Bucket: {s3_bucket}", flush=True)
            print(f"   Region: {s3_region}", flush=True)
            print(f"   Prefix: {s3_prefix}", flush=True)
            if s3_access_key:
                print(f"   Credentials: ✓ Loaded from env (AWS_ACCESS_KEY_ID={s3_access_key[:10]}...)", flush=True)
            else:
                print(f"   Credentials: ⚠️  Not found in env (AWS_ACCESS_KEY_ID)", flush=True)
            
            s3_exporter = S3MetricsExporter(
                bucket=s3_bucket,
                region=s3_region,
                compress=config.s3_export.get('compress', True),
                s3_prefix=s3_prefix,
                local_export_dir=config.s3_export.get('local_export_dir', 'temp-data'),
                access_key_id=s3_access_key,
                secret_access_key=s3_secret_key
            )
            
            if s3_exporter.is_connected:
                print(f"✅ S3 exporter initialized successfully", flush=True)
                print(f"   Session: {s3_exporter.session_id}", flush=True)
                print(f"   Path: {s3_exporter.get_session_path()}", flush=True)
            else:
                print(f"⚠️  S3 exporter failed to connect - metrics will only be saved locally", flush=True)
                
        except Exception as e:
            print(f"❌ Failed to initialize S3 exporter: {e}", flush=True)
            print(f"   Continuing without S3 export - metrics will only be saved locally", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
    else:
        print("ℹ️  S3 export is DISABLED in config", flush=True)
        print("   Metrics will be saved locally only", flush=True)
    
    print("=" * 80, flush=True)
    
    # ✅ INITIALIZE SHAP CALCULATOR (optional, for model explainability)
    logger.info("=" * 80)
    logger.info("🔍 INITIALIZING SHAP CALCULATOR")
    logger.info("=" * 80)
    
    shap_calculator = SHAPCalculator(max_samples=50, sample_size=20)
    
    if shap_calculator.available:
        logger.info(f"✅ SHAP calculator initialized (explainability enabled)")
        logger.info(f"   Max background samples: 50")
        logger.info(f"   Explanation samples: 20")
    else:
        logger.warning(f"⚠️  SHAP library not available - model explanations will be skipped")
        shap_calculator = None
    
    logger.info("=" * 80)
    
    # Create strategy
    logger.info("=" * 80)
    logger.info("🎯 CREATING FEDERATED LEARNING STRATEGY")
    logger.info("=" * 80)
    
    strategy = create_strategy(
        config=config,
        initial_parameters=initial_parameters,
        testloader=testloader,
        malicious_detector=malicious_detector,  # ✅ Pass detector
        s3_exporter=s3_exporter,                # ✅ Pass S3 exporter
        shap_calculator=shap_calculator         # ✅ Pass SHAP calculator
    )
    
    logger.info(f"✅ Strategy created: K8sFederatedStrategy")
    logger.info(f"   Fraction fit: {config.server.fraction_fit}")
    logger.info(f"   Fraction evaluate: {config.server.fraction_evaluate}")
    logger.info(f"   Min fit clients: {config.server.min_fit_clients}")
    
    # Start server
    server_address = f"0.0.0.0:{config.server.port}"
    
    logger.info("=" * 80)
    logger.info("🚀 STARTING FL SERVER")
    logger.info("=" * 80)
    logger.info(f"   Address: {server_address}")
    logger.info(f"   Rounds: {config.num_rounds}")
    logger.info(f"   Clients: {config.num_clients}")
    logger.info("=" * 80)
    
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
    )
    
    logger.info("=" * 80)
    logger.info("✅ FL SERVER FINISHED")
    logger.info("=" * 80)
    
    # Upload final summary to S3 after training completes
    if s3_exporter and strategy:
        try:
            logger.info("📊 Uploading final training summary to S3...")
            summary_success = strategy.upload_final_summary()
            if summary_success:
                logger.info(f"✅ Final summary uploaded: {s3_exporter.get_session_path()}summary.json")
            else:
                logger.warning("⚠️  Final summary upload failed")
        except Exception as e:
            logger.error(f"❌ Error uploading final summary: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
if __name__ == "__main__":
    main()
