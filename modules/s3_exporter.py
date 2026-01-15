"""
S3 Metrics Exporter for FL K8s Dashboard
Handles uploading training metrics to AWS S3 bucket with error handling and compression.
"""

import json
import gzip
import logging
import os
import csv
import io
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logging.warning("boto3 not available. S3 export will be disabled.")


logger = logging.getLogger(__name__)


class S3MetricsExporter:
    """
    Handles export of federated learning metrics to S3 bucket.
    
    Features:
    - Synchronous JSON upload to S3
    - Optional gzip compression
    - Connection status tracking
    - Error handling without crashing training
    """
    
    def __init__(
        self, 
        bucket: str, 
        region: Optional[str] = None,
        session_id: Optional[str] = None,
        compress: bool = True,
        s3_prefix: str = "sessions/",
        local_export_dir: str = "temp-data",
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None
    ):
        """
        Initialize S3 exporter.
        
        Args:
            bucket: S3 bucket name
            region: AWS region (default: from AWS_DEFAULT_REGION env or us-east-1)
            session_id: Unique session identifier for this training run
            compress: Enable gzip compression (default: True)
            s3_prefix: Prefix path in S3 bucket (default: "sessions/")
            local_export_dir: Local directory to save copies (default: "temp-data")
            access_key_id: AWS access key (default: from AWS_ACCESS_KEY_ID env)
            secret_access_key: AWS secret key (default: from AWS_SECRET_ACCESS_KEY env)
        """
        self.bucket = bucket
        # Use provided region or fallback to env variable or default
        self.region = region or os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        self.session_id = session_id or datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
        self.compress = compress
        self.s3_prefix = s3_prefix.rstrip('/') + '/'
        self.local_export_dir = local_export_dir
        self.is_connected = False
        self.s3_client = None
        
        # Track SHAP CSV data in memory
        self.shap_csv_rows = []
        self.shap_csv_headers = None
        
        # Create local export directory if it doesn't exist
        os.makedirs(self.local_export_dir, exist_ok=True)
        
        # Initialize S3 client
        if not BOTO3_AVAILABLE:
            logger.error("boto3 is not installed. S3 export disabled.")
            return
            
        try:
            # Prepare S3 client kwargs with credentials from .env or parameters
            s3_kwargs = {'region_name': self.region}
            
            # Add credentials if provided or from environment
            access_key = access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
            secret_key = secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
            
            if access_key and secret_key:
                s3_kwargs['aws_access_key_id'] = access_key
                s3_kwargs['aws_secret_access_key'] = secret_key
                logger.debug("Using AWS credentials from environment variables")
            
            self.s3_client = boto3.client('s3', **s3_kwargs)
            # Test connection with a simple head_bucket call
            self.s3_client.head_bucket(Bucket=bucket)
            self.is_connected = True
            logger.info(f"✓ S3 exporter initialized: bucket={bucket}, region={self.region}, session={self.session_id}")
        except NoCredentialsError:
            logger.error("✗ AWS credentials not found. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY in .env or pass them as parameters.")
            self.is_connected = False
        except ClientError as e:
            logger.error(f"✗ Failed to connect to S3 bucket '{bucket}': {e}")
            self.is_connected = False
        except Exception as e:
            logger.error(f"✗ Unexpected error initializing S3 client: {e}")
            self.is_connected = False
    
    def upload_json(self, data: Dict[str, Any], s3_key: str = None, filename: str = None, category: str = None) -> bool:
        """
        Upload JSON data to S3 and save local copy.
        
        Args:
            data: Dictionary to serialize as JSON
            s3_key: S3 object key (path within bucket) - if provided, filename and category are ignored
            filename: Name of the JSON file (used with category to construct s3_key)
            category: Subdirectory category (e.g., 'shap_analysis', 'rounds')
            
        Returns:
            bool: True if upload succeeded, False otherwise
        """
        # Construct s3_key from filename and category if s3_key not provided
        if s3_key is None:
            if filename is None:
                raise ValueError("Either s3_key or filename must be provided")
            if category:
                s3_key = f"{category}/{filename}"
            else:
                s3_key = filename
        
        # Serialize to JSON
        json_data = json.dumps(data, indent=2, default=str)
        json_bytes = json_data.encode('utf-8')
        
        # Save local copy
        try:
            local_path = os.path.join(self.local_export_dir, self.session_id, s3_key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'w') as f:
                f.write(json_data)
            logger.debug(f"Local copy saved: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to save local copy to {local_path}: {e}")
        
        if not self.s3_client:
            logger.warning("S3 client not initialized. Skipping upload.")
            return False
        
        try:
            upload_start = datetime.utcnow()
            
            # Optionally compress
            if self.compress:
                json_bytes = gzip.compress(json_bytes)
                content_encoding = 'gzip'
            else:
                content_encoding = None
            
            # Construct full S3 key
            full_key = f"{self.s3_prefix}{self.session_id}/{s3_key}"
            
            # Upload to S3
            extra_args = {'ContentType': 'application/json'}
            if content_encoding:
                extra_args['ContentEncoding'] = content_encoding
            
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=full_key,
                Body=json_bytes,
                **extra_args
            )
            
            upload_duration = (datetime.utcnow() - upload_start).total_seconds()
            logger.debug(f"S3 upload completed in {upload_duration:.2f}s: s3://{self.bucket}/{full_key}")
            
            self.is_connected = True
            return True
            
        except ClientError as e:
            logger.error(f"S3 upload failed (ClientError): {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"S3 upload failed (unexpected error): {e}")
            self.is_connected = False
            return False
    
    def upload_round_data(self, data: Dict[str, Any], round_num: int) -> bool:
        """
        Upload round metrics data to S3.
        
        Args:
            data: Round metrics dictionary
            round_num: Training round number
            
        Returns:
            bool: True if upload succeeded
        """
        s3_key = f"rounds/round_{round_num:03d}.json"
        success = self.upload_json(data, s3_key)
        
        # Also upload as "latest" for easy access
        if success:
            self.upload_json(data, "rounds/round_latest.json")
        
        return success
    
    def upload_round_shap_data(self, data: Dict[str, Any], round_num: int) -> bool:
        """
        DEPRECATED: Use add_shap_row_data and upload_shap_csv_data instead.
        This method is kept for backwards compatibility but does nothing.
        """
        logger.warning("upload_round_shap_data is deprecated. Use add_shap_row_data and upload_shap_csv_data instead.")
        return False
    
    def add_shap_row_data(self, client_id: int, round_num: int, features: Dict[str, Any], 
                          shap_values: Dict[str, Any], main_task_accuracy: float, 
                          main_task_loss: float, predicted_label: str = "benign", 
                          ground_truth_label: str = "benign") -> None:
        """
        Add a row to the cumulative SHAP CSV data.
        
        Args:
            client_id: Client identifier
            round_num: Training round number
            features: Dictionary of client feature values
            shap_values: Dictionary of SHAP feature importance values
            main_task_accuracy: Main task accuracy for this client in this round
            main_task_loss: Main task loss for this client in this round
            predicted_label: Predicted label ("malicious" or "benign")
            ground_truth_label: Ground truth label ("malicious" or "benign")
        """
        # Build the row with all columns
        row = {
            'client_id': client_id,
            'round_num': round_num,
            'main_task_accuracy': main_task_accuracy,
            'main_task_loss': main_task_loss,
            'predicted_label': predicted_label,
            'ground_truth_label': ground_truth_label,
        }
        
        # Add feature columns
        for feature_name, feature_value in features.items():
            row[feature_name] = feature_value if isinstance(feature_value, (int, float)) else str(feature_value)
        
        # Add SHAP columns (prefixed with SHAP_)
        for feature_name, shap_value in shap_values.items():
            row[f'SHAP_{feature_name}'] = shap_value if isinstance(shap_value, (int, float)) else 0.0
        
        self.shap_csv_rows.append(row)
        
        # Initialize headers on first row
        if self.shap_csv_headers is None:
            self.shap_csv_headers = list(row.keys())
    
    def upload_shap_csv_data(self) -> bool:
        """
        Upload all accumulated SHAP CSV rows to S3 as a single CSV file.
        
        Returns:
            bool: True if upload succeeded, False otherwise
        """
        if not self.shap_csv_rows:
            logger.warning("No SHAP rows to upload")
            return False
        
        if not self.shap_csv_headers:
            logger.error("SHAP headers not initialized")
            return False
        
        try:
            # Create CSV in memory
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=self.shap_csv_headers)
            writer.writeheader()
            writer.writerows(self.shap_csv_rows)
            
            csv_content = output.getvalue()
            csv_bytes = csv_content.encode('utf-8')
            
            # Save local copy
            try:
                local_path = os.path.join(self.local_export_dir, self.session_id, 'shap_analysis.csv')
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'w') as f:
                    f.write(csv_content)
                logger.debug(f"Local SHAP CSV saved: {local_path}")
            except Exception as e:
                logger.warning(f"Failed to save local SHAP CSV: {e}")
            
            if not self.s3_client:
                logger.warning("S3 client not initialized. Skipping SHAP CSV upload.")
                return False
            
            upload_start = datetime.utcnow()
            
            # Construct full S3 key
            s3_key = f"{self.s3_prefix}{self.session_id}/shap_analysis.csv"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=csv_bytes,
                ContentType='text/csv'
            )
            
            upload_duration = (datetime.utcnow() - upload_start).total_seconds()
            logger.info(f"S3 SHAP CSV upload completed in {upload_duration:.2f}s: s3://{self.bucket}/{s3_key}")
            logger.info(f"Total SHAP rows: {len(self.shap_csv_rows)}")
            
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload SHAP CSV: {e}")
            self.is_connected = False
            return False
    
    def upload_cumulative_history(self, data: Dict[str, Any]) -> bool:
        """
        Upload cumulative client history data.
        
        Args:
            data: Client history dictionary
            
        Returns:
            bool: True if upload succeeded
        """
        s3_key = "cumulative/client_history.json"
        return self.upload_json(data, s3_key)

    
    def upload_final_summary(self, data: Dict[str, Any]) -> bool:
        """
        Upload final training summary.
        
        Args:
            data: Summary data dictionary
            
        Returns:
            bool: True if upload succeeded
        """
        s3_key = "summary.json"
        return self.upload_json(data, s3_key)
    
    def get_session_path(self) -> str:
        """
        Get the full S3 path for this session.
        
        Returns:
            str: S3 path (e.g., "s3://bucket/sessions/2025-12-08_15-30-45/")
        """
        return f"s3://{self.bucket}/{self.s3_prefix}{self.session_id}/"
    
    def upload_csv(self, rows: List[Dict], s3_key: str, fieldnames: List[str]) -> bool:
        """
        Upload CSV data to S3 with CUMULATIVE append behavior (matching simulation).
        Downloads existing file, appends new rows, and re-uploads.
        
        Args:
            rows: List of dictionaries (new rows to append)
            s3_key: S3 key for the CSV file (e.g., "shap_data.csv")
            fieldnames: List of column names in exact order
            
        Returns:
            bool: True if upload succeeded
        """
        if not rows:
            logger.warning("No CSV rows to upload")
            return False
        
        try:
            upload_start = datetime.utcnow()
            
            # Construct full S3 key
            full_key = f"{self.s3_prefix}{self.session_id}/{s3_key}"
            
            # Step 1: Try to download existing CSV
            existing_rows = []
            
            if self.s3_client:
                try:
                    response = self.s3_client.get_object(Bucket=self.bucket, Key=full_key)
                    existing_csv_data = response['Body'].read()
                    
                    # Decompress if needed
                    if response.get('ContentEncoding') == 'gzip':
                        existing_csv_data = gzip.decompress(existing_csv_data)
                    
                    existing_csv_data = existing_csv_data.decode('utf-8')
                    
                    # Parse existing CSV
                    csv_reader = csv.DictReader(io.StringIO(existing_csv_data))
                    existing_rows = list(csv_reader)
                    logger.debug(f"📥 Downloaded existing CSV with {len(existing_rows)} rows")
                    
                except self.s3_client.exceptions.NoSuchKey:
                    logger.debug(f"CSV file does not exist yet: {full_key}. Creating new file.")
                except Exception as e:
                    logger.warning(f"Could not download existing CSV: {e}. Will create new file.")
            
            # Step 2: Combine existing and new rows
            all_rows = existing_rows + rows
            logger.info(f"📊 Combining {len(existing_rows)} existing + {len(rows)} new = {len(all_rows)} total rows")
            
            # Step 3: Create CSV in memory
            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
            
            csv_data = csv_buffer.getvalue().encode('utf-8')
            
            # Step 4: Optionally compress
            if self.compress:
                csv_data = gzip.compress(csv_data)
                content_encoding = 'gzip'
            else:
                content_encoding = None
            
            # Step 5: Save local copy
            try:
                local_path = os.path.join(self.local_export_dir, self.session_id, s3_key)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'w') as f:
                    f.write(csv_buffer.getvalue())
                logger.debug(f"💾 Local CSV saved: {local_path}")
            except Exception as e:
                logger.warning(f"Failed to save local CSV: {e}")
            
            # Step 6: Upload to S3
            if not self.s3_client:
                logger.warning("S3 client not initialized. Skipping CSV upload.")
                return False
            
            extra_args = {'ContentType': 'text/csv'}
            if content_encoding:
                extra_args['ContentEncoding'] = content_encoding
            
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=full_key,
                Body=csv_data,
                **extra_args
            )
            
            upload_duration = (datetime.utcnow() - upload_start).total_seconds()
            logger.info(f"✅ S3 CSV upload completed in {upload_duration:.2f}s: {len(all_rows)} total rows ({len(rows)} new)")
            logger.debug(f"📍 S3 path: s3://{self.bucket}/{full_key}")
            
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"❌ S3 CSV upload failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_connected = False
            return False
