from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment: str = "stock-forecast"
    mlflow_registered_model_name: str = "stock_forecast_model"
    mlflow_stage: str = "Production"  # or None to always load latest

    # Artifact fallback paths (existing repo artifacts)
    fallback_model_path: str = "Model.h5"
    fallback_feature_scaler_path: str = "Feature_Scaler.pck"
    fallback_target_scaler_path: str = "Target_Scaler.pck"

    # Optional S3 / object storage (works with AWS S3 or MinIO)
    s3_endpoint_url: str | None = "http://minio:9000"
    s3_bucket: str = "stock-mlops"
    s3_region: str = "us-east-1"
    aws_access_key_id: str | None = "minioadmin"
    aws_secret_access_key: str | None = "minioadmin"

    # Prometheus Pushgateway
    pushgateway_url: str = "http://pushgateway:9091"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
