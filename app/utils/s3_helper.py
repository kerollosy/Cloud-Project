import os
import boto3
import io
import logging
import zipfile
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def upload_file_to_s3(file_bytes: bytes, filename: str) -> str:
    bucket_name = os.getenv("S3_BUCKET_NAME")
    region_name = os.getenv("AWS_REGION")
    if not bucket_name or not region_name:
        raise ValueError("S3_BUCKET_NAME or AWS_REGION is missing. Cannot upload file to S3.")

    s3_client = boto3.client('s3', region_name=region_name)
    s3_key = f"resumes/{filename}"
    try:
        file_obj = io.BytesIO(file_bytes)
        
        s3_client.upload_fileobj(file_obj, bucket_name, s3_key)
        
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        return s3_uri
        
    except ClientError as e:
        logger.error(f"AWS Boto3 ClientError while uploading file: {e}")
        raise Exception("Failed to upload file to S3.")

def download_and_extract_model(extract_to_path: str) -> None:
    bucket_name = os.getenv("S3_BUCKET_NAME")
    region_name = os.getenv("AWS_REGION")
    if not bucket_name or not region_name:
        raise ValueError("S3_BUCKET_NAME or AWS_REGION is missing. Cannot download model.")

    s3_client = boto3.client('s3', region_name=region_name)
    s3_key = "models/resume_ner_merged.zip"
    temp_zip_path = "/tmp/resume_ner_merged.zip"

    logger.info(f"Downloading model '{s3_key}' from S3 bucket '{bucket_name}'...")
    
    try:
        s3_client.download_file(bucket_name, s3_key, temp_zip_path)
        logger.info("Download complete. Extracting files...")
        
        os.makedirs(extract_to_path, exist_ok=True)
        
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
            
        os.remove(temp_zip_path)
        logger.info(f"Model successfully extracted to {extract_to_path}")
        
    except ClientError as e:
        logger.error(f"AWS Boto3 ClientError while downloading model: {e}")
        raise Exception("Failed to download model from S3.")