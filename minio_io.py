"""
MinIO helper (optional). If you don't use MinIO, skip this and point cleaner.py to a local CSV path.
"""
from __future__ import annotations
import os
from minio import Minio

def download_from_minio(
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    object_name: str,
    dest_path: str,
    secure: bool = True,
):
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    client.fget_object(bucket, object_name, dest_path)
    return dest_path
