
"""
consumer.py — RabbitMQ worker for the Cleaner

Listens on queue "clean", downloads the Silver CSV (merged.csv) from MinIO for the given corr_id,
runs the cleaner, writes to DuckDB, and logs idempotent metrics.

Env vars expected:
  RABBITMQ_URL              amqp://guest:guest@localhost:5672/%2F
  QUEUE_NAME                clean
  MINIO_ENDPOINT            localhost:9000
  MINIO_ACCESS_KEY          minioadmin
  MINIO_SECRET_KEY          minioadmin
  MINIO_SECURE              false
  GOLD_DB_PATH              ./gold.duckdb    (default if not passed in message)

Message JSON (example):
{
  "type": "clean",
  "corr_id": "2025-09-29T16-12-01Z",
  "xform_bucket": "transform-data",
  "prefix": "crash",
  "object_key": "merged/corr=2025-09-29T16-12-01Z/merged.csv",  # optional; otherwise inferred
  "gold_db_path": "C:/Users/luisg/OneDrive/Desktop/gold.duckdb"
}
"""
from __future__ import annotations
import json, os, time, traceback, tempfile
import duckdb
import pika

from minio_io import download_from_minio
from cleaner import run_clean

def env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1","true","t","yes","y"}

def count_rows(db_path: str) -> int:
    try:
        conn = duckdb.connect(db_path)
        cnt = conn.execute("SELECT COUNT(*) FROM crash_gold.crashes").fetchone()[0]
        conn.close()
        return int(cnt)
    except Exception:
        return -1

def main():
    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F")
    queue_name   = os.getenv("QUEUE_NAME", "clean")

    params = pika.URLParameters(rabbitmq_url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True)

    print(f"[CONSUMER] Connected. Waiting on queue='{queue_name}' ...")

    def handle_message(ch, method, properties, body):
        t0 = time.perf_counter()
        try:
            msg = json.loads(body.decode("utf-8"))
        except Exception:
            print("[ERROR] Invalid JSON message")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

        msg_type = msg.get("type")
        if msg_type != "clean":
            print(f"[SKIP] Unsupported message type: {msg_type}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        corr_id       = msg.get("corr_id")
        xform_bucket  = msg.get("xform_bucket", "transform-data")
        prefix        = msg.get("prefix", "crash")
        object_key    = msg.get("object_key")  # optional
        gold_db_path  = msg.get("gold_db_path", os.getenv("GOLD_DB_PATH", "./gold.duckdb"))

        if not object_key:
            # default convention
            object_key = f"merged/corr={corr_id}/merged.csv"

        # MinIO config
        endpoint   = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        secure     = env_bool("MINIO_SECURE", False)

        # Download to temp path
        tmpdir = tempfile.mkdtemp(prefix="cleaner_")
        local_csv = os.path.join(tmpdir, "merged.csv")
        print(f"[CONSUMER] corr_id={corr_id} downloading s3://{xform_bucket}/{object_key} → {local_csv}")
        try:
            download_from_minio(endpoint, access_key, secret_key, xform_bucket, object_key, local_csv, secure=secure)
        except Exception as e:
            print(f"[ERROR] MinIO download failed: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

        before = count_rows(gold_db_path)
        print(f"[CONSUMER] pre-count rows in crash_gold.crashes: {before}")

        try:
            run_clean(local_csv, gold_db_path)
        except Exception as e:
            print("[ERROR] Cleaner failed:\n", traceback.format_exc())
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

        after = count_rows(gold_db_path)
        dt = time.perf_counter() - t0
        print(f"[CONSUMER] DONE corr_id={corr_id} before={before} after={after} delta={after-before} in {dt:0.2f}s")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue_name, on_message_callback=handle_message)

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    finally:
        connection.close()

if __name__ == "__main__":
    main()
