import time
import pika
import json
import random
from prometheus_client import start_http_server, Summary, Counter

# Prometheus metrics
EXTRACT_DURATION = Summary("extractor_duration_seconds", "Time spent extracting")
EXTRACT_COUNT = Counter("extractor_jobs_total", "Number of extract jobs run")

RABBIT_URL = "amqp://guest:guest@rabbitmq:5672/%2F"
QUEUE_NAME = "transform"

def publish_message(channel):
    """Simulate an extractor sending a job."""
    payload = {
        "corr_id": f"corr-{random.randint(1000, 9999)}",
        "path": "sample/path/to/data.csv"
    }

    channel.basic_publish(
        exchange="",
        routing_key=QUEUE_NAME,
        body=json.dumps(payload),
    )

@EXTRACT_DURATION.time()
def extract_job(channel):
    time.sleep(random.uniform(0.5, 2.0))  # simulate work
    publish_message(channel)
    EXTRACT_COUNT.inc()

def main():
    print("[EXTRACTOR] Starting metrics server on :8000")
    start_http_server(8000)

    params = pika.URLParameters(RABBIT_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    print("[EXTRACTOR] Running extract loop")
    while True:
        extract_job(channel)
        time.sleep(5)

if __name__ == "__main__":
    main()
