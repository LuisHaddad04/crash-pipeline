Chicago Crash ETL Pipeline — End-to-End Data Engineering and Machine Learning System

This project is an end-to-end, containerized ETL and machine learning pipeline built with Docker Compose, RabbitMQ, MinIO, DuckDB, Streamlit, Prometheus, and Grafana.
The goal of the system is to ingest Chicago crash data, clean and process it, store reliable analytical tables, train a predictive model, and provide monitoring capabilities across all components.

1. Project Overview and Problem Statement

The Chicago crash dataset is large, inconsistent, and difficult to analyze in its raw form. My objective was to build a complete pipeline capable of:

Ingesting raw crash data from external sources

Automatically transforming and cleaning data into reliable analytical outputs

Storing high-quality Gold tables in DuckDB

Training and serving a machine learning model to predict crash types

Providing a Streamlit UI for exploration, predictions, and model evaluation

Monitoring the health and behavior of all services using Prometheus and Grafana

This pipeline helps convert messy real-world crash data into something actionable, accessible, and production-ready.

2. Data Flow and System Architecture

The system follows a clear, multi-stage ETL process:

Extractor → MinIO (raw) → Transformer → MinIO (silver)
       → Cleaner → DuckDB (gold) → Streamlit ML Application
Monitoring (Prometheus) ← Metrics from Extractor / Cleaner / MinIO / RabbitMQ
Grafana ← Prometheus dashboards


The Extractor pulls raw datasets into MinIO.

The Transformer merges crash, vehicle, and people files into Silver-layer data.

The Cleaner enforces consistent formatting and writes final Gold-layer records into DuckDB.

Streamlit uses the Gold table to train an ML model and generate predictions.

Prometheus gathers metrics from the system, and Grafana visualizes them.

3. Component Descriptions
Extractor (Go)

The Extractor downloads crash data files from external sources and writes them to MinIO in timestamped folders. It exposes basic Prometheus metrics used for monitoring uptime and general activity.

Transformer (Python)

This service consumes messages from RabbitMQ, loads raw files from MinIO, merges them, and outputs cleaned Silver-level CSVs. It standardizes schemas and prepares the data for the cleaning stage.

Cleaner (Python)

The Cleaner enforces the defined cleaning rules (handling nulls, formatting, conversions), performs final transformations, and writes the processed Gold dataset into DuckDB.

Streamlit Application

The Streamlit app provides a user interface for exploring the data, visualizing trends, training the machine learning model, and generating crash-type predictions.
The model uses features such as weather, lighting, traffic control device, and time-based variables.

Docker Compose

Docker Compose orchestrates all services, including MinIO, RabbitMQ, Prometheus, Grafana, and the pipeline components. It ensures all containers run in the correct network and are able to communicate with each other.

Monitoring (Prometheus + Grafana)

Prometheus scrapes metrics from RabbitMQ, MinIO, and system-level processes. Grafana visualizes these metrics through interactive dashboards.
Metrics used in this project include:

process_cpu_seconds_total

rabbitmq_uptime

minio_node_io_read_bytes

minio_node_io_write_bytes

4. Screenshots 
Streamlit:
<img width="858" height="380" alt="Screenshot 2025-12-04 135331" src="https://github.com/user-attachments/assets/eecba97b-8a69-4ff8-ad95-b31ce437f39c" />
<img width="840" height="513" alt="image" src="https://github.com/user-attachments/assets/56f1b853-f1ae-49e9-a40c-d808655c5583" />
<img width="899" height="343" alt="image" src="https://github.com/user-attachments/assets/05322bd2-84fd-4b74-841d-6523e44c904b" />
<img width="881" height="471" alt="image" src="https://github.com/user-attachments/assets/1d4fd23e-62c5-425f-849a-d9acbc31f6fa" />


5. Architecture Diagram
flowchart LR
    A[Extractor] -->|raw data| B[MinIO]
    B --> C[Transformer]
    C -->|silver data| B
    B --> D[Cleaner]
    D -->|gold tables| E[DuckDB]
    E --> F[Streamlit App]

    A --> G[Prometheus]
    B --> G
    D --> G
    G --> H[Grafana Dashboards]

6. How to Run the Pipeline
Clone the repository
git clone https://github.com/your/repo.git
cd repo

Create a .env file
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
RABBITMQ_DEFAULT_USER=guest
RABBITMQ_DEFAULT_PASS=guest

Create required local folders
mkdir -p grafana_data
mkdir -p prom_data
mkdir -p data

Launch the entire system
docker compose up -d --build

Access each service
Service	URL
Streamlit	http://localhost:8501

Grafana	http://localhost:3000

Prometheus	http://localhost:9090

MinIO Console	http://localhost:9001

RabbitMQ UI	http://localhost:15672
7. Additional Features Added

Throughout the project, I added several enhancements, including:

Custom monitoring dashboards for CPU activity, uptime, and MinIO read/write behavior

DuckDB as an analytical store for reliable and performant querying

A Streamlit interface for predictions and visual exploration

Prometheus exporters for RabbitMQ and MinIO

A multi-stage ETL process using message queues for decoupling services

These additions helped create a more realistic and production-styled data pipeline.

8. Lessons Learned and Challenges

Some of the main challenges included:

Managing limited disk space inside the VirtualBox environment and cleaning unused Docker artifacts

Debugging container networking issues, especially around service discovery and port mapping

Exposing Prometheus metrics correctly for each service

Understanding the interaction between MinIO, RabbitMQ, and the ETL components

Structuring a multi-stage ETL system in a way that feels modular and scalable

If I had more time, I would extend the pipeline by adding:

More custom ETL metrics (rows processed, error counts, job duration)

Automated retries and fault tolerance

A comparison of multiple machine learning models

A feature store for tracking engineered features across pipeline runs
