# syntax=docker/dockerfile:1.6
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies
# apt with cache (faster rebuilds)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    wget curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
# Install torch separately so it caches
RUN pip install --upgrade pip wheel setuptools

# cache pip wheels for requirements
# COPY requirements.txt /tmp/requirements.txt
COPY requirements.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Application setup
COPY . /app/app

# Environment
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
