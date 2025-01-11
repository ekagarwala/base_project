# Use the official Ubuntu base image
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    wget \
    curl \
    git \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN mkdir /app/ && \
    python3.12 -m venv /app/.venv && \
    . /app/.venv/bin/activate && \
    pip install --upgrade pip \
    && pip install \
    scikit-learn \
    jupyter \
    pandas \
    numpy \
    h2o \
    xgboost \
    matplotlib \
    seaborn \
    black \
    mypy \
    isort \
    pytest

# Expose ports 8888 and 54321
EXPOSE 8888
EXPOSE 54321

# Set the default command to python3
CMD ["/app/.venv/bin/python3"]