# Use the official Ubuntu base image
FROM nvidia/cuda:latest

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --extra-index-url https://download.nvidia.com/cpp_cuda/repos/ubuntu20.04/x86_64  xgboost  # Install XGBoost with GPU support

# Install H2O
RUN apt-get update && apt-get install -y default-jre
RUN pip install h2o

# Install Python packages
RUN pip3 install --upgrade pip \
    && pip3 install \
    scikit-learn \
    jupyter \
    pandas \
    numpy \
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
CMD ["python3"]