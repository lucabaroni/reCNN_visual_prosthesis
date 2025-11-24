#!/bin/bash

set -e

# Define environment name
ENV_NAME="orientation_aware_protocol_env"

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment $ENV_NAME already exists. Removing it..."
    conda deactivate 2>/dev/null || true
    conda env remove -n $ENV_NAME -y
fi

echo "Creating Conda environment: $ENV_NAME"
# Create and activate Conda environment
conda create -n $ENV_NAME python=3.9.21 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing PyTorch and dependencies..."
conda install \
  pytorch==1.13.1 \
  torchvision==0.14.1 \
  torchaudio==0.13.1 \
  pytorch-cuda=11.7 \
  mkl=2023.1.0 \
  -c pytorch -c nvidia

# Create a dedicated directory for editable installs
mkdir -p ~/github_packages
cd ~/github_packages

echo "Installing classical_exps from GitHub..."
pip install -e git+https://github.com/lucabaroni/classical_exps.git@package-conversion#egg=classical_exps

echo "Installing nnvision from GitHub..."
pip install -e git+https://github.com/lucabaroni/nnvision.git@model_builder#egg=nnvision

echo "Installing neuralpredictors from GitHub..."
pip uninstall -y neuralpredictors || true
pip install -e git+https://github.com/lucabaroni/neuralpredictors.git@recnn#egg=neuralpredictors

pip install -e git+https://github.com/lucabaroni/featurevis_mod#egg=featurevis

# Return to the original directory
cd -

echo "Installing project dependencies..."
pip install protobuf seaborn wandb ipykernel 

# Install project dependencies (current directory assumed to have the project)
pip install -e .