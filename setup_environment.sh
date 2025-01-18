#!/bin/bash

# Exit script on any error
set -e

# Define environment name
ENV_NAME="recnn_env"

echo "Creating Conda environment: $ENV_NAME"
# Create and activate Conda environment
conda create -n $ENV_NAME python=3.9 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing PyTorch and dependencies..."
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install wandb

echo "Installing classical_exps from GitHub..."
pip install -e git+https://github.com/lucabaroni/classical_exps.git@package-conversion#egg=classical_exps

echo "Installing nnvision from GitHub..."
pip install -e git+https://github.com/lucabaroni/nnvision.git@model_builder#egg=nnvision

echo "Installing neuralpredictors from GitHub..."
# Uninstall existing neuralpredictors and reinstall from GitHub
pip uninstall -y neuralpredictors || true
pip install -e git+https://github.com/lucabaroni/neuralpredictors.git@recnn#egg=neuralpredictors

pip install -e git+https://github.com/lucabaroni/featurevis_mod#egg=featurevis

echo "Installing protobuf and seaborn..." 
pip install protobuf seaborn

echo "Installing project dependencies..."
# Install project dependencies (current directory assumed to have the project)
pip install -e .

echo "Setup completed successfully! Activate the environment with:"
echo "conda activate $ENV_NAME"