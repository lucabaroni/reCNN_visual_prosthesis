# Use the base image with specified version details
FROM sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7

ARG GITHUB_USER
ARG GITHUB_TOKEN

# Set the working directory
WORKDIR /src

# Clone nnvision repository
RUN git config --global credential.helper store && \
    echo "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com" >> ~/.git-credentials && \
    git clone -b model_builder https://github.com/lucabaroni/nnvision.git

# Install nnvision
RUN python3.9 -m pip install -e /src/nnvision

# Uninstall neuralpredictors if already installed
RUN pip uninstall -y neuralpredictors

# Clone neuralpredictors repository
RUN git clone -b recnn https://github.com/lucabaroni/neuralpredictors.git

# Install neuralpredictors
RUN python3.9 -m pip install -e /src/neuralpredictors
# Append neuralpredictors directory to Python path
ENV PYTHONPATH=/src/neuralpredictors:${PYTHONPATH}

# Uncomment if these packages are needed
# RUN python3.9 -m pip install --upgrade pip && \
#     python3.9 -m pip --no-cache-dir install \
#         wandb \
#         moviepy \
#         imageio \
#         configparser==5.0.2 \
#         pillow==9.2.0 \
#         markdown==3.3.4 \
#         numpy==1.21.1 \
#         opencv-python==4.5.4.58 \
#         pathtools==0.1.2 \
#         pytorch-lightning==1.5.2 \
#         pyyaml==5.4.1 \
#         requests==2.26.0 \
#         requests-oauthlib==1.3.0 \
#         scipy==1.7.0 \
#         termcolor==1.1.0 \
#         tqdm==4.61.2 \
#         wget==3.2 \
#         npm \
#         configurable-http-proxy \
#         jupyterhub \
#         batchspawner \
#         plotly  \
#         statsmodels && \
#     python3.9 -m pip install param==1.5.1

# Install protobuf and wandb
RUN pip install protobuf==3.19 wandb

# Add the project directory and install its dependencies
ADD . /project
WORKDIR /project
RUN python3.9 -m pip install -e .
