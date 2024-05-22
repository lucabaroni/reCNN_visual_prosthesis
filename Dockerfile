# Use the base image with specified version details
FROM sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7

WORKDIR /src

RUN git config --global credential.helper store && \
    echo https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com >> ~/.git-credentials && \
    git clone -b model_builder https://github.com/lucabaroni/nnvision.git
    git clone -b recnn https://github.com/lucabaroni/neuralpredictors.git

RUN python3.9 -m pip install -e /src/nnvision 
RUN pip uninstall neuralpredictors
RUN python3.9 -m pip install -e /src/neuralpredictors 

# Upgrade pip and install the required Python packages
RUN python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip --no-cache-dir install \
        wandb \
        moviepy \
        imageio \
        configparser==5.0.2 \
        pillow==9.2.0 \
        markdown==3.3.4 \

        numpy==1.21.1 \
        opencv-python==4.5.4.58 \
        pathtools==0.1.2 \
        pytorch-lightning==1.5.2 \
        pyyaml==5.4.1 \
        requests==2.26.0 \
        requests-oauthlib==1.3.0 \
        scipy==1.7.0 \
        # tensorboard==2.4.1 \
        # tensorboard-plugin-wit==1.8.0 \
        # tensorboardx==2.4 \
        termcolor==1.1.0 \
        tqdm==4.61.2 \
        wget==3.2 \
        npm \
        configurable-http-proxy \
        jupyterhub \
        batchspawner \
        plotly  \
        statsmodels  && \
    python3.9 -m pip install param==1.5.1

RUN pip install protobuf==3.19 wandb
RUN pip install -U torch torchaudio --no-cache-dir

# Add the project directory and install its dependencies
ADD . /project
RUN python3.9 -m pip install -e /project
