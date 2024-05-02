# Start with a base image containing Miniconda installed
FROM continuumio/miniconda3

# Set the working directory in the Docker container to /workspace/tum-traffic-dataset-dev-kit
WORKDIR /workspace/tum-traffic-dataset-dev-kit

# Install any necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libncurses5-dev \
    libncursesw5-dev \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a new Conda environment with Python 3.8 named tum-traffic-dataset-dev-kit
RUN conda create -n tum-traffic-dataset-dev-kit python=3.8 -y
SHELL ["conda", "run", "-n", "tum-traffic-dataset-dev-kit", "/bin/bash", "-c"]

# Install Pytorch env
RUN conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Copy the application's requirements file and install any additional requirements
COPY requirements_docker.txt .
RUN pip install -r requirements_docker.txt

# Install additional tools
RUN conda install -c conda-forge jupyterlab -y && \
    conda install -c conda-forge fvcore && \
    conda install -c conda-forge iopath && \
    conda install conda-forge::ros-sensor-msgs && \
    conda install pytorch3d -c pytorch3d

# Install specific Python packages via pip
RUN pip install -U pip && \
    pip install -v "git+https://github.com/klintan/pypcd.git"

RUN pip install -v numpy==1.19.5

# Copy the rest of your application's source code from the local file system to the filesystem of the container
COPY . /workspace/tum-traffic-dataset-dev-kit/

# Set an environment variable to ensure Python scripts find the modules
ENV PYTHONPATH "${PYTHONPATH}:/workspace/tum-traffic-dataset-dev-kit"

# Expose the port the app runs on
EXPOSE 8888

# Set the default command to execute when creating a new container
CMD ["bash"]
