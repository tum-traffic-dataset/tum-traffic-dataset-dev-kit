# Use the official NVIDIA PyTorch image which includes CUDA and cuDNN.
FROM nvcr.io/nvidia/pytorch:22.04-py3

# Setting up the working directory
WORKDIR /workspace/tum-traffic-dataset-dev-kit

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install specific Python packages via pip
RUN pip install -U pip && \
    pip install fvcore iopath && \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" && \
    pip install "git+https://github.com/klintan/pypcd.git"

# Copy the Python requirements file and install Python dependencies
COPY requirements.txt /workspace/tum-traffic-dataset-dev-kit/
RUN pip install -r requirements.txt

# Copy the rest of the project files into the working directory
COPY . /workspace/tum-traffic-dataset-dev-kit/

# Set the PYTHONPATH environment variable
ENV PYTHONPATH="${PYTHONPATH}:/workspace/tum-traffic-dataset-dev-kit/"

# Expose any ports the app might need
EXPOSE 8888

# Run the Bash Shell when the container starts
CMD ["bash"]
