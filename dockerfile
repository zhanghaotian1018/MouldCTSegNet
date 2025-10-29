# 1. Base image: Match PyTorch 2.8.0 + CUDA 12.9, choose runtime version to reduce size
# For compilation tasks (e.g., custom operators), replace with "pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel"
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# 2. Maintainer and description (optional, for team collaboration)
LABEL maintainer="Haotian Zhang"
LABEL description="Docker env for MouldCTSegNet"

# 3. Basic configuration: Working directory and time zone (to avoid log time confusion)
WORKDIR /workspace
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 4. Install system-level dependencies (support Python libraries such as OpenCV, MedPy, etc.)
# Update package list first (with fault tolerance)
# Correct format: All packages are listed in apt-get install parameters, separated by spaces
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    # Basic tool packages
    git \
    wget \
    unzip \
    build-essential \
    # Basic image processing libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpng-dev \
    libjpeg-dev \
    # X11 graphics libraries required by vtk (newly added, correctly placed in the install list)
    libxrender1 \
    libxt6 \
    libxext6 \
    # Optional: Supplement other X11 libraries that vtk may depend on (if missing in the future)
    libx11-6 \
    libegl1-mesa \
    libegl1-mesa-dev \
    && \  
    # After installation, execute cleanup commands
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 5. Copy local requirements.txt into the container (ensure the file path is at the same level as Dockerfile)
COPY requirements.txt /workspace/requirements.txt

# 6. Install Python dependencies (use domestic mirror for acceleration, avoid version conflicts)
RUN pip install --no-cache-dir \
    -r /workspace/requirements.txt \
    && rm -rf /root/.cache/pip  

# 7. Configure CUDA environment variables (ensure dependencies can correctly identify CUDA path)
ENV CUDA_HOME=/usr/local/cuda-12.9
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 8. Expose common ports (adjust as needed, e.g., TensorBoard, Jupyter)
EXPOSE 6006
EXPOSE 8888

# 9. Startup command: Enter bash terminal by default, modify as needed (e.g., start Jupyter)
CMD ["bash"]