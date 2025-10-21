# 1. 基础镜像：匹配PyTorch 2.8.0 + CUDA 12.9，选择runtime版本减小体积
# 若需编译操作（如自定义算子），可替换为"pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel"
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# 2. 维护者与描述（可选，便于团队协作）
LABEL maintainer="Haotian Zhang"
LABEL description="Docker env for MouldCTSegNet"

# 3. 基础配置：工作目录、时区（避免日志时间错乱）
WORKDIR /workspace
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 4. 安装系统级依赖（支撑Python库运行，如OpenCV、MedPy等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpng-dev \
    libjpeg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. 复制本地requirements.txt到容器内（确保文件路径与Dockerfile同级）
COPY requirements.txt /workspace/requirements.txt

# 6. 安装Python依赖（使用国内源加速，避免版本冲突）
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r /workspace/requirements.txt \
    && rm -rf /root/.cache/pip  # 彻底删除pip缓存

# 7. 配置CUDA环境变量（确保依赖能正确识别CUDA路径）
ENV CUDA_HOME=/usr/local/cuda-12.9
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 8. 暴露常用端口（根据需求调整，如TensorBoard、Jupyter）
EXPOSE 6006
EXPOSE 8888

# 9. 启动命令：默认进入bash终端，可按需修改（如启动Jupyter）
CMD ["bash"]