FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-dev python3-pip \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir uv

WORKDIR /workspace
COPY requirements.txt .
RUN uv pip install --no-cache-dir setuptools wheel

# 1) 先装 torch（按你的 CUDA 选 index，这里以 cu117 + torch1.13 为例）
RUN uv pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu117 \
        torch==1.13.1 torchvision==0.14.1

# 2) 单独装 mmcv-full 的预编译 wheel，find-links 指向匹配 torch/CUDA 的索引
#    mmcv-full 1.7.1 最高支持到 torch1.13
RUN uv pip install --no-cache-dir --no-build-isolation mmcv-full==1.7.1 \
        -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

# 1) 装系统级依赖：C 编译器 + GDAL 开发库
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgdal-dev \
        gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# 2) 让 gdal 的 setup.py 能找到头文件
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV CUDA_HOME=/usr/local/cuda

# 3) 装其余依赖；建议把 mmcv-full 从 requirements.txt 里删掉，避免 uv 又去编源码
RUN uv pip install psutil --no-build-isolation
RUN uv pip install flash-attn --no-build-isolation
COPY override.txt .
RUN uv pip install --override override.txt atorch==0.1.3
RUN uv pip install mmcls==0.25.0
RUN uv pip install mmsegmentation==0.30.0
RUN uv pip install timm==0.6.13
RUN uv pip install gdal==3.4.0
RUN uv pip install scikit-image==0.19.3

CMD ["bash"]
