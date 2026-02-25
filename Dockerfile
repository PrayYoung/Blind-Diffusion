FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# 1) System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git tar \
    libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev \
    patchelf gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2) Install MuJoCo 2.1.0 (mujoco210)
RUN mkdir -p /root/.mujoco && \
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz && \
    tar -xvzf /tmp/mujoco.tar.gz -C /root/.mujoco && \
    rm /tmp/mujoco.tar.gz

# 3) Env vars (make MuJoCo discoverable)
ENV MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so.1
ENV MUJOCO_GL=osmesa

# 4) Workdir
WORKDIR /workspace

# 5) Install uv
RUN pip install --no-cache-dir uv

# 6) Cache deps layer
COPY pyproject.toml README.md ./
COPY src ./src/

# 7) Create venv + install with robomimic
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install -e ".[robomimic]"

# 8) Copy remaining files
COPY . .

CMD ["/bin/bash"]
