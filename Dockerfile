FROM python:3.10-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

# 1) System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git tar \
    libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    patchelf gcc build-essential cmake\
    && rm -rf /var/lib/apt/lists/*

# 2) Workdir
WORKDIR /workspace

# 3) Install uv
RUN pip install --no-cache-dir uv

# 4) Cache deps layer
COPY pyproject.toml README.md ./
COPY src ./src/

# 5) Create venv + install with robomimic
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install -e ".[robomimic]"

# 6) Copy remaining files
COPY . .

# 7) env for robosuite
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

CMD ["/bin/bash"]
