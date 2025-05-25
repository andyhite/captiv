# Multi-stage Dockerfile for Captiv - Optimized for RunPod Production Deployment
# Stage 1: Builder - Build Python wheel and prepare dependencies
FROM python:3.12-slim AS builder

# Build arguments
ARG POETRY_VERSION=2.1.3

# Environment variables for build stage
ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  POETRY_VENV_IN_PROJECT=false \
  POETRY_NO_INTERACTION=1 \
  POETRY_CACHE_DIR=/tmp/poetry_cache

# Install build dependencies
RUN apt-get update --yes && \
  apt-get upgrade --yes && \
  apt-get install --yes --no-install-recommends \
  build-essential \
  pkg-config \
  libffi-dev \
  git \
  ca-certificates && \
  pip install --no-cache-dir poetry==${POETRY_VERSION} && \
  apt-get autoremove -y && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set working directory
WORKDIR /build

# Copy dependency files and install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
  poetry install --only=main --no-root --no-cache && \
  rm -rf $POETRY_CACHE_DIR

# Copy source code and build wheel
COPY src/ ./src/
COPY README.md ./
RUN poetry build --format wheel

# Stage 2: Production - Minimal RunPod-optimized image
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Build arguments
ARG PYTHON_VERSION=3.12

# Environment variables for RunPod
ENV DEBIAN_FRONTEND=noninteractive \
  SHELL=/bin/bash \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu \
  PATH="/opt/venv/bin:$PATH"

# Set shell for better error handling
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install only runtime dependencies in a single layer
RUN apt-get update --yes && \
  apt-get upgrade --yes && \
  apt-get install --yes --no-install-recommends \
  # Core RunPod services
  openssh-server \
  nginx \
  # Essential utilities
  curl \
  wget \
  rsync \
  psmisc \
  bc \
  # Media and ML runtime libraries (minimal set)
  ffmpeg \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  libgomp1 \
  # Python runtime
  software-properties-common && \
  # Install Python
  add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update --yes && \
  apt-get install --yes --no-install-recommends \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-venv \
  python${PYTHON_VERSION}-dev && \
  # Create symlinks
  rm -f /usr/bin/python /usr/bin/python3 && \
  ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
  ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
  # Install pip securely
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
  python get-pip.py --no-cache-dir && \
  rm get-pip.py && \
  pip install --upgrade --no-cache-dir pip setuptools wheel && \
  # Cleanup
  apt-get autoremove -y && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
  echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Install ca-certificates and gnupg separately to avoid architecture conflicts
# Use --fix-broken to handle any dependency issues
RUN apt-get update --yes && \
  DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends --fix-broken \
  ca-certificates \
  gnupg && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  # Manually update ca-certificates to avoid illegal instruction errors
  update-ca-certificates || true

# Create optimized virtual environment
RUN python -m venv /opt/venv

# Copy and install pre-built wheel from builder stage
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
  rm -rf /tmp/*.whl

# Install minimal additional dependencies for RunPod API server
RUN pip install --no-cache-dir \
  python-multipart \
  # Updated Jupyter for RunPod compatibility with security fixes
  jupyterlab==4.2.5 \
  notebook==7.2.2 && \
  # Clean pip cache
  pip cache purge

# Copy entrypoint script
COPY entrypoint.sh /start.sh
RUN chmod +x /start.sh

# Copy Docker configuration files to template location
COPY docker/config/ /opt/captiv-config/

# Create template directories for RunPod workspace sync
RUN mkdir -p /opt/workspace-template/{models,outputs,logs,captiv} && \
  # Copy workspace files to template location
  cp /opt/captiv-config/workspace-readme.md /opt/workspace-template/README.md && \
  cp /opt/captiv-config/health-check.json /opt/workspace-template/captiv/health.json && \
  # Setup NGINX configuration
  cp /opt/captiv-config/nginx-default.conf /etc/nginx/sites-available/default && \
  # Setup RunPod welcome message
  cp /opt/captiv-config/runpod-welcome.txt /etc/runpod.txt && \
  echo 'cat /etc/runpod.txt' >> /root/.bashrc && \
  # Remove SSH host keys (RunPod security pattern)
  rm -f /etc/ssh/ssh_host_* && \
  # Create non-root user for better security
  useradd -m -s /bin/bash -u 1000 captiv && \
  chown -R captiv:captiv /opt/workspace-template && \
  # Set secure permissions
  chmod 755 /opt/workspace-template && \
  # Remove unnecessary setuid/setgid binaries for security
  find /usr/bin /usr/sbin -perm /6000 -type f -exec chmod a-s {} \; || true

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Labels for RunPod
LABEL maintainer="Captiv Team" \
  description="Captiv Image Captioning Application - Production RunPod Image" \
  version="1.0" \
  runpod.template="captiv" \
  org.opencontainers.image.source="https://github.com/andyhite/captiv" \
  org.opencontainers.image.description="Production-optimized Captiv for RunPod GPU environments"

# Expose ports
EXPOSE 7860 8888 22

# Set default command to start script
CMD ["/start.sh"]
