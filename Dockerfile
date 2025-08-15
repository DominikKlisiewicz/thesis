# Debian-based Python 3.12
FROM python:3.12-slim

# Speed up installs and avoid prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# (Light) system deps; enough for common wheels and a few C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libffi-dev libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Default command: open a shell
CMD ["bash"]
