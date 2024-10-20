FROM huggingface/transformers-pytorch-gpu:latest as pytorch_stage 
FROM python:3.11-slim as builder
COPY --from=ghcr.io/astral-sh/uv:0.3.3 /uv /bin/uv

WORKDIR /project


RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && echo "deb http://apt.llvm.org/bullseye/ llvm-10 main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    llvm-10 \
    llvm-10-dev \
    libffi-dev \
    build-essential \
    # Create a symbolic link for llvm-config to point to llvm-config-10
    && ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config \
    # Clean up APT lists to reduce image size
    && rm -rf /var/lib/apt/lists/*

COPY . . 

ENV LLVM_CONFIG=/usr/bin/llvm-config-10

# (Optional) If you need to install specific Python packages or download resources, uncomment and modify as needed
# RUN pip install spacy 
# RUN python -m spacy download en_core_web_sm
# # Spacy has issues being installed via PDM

RUN uv sync --dev

ENTRYPOINT ["/bin/sh", "docker-entrypoint.sh"]
