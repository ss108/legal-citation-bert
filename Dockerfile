FROM huggingface/transformers-pytorch-gpu:latest as pytorch_stage 
FROM python:3.11-slim as builder
COPY --from=ghcr.io/astral-sh/uv:0.3.3 /uv /bin/uv

WORKDIR /project

COPY . . 

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# (Optional) If you need to install specific Python packages or download resources, uncomment and modify as needed
# RUN pip install spacy 
# RUN python -m spacy download en_core_web_sm
# # Spacy has issues being installed via PDM

RUN uv sync --dev

ENTRYPOINT ["/bin/sh", "docker-entrypoint.sh"]
