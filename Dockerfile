FROM huggingface/transformers-pytorch-gpu:latest as pytorch_stage 

FROM python:3.11-slim as builder

WORKDIR /project

# COPY --from=pytorch_stage /opt/conda /opt/conda

COPY docker-entrypoint.sh /project/docker-entrypoint.sh

ENV PDM_HOME=/root/.pdm
# ENV PYTHONPATH "${PYTHONPATH}:/project:/opt/project"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools wheel
RUN pip install pdm
RUN pdm install --global --project .
RUN pdm add torch --global 

RUN pip install spacy 
RUN python -m spacy download en_core_web_sm
# Spacy has issues being installed via PDM

ENTRYPOINT ["/bin/sh", "docker-entrypoint.sh"]