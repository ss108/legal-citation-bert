FROM huggingface/transformers-pytorch-gpu:latest as pytorch_stage 

FROM python:3.11-slim as builder

WORKDIR /project


COPY docker-entrypoint.sh /project/docker-entrypoint.sh

ENV PDM_HOME=/root/.pdm

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

RUN pip install notebook
RUN pip install jupyterlab

RUN jupyter notebook --generate-config

ENV JUPYTER_ENABLE_LAB=yes

COPY ./setup/00_setup.py /root/.ipython/profile_default/startup/00_setup.py

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--port=8888", "--no-browser"]
EXPOSE 8888

ENTRYPOINT ["/bin/sh", "docker-entrypoint.sh"]