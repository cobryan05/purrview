# ARGs must come before first FROM statement to be used in a FROM statement and re-declared after a FROM
ARG PYTORCH_VERSION=2.7.1
ARG CUDA_VERSION=12.8

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn9-runtime AS base

RUN apt-get update \
    && apt-get install -y git libgl1 libglib2.0-0 \
     && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir onnx onnx-simplifier onnx-graphsurgeon onnxruntime-tools onnxruntime pandas opencv-python matplotlib seaborn

WORKDIR /app
COPY convert.py /app/convert.py

ENTRYPOINT ["python", "/app/convert.py"]