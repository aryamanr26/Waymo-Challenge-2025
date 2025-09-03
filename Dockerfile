# 1) Base image with CUDA & cuDNN for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 2) System deps: wget, git, curl, Python prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential bzip2 wget git curl ca-certificates sudo \
      libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3) Install Miniconda (gets us conda + Python 3.11)
ENV MINICONDA_INSTALLER=Miniconda3-latest-Linux-x86_64.sh \
    CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER} -O /tmp/conda.sh \
 && bash /tmp/conda.sh -b -p ${CONDA_DIR} \
 && rm /tmp/conda.sh
ENV PATH=${CONDA_DIR}/bin:$PATH

# 4) Create your `waymo` env with Python 3.11 and CUDA runtime
RUN conda create -y -n waymo \
      python=3.11 \
      cudatoolkit=11.8 -c conda-forge \
    && conda install -y -n waymo -c nvidia/label/cudnn-8.6 cudnn \
    && conda clean -afy

# 5) Install pip packages inside that env
SHELL ["conda", "run", "-n", "waymo", "/bin/bash", "-lc"]
RUN pip install --upgrade pip \
 && pip install --no-cache-dir \
      transformers \
      tensorflow==2.13.0 \
      waymo-open-dataset-tf-2-12-0==1.6.7 \
      opencv-python \
      accelerate

RUN pip install --pre --no-cache-dir \
      torch --index-url https://download.pytorch.org/whl/nightly/cu118 \
      torchvision --index-url https://download.pytorch.org/whl/nightly/cu118 \
      torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

# 6) Install the Google Cloud SDK from Google’s APT repository
USER root
RUN apt-get update && apt-get install -y --no-install-recommends apt-transport-https ca-certificates gnupg \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      | tee /etc/apt/sources.list.d/google-cloud-sdk.list \
 && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
 && apt-get update && apt-get install -y google-cloud-sdk \
 && rm -rf /var/lib/apt/lists/*

# Back to the waymo env as default
SHELL ["conda", "run", "--no-capture-output", "-n", "waymo", "/bin/bash", "-lc"]
RUN conda run -n waymo pip install bitsandbytes>=0.40.0
RUN conda run -n waymo pip install hf_xet
# 7) Copy your code in and set working dir
WORKDIR /app
COPY . /app

# 8) Expose TensorBoard port if you need it
EXPOSE 6006

# 9) Default entrypoint uses your `waymo` env
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "waymo", "python", "train2.py"]


