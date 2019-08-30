ARG IMAGE_CUDA_VERSION
ARG BASE_CONTAINER=nvidia/cuda:${IMAGE_CUDA_VERSION}-cudnn7-runtime-ubuntu16.04
FROM $BASE_CONTAINER
LABEL maintainer="Paridhi Singh <psingh@wavecomp.com>"

ARG IMAGE_CUDA_VERSION
RUN echo ${IMAGE_CUDA_VERSION}

ENV CONDA_DIR=/opt/conda

RUN apt-get update && apt-get -yq dist-upgrade \
 && apt-get install -yq --no-install-recommends \
    wget \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-testresources \

    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    git \
    unzip \
    curl \
    software-properties-common \
 && rm -rf /var/lib/apt/lists/*

# Install conda and check the md5 sum provided on the download site
ENV MINICONDA_VERSION=4.5.12 \
    CONDA_VERSION=4.6.7

RUN cd /tmp && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "866ae9dff53ad0874e1d1a60b1ad1ef8 *Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
    $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
    $CONDA_DIR/bin/conda config --system --set show_channel_urls true && \
    $CONDA_DIR/bin/conda install --quiet --yes conda="${CONDA_VERSION%.*}.*" && \
    $CONDA_DIR/bin/conda update --all --quiet --yes && \
    $CONDA_DIR/bin/conda list python | grep '^python ' | tr -s ' ' | cut -d '.' -f 1,2 | sed 's/$/.*/' >> $CONDA_DIR/conda-meta/pinned && \
    $CONDA_DIR/bin/conda clean --all -f -y

# add conda to path
ENV PATH /opt/conda/bin:$PATH

# neovim
RUN add-apt-repository ppa:neovim-ppa/stable && apt-get update
RUN apt-get update && apt-get install -yq --no-install-recommends \
    neovim \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -yq --no-install-recommends \
    ffmpeg \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy

# opencv
RUN conda install -c conda-forge opencv  && \
    conda clean --all -f -y

# python packages
RUN conda install --yes \
    matplotlib \
    pandas \
    dill \
    pillow \
    lmdb && \
    conda clean --all -f -y

# pytorch
RUN conda install --yes \
    pytorch \
    torchvision \
    cudatoolkit=${IMAGE_CUDA_VERSION} -c pytorch && \
    conda clean --all -f -y

#Tensorflow
#ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

#ARG PYTHON=python3
#ARG PIP=pip3
#RUN ln -s $(which ${PYTHON}) /usr/local/bin/python
#RUN pip install --ignore-installed --upgrade tensorflow-gpu==2.0.0-beta1

conda Tensorflow
RUN conda create --name tf_gpu
RUN activate tf_gpu
RUN conda install tensorflow-estimator==1.14.0
RUN conda install tensorflow-gpu==2.0.0b1


#RUN pip3 install jupyter
RUN conda install jupyter
RUN mkdir /notebooks
CMD jupyter notebook --no-browser --allow-root --ip 0.0.0.0 --port 8088 --notebook-dir=/notebooks


EXPOSE 8088
EXPOSE 1234

