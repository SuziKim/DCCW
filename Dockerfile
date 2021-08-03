# For more information, please refer to https://aka.ms/vscode-docker-python
# FROM python:3.8-slim-buster
FROM ubuntu:18.04
# FROM python:3.6

ENV HOME /root

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# RUN apt-get update \
# && apt install -y software-properties-common \
# && add-apt-repository ppa:deadsnakes/ppa \
# && apt-get install --no-install-recommends -y \
# python3.9 python3-pip

RUN apt-get update \ 
    && apt-get install --no-install-recommends -y \
    python3.7 python3-pip python3-dev \
    libpython3.7-dev
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install setuptools

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \ 
    git \
    ssh \
    gcc \ 
    wget \
    g++ \
    libsuitesparse-dev \
    python-cvxopt \ 
    && rm -rf /var/lib/apt/lists/

RUN apt-get update -y && apt-get install -y python-dev    
RUN apt-get update -y && apt-get install -y libopenblas-dev
RUN apt-get update -y && apt-get install -y libatlas-base-dev
RUN apt-get update -y && apt-get install -y libblas-dev
RUN apt-get update -y && apt-get install -y liblapack-dev 
RUN apt-get update -y && apt-get install -y glpk-utils
RUN apt-get update -y && apt-get install -y libglpk-dev

# RUN wget http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-4.5.3.tar.gz
# RUN tar -xf SuiteSparse-4.5.3.tar.gz
# RUN export CVXOPT_SUITESPARSE_SRC_DIR=$(pwd)/SuiteSparse

# Anaconda installing
# RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh \
#     && bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniconda \
#     && rm -f Miniforge3-Linux-aarch64.sh

# RUN export PATH=$HOME/miniconda/bin:$PATH
# RUN conda install -y -c conda-forge cvxopt

# RUN git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
# RUN pushd SuiteSparse; \
#     git checkout v5.6.0; \
#     popd; \
#     export CVXOPT_SUITESPARSE_SRC_DIR=$(pwd)/SuiteSparse; 

# RUN export CPPFLAGS="-I/usr/include/suitesparse"

# RUN git config --global core.sshCommand 'ssh -o UserKnownHostsFile=/dev/null -o StricHostKeyChecking=no'

RUN git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git $HOME/SuiteSparse
ENV CVXOPT_SUITESPARSE_SRC_DIR $HOME/SuiteSparse
ENV CVXOPT_SUITESPARSE_INC_DIR $HOME/SuiteSparse/UMFPACK
ENV CVXOPT_BUILD_GLPK 1

# Install pip requirements
COPY requirements.txt .

RUN python3 -m pip install cmake
RUN python3 -m pip install -r requirements.txt

RUN git clone https://github.com/fikisipi/elkai.git
RUN python3 -m pip install -e ./elkai

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# EXPOSE 8000
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "DCCWWebDemo.wsgi"]
