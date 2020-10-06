FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
LABEL maintainer="oeslle.lucena@gmail.com"

# Set up environment
RUN DEBIAN_FRONTEND="noninteractive" apt-get update \ 
    && apt-get install -y \ 
    dialog apt-utils \
    curl \
    htop \
    nano \
    tmux \
    git \
    python3 \
    python3-pip \
    protobuf-compiler

# Install TaPAS 
RUN cd /home \
    && git clone https://github.com/google-research/tapas.git \
    && cd tapas \
    && pip3 install --upgrade pip \
    && pip install --upgrade setuptools \
    && pip install --upgrade six \
    && pip install scipy==1.4.1 \
    && pip install oauth2client==4.1.2 \
    && pip install -e . \
    && pip install tox \
    && tox	    

