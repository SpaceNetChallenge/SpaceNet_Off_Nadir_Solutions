FROM nvidia/cuda:9.0-runtime-ubuntu16.04

MAINTAINER Selim Seferbekov <selim.sef@gmail.com>

RUN apt-get update && \
    apt-get install -y curl build-essential libpng12-dev libffi-dev \
      	libboost-all-dev \
		libgflags-dev \
		libgoogle-glog-dev \
		libhdf5-serial-dev \
		libleveldb-dev \
		liblmdb-dev \
		libopencv-dev \
		libprotobuf-dev \
		libsnappy-dev \
		protobuf-compiler \
		python3-rtree \
		git \
		 && \
    apt-get clean && \
    rm -rf /var/tmp /tmp /var/lib/apt/lists/*

RUN curl -sSL -o installer.sh https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh && \
    bash /installer.sh -b -f && \
    rm /installer.sh

ENV PATH "$PATH:/root/anaconda3/bin"

RUN conda install pytorch torchvision -c pytorch
RUN conda install tqdm
RUN pip install opencv-python
RUN pip install https://github.com/SpaceNetChallenge/utilities/tarball/spacenetV3
RUN pip install pygeoif
RUN pip install lightgbm
RUN pip install shapely
RUN pip install pretrainedmodels


WORKDIR /work

COPY . /work/


RUN chmod 777 train.sh
RUN chmod 777 test.sh
