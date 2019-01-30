FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y \
    build-essential \
    bzip2 \
    cmake \
    curl \
    git \
    g++ \
    libboost-all-dev \
    pkg-config \
    rsync \
    software-properties-common \
    sudo \
    tar \
    timidity \
    unzip \
    wget \
    locales \
    zlib1g-dev \
    python3-dev \
    python3 \
    python3-pip \
    python3-tk \
    libjpeg-dev \
    libpng-dev

# Python3
RUN pip3 install pip --upgrade
RUN pip3 install cython  \
  numpy \
  matplotlib 
RUN pip3 install git+https://github.com/crowdai/coco.git#subdirectory=PythonAPI
RUN pip3 install tensorflow-gpu
RUN pip3 install scikit-image
RUN pip3 install keras==2.1.6
RUN pip3 install opencv-python
RUN pip3 install imgaug
RUN pip3 install shapely
RUN pip3 install tifffile
RUN pip3 install tqdm
RUN pip3 install pandas
RUN pip3 install pycocotools
# Unicode support:
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8





COPY src /root/src
RUN chmod a+x /root/src/train.sh && \ 
    chmod a+x /root/src/test.sh
    
ENV PATH $PATH:/root/

#ENV
ENV WORKDIR /root/


COPY spacenet_models.zip /root/
COPY crowdai_data.zip /root/
