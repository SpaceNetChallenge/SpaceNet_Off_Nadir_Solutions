# GPU-based system
FROM nvidia/cuda:9.0-devel-ubuntu16.04
MAINTAINER Kohei <i@ho.lc>

ENV CUDNN_VERSION 7.3.0.29
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get -y update &&\
    apt-get --no-install-recommends -y install \
        apt-utils \
        libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
        libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 \
        &&\
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get -y update &&\
    apt-get --no-install-recommends -y install \
        bc \
        bzip2 \
        git \
        wget \
        unzip \
        libblas-dev \
        liblapack-dev \
        libpng-dev \
        libfreetype6-dev \
        pkg-config \
        ca-certificates \
        libhdf5-serial-dev \
        curl \
        &&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/*

RUN ANACONDA_URL="https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh" &&\
    ANACONDA_FILE="anaconda.sh" &&\
    mkdir -p /opt &&\
    cd /opt &&\
    wget -q --no-check-certificate $ANACONDA_URL -O $ANACONDA_FILE &&\
    echo "1046228398cf2e31e4a3b0bf0b9c9fd6282a1d7c  ${ANACONDA_FILE}" | sha1sum -c - &&\
    bash $ANACONDA_FILE -b -p /opt/conda &&\
    rm $ANACONDA_FILE
ENV PATH "/opt/conda/bin:$PATH"

RUN conda update conda
RUN conda create -n sp4 python=3.6.7=h0371630_0
ENV PATH "/opt/conda/envs/sp4/bin:$PATH"

RUN apt-get -y update &&\
    apt-get --no-install-recommends -y install \
        libxrender-dev \
        libxext-dev \
        libsm6 \
        &&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/*

RUN conda install -n sp4 -c pytorch \
		pytorch=0.4.1=py36_py35_py27__9.0.176_7.1.2_2 \
		torchvision=0.2.1=py36_1 \
        click=7.0=py36_0 \
        tqdm=4.28.1=py36h28b3542_0 \
        rasterio=0.36.0=py36h3f37509_2 \
		libopencv=3.4.2=hb342d67_1 \
        opencv=3.4.2=py36h6fd60c2_1 \
        py-opencv=3.4.2=py36hb342d67_1 \
        shapely=1.6.4=py36h7ef4460_0 \
        geopandas==0.4.0 \
        scikit-image=0.14.0=py36hf484d3e_1 \
        attrs=18.2.0=py36h28b3542_0 \
        scikit-learn=0.20.1=py36hd81dba3_0

RUN pip install \
    albumentations==0.1.2 \
    imgaug==0.2.6 \
    https://github.com/SpaceNetChallenge/utilities/archive/spacenetV3.zip

RUN mkdir -p /root/.torch/models &&\
    ln -s /root/working/models/vgg16-397923af.pth /root/.torch/models/vgg16-397923af.pth

COPY main.py /root/
COPY *.sh /root/

# Models
COPY working/cv.txt /root/working/cv.txt
COPY working/models/vgg16-397923af.pth /root/working/models/vgg16-397923af.pth
COPY working/models/v12_f0/v12_f0_best /root/working/models/v12_f0/v12_f0_best
COPY working/models/v12_f1/v12_f1_best /root/working/models/v12_f1/v12_f1_best
COPY working/models/v12_f2/v12_f2_best /root/working/models/v12_f2/v12_f2_best

RUN chmod a+x /root/train.sh &&\
    chmod a+x /root/test.sh
ENV PATH $PATH:/root/

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /root/
