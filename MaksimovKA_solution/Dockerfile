# use with GPU and nvidia-docker2
FROM tensorflow/tensorflow:1.9.0-devel-gpu-py3
RUN add-apt-repository -y ppa:ubuntugis/ppa; \
apt-get update && apt-get -y install mc python-opencv tmux nano python3-tk gdal-bin python3-gdal less; \
pip3 install --upgrade pip
RUN pip3 install h5py scikit-image scipy pillow
RUN pip3 install opencv-python matplotlib tqdm keras==2.1.6
RUN pip3 install keras-resnet six scipy gdal
RUN pip install albumentations
RUN pip install keras-resnet scikit-learn pandas
RUN pip install jupyter
RUN pip install gdal rasterio
RUN pip install shapely
RUN mkdir /project
COPY ./ /project/
RUN chmod +x /project/train.sh
RUN chmod +x /project/test.sh
WORKDIR /project

ENV PYTHONPATH "${PYTHONPATH}:/project"