# FROM nvcr.io/nvidia/pytorch:22.07-py3
FROM ubuntu:22.04

RUN apt-get update && apt-get -y install sudo
RUN apt-get -y upgrade \
&& apt-get update \
&& DEBIAN_FRONTEND=noninteractive apt-get --fix-missing -y install \
curl \
wget \
git \
build-essential \
zlib1g-dev \
libssl-dev \
libglib2.0-0 \
libsm6 \
libxext6 \
libxrender-dev \
libgl1-mesa-glx \
libgl1-mesa-dev \
vim \
htop \
unzip \
libosmesa6-dev \
libglew-dev \
libgl1-mesa-glx \
xpra \
xserver-xorg-dev \ 
net-tools \
software-properties-common \
tmux \
libopenblas-base \
libffi-dev \
libbz2-dev \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*


# RUN adduser --disabled-password --gecos '' interaction_graph
# RUN adduser interaction_graph sudo
# RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
WORKDIR /home/interaction_graph/
ENV HOME /home/interaction_graph
RUN chmod a+rwx /home/interaction_graph/
# USER interaction_graph

RUN wget https://www.python.org/ftp/python/3.7.12/Python-3.7.12.tgz
RUN tar -xf Python-3.7.12.tgz
WORKDIR /home/interaction_graph/Python-3.7.12
RUN ./configure --enable-optimization
RUN make -j 8
RUN sudo make install
RUN sudo ln -sf /usr/local/bin/python3.7 /usr/bin/python
RUN sudo ln -sf /usr/local/bin/pip3.7 /usr/bin/pip
RUN python -m pip install --upgrade pip
# Install Anaconda
# RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
# RUN bash Anaconda3-2021.05-Linux-x86_64.sh -b -p /home/ubuntu/anaconda3
# RUN rm Anaconda3-2021.05-Linux-x86_64.sh
# # Set path to conda
# ENV PATH /home/ubuntu/anaconda3/bin:$PATH

# Updating Anaconda packages
# RUN conda install conda=4.12.0
# # RUN conda config --remove channels conda-forge
# RUN conda config --add channels conda-forge

# RUN conda install -c conda-forge python=3.7.12

# RUN conda create --name 'dev' -q python=3.7.12
# RUN virtualenv venv
# RUN source ./venv/bin/activate
WORKDIR /home/interaction_graph/
RUN git clone https://github.com/DragonMyth/fairmotion.git
WORKDIR /home/interaction_graph/fairmotion
RUN pip install -e .
WORKDIR /home/interaction_graph/ScaDive
RUN pip install pybullet==3.0.8 ray[rllib]==1.8.0 pandas requests
RUN pip install gym==0.18.0
RUN pip install torch==1.10.0
