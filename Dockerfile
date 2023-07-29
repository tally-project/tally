FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        git \
        libboost-all-dev \
        wget \
        libacl1-dev \
        libncurses5-dev \
        pkg-config \
        zlib1g


RUN cd /home && \
    mkdir /opt/cmake && \
    wget https://cmake.org/files/v3.27/cmake-3.27.0-linux-x86_64.sh && \
    sh cmake-3.27.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

WORKDIR /home/tally

COPY . .

RUN cd cudnn && \
    tar -xvf cudnn-linux-x86_64-8.9.3.28_cuda12-archive.tar.xz && \
    cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include && \
    cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 && \
    chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

RUN cd third_party/cudnn-frontend && \
    cp cudnn-frontend/ /usr/local/cuda/ -r