FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        git \
        wget \
        libacl1-dev \
        libncurses5-dev \
        pkg-config \
        zlib1g \
        g++-10

RUN rm /usr/bin/g++ && \
    ln -s /usr/bin/g++-10 /usr/bin/g++

RUN cd /home && \
    mkdir /opt/cmake && \
    wget https://cmake.org/files/v3.27/cmake-3.27.0-linux-x86_64.sh && \
    sh cmake-3.27.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

RUN cd /home && \
    wget https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.tar.gz && \
    tar xvf boost_1_80_0.tar.gz && \
    cd boost_1_80_0 && \
    ./bootstrap.sh --prefix=/usr/ && \
    ./b2 install

WORKDIR /home/tally

COPY . .

RUN cd cudnn && \
    tar -xvf cudnn-linux-x86_64-8.9.3.28_cuda12-archive.tar.xz && \
    cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include && \
    cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 && \
    chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

RUN cd third_party && \
    cp cudnn-frontend/ /usr/local/cuda/ -r

RUN cd /home/tally && \
    make