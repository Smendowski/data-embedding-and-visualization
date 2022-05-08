FROM ubuntu:20.04

RUN apt update

# Install CPP compiler, lapack, make, libboost and other dependencies
RUN apt-get install build-essential libssl-dev liblapack-dev wget make -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    dpkg-reconfigure tzdata && \
    apt-get install libboost-all-dev -y && \
    apt install git python3 python3-pip -y && \
    pip3 install faiss-cpu

# Copy datasets, labels and graphs required by VisKit
RUN mkdir -p /opt/datasets /opt/labels /opt/graphs
COPY datasets /opt/datasets
COPY labels /opt/labels
COPY graphs /opt/graphs

# Get the VisKit library
RUN mkdir -p /opt/viskit
COPY viskit /opt/viskit/

# Initialize submodules
RUN git submodule init && \
    git submodule update

# Install CMake
RUN mkdir -p /opt/temp && cd /opt/temp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1.tar.gz && \
    tar -zxvf cmake-3.23.1.tar.gz && cd cmake-3.23.1 && \
    ./bootstrap && \
    make && \
    make install

# Build VisKit application
WORKDIR /opt/viskit
RUN cmake ./CMakeLists.txt
RUN make