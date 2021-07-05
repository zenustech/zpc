FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

ENV NCCL_VERSION 2.9.9

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtinfo5 libncursesw5 \
    cuda-cudart-dev-11-3=11.3.109-1 \
    cuda-command-line-tools-11-3=11.3.1-1 \
    cuda-minimal-build-11-3=11.3.1-1 \
    cuda-libraries-dev-11-3=11.3.1-1 \
    cuda-nvml-dev-11-3=11.3.58-1 \
    libnpp-dev-11-3=11.3.3.95-1 \
    libnccl-dev=2.9.9-1+cuda11.3 \
    libcublas-dev-11-3=11.5.1.109-1 \
    libcusparse-dev-11-3=11.6.0.109-1 \
    && rm -rf /var/lib/apt/lists/*

# apt from auto upgrading the cublas package. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold libcublas-dev-11-3 libnccl-dev
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

RUN apt-get -y install clang++-11 clang-11 g++-10 gcc-10 cmake git

RUN cd build && make -j24 && sudo make install