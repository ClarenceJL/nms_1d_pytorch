#!/usr/bin/env bash
# CUDA_PATH=/usr/local/cuda/

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export CPATH=/usr/local/cuda-8.0/include${CPATH:+:${CPATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"

CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 "

# compile nms
cd src
echo "Compiling nms-1d kernels by nvcc..."
nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH

cd ../
python3 build.py
