// ------------------------------------------------------------------
// Non-Maximum Suppression - 1D case
// Adapted from Shaoqing Ren's 2D implementation for Faster R-CNN
// By Claire Li, 2019
// ------------------------------------------------------------------

#include <stdbool.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include "nms_cuda_kernel.h"

#define CUDA_WARN(XXX) \
    do { if (XXX != cudaSuccess) std::cout << "CUDA Error: " << \
        cudaGetErrorString(XXX) << ", at line " << __LINE__ \
<< std::endl; cudaDeviceSynchronize(); } while (0)

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0)) //divide and round-up
int const threadsPerBlock = sizeof(unsigned long long) * 8;  // number of threads per block = number of bits of an ULL
                                                             // This is because we will use a ULL to record the overlapping
                                                             // in each thread.

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[1], b[1]);
  float interS = max(right - left + 1, 0.f);
  float Sa = (a[1] - a[0] + 1);
  float Sb = (b[1] - b[0] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(int n_boxes, float nms_overlap_thresh,float *dev_boxes, unsigned long long *dev_mask) {
  /* The kernel function works INSIDE a THREAD
   * Parameters
   * n_boxes: number of proposals
   * nms_overlap_thresh: overlapping threshold
   * dev_boxes: 2d-array [n_boxes][3]
   * dev_mask: currently empty [n_boxes][n_boxes/threadPerBlock=col_blocks]
   */
  const int row_start = blockIdx.y;   // row index of current block, 0 <= row_start < col_blocks
  const int col_start = blockIdx.x;   // column index of current block, 0 <= col_start < col_blocks

  // if (row_start > col_start) return;  // enable this to avoid redundant computation

  const int row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 3]; // a "__shared__" variable is shared among all threads within this block
                                                     // (my speculation is, although every thread has this line, it is only
                                                     // executed once for each block)
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 3 + 0] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 3 + 0];
    block_boxes[threadIdx.x * 3 + 1] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 3 + 1];
    block_boxes[threadIdx.x * 3 + 2] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 3 + 2];
  }  // retrieve the box descriptor

  // __syncthreads() waits until all threads in the thread block have reached this point and all global and
  // shared memory accesses made by these threads prior to __syncthreads() are visible to all threads in the block.
  __syncthreads();
  // -->> block_boxes is filled up with valid data now

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;  // index of current box
    const float *cur_box = dev_boxes + cur_box_idx * 3;
    int i = 0;
    unsigned long long t = 0;  // number of threads per block = number of bits in 't'
    int start = 0;
    if (row_start == col_start) {  // to compute overlapping
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 3) > nms_overlap_thresh) {
        t |= 1ULL << i;  // bit i is set to 1 if 'cur_box' overlaps with 'block_boxes[i]'
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;  // overlapping condition between box y*threadsPerBlock+t and [x*threadsPerBlock:(x+1)*threadsPerBlock]
  }
}



void nms_cuda_compute(int* keep_out, int *num_out, float* boxes_host, int boxes_num, int boxes_dim,
                      float nms_overlap_thresh) {
  /* Input:
   *   boxes_host: original proposals N * 3
   *   boxes_num: number of proposals N
   *   boxes_dim: number of elements used to describe a proposal (e.g. 1+4 for 2D box, 1+2 for 1D segment)
   *   nms_overlap_thresh: overlapping threshold
   * Output:
   *   keep_out: vector containing indices of kept boxes
   *   num_out: number of kept boxes
   */
  // printf("nms threshold %f\n", nms_overlap_thresh);

  float* boxes_dev = NULL;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock); // total number of blocks needed

  CUDA_CHECK(cudaMalloc(&boxes_dev, boxes_num * boxes_dim * sizeof(float)));      // N x 3
  // copy host memory to device memory
  CUDA_CHECK(cudaMemcpy(boxes_dev, boxes_host, boxes_num * boxes_dim * sizeof(float), cudaMemcpyHostToDevice));
  // mask_dev keeps the overlapping condition between any pair of boxes
  CUDA_CHECK(cudaMalloc(&mask_dev, boxes_num * col_blocks * sizeof(unsigned long long)));  // N x col_blocks (x threadsPerBlock bits)

  dim3 blocks(col_blocks, col_blocks); // col_blocks x col_blocks x 1, we use 2D grid because NMS requires pair-wise operation
  dim3 threads(threadsPerBlock);       // threadsPerBlock x 1 x 1
  // num of works: (col_blocks * col_blocks) x threadsPerBlock = box_num x col_blocks

  nms_kernel<<<blocks, threads>>>(boxes_num, nms_overlap_thresh, boxes_dev, mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0], mask_dev, sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));  // copy device memory to host memory

  std::vector<unsigned long long> remv(col_blocks); // col_blocks (x threadsPerBlock bits), recording whether a box has been removed
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  // create a memory for keep_out on cpu
  // the following computation is done on cpu
  int* keep_out_cpu = new int[boxes_num];

  int num_to_keep = 0;
  int i, j;
  for (i = 0; i < boxes_num; i++) { // enumerate all the boxes
    int nblock = i / threadsPerBlock;   // index of block
    int inblock = i % threadsPerBlock;  // index of thread

    if (!(remv[nblock] & (1ULL << inblock))) {  // if current box has not been removed before
      keep_out_cpu[num_to_keep++] = i;          // add it to the keep list
      unsigned long long *p = &mask_host[0] + i * col_blocks;  // extract the row of overlapping condition between box i and all other boxes
      for (j = nblock; j < col_blocks; j++) {  // blocks before current block has already been considered
        remv[j] |= p[j];  // remove every box overlapping with i
      }
    }
  }
  // printf("kept boxes (should be at least 1): %d\n", num_to_keep);

  // copy keep_out_cpu to keep_out on gpu
  CUDA_WARN(cudaMemcpy(keep_out, keep_out_cpu, boxes_num * sizeof(int), cudaMemcpyHostToDevice));


  // copy num_to_keep to num_out on gpu
  CUDA_WARN(cudaMemcpy(num_out, &num_to_keep, 1 * sizeof(int), cudaMemcpyHostToDevice));

  // release cuda memory
  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(mask_dev));
  // release cpu memory
  delete []keep_out_cpu;
}