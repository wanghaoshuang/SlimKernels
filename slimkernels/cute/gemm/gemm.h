#ifndef GEMM_H
#define GEMM_H

#include <cutlass/gemm/device/gemm.h>

using ElementAccumulator = half;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = half;
using ElementInputB = half;
using ElementOutput = half;

struct MMAarguments {
    cutlass::gemm::GemmCoord problem_size;
    ElementInputA_ *A;
    ElementInputB_ *B;
    ElementAccumulator *C;
    ElementOutput *D;
};

void launch_GEMM_MMA(MMAarguments &arg);

__global__ void gemm_optimized_kernel(
    const float* A,
    const float* B,
    float* C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta
);

__global__ void gemm_tensor_core_kernel(
    const half* A,
    const half* B,
    float* C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta
);

// cuBLAS GEMM实现 (老版本API)
void cublas_gemm(
    const float* A, 
    const float* B, 
    float* C, 
    const int M, 
    const int N, 
    const int K,
    const float alpha,
    const float beta
);

#endif // GEMM_H 