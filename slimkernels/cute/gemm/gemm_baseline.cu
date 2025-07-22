#include <cuda_runtime.h>
#include <cublas.h>
#include <mma.h>
#include "gemm.h"

// 优化的GEMM kernel (使用共享内存)
__global__ void gemm_optimized_kernel(
    const float* A, 
    const float* B, 
    float* C, 
    const int M, 
    const int N, 
    const int K,
    const float alpha,
    const float beta
) {
    const int TILE_SIZE = 32;
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // 加载tile到共享内存
        int tileRow = tile * TILE_SIZE + threadIdx.y;
        int tileCol = tile * TILE_SIZE + threadIdx.x;
        
        if (row < M && tileCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + tileCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (tileRow < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[tileRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // 计算tile内的点积
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Tensor Core优化的GEMM kernel
__global__ void gemm_tensor_core_kernel(
    const half* A, 
    const half* B, 
    float* C, 
    const int M, 
    const int N, 
    const int K,
    const float alpha,
    const float beta
) {
    // 使用16x16x16的Tensor Core tile
    using namespace nvcuda::wmma;
    
    // 每个warp处理16x16的输出tile
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 16;
    
    // 计算当前warp负责的输出tile位置
    int tileM = blockIdx.y * TILE_M;
    int tileN = blockIdx.x * TILE_N;
    
    // 声明fragment
    fragment<matrix_a, TILE_M, TILE_N, TILE_K, half, row_major> a_frag;
    fragment<matrix_b, TILE_M, TILE_N, TILE_K, half, col_major> b_frag;
    fragment<accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;
    
    // 初始化累加器
    fill_fragment(c_frag, 0.0f);
    
    // 遍历K维度
    for (int k = 0; k < K; k += TILE_K) {
        // 加载A和B的tile
        load_matrix_sync(a_frag, A + tileM * K + k, K);
        load_matrix_sync(b_frag, B + k * N + tileN, N);
        
        // Tensor Core矩阵乘法
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 存储结果
    store_matrix_sync(C + tileM * N + tileN, c_frag, N, nvcuda::wmma::mem_row_major);
} 

// cuBLAS错误检查宏 (老版本API)
#define CUBLAS_CHECK(call) \
    do { \
        call; \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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
) {
    // 老版本cuBLAS API
    // sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // 其中: 'N'表示不转置, 'T'表示转置
    CUBLAS_CHECK(cublasSgemm('N', 'N', M, N, K, alpha, A, M, B, K, beta, C, M));
}