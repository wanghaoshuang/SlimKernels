#include <cuda_runtime.h>
#include <cublas.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <tuple>

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

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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

// 生成随机数据
void generate_random_data(float* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}



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

// 性能测试函数
void benchmark_gemm(int M, int K, int N, int num_runs = 100) {
    std::cout << "=== GEMM Performance Benchmark ===" << std::endl;
    std::cout << "Matrix dimensions: A(" << M << "x" << K << ") * B(" << K << "x" << N << ") = C(" << M << "x" << N << ")" << std::endl;
    
    // 分配主机内存
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_ref(M * N);
    
    // 生成随机数据
    generate_random_data(h_A.data(), M * K);
    generate_random_data(h_B.data(), K * N);
    generate_random_data(h_C.data(), M * N);
    std::memcpy(h_C_ref.data(), h_C.data(), M * N * sizeof(float));
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 配置kernel参数
    dim3 block_size(32, 32);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // 预热
    // for (int i = 0; i < 10; ++i) {
    //     gemm_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    // }
    // CUDA_CHECK(cudaDeviceSynchronize());
    
    // 测试基础kernel性能
    // auto start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < num_runs; ++i) {
    //     gemm_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    // }
    // CUDA_CHECK(cudaDeviceSynchronize());
    // auto end = std::chrono::high_resolution_clock::now();
    
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // double avg_time_ms = duration.count() / 1000.0 / num_runs;
    
    // // 计算理论性能
    // double flops = 2.0 * M * N * K;
    // double gflops = (flops / avg_time_ms) / 1e6;
    
    // std::cout << "Basic GEMM Kernel:" << std::endl;
    // std::cout << "  Average time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
    // std::cout << "  Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    
    // 测试优化kernel性能
    CUDA_CHECK(cudaMemcpy(d_C, h_C_ref.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        gemm_optimized_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / 1000.0 / num_runs;

    // 计算理论性能
    double flops = 2.0 * M * N * K;
    double gflops = (flops / avg_time_ms) / 1e6;
    
    std::cout << "Optimized GEMM Kernel:" << std::endl;
    std::cout << "  Average time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
    std::cout << "  Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    
    
    // 测试cuBLAS性能
    CUDA_CHECK(cudaMemcpy(d_C, h_C_ref.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        cublas_gemm(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    avg_time_ms = duration.count() / 1000.0 / num_runs;
    gflops = (flops / avg_time_ms) / 1e6;
    
    std::cout << "cuBLAS GEMM:" << std::endl;
    std::cout << "  Average time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms" << std::endl;
    std::cout << "  Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    
    // 清理内存
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
}



int main() {
    // 设置GPU设备
    int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // 获取GPU信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB" << std::endl;
    
    // 性能测试
    std::vector<std::tuple<int, int, int>> test_cases = {
    std::make_tuple(512, 512, 512),
    std::make_tuple(1024, 1024, 1024),
    std::make_tuple(2048, 2048, 2048),
    std::make_tuple(4096, 4096, 4096)
    };
    
    for (const auto& [M, K, N] : test_cases) {
        benchmark_gemm(M, K, N, 50);
        std::cout << std::endl;
    }
    
    return 0;
}
