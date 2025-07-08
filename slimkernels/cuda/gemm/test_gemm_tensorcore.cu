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
#include <mma.h>
#include "gemm.h"


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



// 生成随机数据
void generate_random_data(float* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

// 生成随机half数据
void generate_random_half_data(half* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (int i = 0; i < size; ++i) {
        data[i] = __float2half(dis(gen));
    }
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
    std::vector<half> h_A_half(M * K);
    std::vector<half> h_B_half(K * N);
    
    // 生成随机数据
    generate_random_data(h_A.data(), M * K);
    generate_random_data(h_B.data(), K * N);
    generate_random_data(h_C.data(), M * N);
    std::memcpy(h_C_ref.data(), h_C.data(), M * N * sizeof(float));
    
    // 转换为half精度
    for (int i = 0; i < M * K; ++i) {
        h_A_half[i] = __float2half(h_A[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        h_B_half[i] = __float2half(h_B[i]);
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    half *d_A_half, *d_B_half;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_half, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B_half, K * N * sizeof(half)));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_half, h_A_half.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_half, h_B_half.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));
    
    // 配置kernel参数
    dim3 block_size(32, 32);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);
    
    // Tensor Core kernel配置 (16x16 tile)
    dim3 tensor_block_size(256); // 8 warps
    dim3 tensor_grid_size((N + 15) / 16, (M + 15) / 16);

    float alpha = 1.0f;
    float beta = 0.0f;
    
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
    
    // 测试Tensor Core kernel性能 (纯half)
    CUDA_CHECK(cudaMemcpy(d_C, h_C_ref.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    MMAarguments mmaArg{
                {M, N, K}, // problem shape
                d_A_half,
                d_B_half,
                d_C,
                d_C
            };
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        launch_GEMM_MMA(mmaArg);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    avg_time_ms = duration.count() / 1000.0 / num_runs;
    gflops = (flops / avg_time_ms) / 1e6;
    
    std::cout << "Tensor Core GEMM (Half):" << std::endl;
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
    CUDA_CHECK(cudaFree(d_A_half));
    CUDA_CHECK(cudaFree(d_B_half));
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
