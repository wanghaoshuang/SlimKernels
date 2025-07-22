#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cute/tensor.hpp>
#include <cute/underscore.hpp>
#include <cute/numeric/integral_constant.hpp>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "gemm.h"

using namespace cute;

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = float;          // <- data type of elements in input matrix A
using ElementInputB = float;          // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

const int  THREADS_PER_WARP=32;

struct GemmConfig {
    static constexpr int BLOCK_DIM_M = 128;
    static constexpr int BLOCK_DIM_N = 128;
    static constexpr int BLOCK_DIM_K = 8;
    static constexpr int WARP_TILE_DIM_M = 64;
    static constexpr int WARP_TILE_DIM_N = 64;
    static constexpr int THREAD_NUM = 256;
    static constexpr int TENSOR_CORE_M = 16;
    static constexpr int TENSOR_CORE_N = 8;
    static constexpr int TENSOR_CORE_K = 8;
};


// threadblockShape 128 64 8
// warpShape 64 32 8
// every block has 4 warps
template<typename Config>
__global__ void GEMM_MMA(MMAarguments arg){
    auto shape_MNK = make_shape(arg.problem_size.m(), arg.problem_size.n(), arg.problem_size.k());
    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(arg.A), select<0,2>(shape_MNK), make_stride(arg.problem_size.k(), _1{})); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(arg.B), select<1,2>(shape_MNK), make_stride(arg.problem_size.k(), _1{})); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(arg.C), select<0,1>(shape_MNK), make_stride(_1{}, arg.problem_size.m())); // (M,N)

    // Define CTA tile sizes (static)
    auto cta_tiler = make_shape(Int<Config::BLOCK_DIM_M>{}, Int<Config::BLOCK_DIM_N>{}, Int<Config::BLOCK_DIM_K>{});                   // (BLK_M, BLK_N, BLK_K)
    
    // Define the smem layouts (static)
    auto sA_layout = make_layout(make_shape(Int<Config::BLOCK_DIM_M>{},Int<Config::BLOCK_DIM_K>{}), LayoutRight{});   // (m,k) -> smem_idx; k-major
    auto sB_layout = make_layout(make_shape(Int<Config::BLOCK_DIM_N>{},Int<Config::BLOCK_DIM_K>{}), LayoutRight{});   // (n,k) -> smem_idx; k-major

    // Define the thread layouts (static)
    auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});  // (m,k) -> thr_idx; k-major
    auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});  // (n,k) -> thr_idx; k-major
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));                 // (m,n) -> thr_idx; m-major

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    // Shared memory buffers
    __shared__ ElementInputA smemA[cosize_v<decltype(sA_layout)>];
    __shared__ ElementInputB smemB[cosize_v<decltype(sB_layout)>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

    //
    // Partition the copying of A and B tiles across the threads
    //

    // TUTORIAL: Example of simple raked partitioning of ThreadLayouts tA|tB over data A|B tiles

    Tensor tAgA = local_partition(gA, tA, threadIdx.x);                  // (THR_M,THR_K,k)
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K)

    Tensor tBgB = local_partition(gB, tB, threadIdx.x);                  // (THR_N,THR_K,k)
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)

    //
    // Define A/B partitioning and C accumulators
    //

    // TUTORIAL: Example of partitioning via projections of a ThreadLayout tC

    // Partition sA (BLK_M, BLK_K) by the rows of tC
    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K)
    // Partition sB (BLK_N, BLK_K) by the cols of tC
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K)
    // Partition gC (M,N) by the tile of tC
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

    // Allocate the accumulators -- same shape/layout as the partitioned data
    Tensor tCrC = make_tensor_like(tCgC);

    // Clear the accumulators
    clear(tCrC); 

    auto K_TILE_MAX = size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // Copy gmem to smem with tA|tB thread-partitioned tensors
    copy(tAgA(_,_,k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
    copy(tBgB(_,_,k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)
    cp_async_fence();        // Label the end of (potential) cp.async instructions
    cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
    __syncthreads();         // Wait for all threads to write to smem
    // Compute gemm on tC thread-partitioned smem
    gemm(tCsA, tCsB, tCrC);            // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)
    __syncthreads();         // Wait for all threads to read from smem
  }
}

void launch_GEMM_MMA(MMAarguments &arg){
    dim3 grid,block;
    using Config = GemmConfig;
    grid.x = (arg.problem_size.n()+Config::BLOCK_DIM_N-1)/Config::BLOCK_DIM_N;
    grid.y = (arg.problem_size.m()+Config::BLOCK_DIM_M-1)/Config::BLOCK_DIM_M;
    grid.z = 1;

    block.x = Config::THREAD_NUM;
    block.y = 1;
    block.z = 1;

    GEMM_MMA<Config><<<grid,block>>>(arg);
}