/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/device_kernel.h"
#include "../atom/mma_traits_sm90_gmma.hpp"
#include "../algorithm/gemm.hpp"

using namespace cute;

// ============== Debug Print Utilities ==============
#ifdef DEBUG_MODE

#define DEBUG_PRINT_VAR(var) \
  do { if(thread0()) { print("  " #var " : "); print(var); print("\n"); } } while(0)

#define DEBUG_SECTION_BEGIN(title) \
  do { if(thread0()) { print("******************" title "*********************\n"); } } while(0)

#define DEBUG_PRINT_MATRIX_A(s2r_atom_a, s2r_copy_a, s2r_thr_copy_a, mA, gA, sA, tAgA, tAsA, tCsA, tCrA, tXsA, tXrA) \
  do { \
    DEBUG_SECTION_BEGIN("A Matrix"); \
    DEBUG_PRINT_VAR(s2r_atom_a); \
    DEBUG_PRINT_VAR(s2r_copy_a); \
    DEBUG_PRINT_VAR(s2r_thr_copy_a); \
    DEBUG_PRINT_VAR(mA); \
    DEBUG_PRINT_VAR(gA); \
    DEBUG_PRINT_VAR(sA); \
    DEBUG_PRINT_VAR(tAgA); \
    DEBUG_PRINT_VAR(tAsA); \
    DEBUG_PRINT_VAR(tCsA); \
    DEBUG_PRINT_VAR(tCrA); \
    DEBUG_PRINT_VAR(tXsA); \
    DEBUG_PRINT_VAR(tXrA); \
  } while(0)

#define DEBUG_PRINT_MATRIX_B(mB, gB, sB, tBgB, tBsB, tCsB, tCrB) \
  do { \
    DEBUG_SECTION_BEGIN("B Matrix"); \
    DEBUG_PRINT_VAR(mB); \
    DEBUG_PRINT_VAR(gB); \
    DEBUG_PRINT_VAR(sB); \
    DEBUG_PRINT_VAR(tBgB); \
    DEBUG_PRINT_VAR(tBsB); \
    DEBUG_PRINT_VAR(tCsB); \
    DEBUG_PRINT_VAR(tCrB); \
  } while(0)

#define DEBUG_PRINT_MATRIX_C(mC, gC, tCgC, tCrC) \
  do { \
    DEBUG_SECTION_BEGIN("C Matrix"); \
    DEBUG_PRINT_VAR(mC); \
    DEBUG_PRINT_VAR(gC); \
    DEBUG_PRINT_VAR(tCgC); \
    DEBUG_PRINT_VAR(tCrC); \
  } while(0)

#else  // !DEBUG_MODE

#define DEBUG_PRINT_VAR(var) ((void)0)
#define DEBUG_SECTION_BEGIN(title) ((void)0)
#define DEBUG_PRINT_MATRIX_A(...) ((void)0)
#define DEBUG_PRINT_MATRIX_B(...) ((void)0)
#define DEBUG_PRINT_MATRIX_C(...) ((void)0)

#endif  // DEBUG_MODE
// ============== End Debug Print Utilities ==============


template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};



// SM90 Register Configuration:
// - Producer (1 warp, 32 threads): 40 registers each = 1,280 total
// - Consumer (4 warps, 128 threads): 232 registers each = 29,696 total
// - Total per block: 30,976 registers
// - SM90 has 65,536 registers per SM -> allows ~2 blocks per SM
constexpr int ProducerRegCount = 40;
constexpr int ConsumerRegCount = 232;

template <class ProblemShape, class ACtaTiler, class BCtaTiler, class CCtaTiler,
          class TA, class ASmemLayout, class TmaA, class S2RAtomA,
          class TB, class BSmemLayout, class TmaB,
          class TC, class CStride, class TiledMma, class TiledMmaA,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(160, 2)  // 5 warps: 1 producer + 4 consumers, target 2 blocks/SM
void
gemm_device(ProblemShape shape_MNK, ACtaTiler a_cta_tiler, BCtaTiler b_cta_tiler, CCtaTiler c_cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a, S2RAtomA s2r_atom_a,
            TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
            TC      * C, CStride dC, TiledMma mma, TiledMmaA mma_a,
            Alpha alpha, Beta beta)
{
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<4>{});                   // (M, A_K, N, B_K)

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  auto [M, A_K, N, B_K] = shape_MNK;
  auto gA_layout = make_layout(make_shape(M, A_K), make_stride(A_K, Int<1>{}));
  Tensor mA = tma_a.get_tma_tensor(gA_layout.shape()); 
  auto gB_layout = make_layout(make_shape(M, B_K), make_stride(A_K, Int<1>{}));
  Tensor mB = tma_b.get_tma_tensor(gB_layout.shape());              
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, a_cta_tiler, cta_coord, Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, b_cta_tiler, cta_coord, Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, c_cta_tiler, cta_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), BSmemLayout{}); // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles using TMA
  //

  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));  // (TMA,k) and (TMA,PIPE)

  auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{}); 
  Layout cta_layout_mnk = make_layout(cluster_shape);
  auto cta_coord_mnk = cta_layout_mnk.get_flat_coord(cute::block_rank_in_cluster());                                  
  uint16_t mcast_mask_b = create_tma_multicast_mask<0>(cta_layout_mnk, cta_coord_mnk);
  bool is_elected_for_b = (get<0>(cta_coord_mnk) == 0);
  auto [tBgB, tBsB] = tma_partition(tma_b, 
    get<0>(cta_coord_mnk),           // M 方向的 CTA 坐标
    make_layout(size<0>(cluster_shape)), // M 方向 layout (2)
    group_modes<0,2>(sB), group_modes<0,2>(gB));

  // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

  
  //
  // Define A/B partitioning and C accumulators
  //

  // Producer: warp 4 (threads 128-159), Consumer: warps 0-3 (threads 0-127)
  bool is_producer = (threadIdx.x >= 128);
  bool is_consumer = (threadIdx.x < 128);
  
  // SM90 Dynamic Register Reallocation:
  // - Producer warp reduces to 40 registers (only needs TMA operations)
  // - Consumer warp group increases to 232 registers (needs WGMMA accumulators)
  if (is_producer) {
    cutlass::arch::warpgroup_reg_dealloc<ProducerRegCount>();
  } else {
    cutlass::arch::warpgroup_reg_alloc<ConsumerRegCount>();
  }
  
  // MMA thread index for consumers (0-127 mapping from threads 0-127)
  int mma_thread_idx = threadIdx.x;

  // Consumer threads participate in MMA operations
  // Use mma_thread_idx to get correct MMA slice for consumer threads
  ThrMMA thr_mma_a = mma_a.get_slice(is_consumer ? mma_thread_idx : 0);
  Tensor tCsA = thr_mma_a.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrA = thr_mma_a.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
  
  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma_a);
  ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(is_consumer ? mma_thread_idx : 0);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

  ThrMMA thr_mma = mma.get_slice(is_consumer ? mma_thread_idx : 0);
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)
  auto tCrB_reshaped = make_tensor(
    tCrB.data(),
    make_layout(
        make_shape(_2{}, _1{}, _8{}, size<3>(tCsB)),  // 添加 pipeline 维度
        make_stride(_2{}, _0{}, _0{}, Int<2048>{})
    )
  );

  // Allocate the accumulators -- same size as the projected data
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  // Clear the accumulators (only consumers need this)
  if (is_consumer) {
    clear(tCrC);
  }

  // Debug print (only when compiled with -DDEBUG_MODE)
  DEBUG_PRINT_VAR(gA_layout);
  DEBUG_PRINT_MATRIX_A(s2r_atom_a, s2r_copy_a, s2r_thr_copy_a, mA, gA, sA, tAgA, tAsA, tCsA, tCrA, tXsA, tXrA);
  DEBUG_PRINT_MATRIX_B(mB, gB, sB, tBgB, tBsB, tCsB, tCrB);
  DEBUG_PRINT_VAR(tCrB_reshaped);
  DEBUG_PRINT_MATRIX_C(mC, gC, tCgC, tCrC);
  
#if 1
  //
  // PREFETCH
  //

  auto K_PIPE_MAX = size<1>(tAsA);
  // Total count of tiles (use A's k-tile count as main loop count)
  int k_tile_count = size<1>(tAgA);
  // Current tile index in gmem to read from
  int k_tile = 0;

  // Initialize Barriers
  // Producer warp is warp 0, use lane_predicate to elect one thread
  int lane_predicate = cute::elect_one_sync();
  uint64_t* producer_mbar = smem.tma_barrier;
  uint64_t* consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA
  using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    // Producer warp (warp 0) initializes barriers
    if (is_producer && lane_predicate) {
      ProducerBarType::init(&producer_mbar[pipe],   1);
      ConsumerBarType::init(&consumer_mbar[pipe], 128);  // 4 consumer warps = 128 threads
    }
  }
  // Ensure barrier init is complete on all CTAs
  cluster_sync();

  // Total number of k-tiles
  int k_tile_max = size<1>(tAgA);

  // Start async loads for valid pipes only (producer warp only)
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
  {
    if (is_producer && lane_predicate)
    {
      if (k_tile < k_tile_max) {
        // Set expected Tx Bytes and issue TMA
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
        copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
        copy(tma_b.with(producer_mbar[pipe], mcast_mask_b), tBgB(_,k_tile), tBsB(_,pipe));
      } else {
        // No more tiles - expect 0 bytes so barrier completes immediately
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], 0);
      }
    }
    --k_tile_count;
    ++k_tile;
  }


  //
  // PIPELINED MAIN LOOP
  //

  // A PipelineState is a circular pipe index [.index()] and a pipe phase [.phase()]
  //   that flips each cycle through K_PIPE_MAX.
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();             // TMA writes (producer)
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();             // MMA reads (consumer)
  
  
  // Producer warp (warp 0) - only does TMA loading
  if (is_producer) {
    CUTE_NO_UNROLL
    while (k_tile_count > -K_PIPE_MAX)
    {
      if (lane_predicate)
      {
        int pipe = write_state.index();
        // Wait for Consumer to complete consumption
        ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
        // Set expected Tx Bytes after each reset / init
        if (k_tile < k_tile_max) {
          ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
          copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
          copy(tma_b.with(producer_mbar[pipe], mcast_mask_b), tBgB(_,k_tile), tBsB(_,pipe));
        } else {
          // No more tiles - expect 0 bytes so barrier completes immediately
          ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], 0);
        }
        ++write_state;
      }
      --k_tile_count;
      ++k_tile;
    }
  }
  // Consumer warps (warps 1-4) - only do WGMMA computation
  else {
    CUTE_NO_UNROLL
    while (k_tile_count > -K_PIPE_MAX)
    {
      // Wait for Producer to complete
      int read_pipe = read_state.index();
      ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());
      // Copy A from shared memory to registers
      copy(s2r_copy_a, tXsA(_,_,_,read_pipe), tXrA);
      // MMAs to cover 1 K_TILE
      warpgroup_arrive();
      // warpgroup_fence_operand(tCrC);
      // warpgroup_fence_operand(tCrA);
      auto tCrB_k = tCrB_reshaped(_,_,_,read_pipe);
      cute::gemm(mma, tCrA, tCrB_k, tCrC);     // (V,M) x (V,N) => (V,M,N)
      warpgroup_commit_batch();
      // Wait for all MMAs in a K_TILE to complete
      warpgroup_wait<0>();
      // warpgroup_fence_operand(tCrC);
      // Notify that consumption is done
      ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
      ++read_state;

      --k_tile_count;
      ++k_tile;
    }
  }

  //
  // Epilogue (unpredicated)
  //

  // axpby(alpha, tCrC, beta, tCgC);
#endif
}


// Setup params for a TN GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int a_k, int b_k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  auto prob_shape = make_shape(m, a_k, n, b_k);

  auto a_cta_tiler = make_shape(_64{}, _64{}, _128{}); // M X K
  auto b_cta_tiler = make_shape(_64{}, _64{}, _512{}); // N X K
  auto c_cta_tiler = make_shape(_64{}, _64{}, _512{}); // M X N
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(size<0>(a_cta_tiler), size<2>(a_cta_tiler), bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(size<0>(b_cta_tiler), size<2>(b_cta_tiler), bP));
  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));

  // Define the TMA atoms
  auto gA_layout = make_layout(make_shape(m, a_k), make_stride(a_k, Int<1>{}));
  Tensor mA = make_tensor(A, gA_layout);
  auto gB_layout = make_layout(make_shape(m, b_k), make_stride(b_k, Int<1>{}));
  Tensor mB = make_tensor(B, gB_layout);

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(size<0>(a_cta_tiler), size<2>(a_cta_tiler)));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD_MULTICAST{}, mB, sB(_,_,0), make_shape(size<0>(b_cta_tiler), size<2>(b_cta_tiler)), Int<2>{});

  Copy_Atom<SM75_U32x2_LDSM_N, TA> s2r_atom_a;
  TiledMMA tiled_mma_a = make_tiled_mma(SM90_64x64x64_F16I2E4M3_RS_TN_A<GMMA::ScaleIn::One, GMMA::ScaleIn::One>{});
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x64_F16I2E4M3_RS_TN<GMMA::ScaleIn::One, GMMA::ScaleIn::One>{});

  // Launch parameter setup
  // 5 warps: 1 producer (TMA) + 4 consumers (WGMMA)
  dim3 dimBlock(160);
  dim3 dimCluster(2, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, size<0>(c_cta_tiler))), dimCluster.x),
               round_up(size(ceil_div(n, size<1>(c_cta_tiler))), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device<decltype(prob_shape), decltype(a_cta_tiler), decltype(b_cta_tiler), decltype(c_cta_tiler),
                                           TA, decltype(sA), decltype(tmaA), decltype(s2r_atom_a),
                                           TB, decltype(sB), decltype(tmaB),
                                           TC, decltype(dC), decltype(tiled_mma), decltype(tiled_mma_a),
                                           decltype(alpha), decltype(beta)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, a_cta_tiler, b_cta_tiler, c_cta_tiler,
                                                             A, tmaA, s2r_atom_a,
                                                             B, tmaB,
                                                             C, dC, tiled_mma, tiled_mma_a,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}

int main(int argc, char** argv)
{
  cudaDeviceProp props;
  int current_device_id;
  cudaGetDevice(&current_device_id);
  cudaGetDeviceProperties(&props, current_device_id);
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major != 9) {
    std::cout << "This example requires NVIDIA's Hopper Architecture GPU with compute capability 90a" << std::endl;
    // Return 0 so tests pass if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  int m = 5120;
  int n = 5120;
  int b_k = 4096;
  int a_k = 4096/4;

  using TA = cute::uint8_t;
  using TB = cute::float_e4m3_t;
  using TC = cute::half_t;
  using TI = cute::half_t;

  TI alpha = TI(1.0f);
  TI beta  = TI(0.0f);

  thrust::host_vector<TA> h_A(m*a_k);
  thrust::host_vector<TB> h_B(n*b_k);
  thrust::host_vector<TC> h_C(m*n);

  // Initialize the tensors
  for (int j = 0; j < m*a_k; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1));
  for (int j = 0; j < n*b_k; ++j) h_B[j] = TB(int((rand() % 2) ? 1 : -1));
  for (int j = 0; j < m*n; ++j) h_C[j] = TC(0);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  double gflops = (2.0*m*n*b_k) * 1e-9;
#ifdef DEBUG_MODE
  const int timing_iterations = 1;
#else
  const int timing_iterations = 100;
#endif

  GPU_Clock timer;

  int ldA = a_k, ldB = b_k, ldC = m;

  d_C = h_C;
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm_tn(m, n, a_k, b_k,
            alpha,
            d_A.data().get(), ldA,  // A now in packed 64x16 tile layout
            d_B.data().get(), ldB,  // B now in packed 64x64 tile layout
            beta,
            d_C.data().get(), ldC);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

#else
  std::cout << "CUTLASS_ARCH_MMA_SM90_SUPPORTED must be enabled, but it is not. Test is waived \n" << std::endl;
#endif

  return 0;
}
