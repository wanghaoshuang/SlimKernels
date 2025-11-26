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
#include "cutlass/device_kernel.h"
#include "../atom/mma_traits_sm90_gmma.hpp"
#include "../algorithm/gemm.hpp"

using namespace cute;

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

template <class ProblemShape, class ACtaTiler, class BCtaTiler, class CCtaTiler,
          class TA, class ASmemLayout, class TmaA, class S2RAtomA,
          class TB, class BSmemLayout, class TmaB,
          class TC, class CStride, class TiledMma, class TiledMmaA,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
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
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, A_K));               // (M, A_K) TMA Tensor
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, B_K));               // (N, B_K) TMA Tensor
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);    // (M, N)

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

  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));  // (TMA,k) and (TMA,PIPE)

  // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

  
  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma_a = mma_a.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma_a.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrA = thr_mma_a.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
  
  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma_a);
  ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)

  // Allocate the accumulators -- same size as the projected data
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  // Clear the accumulators
  clear(tCrC);

#if 0
  if(thread0()) {
    print("******************A Matrix*********************\n");
    print("  s2r_atom_a : "); print(  s2r_atom_a); print("\n");
    print("  s2r_copy_a : "); print(  s2r_copy_a); print("\n");
    print("  s2r_thr_copy_a : "); print(  s2r_thr_copy_a); print("\n");
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCrA : "); print(tCrA); print("\n");
    print("tXsA : "); print(tXsA); print("\n");
    print("tXrA : "); print(tXrA); print("\n");
  }

  if(thread0()) {
    print("******************B Matrix*********************\n");
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCrB : "); print(tCrB); print("\n");
  }

  if(thread0()) {
    print("******************C Matrix*********************\n");
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif
  
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
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t* producer_mbar = smem.tma_barrier;
  uint64_t* consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA
  using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::init(&producer_mbar[pipe],   1);
      ConsumerBarType::init(&consumer_mbar[pipe], 128);
    }
  }
  // Ensure barrier init is complete on all CTAs
  cluster_sync();

  // Total number of k-tiles
  int k_tile_max = size<1>(tAgA);

  // Start async loads for valid pipes only
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
  {
    if ((warp_idx == 0) && lane_predicate)
    {
      if (k_tile < k_tile_max) {
        // Set expected Tx Bytes and issue TMA
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
        copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
        copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
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
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();             // TMA writes
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();             // MMA  reads
  auto tCrB_reshaped = make_tensor(
    tCrB.data(),
    make_layout(
        make_shape(_2{}, _1{}, _1{}, K_PIPE_MAX),  // 添加 pipeline 维度
        make_stride(_2{}, _0{}, _0{}, _256{})
    )
  );
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
    warpgroup_fence_operand(tCrC);
    warpgroup_fence_operand(tCrA);
    auto tCrB_k = tCrB_reshaped(_,_,_,read_pipe);
    cute::gemm(mma, tCrA, tCrB_k, tCrC);     // (V,M) x (V,N) => (V,M,N)
    warpgroup_commit_batch();
    // Wait for all MMAs in a K_TILE to complete
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    
    // Notify that consumption is done
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    ++read_state;

    if ((warp_idx == 0) && lane_predicate)
    {
      int pipe = write_state.index();
      // Wait for Consumer to complete consumption
      ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
      // Set expected Tx Bytes after each reset / init
      if (k_tile < k_tile_max) {
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
        copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
        copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
      } else {
        // No more tiles - expect 0 bytes so barrier completes immediately
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], 0);
      }
      ++write_state;
    }
    --k_tile_count;
    ++k_tile;
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
  auto prob_shape = make_shape(m, a_k, n, b_k);                     // (M, A_K, N, B_K)

  auto a_cta_tiler = make_shape(_64{}, _64{}, _16{});                   // (BLK_M, BLK_N, BLK_K)
  auto b_cta_tiler = make_shape(_64{}, _64{}, _64{});                   // (BLK_M, BLK_N, BLK_K)
  auto c_cta_tiler = make_shape(_64{}, _64{}, _64{});                   // (BLK_M, BLK_N)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  auto bM = Int<64>{};
  auto bN = Int<64>{};
  auto bKA = Int<16>{};
  auto bKB = Int<64>{};
  auto bP = Int<1>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_INTER_Atom<TA>{}, make_shape(bM, bKA, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW64_Atom<TB>{}, make_shape(bN, bKB, bP));
  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));

  // Define the TMA atoms
  Tensor mA = make_tensor(A, make_shape(m, a_k), dA);
  Tensor mB = make_tensor(B, make_shape(n, b_k), dB);

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM, bKA));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN, bKB));

  Copy_Atom<SM75_U32x2_LDSM_N, TA> s2r_atom_a;
  TiledMMA tiled_mma_a = make_tiled_mma(SM90_64x64x64_F16I2E4M3_RS_TN_A<GMMA::ScaleIn::One, GMMA::ScaleIn::One>{});
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x64_F16I2E4M3_RS_TN<GMMA::ScaleIn::One, GMMA::ScaleIn::One>{});

#if 0
  print("  tmaA : "); print(  tmaA); print("\n");
  print("  tmaB : "); print(  tmaB); print("\n");
  print("  s2r_atom_a : "); print(  s2r_atom_a); print("\n");
  print("  tiled_mma : "); print(  tiled_mma); print("\n");
  print("  sA : "); print(  sA); print("\n");
  print("  sB : "); print(  sB); print("\n");
#endif

  //
  // Setup and Launch
  //

  // Launch parameter setup
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(1, 1, 1);
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
  // int m = 64;
  // int n = 64;
  // int b_k = 64*8;
  // int a_k = 64*8/4;
  

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

  const int timing_iterations = 100;
  GPU_Clock timer;

  int ldA = a_k, ldB = b_k, ldC = m;


  // Run once
  d_C = h_C;
  gemm_tn(m, n, a_k, b_k,
       alpha,
       d_A.data().get(), ldA,
       d_B.data().get(), ldB,
       beta,
       d_C.data().get(), ldC);
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm_tn(m, n, a_k, b_k,
         alpha,
         d_A.data().get(), ldA,
         d_B.data().get(), ldB,
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
