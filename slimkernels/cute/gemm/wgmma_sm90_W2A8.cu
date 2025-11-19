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

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "../atom/mma_traits_sm90_gmma.hpp"

using namespace cute;

template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape, class ACtaTiler,class BCtaTiler,class CCtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class TiledMma, class TiledMmaA,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, ACtaTiler a_cta_tiler, BCtaTiler b_cta_tiler, CCtaTiler c_cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, TiledMma mma, TiledMmaA mma_a,
            Alpha alpha, Beta beta)
{
  // Preconditions

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,1>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<2,3>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,2>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, a_cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, b_cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, c_cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), BSmemLayout{}); // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles across the threads
  //

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor sA_ = as_position_independent_swizzle_tensor(sA);
  Tensor tAsA = thr_copy_a.partition_D(sA_);                           // (CPY,CPY_M,CPY_K,PIPE)
#if 0
  if(thread0()) {
    print("gA : "); print(gA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("thr_copy_a : "); print(thr_copy_a); print("\n");
    print("copy_a : "); print(copy_a); print("\n");
  }
#endif


  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor sB_ = as_position_independent_swizzle_tensor(sB);
  Tensor tBsB = thr_copy_b.partition_D(sB_);                           // (CPY,CPY_N,CPY_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K


  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma_a = mma_a.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma_a.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrA = thr_mma_a.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
  
  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma_a);
  ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
  // Tensor tXrA = s2r_thr_copy_a.partition_D(sA(_,_,0));
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

  Tensor recast_tensor = recast<uint128_t>(tXsA);

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)

  // Allocate the accumulators -- same size as the projected data
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  

#if 0
  CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCsA)));              // MMA_M
  CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCsB)));              // MMA_N
  CUTE_STATIC_ASSERT_V((size<2>(tCsA) == size<2>(tCsB)));              // MMA_K



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
    print("recast_tensor : "); print(recast_tensor); print("\n");
  }
#endif
#if 0
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


#if 0
  // Total number of k-tiles
  auto K_TILE_MAX  = size<3>(tAgA);
  // Number of pipelined k-tiles in smem
  auto K_PIPE_MAX  = size<3>(tAsA);
  // Prefetch all but the last
  CUTE_UNROLL
  for (int k = 0; k < K_PIPE_MAX-1; ++k)
  {
    copy(copy_a, tAgA(_,_,_,k), tAsA(_,_,_,k));
    copy(copy_b, tBgB(_,_,_,k), tBsB(_,_,_,k));
    cp_async_fence(); // cp.async.commit_group;
  }

  // Clear the accumulators
  clear(tCrC);

  __syncthreads();

  int k_pipe_read  = 0;
  int k_pipe_write = K_PIPE_MAX-1;

  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    int k_tile_next = k_tile + (K_PIPE_MAX-1);
    k_tile_next = (k_tile_next >= K_TILE_MAX) ? K_TILE_MAX-1 : k_tile_next; 

    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe_write));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe_write));
    cp_async_fence(); // cp.async.commit_group;
    // Advance k_pipe_write
    ++k_pipe_write;
    k_pipe_write = (k_pipe_write == K_PIPE_MAX) ? 0 : k_pipe_write;

    // Wait on all cp.async -- optimize by pipelining to overlap GMEM reads
    cp_async_wait<0>();
    copy(s2r_atom_a, tXsA(_,_,_, k_pipe_read), tXrA);
    warpgroup_fence_operand(tCrC);
    warpgroup_fence_operand(tCrA);

    warpgroup_arrive(); // wgmma.fence.sync.aligned;
    auto tCrB_k = tCrB(_,_,_,k_pipe_read);
    auto tCrB_reshaped = make_tensor(
      tCrB_k.data(),
      make_layout(
          Shape<_2, _2, _2, Shape<_1, _3>>{},
          Stride<_512, _64, _256, Stride<_0, _1024>>{}
      )
    );
    cute::gemm(mma, tCrA, tCrB_reshaped, tCrC);
    warpgroup_commit_batch(); // wgmma.commit_group.sync.aligned;
    // Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
    warpgroup_wait<0>(); // wgmma.wait_group.sync.aligned %0
    warpgroup_fence_operand(tCrC);
    warpgroup_fence_operand(tCrA);

    // Advance k_pipe_read
    ++k_pipe_read;
    k_pipe_read = (k_pipe_read == K_PIPE_MAX) ? 0 : k_pipe_read;
  }

#endif

  //
  // Epilogue
  //

  // axpby(alpha, tCrC, beta, tCgC);
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
  auto a_cta_tiler = make_shape(_128{}, _128{}, _32{});                   // (BLK_M, BLK_N, BLK_K)
  auto b_cta_tiler = make_shape(_128{}, _128{}, _128{});                   // (BLK_M, BLK_N, BLK_K)
  auto c_cta_tiler = make_shape(_128{}, _128{}, _128{});                   // (BLK_M, BLK_N)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_INTER_Atom<TA>{}, make_shape(size<0>(a_cta_tiler),size<2>(a_cta_tiler),bP));
  auto sB = tile_to_shape(GMMA::Layout_K_INTER_Atom<TB>{}, make_shape(size<1>(b_cta_tiler),size<2>(b_cta_tiler),bP));
  size_t smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));

  // Define the thread layouts (static)
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                    Layout<Shape<_64,_2>,Stride<_2,_1>>{}, // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_16>>{});              // Val layout  1x8
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                    Layout<Shape<_32,_4>,Stride<_4,_1>>{}, // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_16>>{});              // Val layout  1x8

  Copy_Atom<SM75_U32x2_LDSM_N, TA> s2r_atom_a;
  // Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, TA> s2r_atom_a;
  TiledMMA tiled_mma_a = make_tiled_mma(SM90_64x64x64_F16I2E4M3_RS_TN_A<GMMA::ScaleIn::One, GMMA::ScaleIn::One>{});
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x64_F16I2E4M3_RS_TN<GMMA::ScaleIn::One, GMMA::ScaleIn::One>{});

#if 0
  print("  copyA : "); print(  copyA); print("\n");
  print("  s2r_atom_a : "); print(  s2r_atom_a); print("\n");
  print("  copyB : "); print(  copyB); print("\n");
  print("  tiled_mma : "); print(  tiled_mma); print("\n");
  print("  sA : "); print(  sA); print("\n");
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
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
                                           TA, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_a),
                                           TB, decltype(dB), decltype(sB), decltype(copyB),
                                           TC, decltype(dC), decltype(tiled_mma), decltype(tiled_mma_a),
                                           decltype(alpha), decltype(beta)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, a_cta_tiler, b_cta_tiler, c_cta_tiler,
                                                             A, dA, sA, copyA, s2r_atom_a,
                                                             B, dB, sB, copyB,
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

  // int m = 512;
  // int n = 512;
  // int a_k = 128;
  // int b_k = 512;

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
