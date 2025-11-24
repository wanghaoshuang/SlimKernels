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

template <class ProblemShape, class ACtaTiler, class BCtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB>
__global__ static
__launch_bounds__(128)
void
gemm_device(ProblemShape shape_MNK, ACtaTiler a_cta_tiler, BCtaTiler b_cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b)
{

  // Represent the full tensors
  auto gA_layout = tile_to_shape(Layout<Shape<_64, _16>,Stride<_16,_1>>{}, select<0,1>(shape_MNK));
  Tensor mA = make_tensor(make_gmem_ptr(A), gA_layout); // (M,K)
  // Tensor mB = make_tensor(make_gmem_ptr(B), select<2,3>(shape_MNK), dB); // (N,K)
  auto gB_layout = tile_to_shape(Layout<Shape<_64, _64>,Stride<_64,_1>>{}, select<2,3>(shape_MNK));
  Tensor mB = make_tensor(make_gmem_ptr(B), gB_layout); 
  
  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, a_cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, b_cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  
  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), BSmemLayout{}); // (BLK_N,BLK_K,PIPE)
  
  // Partition the copying of A and B tiles across the threads
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor sA_ = as_position_independent_swizzle_tensor(sA);
  Tensor tAsA = thr_copy_a.partition_D(sA_);                           // (CPY,CPY_M,CPY_K,PIPE)
  
  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor sB_ = as_position_independent_swizzle_tensor(sB);
  Tensor tBsB = thr_copy_b.partition_D(sB_);                           // (CPY,CPY_N,CPY_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

  if(thread0()) {
    print("******************A Matrix*********************\n");
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
  }
  
  if(thread0()) {
    print("******************B Matrix*********************\n");
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
  }
  // Copy from global memory to shared memory
  copy(copy_b, tBgB(_,_,_,0), tBsB(_,_,_,0));
  copy(copy_a, tAgA(_,_,_,0), tAsA(_,_,_,0));
  cp_async_fence();
  cp_async_wait<0>();
}


// Setup params for testing A and B matrix copy
template <class TA, class TB>
void
test_copy_ab(int m, int n, int a_k, int b_k, 
             TA const* A, int ldA,
             TB const* B, int ldB, 
             cudaStream_t stream = 0)
{
  auto prob_shape = make_shape(m, a_k, n, b_k);                     // (M, A_K, N, B_K)
  auto a_cta_tiler = make_shape(_64{}, _64{}, _16{});              // (BLK_M, BLK_K, k)
  auto b_cta_tiler = make_shape(_64{}, _64{}, _64{});              // (BLK_N, BLK_K, k)
  
  // Define TN strides
  auto dA = make_stride(ldA, Int<1>{});                            // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                            // (dN, dK)
  
  auto bP = Int<2>{};  // Pipeline
  
  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_INTER_Atom<TA>{}, make_shape(size<0>(a_cta_tiler),size<2>(a_cta_tiler),bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW64_Atom<TB>{}, make_shape(size<0>(b_cta_tiler),size<2>(b_cta_tiler),bP));
  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  
  // Define the thread layouts (static)
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint64_t>, TA>{},
                                    Layout<Shape<_64,_2>,Stride<_2,_1>>{}, 
                                    Layout<Shape< _1,_8>>{});
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                    Layout<Shape<_32,_4>,Stride<_4,_1>>{},
                                    Layout<Shape< _1,_16>>{});
  
  dim3 dimBlock(128);
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, size<0>(a_cta_tiler))), dimCluster.x),
               round_up(size(ceil_div(n, size<0>(b_cta_tiler))), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device<decltype(prob_shape), decltype(a_cta_tiler), decltype(b_cta_tiler),
                                           TA, decltype(dA), decltype(sA), decltype(copyA),
                                           TB, decltype(dB), decltype(sB), decltype(copyB)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));
  
  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, a_cta_tiler, b_cta_tiler,
                                                             A, dA, sA, copyA,
                                                             B, dB, sB, copyB);
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
    
    int m = 4096;
    int n = 4096;
    int b_k = 4096;
    

    // Parse command line arguments
    if (argc > 1) {
        b_k = atoi(argv[1]);
        if (b_k <= 0) {
            std::cerr << "Error: b_k must be a positive integer" << std::endl;
            std::cerr << "Usage: " << argv[0] << " [b_k] [a_k]" << std::endl;
            return -1;
        }
    }
    
    int a_k = b_k/4;

    using TA = cute::uint8_t;
    using TB = cute::float_e4m3_t;
    
    thrust::host_vector<TA> h_A(m*a_k);
    thrust::host_vector<TB> h_B(n*b_k);
    
    // Initialize the A and B tensors
    for (int j = 0; j < m*a_k; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < n*b_k; ++j) h_B[j] = TB(int((rand() % 2) ? 1 : -1));
    
    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    
    int ldA = a_k;
    int ldB = b_k;
    
    // Run once
    test_copy_ab(m, n, a_k, b_k, 
                 d_A.data().get(), ldA,
                 d_B.data().get(), ldB);
    CUTE_CHECK_LAST();
    printf("A and B matrix copy test completed successfully\n");
    return 0;
}
