
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
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/device_kernel.h"
#include "../atom/mma_traits_sm90_gmma.hpp"

using namespace cute;

template <class ElementA,
          class SmemLayoutA>  // (M,K,P)
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  
  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
};

// Forward declaration of gemm_device
template <class ProblemShape, class ACtaTiler,
          class TA, class ASmemLayout, class TmaA>
__global__ static
__launch_bounds__(128)
void
gemm_device(ProblemShape shape_MNK, ACtaTiler a_cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a);

// Setup params for testing A matrix copy using TMA
template <class TA>
void
test_copy_a(int m, int a_k, 
            TA const* A,
            cudaStream_t stream = 0)
{
  auto prob_shape = make_shape(m, a_k);                             // (M, A_K)
  auto a_cta_tiler = make_shape(_64{}, _64{}, _128{});              // (BLK_M, BLK_N, BLK_K)
  
  auto bM = Int<64>{};
  auto bKA = Int<128>{};
  auto bP = Int<2>{};  // Pipeline
  
  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_INTER_Atom<TA>{}, make_shape(_64{}, _128{}, bP));
  int smem_size = int(sizeof(SharedStorage<TA, decltype(sA)>));
  
  // Define the TMA atoms
  // auto gA_layout = tile_to_shape(Layout<Shape<_64, _128>, Stride<_128, _1>>{}, make_shape(m, a_k));
  // Tensor mA = make_tensor(A, gA_layout);

  auto gA_layout = make_layout(make_shape(m, a_k), make_stride(a_k, Int<1>{}));
  Tensor mA = make_tensor(A, gA_layout);
  
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(_64{}, _128{}));
  
  // Launch parameter setup
  dim3 dimBlock(128);
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, size<0>(a_cta_tiler))), dimCluster.x), 1);
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device<decltype(prob_shape), decltype(a_cta_tiler),
                                           TA, decltype(sA), decltype(tmaA)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));
  // Kernel Launch using cluster launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, a_cta_tiler,
                                                             A, tmaA);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}


template <class ProblemShape, class ACtaTiler,
          class TA, class ASmemLayout, class TmaA>
__global__ static
__launch_bounds__(128)
void
gemm_device(ProblemShape shape_MK, ACtaTiler a_cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a)
{
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MK) == Int<2>{});  // (M, A_K)
  static_assert(is_static<ASmemLayout>::value);

  // Represent the full tensors using TMA descriptors
  auto [M, A_K] = shape_MK;
  // auto gA_layout = tile_to_shape(Layout<Shape<_64, _128>, Stride<_128, _1>>{}, make_shape(M, A_K));
  auto gA_layout = make_layout(make_shape(M, A_K), make_stride(A_K, Int<1>{}));
  Tensor mA = tma_a.get_tma_tensor(gA_layout.shape());
  
  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, a_cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
  
  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, ASmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{}); // (BLK_M,BLK_K,PIPE)
  
  //
  // Partition the copying of A tiles using TMA
  //
  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));  // (TMA,k) and (TMA,PIPE)

  // The TMA is responsible for copying everything in mode-0 of tAsA
  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)));

  // if(thread0() && block0()) {
  //   print("******************A Matrix (TMA)*********************\n");
  //   print("  gA_layout : "); print(  gA_layout); print("\n");
  //   print("  mA : "); print(  mA); print("\n");
  //   print("  gA : "); print(  gA); print("\n");
  //   print("  sA : "); print(  sA); print("\n");
  //   print("tAgA : "); print(tAgA); print("\n");
  //   print("tAsA : "); print(tAsA); print("\n");
  //   print("tma_transaction_bytes : "); print(tma_transaction_bytes); print("\n");
  //   print("gA tensor: ");cute::print_tensor(gA(_,_,0));print("\n");
  // }

#if 1
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t* tma_mbar = smem.tma_barrier;
  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
  if ((warp_idx==0) && lane_predicate) {
    ProducerBarType::init(&tma_mbar[0], 1);
  }
  cluster_sync();
  int k_tile_count = size<1>(tAgA);
  constexpr int K_PIPE_MAX = 1;  // 单级 pipeline
  auto state = cutlass::PipelineState<K_PIPE_MAX>();  // 状态对象
  for (int k=0; k< k_tile_count; ++k){
    if ((warp_idx==0) && lane_predicate) {
      ProducerBarType::arrive_and_expect_tx(&tma_mbar[0], tma_transaction_bytes);
      copy(tma_a.with(tma_mbar[0]), tAgA(_,k), tAsA(_,0));
    }
    // Wait for TMA to complete
    ProducerBarType::wait(&tma_mbar[0], state.phase());  // phase 0
    ++state;
  }
#endif
}

int main(int argc, char** argv)
{
    cudaDeviceProp props;
    int current_device_id;
    cudaGetDevice(&current_device_id);
    cudaGetDeviceProperties(&props, current_device_id);

    int m = 5120;
    int a_k = 4096/4;

    using TA = cute::uint8_t;
    
    thrust::host_vector<TA> h_A(m*a_k);
    
    // Initialize the A tensor
    for (int j = 0; j < m*a_k; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1));
    
    thrust::device_vector<TA> d_A = h_A;
    for (int i=0;i<100;i++){
      // Run TMA copy test with packed data
      test_copy_a(m, a_k, d_A.data().get());
      CUTE_CHECK_LAST();
    }   
    
    printf("TMA copy test completed successfully!\n");

    return 0;
}
