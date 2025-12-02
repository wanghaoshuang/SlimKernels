
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
          class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
  
  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
};

// Forward declaration of gemm_device
template <class ProblemShape, class ACtaTiler, class BCtaTiler,
          class TA, class ASmemLayout, class TmaA,
          class TB, class BSmemLayout, class TmaB>
__global__ static
__launch_bounds__(128)
void
gemm_device(ProblemShape shape_MNK, ACtaTiler a_cta_tiler, BCtaTiler b_cta_tiler,
            TA const* A, TmaA const tma_a,
            TB const* B, TmaB const tma_b);
// gemm_device(ProblemShape shape_MNK, ACtaTiler a_cta_tiler, BCtaTiler b_cta_tiler,
//             TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
//             TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b);

// Setup params for testing A and B matrix copy using TMA
template <class TA, class TB>
void
test_copy_ab(int m, int n, int a_k, int b_k, 
             TA const* A,
             TB const* B, 
             cudaStream_t stream = 0)
{
  auto prob_shape = make_shape(m, a_k, n, b_k);                     // (M, A_K, N, B_K)
  auto a_cta_tiler = make_shape(_64{}, _64{}, _16{});              // (BLK_M, BLK_N, BLK_K)
  auto b_cta_tiler = make_shape(_64{}, _64{}, _64{});              // (BLK_M, BLK_N, BLK_K)
  
  auto bM = Int<64>{};
  auto bN = Int<64>{};
  auto bKA = Int<16>{};
  auto bKB = Int<64>{};
  auto bP = Int<2>{};  // Pipeline
  
  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_INTER_Atom<TA>{}, make_shape(_64{}, _16{}, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW64_Atom<TB>{}, make_shape(_64{}, _64{}, bP));
  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  
  // Define the TMA atoms
  // A: packed 64x16 tile layout
  auto gA_layout = tile_to_shape(Layout<Shape<_64, _16>, Stride<_16, _1>>{}, make_shape(m, a_k));
  Tensor mA = make_tensor(A, gA_layout);
  // Tensor mA = make_tensor(A, make_shape(m, a_k), make_stride(a_k, Int<1>{}));
  // B: packed 64x64 tile layout
  auto gB_layout = tile_to_shape(Layout<Shape<_64, _64>, Stride<_64, _1>>{}, make_shape(n, b_k));
  Tensor mB = make_tensor(B, gB_layout);
  // Tensor mB = make_tensor(B, make_shape(n, b_k), make_stride(b_k, Int<1>{}));
  
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(make_shape(_64{}, _1{}), make_shape(_16{}, _1{})));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(make_shape(_64{}, _1{}), make_shape(_64{}, _1{})));

  // Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0));
  // Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0));
  
  // Launch parameter setup
  dim3 dimBlock(128);
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, size<0>(a_cta_tiler))), dimCluster.x),
               round_up(size(ceil_div(n, size<0>(b_cta_tiler))), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device<decltype(prob_shape), decltype(a_cta_tiler), decltype(b_cta_tiler),
                                           TA, decltype(sA), decltype(tmaA),
                                           TB, decltype(sB), decltype(tmaB)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  gemm_device<decltype(prob_shape), decltype(a_cta_tiler), decltype(b_cta_tiler),
            TA, decltype(sA), decltype(tmaA),
            TB, decltype(sB), decltype(tmaB)>
    <<<dimGrid, dimBlock, smem_size, stream>>>(
        prob_shape, a_cta_tiler, b_cta_tiler,
        A, tmaA, B, tmaB);
  // Kernel Launch using cluster launch
  // cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
  //                                                            prob_shape, a_cta_tiler, b_cta_tiler,
  //                                                            A, tmaA,
  //                                                            B, tmaB);
  cutlass::Status status = cutlass::Status::kSuccess;
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}


template <class ProblemShape, class ACtaTiler, class BCtaTiler,
          class TA, class ASmemLayout, class TmaA,
          class TB, class BSmemLayout, class TmaB>
__global__ static
__launch_bounds__(128)
void
gemm_device(ProblemShape shape_MNK, ACtaTiler a_cta_tiler, BCtaTiler b_cta_tiler,
            TA const* A, TmaA const tma_a,
            TB const* B, TmaB const tma_b)
// gemm_device(ProblemShape shape_MNK, ACtaTiler a_cta_tiler, BCtaTiler b_cta_tiler,
//             TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
//             TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b)
{
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<4>{});  // (M, A_K, N, B_K)
  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);

  // // Represent the full tensors using TMA descriptors
  // auto [M, A_K, N, B_K] = shape_MNK;
  // auto gA_layout = tile_to_shape(Layout<Shape<_64, _16>, Stride<_16, _1>>{}, make_shape(M, A_K));
  // Tensor mA = tma_a.get_tma_tensor(gA_layout.shape());
  // auto gB_layout = tile_to_shape(Layout<Shape<_64, _64>, Stride<_64, _1>>{}, make_shape(N, B_K));
  // Tensor mB = tma_b.get_tma_tensor(gB_layout.shape());
  
  // // Get the appropriate blocks for this thread block
  // auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  // Tensor gA = local_tile(mA, a_cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
  // Tensor gB = local_tile(mB, b_cta_tiler, cta_coord, Step< X, _1, _1>{});  // (BLK_N,BLK_K,k)
  // // Shared memory tensors
  // extern __shared__ char shared_memory[];
  // using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  // SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  // Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{}); // (BLK_M,BLK_K,PIPE)
  // Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), BSmemLayout{}); // (BLK_N,BLK_K,PIPE)
  
  // //
  // // Partition the copying of A and B tiles using TMA
  // //
  // auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
  //                                   group_modes<0,2>(sA), group_modes<0,2>(gA));  // (TMA,k) and (TMA,PIPE)

  // auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
  //                                   group_modes<0,2>(sB), group_modes<0,2>(gB));  // (TMA,k) and (TMA,PIPE)

  // // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
  // constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)));
                                      // + sizeof(make_tensor_like(tensor<0>(tBsB)));

  

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
  
  // if(thread0()) {
  //   print("******************B Matrix (TMA)*********************\n");
  //   print("  mB : "); print(  mB); print("\n");
  //   print("  gB : "); print(  gB); print("\n");
  //   print("  sB : "); print(  sB); print("\n");
  //   print("tBgB : "); print(tBgB); print("\n");
  //   print("tBsB : "); print(tBsB); print("\n");
  // }

#if 0
  int warp_idx = cutlass::canonical_warp_idx_sync();  // 添加这行
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
      // copy(tma_b.with(tma_mbar[0]), tBgB(_,0), tBsB(_,0));
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
    int n = 5120;
    int b_k = 4096;
    int a_k = 4096/4;

    using TA = cute::uint8_t;
    using TB = cute::float_e4m3_t;
    
    thrust::host_vector<TA> h_A(m*a_k);
    thrust::host_vector<TB> h_B(n*b_k);
    
    // Initialize the A and B tensors
    for (int j = 0; j < m*a_k; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1));
    for (int j = 0; j < n*b_k; ++j) h_B[j] = TB(int((rand() % 2) ? 1 : -1));
    
    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    for (int i=0;i<10;i++){
      // Run TMA copy test with packed data
      test_copy_ab(m, n, a_k, b_k, d_A.data().get(), d_B.data().get());
      CUTE_CHECK_LAST();
    }   
    
    printf("TMA copy test completed successfully!\n");

    return 0;
}
