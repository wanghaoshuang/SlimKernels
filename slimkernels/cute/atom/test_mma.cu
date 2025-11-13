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
#include "mma_traits_sm90_gmma.hpp"

using namespace cute;
using TA = cute::float_e4m3_t;
using TB = cute::float_e4m3_t;

template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
};

template<typename ASmemLayout, typename BSmemLayout, typename TiledMMA, typename TiledMMA_A, typename S2RAtomA>
__global__ void my_mma_kernel(TiledMMA mma, TiledMMA_A mma_A, S2RAtomA s2r_atom_a) {
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), BSmemLayout{}); // (BLK_M,BLK_K,PIPE)


    ThrMMA thr_mma_A = mma_A.get_slice(0);
    Tensor tCsA = thr_mma_A.partition_A(sA);                           // (MMA,MMA_M,MMA_K,PIPE)
    // Allocate registers for pipelining
    // Tensor tCrA = thr_mma.make_fragment_A(tCsA);                   // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrA = thr_mma_A.partition_fragment_A(sA(_,_,0));           // (MMA,MMA_M,MMA_K)
    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma_A);
    ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                    // (CPY,MMA_M,MMA_K,PIPE)
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                     // (CPY,MMA_M,MMA_K)
    if(thread0()) {
      // print("  mma : "); print(  mma_A); print("\n");
      print("  s2r_atom_a : "); print(  s2r_atom_a); print("\n");
      // print("  s2r_copy_a : "); print(  s2r_copy_a); print("\n");
      print("  s2r_thr_copy_a : "); print(  s2r_thr_copy_a); print("\n");
      print("  sA : "); print(  sA); print("\n");
      print("  tXsA : "); print(  tXsA); print("\n");
      print("  tXrA : "); print(  tXrA); print("\n");
    }

    ThrMMA thr_mma = mma.get_slice(0);    
    Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)
    
    // auto tCsB_0 = local_tile(tCsB, Shape<Shape<_64,_32>,_2,_2,Shape<_1,_1>>{}, make_coord(0,0,0,0));
    // auto tCsB_1 = local_tile(tCsB, Shape<Shape<_64,_32>,_2,_2,Shape<_1,_1>>{}, make_coord(1,0,0,0));   
    // Tensor tCrB_0 = thr_mma.make_fragment_B(tCsB_0);                         // (MMA,MMA_N,MMA_K,PIPE)
    // Tensor tCrB_1 = thr_mma.make_fragment_B(tCsB_1);                         // (MMA,MMA_N,MMA_K,PIPE)

    if(thread0()) {
      print("  sB : "); print(  sB); print("\n");
      print("  tCsB : "); print(  tCsB); print("\n");
      print("  tCrB : "); print(  tCrB); print("\n");
      // print("  tCsB_0 : "); print(  tCsB_0); print("\n");
      // print("  tCsB_1 : "); print(  tCsB_1); print("\n");
      // print("  tCrB_0 : "); print(  tCrB_0); print("\n");
      // print("  tCrB_1 : "); print(  tCrB_1); print("\n");
    }
}

int main(int argc, char** argv)
{
    
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<128>{};
    auto aK = Int<32>{};
    auto bP = Int<1>{};  // Pipeline
    TiledMMA tiled_mma_a = make_tiled_mma(SM90_64x64x64_F16I2E4M3_RS_TN_A<GMMA::ScaleIn::One, GMMA::ScaleIn::One>{});
    TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x64_F16I2E4M3_RS_TN<GMMA::ScaleIn::One, GMMA::ScaleIn::One>{});
    Copy_Atom<SM75_U32x2_LDSM_N, TB> s2r_atom_a;
    auto sA = tile_to_shape(GMMA::Layout_K_SW32_Atom<TA>{}, make_shape(bM,aK,bP));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));
    size_t shared_mem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
    int grid_size = 8;
    int block_size = size(tiled_mma);
    my_mma_kernel<decltype(sA), decltype(sB), decltype(tiled_mma), decltype(tiled_mma_a), decltype(s2r_atom_a)><<<grid_size, block_size, shared_mem_size>>>(tiled_mma, tiled_mma_a, s2r_atom_a);
    // Check the kernel launch for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();
    
}