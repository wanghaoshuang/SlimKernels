#pragma once

#include <cute/config.hpp>
#include <cute/util/type_traits.hpp>
#include <cute/algorithm/functional.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/gemm.hpp>

namespace cute
{
// Dispatch [5]: (V,M,K) x (V,N,K) => (V,M,N)
// for w2a8 kernel
template <class MMA,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout,
          __CUTE_REQUIRES(DLayout::rank == 3 && is_rmem<TD>::value &&
                          ALayout::rank == 3 && is_rmem<TA>::value &&
                          BLayout::rank == 3 && is_smem<TB>::value &&
                          CLayout::rank == 3 && is_rmem<TC>::value)>
CUTE_HOST_DEVICE
void
gemm(MMA_Atom<MMA>       const& mma,
     Tensor<TD, DLayout>      & D,  // (V,M,N) Logical data
     Tensor<TA, ALayout> const& A,  // (V,M,K) Logical data
     Tensor<TB, BLayout> const& B,  // (V,N,K) Logical data
     Tensor<TC, CLayout> const& C)  // (V,M,N) Logical data
{
  // tCrA : ptr[8b](0x7f9d94fffbc0) o ((_4,_2),_2,_2):((_1,_4),_8,_16)
  // tCrB : GMMA::DescriptorIterator o (_2,_2,_2):(_512,_64,_256)
  // tCrC : ptr[16b](0x7f9d94fffbe0) o ((_2,_2,_8),_2,_2):((_1,_2,_4),_32,_64)

  CUTE_STATIC_ASSERT_V(size<1>(A) == size<1>(C));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<1>(B) == size<2>(C));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<2>(A) == size<2>(B));  // AK == BK
  CUTE_STATIC_ASSERT_V(size<0>(C) == size<0>(D) && size<1>(C) == size<1>(D) && size<2>(C) == size<2>(D));

  auto rA = MMA_Atom<MMA>::make_fragment_A(A);
  auto rB = MMA_Atom<MMA>::make_fragment_B(B);

  auto K = size<2>(A);

  CUTE_UNROLL
  for (int k = 0; k < K; ++k)
  {
    // if(thread0()) {
    //     print("k: "); print(k); print("\n");
    // }
    copy(A(_,_,k), rA(_,_,k));
    copy(B(_,_,k), rB(_,_,k));
    // Thread-level register gemm for k
    gemm(mma, D, rA(_,_,k), rB(_,_,k), C);
  }
}

} // namespace cute