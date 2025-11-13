

#pragma once

#include <cute/layout.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cutlass/half.h>
#include <cutlass/float8.h>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

#include "mma_sm90_gmma.hpp"

namespace cute {
////////////////////////////////////////////////////////////////////////////////////////////////////

// Tag types to distinguish different configurations
struct Tag_K16 {};
struct Tag_K64 {};

// 使用继承来创建不同的类型，同时保持与底层类型的兼容性
template <
    GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
    GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x64x64_F16I2E4M3_RS_TN_A 
    : public SM90::GMMA::MMA_64x64x64_F16I2E4M3_RS_TN<scaleA, scaleB>, 
      public Tag_K16
{
    constexpr SM90_64x64x64_F16I2E4M3_RS_TN_A() = default;
};

template <
    GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
    GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x64x64_F16I2E4M3_RS_TN 
    : public SM90::GMMA::MMA_64x64x64_F16I2E4M3_RS_TN<scaleA, scaleB>, 
      public Tag_K64
{
    constexpr SM90_64x64x64_F16I2E4M3_RS_TN() = default;
};

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x64_F16I2E4M3_RS_TN_A<scaleA, scaleB>>
{
    using ValTypeD = half_t;
    using ValTypeA = uint8_t;
    using ValTypeB = float_e4m3_t;
    using ValTypeC = half_t;

    using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;
    using Shape_MNK = Shape<_64, _64, _16>;
    using ThrID = Layout<_128>;
    using ALayout = GMMA::ALayout_64x16;
    using BLayout = GMMA::ABLayout<64, 16>;  // 注意：这里应该是 16，对应 K=16
    using CLayout = GMMA::CLayout_64x64;

    GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x64_F16I2E4M3_RS_TN<scaleA, scaleB>>
{
    using ValTypeD = half_t;
    using ValTypeA = uint8_t;
    using ValTypeB = float_e4m3_t;
    using ValTypeC = half_t;

    using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;
    using Shape_MNK = Shape<_64, _64, _32>;
    using ThrID = Layout<_128>;
    using ALayout = GMMA::ALayout_64x16;
    using BLayout = GMMA::ABLayout<64, 32>;  // 对应 K=64; used by mma.make_fragment_B(), so it should be actual shape of wgmma ptx instruction
    using CLayout = GMMA::CLayout_64x64;

    GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
}