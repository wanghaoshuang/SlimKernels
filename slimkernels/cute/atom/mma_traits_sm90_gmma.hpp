

#pragma once

#include <cute/layout.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cutlass/half.h>
#include <cutlass/float8.h>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

#include "mma_sm90_gmma.hpp"

namespace cute {
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
    GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
using SM90_64x64x64_F16I2E4M3_RS_TN = SM90::GMMA::MMA_64x64x64_F16I2E4M3_RS_TN<scaleA, scaleB>;

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB>
struct MMA_Traits<SM90_64x64x64_F16I2E4M3_RS_TN<scaleA, scaleB>>
{
    using ValTypeD = half_t;
    using ValTypeA = uint8_t;
    using ValTypeB = float_e4m3_t;
    using ValTypeC = half_t;

    using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

    using Shape_MNK = Shape<_64, _64, _64>;
    using ThrID = Layout<_128>;
    using ALayout = GMMA::ALayout_64x16;
    using BLayout = GMMA::ABLayout<64, 64>;
    using CLayout = GMMA::CLayout_64x64;

    GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
}