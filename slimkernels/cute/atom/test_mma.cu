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

int main(int argc, char** argv)
{
    TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x64_F16I2E4M3_RS_TN<GMMA::ScaleIn::One, GMMA::ScaleIn::One>{});
    print("  tiled_mma : "); print(  tiled_mma); print("\n");
}