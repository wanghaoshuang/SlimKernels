#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "gemm.h"

#include <iostream>
#include <functional>

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = float;          // <- data type of elements in input matrix A
using ElementInputB = float;          // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

const int  THREADS_PER_WARP=32;

struct GemmConfig {
    static constexpr int BLOCK_DIM_M = 128;
    static constexpr int BLOCK_DIM_N = 128;
    static constexpr int WARP_TILE_DIM_M = 64;
    static constexpr int WARP_TILE_DIM_N = 64;
    static constexpr int THREAD_NUM = 128;
    static constexpr int TENSOR_CORE_M = 16;
    static constexpr int TENSOR_CORE_N = 8;
    static constexpr int TENSOR_CORE_K = 8;
};


template<typename GemmConfig>
struct Index{

    constexpr int BLOCK_DIM_M = GemmConfig::BLOCK_DIM_M;
    constexpr int BLOCK_DIM_N = GemmConfig::BLOCK_DIM_N;
    constexpr int WARP_TILE_DIM_M = GemmConfig::WARP_TILE_DIM_M;
    constexpr int WARP_TILE_DIM_N = GemmConfig::WARP_TILE_DIM_N;
    constexpr int THREAD_NUM = GemmConfig::THREAD_NUM;
    constexpr int TENSOR_CORE_M = GemmConfig::TENSOR_CORE_M;
    constexpr int TENSOR_CORE_N = GemmConfig::TENSOR_CORE_N;
    constexpr int TENSOR_CORE_K = GemmConfig::TENSOR_CORE_K;
    

    const int warpTilesCountM = BLOCK_DIM_M / WARP_TILE_DIM_M;
    const int warpTilesCountN = BLOCK_DIM_N / WARP_TILE_DIM_N;
    static const int threadTilesCountM = WARP_TILE_DIM_M / TENSOR_CORE_M;
    static const int threadTilesCountN = WARP_TILE_DIM_N / TENSOR_CORE_N;
    
    int blockIdx_x,blockIdx_y;
    int warpIdx;
    int laneIdx;
    int warpTileIdxM;
    int warpTileIdxN;

    // Define the address used by ldmatrix instruction to load A and B from shared memory to registers.
    int a[threadTilesCountM];
    int b[threadTilesCountN];

    __device__ Index(){
        warpIdx = threadIdx.x / THREADS_PER_WARP;
        laneIdx = threadIdx.x % THREADS_PER_WARP;
        warpTileIdxM = warpIdx / warpTilesCountN;
        warpTileIdxN = warpIdx % warpTilesCountM;

        // blockIdx_x = blockIdx.x >> logtile;
        // blockIdx_y = (blockIdx.y << logtile) + ((blockIdx.x) & ((1 << (logtile)) - 1));
        blockIdx_x = blockIdx.x;
        blockIdx_y = blockIdx.y;
        
        // for ldmatrix.sync.aligned.x4.m8n8.shared.b16
        int swizzle_offset = (((laneIdx>>2)^(laneIdx>>4))&1)*4;
        for(int i=0;i<threadTilesCountM;i++){
            // a[k][i] = (rowwarp*64+i*16+laneidx%16)*8+(((laneidx>>2)^(laneidx>>4))&1)*4+k*128*8;
            a[i] = (warpTileIdxM*WARP_TILE_DIM_M+i*TENSOR_CORE_M+laneIdx%TENSOR_CORE_M)*TENSOR_CORE_N + swizzle_offset;
        }

        swizzle_offset = (((laneIdx>>2)^(laneIdx>>3))&1)*4;
        for(int i=0;i<threadTilesCountN;i++){
            b[i] = (warpTileIdxM*WARP_TILE_DIM_N+i*TENSOR_CORE_N+laneIdx%TENSOR_CORE_N)*TENSOR_CORE_N + swizzle_offset;
        }
        
    }
};

template<typename Config>
__device__ void loadtileC(MMAarguments &arg,ElementOutput *C_fragment){
    // iter = 128 * 64 / 128
    const int c_num_per_thread = Config::BLOCK_DIM_M * Config::BLOCK_DIM_N / Config::THREAD_NUM;
    for(int i=0;i<c_num_per_thread;i++){
        int tileIdx = threadIdx.x*c_num_per_thread + i;
        int rowC = tileIdx / Config::BLOCK_DIM_N + blockIdx.y * Config::BLOCK_DIM_M;
        int colC = tileIdx % Config::BLOCK_DIM_N + blockIdx.x * Config::BLOCK_DIM_N;
        C_fragment[i] =  rowC<arg.problem_size.m()&&colC<arg.problem_size.n() ? arg.C[rowC*arg.problem_size.n()+colC] : ElementOutput(0);
    }
}

template<typename Config>
__device__ void loadtileA(MMAarguments &arg,ElementInputA *A,int idx){
    // iter = 128 * 8 / 128
    const int a_num_per_thread = Config::BLOCK_DIM_M * Config::TENSOR_CORE_K / Config::THREAD_NUM;
    for(int i=0;i<a_num_per_thread;i++){
        int tileIdx = ((threadIdx.x/32)*(32*a_num_per_thread)+ (threadIdx.x % 32) + i*32)%(Config::BLOCK_DIM_M * Config::TENSOR_CORE_K);
        int rowA = tileIdx/Config::TENSOR_CORE_K + blockIdx.y * Config::BLOCK_DIM_M;
        int colA = tileIdx%Config::TENSOR_CORE_K + idx*Config::TENSOR_CORE_K; 

        A[tileIdx] = rowA<arg.problem_size.m()&&colA<arg.problem_size.k() ? arg.A[rowA*arg.problem_size.k()+colA] : ElementInputA(0);
    }
}

template<typename Config>
__device__ void loadtileB(MMAarguments &arg,ElementInputB *B,int idx){
    // iter = 64 * 8 / 128
    const int b_num_per_thread = Config::BLOCK_DIM_N * Config::TENSOR_CORE_K / Config::THREAD_NUM;
    for(int i=0;i<b_num_per_thread;i++){
        int tileIdx = ((threadIdx.x/32)*(32*b_num_per_thread)+ (threadIdx.x % 32) + i*32)%(Config::BLOCK_DIM_N * Config::TENSOR_CORE_K);
        int rowB = idx*Config::TENSOR_CORE_K + tileIdx%Config::TENSOR_CORE_K;
        int colB = blockIdx.x*Config::BLOCK_DIM_N + tileIdx/Config::TENSOR_CORE_K;

        B[tileIdx] = rowB<arg.problem_size.k()&&colB<arg.problem_size.n() ? arg.B[colB*arg.problem_size.k()+rowB] : ElementInputB(0);
    }
}

template<typename Config>
__device__ void storetile(MMAarguments &arg,ElementOutput *D){
    // iter = 128 * 64 / 128
    const int d_num_per_thread = Config::BLOCK_DIM_M * Config::BLOCK_DIM_N / Config::THREAD_NUM;
    for(int i=0;i<d_num_per_thread;i++){
        int tileIdx = threadIdx.x*d_num_per_thread + i;
        int rowD = tileIdx / Config::BLOCK_DIM_N + blockIdx.y * Config::BLOCK_DIM_M;
        int colD = tileIdx % Config::BLOCK_DIM_N + blockIdx.x * Config::BLOCK_DIM_N;

        if(rowD<arg.problem_size.m()&&colD<arg.problem_size.n()){
            arg.D[rowD*arg.problem_size.n()+colD] = D[i];
        }
    }
}

template<typename Config>
__device__ void mma_tile(MMAarguments &arg,
                        ElementInputA *A,
                        ElementInputB *B,
                        ElementAccumulator *C,
                        ElementOutput *D,
                        Index<Config> &index){

    ElementInputA a[4];
    ElementInputB b[2];

    for(int rowtile=0;rowtile<index.threadTilesCountM;rowtile++){
        for(int coltile=0;coltile<index.threadTilesCountN;coltile++){
            int tileidx = rowtile*index.threadTilesCountN+coltile;
            asm volatile(
                    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" 
                    : 
                    "=r"(*(uint32_t*)&a[0]), 
                    "=r"(*(uint32_t*)&a[1]), 
                    "=r"(*(uint32_t*)&a[2]), 
                    "=r"(*(uint32_t*)&a[3])
                    : 
                    "r"((uint32_t)__cvta_generic_to_shared(&A[index.a[rowtile]])) 
                );

            asm volatile(
                    "ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {%0, %1}, [%2];" 
                    : 
                    "=r"(*(uint32_t*)&b[0]), 
                    "=r"(*(uint32_t*)&b[1])
                    : 
                    "r"((uint32_t)__cvta_generic_to_shared(&B[index.b[coltile]])) 
                );

            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : 
                "=f"(C[tileidx*4+0]),  // D[0]
                "=f"(C[tileidx*4+1]),  // D[1]
                "=f"(C[tileidx*4+2]),  // D[2]
                "=f"(C[tileidx*4+3])   // D[3]
                : 
                "r"(*reinterpret_cast<uint32_t const *>(&a[0])),
                "r"(*reinterpret_cast<uint32_t const *>(&a[1])),
                "r"(*reinterpret_cast<uint32_t const *>(&a[2])),
                "r"(*reinterpret_cast<uint32_t const *>(&a[3])),
                "r"(*reinterpret_cast<uint32_t const *>(&b[0])),
                "r"(*reinterpret_cast<uint32_t const *>(&b[1])),
                "f"(C[tileidx*4+0]),   // C[0]
                "f"(C[tileidx*4+1]),   // C[1]
                "f"(C[tileidx*4+2]),   // C[2]
                "f"(C[tileidx*4+3])    // C[3]
            );
        }
    }
}

// threadblockShape 128 64 8
// warpShape 64 32 8
// every block has 4 warps

template<typename Config>
__global__ void GEMM_MMA(MMAarguments arg){
    __shared__ ElementInputA tileA[3][Config::BLOCK_DIM_M*Config::TENSOR_CORE_K]; // 3*128*8
    __shared__ ElementInputB tileB[3][Config::TENSOR_CORE_K*Config::BLOCK_DIM_N]; // 3*8*64
    // __shared__ ElementOutput tileC[128*64];
    ElementOutput C_fragment[Config::BLOCK_DIM_M * Config::BLOCK_DIM_N / Config::THREAD_NUM]; // 128*64/128
    const int iters = (arg.problem_size.k() + Config::TENSOR_CORE_K - 1) / Config::TENSOR_CORE_K;
    struct Index<Config> index;

    loadtileC<Config>(arg,C_fragment);

    loadtileA<Config>(arg,tileA[0],0);
    loadtileB<Config>(arg,tileB[0],0);
    asm("cp.async.commit_group;\n"::);

    loadtileA<Config>(arg,tileA[1],1);
    loadtileB<Config>(arg,tileB[1],1);
    asm("cp.async.commit_group;\n"::);
    
    for(int i=0;i<iters;i++){
        asm("cp.async.wait_group 1;\n"::);
        __syncthreads();
        mma_tile<Config>(arg,tileA[i%3],tileB[i%3],C_fragment,C_fragment, index);
        if (i+2<iters){
            loadtileA<Config>(arg,tileA[(i+2)%3],i+2);
            loadtileB<Config>(arg,tileB[(i+2)%3],i+2);
            asm("cp.async.commit_group;\n"::);
        }
    }

    storetile<Config>(arg,C_fragment);
}

void launch_GEMM_MMA(MMAarguments &arg){
    dim3 grid,block;

    grid.x = (arg.problem_size.n()+BLOCK_SHAPE_N-1)/BLOCK_SHAPE_N;
    grid.y = (arg.problem_size.m()+BLOCK_SHAPE_M-1)/BLOCK_SHAPE_M;
    grid.z = 1;

    block.x = THREAD_NUM;
    block.y = 1;
    block.z = 1;

    GEMM_MMA<GemmConfig><<<grid,block>>>(arg);
}