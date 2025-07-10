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

// Shape of MMA operator
const int TENSOR_CORE_M = 16;
const int TENSOR_CORE_N = 8;
const int TENSOR_CORE_K = 8;

template<int BLOCK_SHAPE_M,int BLOCK_SHAPE_N,int WARP_SHAPE_M,int WARP_SHAPE_N,int THREAD_NUM>
__device__ void loadtileC(MMAarguments &arg,ElementOutput *C_fragment){
    // iter = 128 * 64 / 128
    const int c_num_per_thread = BLOCK_SHAPE_M * BLOCK_SHAPE_N / THREAD_NUM;
    for(int i=0;i<c_num_per_thread;i++){
        int tileIdx = threadIdx.x*c_num_per_thread + i;
        int rowC = tileIdx / BLOCK_SHAPE_N + blockIdx.y * BLOCK_SHAPE_M;
        int colC = tileIdx % BLOCK_SHAPE_N + blockIdx.x * BLOCK_SHAPE_N;
        C_fragment[i] =  rowC<arg.problem_size.m()&&colC<arg.problem_size.n() ? arg.C[rowC*arg.problem_size.n()+colC] : ElementOutput(0);
    }
}

template<int BLOCK_SHAPE_M,int BLOCK_SHAPE_N,int WARP_SHAPE_M,int WARP_SHAPE_N, int THREAD_NUM>
__device__ void loadtileA(MMAarguments &arg,ElementInputA *A,int idx){
    // iter = 128 * 8 / 128
    const int a_num_per_thread = BLOCK_SHAPE_M * TENSOR_CORE_K / THREAD_NUM;
    for(int i=0;i<a_num_per_thread;i++){
        int tileIdx = ((threadIdx.x/32)*(32*a_num_per_thread)+ (threadIdx.x % 32) + i*32)%(BLOCK_SHAPE_M * TENSOR_CORE_K);
        int rowA = tileIdx/TENSOR_CORE_K + blockIdx.y * BLOCK_SHAPE_M;
        int colA = tileIdx%TENSOR_CORE_K + idx*TENSOR_CORE_K; 

        A[tileIdx] = rowA<arg.problem_size.m()&&colA<arg.problem_size.k() ? arg.A[rowA*arg.problem_size.k()+colA] : ElementInputA(0);
    }
}

template<int BLOCK_SHAPE_M,int BLOCK_SHAPE_N,int WARP_SHAPE_M,int WARP_SHAPE_N, int THREAD_NUM>
__device__ void loadtileB(MMAarguments &arg,ElementInputB *B,int idx){
    // iter = 64 * 8 / 128
    const int b_num_per_thread = BLOCK_SHAPE_N * TENSOR_CORE_K / THREAD_NUM;
    for(int i=0;i<b_num_per_thread;i++){
        int tileIdx = ((threadIdx.x/32)*(32*b_num_per_thread)+ (threadIdx.x % 32) + i*32)%(BLOCK_SHAPE_N * TENSOR_CORE_K);
        int rowB = idx*TENSOR_CORE_K + tileIdx%TENSOR_CORE_K;
        int colB = blockIdx.x*BLOCK_SHAPE_N + tileIdx/TENSOR_CORE_K;

        B[tileIdx] = rowB<arg.problem_size.k()&&colB<arg.problem_size.n() ? arg.B[colB*arg.problem_size.k()+rowB] : ElementInputB(0);
    }
}

template<int BLOCK_SHAPE_M,int BLOCK_SHAPE_N,int WARP_SHAPE_M,int WARP_SHAPE_N,int THREAD_NUM>
__device__ void storetile(MMAarguments &arg,ElementOutput *D){
    // iter = 128 * 64 / 128
    const int d_num_per_thread = BLOCK_SHAPE_M * BLOCK_SHAPE_N / THREAD_NUM;
    for(int i=0;i<d_num_per_thread;i++){
        int tileIdx = threadIdx.x*d_num_per_thread + i;
        int rowD = tileIdx / BLOCK_SHAPE_N + blockIdx.y * BLOCK_SHAPE_M;
        int colD = tileIdx % BLOCK_SHAPE_N + blockIdx.x * BLOCK_SHAPE_N;

        if(rowD<arg.problem_size.m()&&colD<arg.problem_size.n()){
            arg.D[rowD*arg.problem_size.n()+colD] = D[i];
        }
    }
}

template<int BLOCK_SHAPE_M,int BLOCK_SHAPE_N,int WARP_SHAPE_M,int WARP_SHAPE_N,int THREAD_NUM>
__device__ void mma_tile(MMAarguments &arg,ElementInputA *A,ElementInputB *B,ElementAccumulator *C,ElementOutput *D){
    const int warpidx = threadIdx.x / THREADS_PER_WARP;
    const int rowwarp = warpidx / 2;
    const int colwarp = warpidx % 2;
    const int laneidx = threadIdx.x % THREADS_PER_WARP;

    int a[4],b[2];
    const int tile_m = WARP_SHAPE_M / TENSOR_CORE_M;
    const int tile_n = WARP_SHAPE_N / TENSOR_CORE_N;
    const int tile_size = tile_m * tile_n; // one warp computed one tile
    const int num_per_thread = TENSOR_CORE_M * TENSOR_CORE_N / THREADS_PER_WARP; //4
    for(int tileidx=0;tileidx<tile_size;tileidx++){
        int rowtile = tileidx / tile_n;
        int coltile = tileidx % tile_n;

        a[0] = (rowwarp*WARP_SHAPE_M+rowtile*TENSOR_CORE_M+laneidx/num_per_thread)*TENSOR_CORE_K + laneidx%num_per_thread;
        a[1] = a[0] + TENSOR_CORE_N*TENSOR_CORE_K;
        a[2] = a[0] + 4;
        a[3] = a[1] + 4;

        b[0] = (colwarp*WARP_SHAPE_N+coltile*TENSOR_CORE_N+laneidx/num_per_thread)*TENSOR_CORE_K + laneidx%num_per_thread;
        b[1] = b[0] + 4;

        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : 
            "=f"(C[tileidx*4+0]),  // D[0]
            "=f"(C[tileidx*4+1]),  // D[1]
            "=f"(C[tileidx*4+2]),  // D[2]
            "=f"(C[tileidx*4+3])   // D[3]
            : 
            "r"(*reinterpret_cast<uint32_t const *>(&A[a[0]])),   // A[0]
            "r"(*reinterpret_cast<uint32_t const *>(&A[a[1]])),   // A[1]
            "r"(*reinterpret_cast<uint32_t const *>(&A[a[2]])),   // A[2]
            "r"(*reinterpret_cast<uint32_t const *>(&A[a[3]])),   // A[3]
            "r"(*reinterpret_cast<uint32_t const *>(&B[b[0]])),   // B[0]
            "r"(*reinterpret_cast<uint32_t const *>(&B[b[1]])),   // B[1]
            "f"(C[tileidx*4+0]),   // C[0]
            "f"(C[tileidx*4+1]),   // C[1]
            "f"(C[tileidx*4+2]),   // C[2]
            "f"(C[tileidx*4+3])    // C[3]
        );
    }
}

// threadblockShape 128 64 8
// warpShape 64 32 8
// every block has 4 warps

template<int BLOCK_SHAPE_M,int BLOCK_SHAPE_N,int WARP_SHAPE_M,int WARP_SHAPE_N,int THREAD_NUM>
__global__ void GEMM_MMA(MMAarguments arg){
    __shared__ ElementInputA tileA[3][BLOCK_SHAPE_M*TENSOR_CORE_K]; // 3*128*8
    __shared__ ElementInputB tileB[3][TENSOR_CORE_K*BLOCK_SHAPE_N]; // 3*8*64
    // __shared__ ElementOutput tileC[128*64];
    ElementOutput C_fragment[BLOCK_SHAPE_M * BLOCK_SHAPE_N / THREAD_NUM]; // 128*64/128
    const int iters = (arg.problem_size.k() + TENSOR_CORE_K - 1) / TENSOR_CORE_K;
    
    loadtileC<BLOCK_SHAPE_M,BLOCK_SHAPE_N,WARP_SHAPE_M,WARP_SHAPE_N,THREAD_NUM>(arg,C_fragment);

    loadtileA<BLOCK_SHAPE_M,BLOCK_SHAPE_N,WARP_SHAPE_M,WARP_SHAPE_N,THREAD_NUM>(arg,tileA[0],0);
    loadtileB<BLOCK_SHAPE_M,BLOCK_SHAPE_N,WARP_SHAPE_M,WARP_SHAPE_N,THREAD_NUM>(arg,tileB[0],0);
    asm("cp.async.commit_group;\n"::);

    loadtileA<BLOCK_SHAPE_M,BLOCK_SHAPE_N,WARP_SHAPE_M,WARP_SHAPE_N,THREAD_NUM>(arg,tileA[1],1);
    loadtileB<BLOCK_SHAPE_M,BLOCK_SHAPE_N,WARP_SHAPE_M,WARP_SHAPE_N,THREAD_NUM>(arg,tileB[1],1);
    asm("cp.async.commit_group;\n"::);
    
    for(int i=0;i<iters;i++){
        asm("cp.async.wait_group 1;\n"::);
        __syncthreads();
        mma_tile<BLOCK_SHAPE_M,BLOCK_SHAPE_N,WARP_SHAPE_M,WARP_SHAPE_N,THREAD_NUM>(arg,tileA[i%3],tileB[i%3],C_fragment,C_fragment);
        if (i+2<iters){
            loadtileA<BLOCK_SHAPE_M,BLOCK_SHAPE_N,WARP_SHAPE_M,WARP_SHAPE_N,THREAD_NUM>(arg,tileA[(i+2)%3],i+2);
            loadtileB<BLOCK_SHAPE_M,BLOCK_SHAPE_N,WARP_SHAPE_M,WARP_SHAPE_N,THREAD_NUM>(arg,tileB[(i+2)%3],i+2);
            asm("cp.async.commit_group;\n"::);
        }
    }

    storetile<BLOCK_SHAPE_M,BLOCK_SHAPE_N,WARP_SHAPE_M,WARP_SHAPE_N,THREAD_NUM>(arg,C_fragment);
}

void launch_GEMM_MMA(MMAarguments &arg){
    dim3 grid,block;
    // threadblockShape 128 64 8
    // warpShape 64 32 8
    // every block has 4 warps
    const int BLOCK_SHAPE_M = 128;
    const int BLOCK_SHAPE_N = 128;
    const int WARP_SHAPE_M = 64;
    const int WARP_SHAPE_N = 64;
    const int THREAD_NUM = 128;

    grid.x = (arg.problem_size.n()+BLOCK_SHAPE_N-1)/BLOCK_SHAPE_N;
    grid.y = (arg.problem_size.m()+BLOCK_SHAPE_M-1)/BLOCK_SHAPE_M;
    grid.z = 1;

    block.x = THREAD_NUM;
    block.y = 1;
    block.z = 1;


    GEMM_MMA<BLOCK_SHAPE_M, BLOCK_SHAPE_N, WARP_SHAPE_M, WARP_SHAPE_N, THREAD_NUM><<<grid,block>>>(arg);
}