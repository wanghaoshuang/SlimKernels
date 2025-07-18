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


// Shape of MMA operator
const int M = 16;
const int N = 8;
const int K = 8;

__device__ void loadtileC(MMAarguments &arg,ElementOutput *C_fragment){
    // iter = 128 * 64 / 128
    for(int i=0;i<64;i++){
        int tileIdx = threadIdx.x*64 + i;
        int rowC = tileIdx / 64 + blockIdx.y * 128;
        int colC = tileIdx % 64 + blockIdx.x * 64;

        C_fragment[i] =  rowC<arg.problem_size.m()&&colC<arg.problem_size.n() ? arg.C[rowC*arg.problem_size.n()+colC] : ElementOutput(0);
    }
}

__device__ void loadtileA(MMAarguments &arg,ElementInputA *A,int idx){
    // iter = 128 * 8 / 128
    for(int i=0;i<8;i++){
        int tileIdx = threadIdx.x*8 + i;
        int rowA = tileIdx/K + blockIdx.y * 128;
        int colA = tileIdx%K + idx*K; 

        A[tileIdx] = rowA<arg.problem_size.m()&&colA<arg.problem_size.k() ? arg.A[rowA*arg.problem_size.k()+colA] : ElementInputA(0);
    }
}

__device__ void loadtileB(MMAarguments &arg,ElementInputB *B,int idx){
    // iter = 64 * 8 / 128
    for(int i=0;i<4;i++){
        int tileIdx = threadIdx.x*4 + i;
        int rowB = idx*K + tileIdx%K;
        int colB = blockIdx.x*64 + tileIdx/K;

        B[tileIdx] = rowB<arg.problem_size.k()&&colB<arg.problem_size.n() ? arg.B[colB*arg.problem_size.k()+rowB] : ElementInputB(0);
    }
}

__device__ void storetile(MMAarguments &arg,ElementOutput *D){
    // iter = 128 * 64 / 128
    for(int i=0;i<64;i++){
        int tileIdx = threadIdx.x*64 + i;
        int rowD = tileIdx / 64 + blockIdx.y * 128;
        int colD = tileIdx % 64 + blockIdx.x * 64;

        if(rowD<arg.problem_size.m()&&colD<arg.problem_size.n()){
            arg.D[rowD*arg.problem_size.n()+colD] = D[i];
        }
    }
}

__device__ void mma_tile(MMAarguments &arg,ElementInputA *A,ElementInputB *B,ElementAccumulator *C,ElementOutput *D){
    const int warpidx = threadIdx.x / 32;
    const int rowwarp = warpidx / 2;
    const int colwarp = warpidx % 2;
    const int laneidx = threadIdx.x % 32;

    int a[4],b[2];

    for(int tileidx=0;tileidx<16;tileidx++){
        int rowtile = tileidx / 4;
        int coltile = tileidx % 4;

        a[0] = (rowwarp*64+rowtile*M+laneidx/4)*K + laneidx%4;
        a[1] = a[0] + 8*K;
        a[2] = a[0] + 4;
        a[3] = a[1] + 4;

        b[0] = (colwarp*32+coltile*N+laneidx/4)*K + laneidx%4;
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

__global__ void GEMM_MMA(MMAarguments arg){
    __shared__ ElementInputA tileA[3][128*8];
    __shared__ ElementInputB tileB[3][8*64];
    // __shared__ ElementOutput tileC[128*64];
    ElementOutput C_fragment[64];
    const int iters = (arg.problem_size.k() + K - 1) / K;
    
    loadtileC(arg,C_fragment);

    loadtileA(arg,tileA[0],0);
    loadtileB(arg,tileB[0],0);
    asm("cp.async.commit_group;\n"::);

    loadtileA(arg,tileA[1],1);
    loadtileB(arg,tileB[1],1);
    asm("cp.async.commit_group;\n"::);
    
    for(int i=0;i<iters;i++){
        asm("cp.async.wait_group 1;\n"::);
        __syncthreads();
        mma_tile(arg,tileA[i%3],tileB[i%3],C_fragment,C_fragment);
        if (i+2<iters){
            loadtileA(arg,tileA[(i+2)%3],i+2);
            loadtileB(arg,tileB[(i+2)%3],i+2);
            asm("cp.async.commit_group;\n"::);
        }
    }

    storetile(arg,C_fragment);
}

void launch_GEMM_MMA(MMAarguments &arg){
    dim3 grid,block;
    // threadblockShape 128 64 8
    // warpShape 64 32 8
    // every block has 4 warps

    grid.x = (arg.problem_size.n()+64-1)/64;
    grid.y = (arg.problem_size.m()+128-1)/128;
    grid.z = 1;

    block.x = 128;
    block.y = 1;
    block.z = 1;

    GEMM_MMA<<<grid,block>>>(arg);
}