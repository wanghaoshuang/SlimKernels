NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 -lcublas
ARCH = sm_90
CUTLASS_INCLUDE = /root/paddlejob/workspace/env_run/output/whs/complier/cutlass_gemv/cutlass/include

all: test_gemm_baseline test_gemm_reg

test_gemm_baseline: test_gemm.cu gemm_baseline.cu gemm_tensorcore.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUTLASS_INCLUDE) -arch=$(ARCH) test_gemm.cu gemm_baseline.cu gemm_tensorcore.cu -o test_gemm_baseline

test_gemm_reg: test_gemm.cu gemm_baseline.cu gemm_reg.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUTLASS_INCLUDE) -arch=$(ARCH) test_gemm.cu gemm_baseline.cu gemm_reg.cu -o test_gemm_reg

test_gemm_prefetch: test_gemm.cu gemm_baseline.cu gemm_prefetch.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUTLASS_INCLUDE) -arch=$(ARCH) test_gemm.cu gemm_baseline.cu gemm_prefetch.cu -o test_gemm_prefetch

clean:
	rm -f test_gemm_baseline test_gemm_reg test_gemm_prefetch

run: test_gemm_prefetch
	./test_gemm

.PHONY: all clean run
