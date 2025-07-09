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

test_gemm_shape: test_gemm.cu gemm_baseline.cu gemm_shape.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUTLASS_INCLUDE) -arch=$(ARCH) test_gemm.cu gemm_baseline.cu gemm_shape.cu -o test_gemm_shape

clean:
	rm -f test_gemm_baseline test_gemm_reg test_gemm_prefetch test_gemm_shape

run: test_gemm_shape
	./test_gemm_shape

.PHONY: all clean run
