NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 -lcublas
ARCH = sm_90
CUTLASS_INCLUDE = /root/paddlejob/workspace/env_run/output/whs/complier/cutlass_gemv/cutlass/include

all: test_gemm

test_gemm_baseline: test_gemm_tensorcore.cu gemm_baseline.cu gemm_tensorcore.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUTLASS_INCLUDE) -arch=$(ARCH) test_gemm_tensorcore.cu gemm_baseline.cu gemm_tensorcore.cu -o test_gemm

test_gemm_reg: test_gemm_tensorcore.cu gemm_baseline.cu gemm_reg.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUTLASS_INCLUDE) -arch=$(ARCH) test_gemm_tensorcore.cu gemm_baseline.cu gemm_tensorcore.cu -o test_gemm

debug: test_gemm.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUTLASS_INCLUDE) -arch=$(ARCH) -g -G $< -o test_gemm_debug

clean:
	rm -f test_gemm_baseline test_gemm_reg test_gemm_debug

run: test_gemm
	./test_gemm

.PHONY: all debug clean run
