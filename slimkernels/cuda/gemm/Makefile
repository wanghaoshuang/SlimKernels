NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 -lcublas
ARCH = sm_90

all: test_gemm

test_gemm: test_gemm_tensorcore.cu
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) $< -o $@

debug: test_gemm.cu
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) -g -G $< -o test_gemm_debug

clean:
	rm -f test_gemm test_gemm_debug

run: test_gemm
	./test_gemm

.PHONY: all debug clean run
