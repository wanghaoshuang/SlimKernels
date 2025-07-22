NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 -lcublas
ARCH = sm_90
CUTLASS_INCLUDE = /root/paddlejob/workspace/env_run/output/whs/cutlass/include

TARGETS = \
    test_baseline

all: $(TARGETS)

test_baseline: test_gemm.cu gemm_baseline.cu gemm_cute.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUTLASS_INCLUDE) -arch=$(ARCH) $^ -o $@

clean:
	rm -f $(TARGETS)

run: test_baseline
	./test_baseline

.PHONY: all clean run
