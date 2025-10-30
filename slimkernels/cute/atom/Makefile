NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O0 -g -G -lcublas -lineinfo --expt-relaxed-constexpr
ARCH = sm_90a
CUTLASS_INCLUDE = -I/root/paddlejob/workspace/env_run/output/whs/cutlass/include -I/root/paddlejob/workspace/env_run/output/whs/cutlass/tools/util/include/

TARGETS = \
	test_mma

all: $(TARGETS)

test_mma: test_mma.cu
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDE) -arch=$(ARCH) $^ -o $@

clean:
	rm -f $(TARGETS)

run: test_mma
	./test_mma

.PHONY: all clean run
