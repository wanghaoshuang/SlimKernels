NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 -lineinfo -DNDEBUG -lcublas --expt-relaxed-constexpr
ARCH = sm_90a
CUTLASS_INCLUDE = -I/root/paddlejob/workspace/env_run/output/whs/cutlass/include -I/root/paddlejob/workspace/env_run/output/whs/cutlass/tools/util/include/

TARGETS = \
    test_baseline \
	test_tensorcore \
	interleave

all: $(TARGETS)

test_baseline: test_gemm.cu gemm_cute.cu
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDE) -arch=$(ARCH) $^ -o $@

test_tensorcore: test_gemm.cu gemm_tensorcore.cu
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDE) -arch=$(ARCH) $^ -o $@

interleave: interleave.cu
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDE) -arch=$(ARCH) $^ -o $@

wgmma: wgmma_sm90.cu
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDE) -arch=$(ARCH) $^ -o $@

wgmma_fp8: wgmma_sm90_fp8.cu
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDE) -arch=$(ARCH) $^ -o $@

wgmma_w2a8: wgmma_sm90_W2A8.cu
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDE) -arch=$(ARCH) $^ -o $@

test_cute: test_cute.cu
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDE) -arch=$(ARCH) $^ -o $@

test_bc: test_bc.cu
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDE) -arch=$(ARCH) $^ -o $@

clean:
	rm -f $(TARGETS)

run: gemm_tensorcore
	./gemm_tensorcore

.PHONY: all clean run
