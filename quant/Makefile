NVCC = /usr/local/cuda/bin/nvcc
CUBLAS = -lcublas
ARCH = -arch=sm_75

TARGET = quant_bench

SRCS = src/quant_bench.cu
HDRS = src/gemm_fp16.cuh src/gemm_int8_dequant.cuh src/gemm_int4_dequant.cuh

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(ARCH) $(CUBLAS) -O2 --use_fast_math -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)

bench: $(TARGET)
	./$(TARGET) 1024 1024 1024

.PHONY: all clean bench
