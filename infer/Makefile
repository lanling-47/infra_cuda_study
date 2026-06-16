NVCC = /usr/local/cuda/bin/nvcc
CUBLAS = -lcublas
ARCH = -arch=sm_75

TARGET = infer_bench

SRCS = src/infer_bench.cu
HDRS = src/layernorm.cuh src/softmax.cuh src/rope.cuh src/flash_attention.cuh src/elementwise.cuh

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(ARCH) $(CUBLAS) -O2 -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)

bench: $(TARGET)
	./$(TARGET)

.PHONY: all clean bench
