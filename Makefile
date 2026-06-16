NVCC = /usr/local/cuda/bin/nvcc
CUBLAS = -lcublas
ARCH = -arch=sm_75

TARGET = gemm

all: $(TARGET)

$(TARGET): src/gemm.cu src/gemm_naive.cuh src/gemm_shared.cuh
	$(NVCC) $(ARCH) $(CUBLAS) -O2 -o $(TARGET) src/gemm.cu

clean:
	rm -f $(TARGET)

bench: $(TARGET)
	./$(TARGET) 1024 1024 1024

.PHONY: all clean bench
