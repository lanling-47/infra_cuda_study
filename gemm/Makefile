NVCC = /usr/local/cuda/bin/nvcc
CUBLAS = -lcublas
ARCH = -arch=sm_75

TARGET = gemm

SRCS = src/gemm.cu
HDRS = src/gemm_naive.cuh src/gemm_shared.cuh src/gemm_reg_tile.cuh \
       src/gemm_vec_load.cuh src/gemm_dbl_buf.cuh src/gemm_wmma.cuh

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(ARCH) $(CUBLAS) -O2 --use_fast_math -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)

bench: $(TARGET)
	./$(TARGET) 1024 1024 1024

.PHONY: all clean bench
