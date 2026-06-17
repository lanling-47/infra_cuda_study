#!/bin/bash
# 安装 nano-vllm sm_75 兼容层 (含 CUDA KV store kernel)

set -e

NANO_VLLM_DIR="/root/cuda-lab/nano-vllm"
COMPAT_DIR="$NANO_VLLM_DIR/nanovllm/compat"

echo "=========================================="
echo "nano-vllm sm_75 兼容层安装"
echo "=========================================="

# 1. 创建 compat 目录
echo "[1/4] 创建 compat 目录..."
mkdir -p "$COMPAT_DIR"
touch "$COMPAT_DIR/__init__.py"

# 2. 备份原始 attention.py
echo "[2/4] 备份原始 attention.py..."
if [ ! -f "$NANO_VLLM_DIR/nanovllm/layers/attention.py.bak" ]; then
    cp "$NANO_VLLM_DIR/nanovllm/layers/attention.py" "$NANO_VLLM_DIR/nanovllm/layers/attention.py.bak"
    echo "  ✓ 已备份为 attention.py.bak"
else
    echo "  (备份已存在)"
fi

# 3. 复制兼容文件
echo "[3/4] 复制兼容文件..."
cp /root/cuda-lab/nano_vllm_compat/flash_attn_compat.py    "$COMPAT_DIR/"
cp /root/cuda-lab/nano_vllm_compat/kv_store.py             "$COMPAT_DIR/"
cp /root/cuda-lab/nano_vllm_compat/kv_store_cuda.cu        "$COMPAT_DIR/"
cp /root/cuda-lab/nano_vllm_compat/kv_store_cuda_ext.cpp   "$COMPAT_DIR/"
cp /root/cuda-lab/nano_vllm_compat/paged_attn.py           "$COMPAT_DIR/"
cp /root/cuda-lab/nano_vllm_compat/paged_attention.cu      "$COMPAT_DIR/"
cp /root/cuda-lab/nano_vllm_compat/paged_attention_ext.cpp "$COMPAT_DIR/"
cp /root/cuda-lab/nano_vllm_compat/int8_linear.py          "$COMPAT_DIR/"
cp /root/cuda-lab/nano_vllm_compat/int8_quantize.py        "$COMPAT_DIR/"
cp /root/cuda-lab/nano_vllm_compat/model_runner_int8.py    "$COMPAT_DIR/"
cp /root/cuda-lab/nano_vllm_compat/speculative.py          "$COMPAT_DIR/"
cp /root/cuda-lab/nano_vllm_compat/attention_sm75.py       "$NANO_VLLM_DIR/nanovllm/layers/attention.py"
echo "  ✓ 已安装"

# 4. 验证
echo "[4/4] 验证安装..."
cd "$NANO_VLLM_DIR"
python3 -c "
import sys
sys.path.insert(0, '.')
from nanovllm.layers.attention import Attention, FLASH_ATTN_AVAILABLE
print(f'  FLASH_ATTN_AVAILABLE: {FLASH_ATTN_AVAILABLE}')
print('  ✓ attention 模块加载成功')
"

echo ""
echo "=========================================="
echo "安装完成!"
echo "=========================================="
echo ""
echo "如需恢复原始版本："
echo "  cp $NANO_VLLM_DIR/nanovllm/layers/attention.py.bak $NANO_VLLM_DIR/nanovllm/layers/attention.py"
