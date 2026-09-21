#!/usr/bin/env bash
set -euo pipefail

# NVIDIA H20 MIG GPU version of the SDC MoE smoke script.
# Topology: CP4 x TP8 x DP1 x PP1 = 32 processes
# MoE:      EP16 x ETP2 x EDP1 x PP1 = 32 processes
# Model:    1 layer / hidden 8192 / heads 64 / seq 8192 / FFN 32768
#           experts 16 / top-k 2 / bf16
#
# 跑脚本命令  DISPATCHER=alltoallsdc bash moe_cp4tp8_ep16_etp2_gpu.sh


unset NUM_EXPERTS
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONHASHSEED=1234
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NCCL_DEBUG=ERROR
export NCCL_BYPASS_MIG_DUPLICATE_CHECK=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MEGATRON_LOG_LEVEL=ERROR
export TE_LOG_LEVEL=ERROR
# alltoall_ep_etp 通信版本的 ETP backward 与原始 alltoall_ep 通信版本数值对齐
export USE_ORDERED_ETP_BACKWARD=1 # 按原始 alltoall_ep 的顺序归并 ETP 梯度，不增加通信
DISPATCHER="${DISPATCHER:-alltoall}" # 跑脚本命令  DISPATCHER=alltoallsdc bash moe_cp4tp8_ep16_etp2_gpu.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_NODES=1
NODE_RANK=0
GPUS_PER_NODE=32
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

# In MIG mode CUDA must enumerate the 32 MIG instances explicitly. The
# container sets this at startup; the fallback also works when run manually.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi -L | sed -n 's/.*UUID: \(MIG-[^)]*\).*/\1/p' | paste -sd, -)"
fi
VISIBLE_DEVICE_COUNT="$(printf '%s\n' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | sed '/^$/d' | wc -l)"
if [ "$VISIBLE_DEVICE_COUNT" -ne "$GPUS_PER_NODE" ]; then
    echo "ERROR: expected $GPUS_PER_NODE CUDA devices, found $VISIBLE_DEVICE_COUNT"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    exit 1
fi

MASTER_ADDR="localhost"
MASTER_PORT=7896

tp_size=8
cp_size=4
pp_size=1
ep_size=16
etp_size=2

if [ $((tp_size * cp_size)) -ne $((ep_size * etp_size)) ]; then
    echo "ERROR: tp*cp=$((tp_size * cp_size)) != ep*etp=$((ep_size * etp_size))"
    exit 1
fi

NUM_EXPERTS=16
TOP_K=2

N_LAYERS=1
HIDDEN=8192
FFN_HIDDEN_SIZE=32768
N_HEADS=64
SEQ_LEN=8192
MICRO_BATCH=1
GLOBAL_BATCH=1
NUM_ITERS=2

# Paths on the H20 host/container. The /data4 mount is shared with the host.
DATA_PATH=/data4/Algorithm/modelzoo/dataset/llama2/redpajama_text_document
TOKENIZER_PATH=/data4/Algorithm/modelzoo/dataset/llama2/tokenizer/llama2_tokenizer.model
TOKENIZER_TYPE=Llama2Tokenizer
CHECKPOINT_PATH=/data4/zhengqun/20260902/tests/ckp

cd ../../code/Megatron-LM
# mkdir -p ../log

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

GPU_ARGS=(
    --seed 1234
    --no-barrier-with-level-1-timing
    --disable-bias-linear
    --distributed-timeout-minutes 90
    --no-gradient-accumulation-fusion
    --make-vocab-size-divisible-by 1024
    --position-embedding-type rope
    --no-rope-fusion
    --no-bias-dropout-fusion
    --untie-embeddings-and-output-weights
    --attention-dropout 0
    --hidden-dropout 0
    --no-persist-layer-norm
)

GPT_MODEL_ARGS=(
    --use-mcore-models
    --num-layers "$N_LAYERS"
    --hidden-size "$HIDDEN"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-attention-heads "$N_HEADS"
    --num-query-groups 8
    --group-query-attention
    --seq-length "$SEQ_LEN"
    --max-position-embeddings "$SEQ_LEN"
    # --swiglu # moeSDC_io的md文档是用swiglu得到的数据，因此groupedMLP 权重第2维是32768
)

MOE_ARGS=(
    --num-experts "$NUM_EXPERTS"
    --moe-router-topk "$TOP_K"
    --moe-token-dispatcher-type "$DISPATCHER"
    --moe-router-score-function softmax
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
)

if [ "${SEQ_AUX:-0}" = "1" ]; then
    MOE_ARGS+=(--moe-router-load-balancing-type seq_aux_loss)
fi

TRAINING_ARGS=(
    --deterministic-mode
    --no-masked-softmax-fusion
    --normalization RMSNorm
    --micro-batch-size "$MICRO_BATCH"
    --global-batch-size "$GLOBAL_BATCH"
    --train-iters "$NUM_ITERS"
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --init-method-std 0.02
    --clip-grad 1.0
    --bf16
    --lr 0.00015
    --lr-decay-style constant
    --min-lr 1.5e-5
    --lr-warmup-fraction 0.0
    --lr-decay-iters 430000
    --optimizer adam
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$tp_size"
    --sequence-parallel
    --pipeline-model-parallel-size "$pp_size"
    --context-parallel-size "$cp_size"
    --cp-comm-type p2p
    --expert-model-parallel-size "$ep_size"
    --expert-tensor-parallel-size "$etp_size"
)

DATA_ARGS=(
    --dataloader-type single
    --vocab-size 65536
    --data-path "$DATA_PATH"
    --split 1,0,0
    --tokenizer-model "$TOKENIZER_PATH"
    --tokenizer-type "$TOKENIZER_TYPE"
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1
    --eval-interval 1000
    --ckpt-format torch
    --eval-iters 0
    # --save "$CHECKPOINT_PATH"
    # --load "$CHECKPOINT_PATH"
    # --use-checkpoint-opt-param-scheduler
)

TE_ARGS=(
    --transformer-impl transformer_engine
)

LOGFILE="/data4/zhengqun/20260902/tests/logs/cp4tp8_ep${ep_size}etp${etp_size}_${DISPATCHER}_gpu_$(date +%Y%m%d_%H%M%S).log"

torchrun "${DISTRIBUTED_ARGS[@]}" pretrain_gpt.py \
    "${GPU_ARGS[@]}" \
    "${GPT_MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}" \
    "${TE_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    2>&1 | tee "$LOGFILE"
