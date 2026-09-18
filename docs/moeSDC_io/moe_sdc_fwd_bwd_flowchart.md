# SDC MoE 前/反向全过程详解（单棵整合树 · 三层 · shape/dtype/公式/四项元数据）


## 1. 基线速览

### 1.1 符号

```
  符号      值     公式                              说明
  ────────────────────────────────────────────────────────────────────────────
  N         256    B·S/(CP·TP) = 1·8192/(4·8)        本 rank 进入 MoE 的 token 数
  M         512    N·K = 256·2                      token-expert 任务数（ETP 合并后）
  M'        1024   N·K·ETP = 256·2·2                 ETP 扩展后行数（本 rank 发出量）
  N_recv    1308   Σ output_splits                   A2A 后本 rank 收到量（逐 rank 变长）
  C         32     EP·ETP = 16·2                     splits 段数 / 重排 chunk 数（L=1 时同为 32）
  E/L/K    16/1/2  E=EP·L / L=E/EP / 配置            全局 expert / 本地 expert / topk
  EP/ETP   16/2    配置                              expert 并行度 / expert 张量并行度
  H/H_ffn  8192/32768  配置                          hidden / FFN intermediate
```

### 1.2 四项通信元数据（本 rank 实测）

```
  名称                  dtype         定义                                    实测值（rank0）
  ──────────────────────────────────────────────────────────────────────────────────────────
  num_out_tokens        int           size(0)·topk·tp_size = 256·2·2          1024   恒定，32 rank 相同
  input_splits          int64[32]     num_tokens_per_target_rank_local_expert   Σ=1024 每 rank 皆同
                                      .sum(dim=1)
  output_splits         int64[32]     num_global_tokens_per_local_expert        Σ=1308 全组 377~1308
                                      .sum(dim=1)
  tokens_per_expert     int64[L]=[1]  num_global_tokens_per_local_expert        1308 = Σ output_splits
                                      .sum(dim=0)

  input_splits  = 53;53;32;32;39;39;25;25;26;26;32;32;30;30;26;26;
                  33;33;30;30;41;41;31;31;21;21;17;17;51;51;25;25        Σ=1024
                  ↑ 两两相等（ETP=2 的孪生列），32 个 rank 恒为 Σ=1024

  output_splits = 53;45;37;51;45;46;30;42;47;38;46;33;33;40;38;32;
                  31;34;46;49;33;37;47;44;45;43;41;42;37;36;44;43        Σ=1308
                  ↑ 逐 rank 不同；全组 Σ=32768，均值回到 1024
```

### 1.3 尺寸变化骨架（四条主脉，细节见 §2/§3）

```
  行数（token 数）   256 ──×K·ETP──► 1024 ──A2A──► 1308 ──A2A──► 1024 ──÷ETP──► 512 ──÷K──► 256
                          permute       dispatch      combine      collapse     unpermute

  列宽（hidden 宽）  8192 ──fc1──► 32768 ──act──► 16384 ──fc2──► 8192
                          ×2·H_ffn/ETP   ÷2      H_ffn/ETP  H

  ETP 维度           E=16 ──expand──► E·ETP=32 ──collapse 的反向──► 16
```

---

## 2. 前向全过程（单棵主树）

```
GPTModel.forward  ·  CP4 × TP8 × PP1 = 32 进程                                  [module]
  │
  ▼
TransformerLayer（本次共 1 层）
  │
  ├── Self-Attention（本文不展开）
  │
  └──► MoELayer.forward                                                        [module B5]
        in  hidden_states [256,1,8192] bf16  grad=T        = [N,1,H]
        in  residual      [256,1,8192] bf16
        │
        │   注：MoELayer.forward = router_and_preprocess → dispatch → experts_compute → combine
        │
        ▼
  ┌── ① router_and_preprocess
  │
  │  [module B2] router_and_preprocess
  │    in  hidden_states [256,1,8192] bf16                     【traced】
  │    │
  │    ├─► [module B1] router  (TopKRouter)
  │    │     in  arg0 [256,1,8192] bf16      ← 位置实参①
  │    │     out out0 [256,16]     bf16      ← scores，即 probs
  │    │     out out1 [256,16]     bool      ← routing_map
  │    │     行 256→256： N→N                      公式 B·S/(CP·TP) = 1·8192/32 = 256
  │    │     列 8192→16： H→E                      公式 E = EP·L = 16·1 = 16
  │    │     维度 3→2：   [256,1,8192] → [256,8192]（router 内部 view）
  │    │
  │    └─► [stage A2] dispatch_preprocess
  │          in  hidden_states [256,1,8192] bf16
  │          in  routing_map   [256,16]     bool
  │          in  probs         [256,16]     bf16
  │          │  view(-1, H) → [256,8192]
  │          │
  │          ├─► [stage A1 / op A9] preprocess
  │          │    │
  │          │    ├─ _expand_to_ep_etp_targets(routing_map)
  │          │    │    in  [256,16] bool  →  out [256,32] bool
  │          │    │    行 256→256： 不变
  │          │    │    列 16→32：    E→E·ETP = 16·2 = 32
  │          │    │
  │          │    ├─ expanded_routing_map [256,32] bool
  │          │    │
  │          │    ├─ sum(dim=0) ──► num_tokens_per_target_rank_local_expert [32,1] int64
  │          │    │                  语义：我准备发给每个 (ep,etp) rank 多少行
  │          │    │                  行 32 = EP·ETP = 16·2
  │          │    │
  │          │    ├─ counts A2A（只传 32 个 int64，极小）
  │          │    │    ──► num_global_tokens_per_local_expert [32,1] int64
  │          │    │        语义：每个 (ep,etp) rank 准备发给我多少行
  │          │    │
  │          │    ├─ ◆ input_splits       int64[32]  Σ=1024
  │          │    │    = num_tokens_per_target_rank_local_expert.sum(dim=1)
  │          │    │      Σ = 1024 = M' = N·K·ETP = 256·2·2   ← 32 个 rank 全部相同
  │          │    │      含义：我发给每个 tp_ep rank 的行数（怎么切着发）
  │          │    │
  │          │    ├─ ◆ output_splits      int64[32]  Σ=1308
  │          │    │    = num_global_tokens_per_local_expert.sum(dim=1)
  │          │    │      Σ = 1308（rank0）＝ 全组路由到本地 expert 的去重任务数
  │          │    │      含义：每个 tp_ep rank 发给我的行数（怎么拼着收）
  │          │    │      均衡值应为 Σ_r M'[r]/ETP/E = 32768/2/16 = 1024，1308 是热点
  │          │    │
  │          │    ├─ ◆ num_out_tokens     int = 1024
  │          │    │    = size(0)·topk·tp_size = 256·2·2，恒定，与路由结果无关
  │          │    │      作用：permute 按它开 [1024,8192] 的缓冲区
  │          │    │
  │          │    └─ ◆ tokens_per_expert  int64[1] = 1308
  │          │         = num_global_tokens_per_local_expert.sum(dim=0)
  │          │           长度 = L = E/EP = 16/16 = 1
  │          │           含义：我的本地 expert 一共收到多少行（A2A 之后才成立）
  │          │
  │          ├─► [op A9] _build_unpermute_mapping
  │          │     in  routing_map [256,16] bool
  │          │     out reversed_local_input_permutation_mapping [512] int64
  │          │     行 512 = M = N·K = 256·2
  │          │     非 fused 路径（脚本未开 --moe-permute-fusion）
  │          │     fused 路径输出 [E,N] = [16,256]，用法不同 → 实现时必须带 fused 分支
  │          │
  │          ├─► [op A9] _expand_to_ep_etp_targets(probs)
  │          │     in  probs [256,16] bf16  →  out probs [256,32] bf16
  │          │     行 256→256： 不变      列 16→32： E→E·ETP
  │          │     （与 routing_map 那次同一 kernel，只是 dtype 是 bf16 而非 bool）
  │          │
  │          ├─► [stage A11] _maybe_dtoh_and_synchronize("before_permutation_1")
  │          │     四项元数据 device 由 cuda:0 → cpu_numpy（取值不变）
  │          │     trace 在此再记一次（stage=dtoh），用于核对 DtoH 时机
  │          │
  │          └─► [op C1] permute
  │                out permutated_local_input_tokens [1024,8192] bf16
  │                out permuted_probs                 [1024]      bf16
  │                行 256→1024： N→M' = N·K·ETP = 256·2·2 = 1024（×4）
  │                列 8192→8192： 不变
  │                依据 expanded_routing_map [256,32] 复制派生：
  │                  topk K=2 × ETP=2 = 4 行/token
  │                ◆ 行数恒等于 num_out_tokens = 1024
  │          │
  │          ▼
  │    [module B2] 返回  hidden_states [1024,8192] bf16
  │                        probs        [1024]      bf16
  │                        residual     [256,1,8192] bf16  ← 旁路，不参与 MoE 计算
  └────────────────────────────────────────────────────────────────────────────────────────────────
        │
        ▼
  ┌── ② dispatch（含第一次 AllToAll）
  │
  │  [stage A3 / op C4] token_dispatch
  │    in  permutated_local_input_tokens [1024,8192] bf16          【traced】
  │    in  permuted_probs                [1024]      bf16
  │    │
  │    ├─► [stage A10] _OrderedEtpGradReduction.apply  （前向恒等，仅为反向建图）
  │    │     in  ordered_etp_input [1024,8192] / [1024]  bf16
  │    │     out 同 shape（前向不做任何数值改动）
  │    │     行 1024→1024： 不变
  │    │     条件：USE_ORDERED_ETP_BACKWARD=1 且 tp_size>1（本次已启用）
  │    │
  │    └─► all_to_all(tp_ep_group, x, output_splits, input_splits)
  │         签名：all_to_all(group, input_, output_split_sizes, input_split_sizes)
  │         ③=output_split_sizes 描述"收到的张量怎么切" → 拼接依据
  │         ④=input_split_sizes  描述"发出的张量怎么切" → 切块依据
  │         （与参数名的 input/output 反着理解，最易看错）
  │          │
  │          │  ① 按 input_splits 切成 C=32 块（tokens 与 probs 各通信一次）
  │          │
  │          │     [1024,8192] ──切──►  r0 … r31
  │          │                          53 53 32 32 … 51 25 25      ← input_splits[i]
  │          │                          32 块之和 Σ = 1024 = M'
  │          │
  │          │  ② torch.distributed.all_to_all_single(out, in, o_sz, i_sz, group)
  │          │
  │          │  ③ 各 rank 的块按 output_splits 拼成本 rank 的新张量
  │          │
  │          │     r0 … r31 ──拼──►  [1308,8192]
  │          │     53 45 37 51 … 36 44 43                          ← output_splits[i]
  │          │     32 块之和 Σ = 1308 = N_recv
  │          │
  │          out global_input_tokens [1308,8192] bf16
  │          out global_probs        [1308]      bf16
  │          行 1024→1308： M'(Σ input_splits) → N_recv(Σ output_splits)
  │          列 8192→8192： 不变
  │          ◆ tokens_per_expert = 1308 = Σ output_splits（此处才成立）
  │
  │    ⚠ 1024 = "我发多少"（恒定）  ≠  1308 = "我收多少"（变长）
  │      正确对照：Σ output_splits == rows(global_input_tokens) == Σ tokens_per_expert
  └────────────────────────────────────────────────────────────────────────────────────────────────
        │
        ▼
  ┌── ③ experts_compute（重排 + experts 内部）
  │
  │  [stage A4 / op C3] dispatch_postprocess · sort_chunks_by_idxs
  │    out global_input_tokens [1308,8192] bf16
  │    out global_probs        [1308]      bf16
  │    行 1308→1308： Σ output_splits → Σ output_splits（仅重排，不增不减）
  │    列 8192→8192： 不变
  │    切块表 = num_global_tokens_per_local_expert.ravel()（C=32 段）
  │    排序索引 = sort_input_by_local_experts（把 chunk 按本地 expert 归位）
  │    │
  │    ▼
  │  [module B3] experts · TEGroupedMLP
  │    in  arg0 = permuted_local_hidden_states [1308,8192] bf16
  │    in  arg1 = tokens_per_expert            [1]         int64 (cpu)
  │    in  arg2 = permuted_probs               [1308]      bf16
  │    out out0 = output                       [1308,8192] bf16
  │    out out1 = output_bias                  None（本配置无 bias）
  │    │
  │    ├─► [op E1] TEGroupedMLP.forward
  │    │     in  permuted_local_hidden_states [1308,8192] bf16
  │    │     in  tokens_per_expert            [1]         int64 (cpu)
  │    │     in  permuted_probs               [1308]      bf16
  │    │     ──► permuted_probs.unsqueeze(-1) [1308,1]    bf16   ← intermediate
  │    │
  │    ├─► [op E3] TEGroupedMLP.linear_fc1   (TE column-parallel)
  │    │     in  permuted_local_hidden_states [1308,8192]  bf16
  │    │     out intermediate_parallel        [1308,32768] bf16
  │    │     行 1308→1308： N_recv → N_recv（不变）
  │    │     列 8192→32768： H → 2·H_ffn/ETP = 2·32768/2 = 32768
  │    │     末维受 ETP 切分（fc1 沿 axis 0 分片）
  │    │
  │    ├─► [op E4] TEGroupedMLP.activation   (GLU / swiglu，再 × permuted_probs)
  │    │     out intermediate_parallel        [1308,16384] bf16
  │    │     行 1308→1308： N_recv → N_recv（不变）
  │    │     列 32768→16384： 2·H_ffn/ETP → H_ffn/ETP
  │    │                      = 2·32768/2 → 32768/2 = 32768 → 16384（÷2）
  │    │     两步：① glu 把末维对半切，act(x0)*x1 得 H_ffn/ETP
  │    │           ② × permuted_probs（broadcast [1308,1]）
  │    │           中间有 dtype 提升（× probs），最终 cast 回 bf16
  │    │
  │    ├─► [op E5] TEGroupedMLP.linear_fc2   (TE row-parallel)
  │    │     out output                        [1308,8192]  bf16
  │    │     行 1308→1308： N_recv → N_recv（不变）
  │    │     列 16384→8192： H_ffn/ETP → H = 32768/2 → 8192
  │    │     ⚠ 这是 ETP 的部分和，需等 combine 侧的 collapse 才是完整值
  │    │
  │    ▼  experts 内部尺寸全表（行恒为 N_recv = 1308）
  │    ┌──────────────────┬────────────────┬──────────────────────────────────────┐
  │    │ 步骤              │ 行              │ 列（公式 → 数值）                     │
  │    ├──────────────────┼────────────────┼──────────────────────────────────────┤
  │    │ 入口              │ 1308 = N_recv   │ 8192 = H                             │
  │    │ linear_fc1        │ 1308（不变）     │ 2·H_ffn/ETP = 2·32768/2 = 32768      │
  │    │ activation        │ 1308（不变）     │ H_ffn/ETP   = 32768/2   = 16384      │
  │    │ linear_fc2        │ 1308（不变）     │ H           = 8192                   │
  │    └──────────────────┴────────────────┴──────────────────────────────────────┘
  │    列走 H → 2·H_ffn/ETP → H_ffn/ETP → H 一圈回到原点；
  │    只看模块边界 [1308,8192]→[1308,8192] 完全看不出这四条契约。
  │
  │  [module B3] experts_compute 汇总
  │    in  dispatched_input [1308,8192] bf16
  │    in  permuted_probs   [1308]      bf16
  │    out expert_output    [1308,8192] bf16
  └────────────────────────────────────────────────────────────────────────────────────────────────
        │
        ▼
  ┌── ④ combine（第二次 AllToAll + 收拢回 256）
  │
  │  [stage A5 / op C3] combine_preprocess · sort_chunks_by_idxs
  │    in  expert output       [1308,8192] bf16
  │    out hidden_states       [1308,8192] bf16
  │    行 1308→1308： Σ output_splits（仅逆重排回 A2A 块序）
  │    列 8192→8192： 不变
  │    切块表 = sorted_split_sizes（num_global_tokens_per_local_expert 经 permute(2,1,0)）
  │    排序索引 = restore_output_by_local_experts
  │    │
  │    ▼
  │  [stage A6 / op C4] token_combine · all_to_all   ═══ 第二次集合通信 ═══
  │    in  hidden_states [1308,8192] bf16
  │    切块表与 dispatch 阶段互换：
  │      ┌──────────┬──────────┬─────────────────┬──────────────────┐
  │      │ 通信      │ 输入行数  │ input_split_size │ output_split_size │
  │      ├──────────┼──────────┼─────────────────┼──────────────────┤
  │      │ dispatch │ 1024     │ input_splits    │ output_splits     │
  │      │          │          │ Σ=1024          │ Σ=1308            │
  │      ├──────────┼──────────┼─────────────────┼──────────────────┤
  │      │ combine  │ 1308     │ output_splits   │ input_splits      │
  │      │          │          │ Σ=1308          │ Σ=1024            │
  │      └──────────┴──────────┴─────────────────┴──────────────────┘
  │    效果：各 rank 把当初收的 1308 行原路退回，收回自己当初发出的 1024 行
  │    out permutated_local_input_tokens [1024,8192] bf16
  │    行 1308→1024： Σ output_splits(1308) → Σ input_splits(1024)
  │    列 8192→8192： 不变
  │    │
  │    ▼
  │  [stage A7] combine_postprocess
  │    in  permutated_local_input_tokens [1024,8192] bf16
  │    │
  │    ├─► [stage A8 / op A9] _collapse_etp_partial_outputs
  │    │     in  permutated_local_input_tokens [1024,8192] bf16
  │    │     out collapsed_tokens               [512,8192]  bf16
  │    │     行 1024→512： M' → M = M'/ETP = N·K·ETP/ETP = 256·2 = 512
  │    │                    即 1024/2 = 512
  │    │     列 8192→8192： 不变
  │    │     语义：同 (ep, local_expert) 的 ETP=2 个 chunk 相加
  │    │           → 把"同一 expert 的 2 个副本各算了一半"合并成完整结果
  │    │     切 C=EP·ETP·L=32 段；求和中间 dtype 取 probs.dtype
  │    │
  │    └─► [op C2] unpermute
  │          out output [256,1,8192] bf16
  │          行 512→256： M → N = M/K = N·K/K = 256，即 512/2 = 256
  │          维度 2→3：  [M,H]=[512,8192] → [N,1,H]=[256,1,8192]
  │          按 reversed_local_input_permutation_mapping [512] 反向 scatter
  │          每个 token 的 K=2 个 expert 贡献相加（scatter-add）
  │          │
  │          ▼
  │    [module B4] moe_layer.combine  out output [256,1,8192] bf16
  │    与进入 MoE 的输入 [256,1,8192] 形状完全一致（残差相加后出 MoE）
  └────────────────────────────────────────────────────────────────────────────────────────────────
        │
        ▼
   MoELayer 输出 [256,1,8192] bf16  →  与 residual 相加  →  下一层 / Final LayerNorm
```

---

## 3. 反向全过程（单棵主树，与 §2 严格镜像）

```
  上游梯度（来自下一层 / 残差路径）  [256,1,8192] bf16
    │
    │   注：autograd 逆序回传，行数走 256 → 512 → 1024 → 1308 → 1024 → 256
    │
    ▼
  ┌── ① combine_postprocess 反向
  │
  │  [op C2] unpermute.backward
  │    in  grad_output              [256,1,8192] bf16        【infer】
  │    out grad collapsed_tokens    [512,8192]   bf16        【traced】role=grad_output
  │    行 256→512： N → M = N·K = 256·2 = 512
  │    列 8192→8192： 不变
  │    机制：正向是 K 路 scatter-add 到 token 维，反向按 mapping [512] 把
  │          token 的梯度广播回它的 K=2 个 expert 任务
  │
  │  [stage A8 / op A9] _collapse_etp_partial_outputs.backward
  │    in  grad collapsed_tokens          [512,8192]  bf16   【traced】
  │    out grad permutated_local_input_tokens [1024,8192] bf16 【traced】role=grad_output
  │    行 512→1024： M → M' = M·ETP = 512·2 = 1024
  │    列 8192→8192： 不变
  │    机制：正向 sum 的反向 = replicate
  │          把每块 grad 原样复制给该 (ep,l) 的 ETP=2 个 chunk
  │    ⚠ 必须保留 input_dtype → reduce_dtype → input_dtype 的 cast 链路
  │
  └───────────────────────────────────────────────
    │
    ▼
  ┌── ② combine（第二次 AllToAll 的反向）
  │
  │  [stage A6 / op C4] token_combine · all_to_all.backward
  │    in  grad permutated_local_input_tokens [1024,8192] bf16 【traced】
  │    out grad hidden_states                 [1308,8192] bf16 【traced】role=grad_output
  │    行 1024→1308： Σ input_splits → Σ output_splits
  │    列 8192→8192： 不变
  │    机制：_AllToAll.backward 把正向的两个切块表对调后再跑一次通信
  │          （见 §3.1 展开），于是每块 grad 被送回它当初来源的 rank
  │
  │  [stage A5 / op C3] combine_preprocess · sort_chunks_by_idxs.backward
  │    out grad hidden_states [1308,8192] bf16                【traced】
  │    行 1308→1308： 仅逆重排，不增不减
  │    列 8192→8192： 不变
  │
  └───────────────────────────────────────────────
    │
    ▼
  ┌── ③ experts_compute 反向
  │
  │  [module B3] experts backward · TEGroupedMLP
  │    grad_out0 = [1308,8192] bf16    【traced】
  │    grad_out1 = None（无 bias）
  │    │
  │    ├─ [op E5] TEGroupedMLP.linear_fc2 反向
  │    │    grad_output                     [1308,8192]  bf16 【traced】
  │    │    行 1308→1308 不变；列 8192→16384
  │    │    列公式： H → H_ffn/ETP = 8192 → 32768/2 = 16384
  │    │    同时产出权重梯度（DW，见 D 段插桩点）
  │    │
  │    ├─ [op E4] activation 反向（GLU + ×probs）
  │    │    grad 16384 → 32768：H_ffn/ETP → 2·H_ffn/ETP
  │    │    两路 grad 拼接回 fc1 输出的 [1308,32768]
  │    │    行 1308→1308 不变
  │    │
  │    └─ [op E3] TEGroupedMLP.linear_fc1 反向
  │         grad_output                     [1308,32768] bf16 【traced】
  │         行 1308→1308 不变；列 32768→8192
  │         列公式： 2·H_ffn/ETP → H = 32768 → 8192
  │
  │    grad_in0 = [1308,8192] bf16    ← 对 permuted_local_hidden_states 的梯度
  │    grad_in1 = [1]         int64 (cpu)  ⚠ 占位张量而非 None，不是有效数值梯度，应忽略
  │    grad_in2 = [1308]      bf16    ← 对 permuted_probs 的梯度
  │
  │  [stage A4 / op C3] dispatch_postprocess · sort_chunks_by_idxs.backward
  │    grad global_input_tokens [1308,8192] bf16              【traced】
  │    grad global_probs        [1308]      bf16              【traced】
  │    行 1308→1308： 仅逆重排      列 8192→8192： 不变
  │
  └───────────────────────────────────────────────
    │
    ▼
  ┌── ④ dispatch（第一次 AllToAll 的反向 + ETP 归并）
  │
  │  [stage A3 / op C4] token_dispatch · all_to_all.backward
  │    in  grad global_input_tokens [1308,8192] bf16          【traced】
  │    out grad permutated          [1024,8192] bf16          【traced】
  │    行 1308→1024： Σ output_splits → Σ input_splits
  │    列 8192→8192： 不变
  │
  │  [stage A10] _OrderedEtpGradReduction.backward
  │    chunk 数 = EP·ETP·L = 16·2·1 = 32
  │    in  ordered_etp_input        [1024,8192] bf16           【traced】role=grad_output
  │    out ordered_etp_grad_output  [1024,8192] bf16           【traced】role=grad_input
  │    out ordered_etp_grad_output  [1024]      bf16           【traced】role=grad_input
  │    行 1024→1024： 不变（只做块内归并，不增删行）
  │    机制：固定 (ep_rank, local_expert)，把 tp_rank=1..ETP-1 的 chunk 累加进
  │          tp_rank=0 的 chunk，其余 chunk 置零
  │          即"把同一 expert 的 ETP 两路梯度合成一路"
  │
  │  [stage A2 / op C1] dispatch_preprocess · permute.backward
  │    grad permutated_local_input_tokens [1024,8192] bf16      【traced】
  │    grad permuted_probs                [1024]      bf16      【traced】
  │    行 1024→1024；列 8192→8192（逆 permute，按原索引 scatter 回 token 序）
  │
  │  [infer] 未单独插桩的收尾步骤（由 autograd 内部完成）
  │    · _expand_to_ep_etp_targets.backward：
  │        列 32→16（公式 E·ETP→E），沿 ETP 维求和
  │        ──► grad probs    [256,16]     bf16
  │    · _build_unpermute_mapping： 纯索引，无梯度
  │    · permute 对 hidden 的梯度 + view 还原：
  │        行 1024→256；列 8192→8192；维度 2→3
  │        ──► grad hidden_states [256,1,8192] bf16
  │
  └───────────────────────────────────────────────
    │
    ▼
  ┌── ⑤ router 反向
  │
  │  [module B1] router backward
  │    grad_in0  = [256,1,8192] bf16   ← 回到 MoE 层输入形状
  │    grad_out0 = [256,16]     bf16   ← 对 probs
  │    grad_out1 = None                ← routing_map 是 bool，不可微
  │
  └───────────────────────────────────────────────
    │
    ▼
  MoELayer 输入梯度 [256,1,8192] bf16 → 继续向上一层回传
```

### 3.1 反向的两个关键机制（展开）

```
  ── A. 两次 AllToAll 的反向：切块表对调，各跑一次通信 ──────────────────────────

    i_sz = input_split_sizes   （描述"我发出的张量怎么切"）
    o_sz = output_split_sizes  （描述"我收到的张量怎么切"）

      前向 dispatch：  in[1024]  i_sz=input_splits   o_sz=output_splits   ->  out[1308]
      反向 dispatch：  in[1308]  i_sz=output_splits  o_sz=input_splits    ->  out[1024]

      前向 combine：   in[1308]  i_sz=output_splits  o_sz=input_splits    ->  out[1024]
      反向 combine：   in[1024]  i_sz=input_splits   o_sz=output_splits   ->  out[1308]

    规律：反向 = 把前向的两个切块表对调，其余不变，行数自然回到原值。
    实现：mappings.py 的 _AllToAll.backward
            return _AllToAll.apply(group, *grad_output,
                    ctx.input_split_sizes, ctx.output_split_sizes)
          —— 两个 splits 实参顺序对调。

  ── B. 两处 ETP 归并：互为反操作 ──────────────────────────────────────────────

      前向 collapse：     [1024]  -- 同 (ep,l) 的 ETP=2 chunk 相加 -->  [512]
      反向 collapse：     [512]   -- 每块 grad 复制给 ETP=2 chunk -->  [1024]  (replicate)

      反向 ordered-ETP：  [1024]  -- 同 (ep,l) 的 2 路 grad 累加 -->   [1024]  (行数不变)

    由 USE_ORDERED_ETP_BACKWARD=1 触发（本 run 已启用）。
    v2 产物实测 6 条 ordered-ETP 事件：fwd 2 条（tokens/probs 各 1）+ bwd 4 条。
```

### 3.2 traced / infer 对照

```
  反向步骤                              事件来源        实测 shape（tokens 支路）
  ───────────────────────────────────────────────────────────────────────────────
  unpermute.backward 产出                traced          [512,8192]  bf16
  collapse.backward 产出                 traced          [1024,8192] bf16
  token_combine A2A.backward 产出        traced          [1308,8192] bf16
  combine_preprocess sort.backward 产出  traced          [1308,8192] bf16
  experts fc2 / fc1 的 grad_output       traced          [1308,8192] / [1308,32768]
  experts module grad_in0 / grad_in2     traced          [1308,8192] / [1308]
  dispatch_postprocess sort.backward     traced          [1308,8192] bf16
  token_dispatch A2A.backward 产出       traced          [1024,8192] bf16
  ordered-ETP grad_input                 traced          [1024,8192] / [1024]
  permute.backward 产出                  traced          [1024,8192] / [1024]
  expand.backward / permute 的 hidden 支路 infer           [256,16] / [256,1,8192]
  router grad_in0 / grad_out0            traced          [256,1,8192] / [256,16]
  ───────────────────────────────────────────────────────────────────────────────
  说明：tensor.register_hook 只挂在被显式 watch 的张量上；unpermute / expand /
        router 内部 view 等未挂 hook 的环节由 autograd 自动完成，故标 infer。
```

---

## 4. 附录

### 4.1 尺寸变化总表（行 = token 数，列 = 宽度）

```
  阶段 / 算子                     行变化（公式 → 数值）                     列变化（公式 → 数值）
  ──────────────────────────────────────────────────────────────────────────────────────────
  router                          N → N                      256→256       H → E              8192→16
  expand (routing_map/probs)      N → N                      256→256       E → E·ETP          16→32
  _build_unpermute_mapping        [N,E] → [M]                →512          1 维               —
  permute                         N → M' = N·K·ETP           256→1024      H → H              8192→8192
  dispatch A2A                    M' → N_recv = Σ out_splits 1024→1308     H → H              8192→8192
  sort (dispatch_postprocess)     N_recv → N_recv            1308→1308     H → H              8192→8192
  linear_fc1                      N_recv → N_recv            1308→1308     2·H_ffn/ETP        2·32768/2=32768
  activation                      N_recv → N_recv            1308→1308     H_ffn/ETP          32768/2=16384
  linear_fc2                      N_recv → N_recv            1308→1308     H                  8192
  sort (combine_preprocess)       N_recv → N_recv            1308→1308     H → H              8192→8192
  combine A2A                     N_recv → M'                1308→1024     H → H              8192→8192
  collapse                        M' → M = M'/ETP            1024→512      H → H              8192→8192
  unpermute                       M → N = M/K                512→256       H → H，维度 2→3    8192→8192
  ──────────────────────────────────────────────────────────────────────────────────────────
  反向（镜像）
  unpermute bwd                   N → M                      256→512       H → H              8192→8192
  collapse bwd                    M → M' = M·ETP             512→1024      H → H              8192→8192
  combine A2A bwd                 M' → N_recv                1024→1308     H → H              8192→8192
  experts fc2 bwd                 N_recv → N_recv            1308→1308     H_ffn/ETP          16384
  experts fc1 bwd                 N_recv → N_recv            1308→1308     2·H_ffn/ETP        32768
  dispatch A2A bwd                N_recv → M'                1308→1024     H → H              8192→8192
  ordered-ETP bwd                 M' → M'                    1024→1024     H → H              8192→8192
  expand bwd                      不变                        256            E·ETP → E          32→16
```

### 4.2 dtype 总表

```
  dtype          出现位置                                                     备注
  ────────────────────────────────────────────────────────────────────────────────────
  bf16           所有 tokens / probs / experts 中间张量与梯度                   本 run 未开 fp32 router
  bool           routing_map / expanded_routing_map                           无梯度（grad_out1 = None）
  int64 cuda:0   input_splits / output_splits / num_tokens_per_target_rank_*   DtoH 后变 cpu_numpy
  int            num_out_tokens                                                标量
  int64 cpu      experts 入口 tokens_per_expert [1]                            在 experts.py:750 被 .tolist()
  int64 cpu      experts 反向 grad_in1 [1]                                     ⚠ 占位张量，非 None
  int64 cuda:0   reversed_local_input_permutation_mapping [512]                非 fused 路径
```

### 4.3 运行快照（`index_rank0000.json`）

```
  created                  2026-09-15 11:47:36
  rank / world_size        0 / 32
  MOE_SDC_TRACE            1            MOE_SDC_TRACE_BWD      1
  MOE_SDC_TRACE_LEVEL      full         MOE_SDC_TRACE_LAYERS   module,stage,op
  MOE_SDC_TRACE_RANKS      all          MOE_SDC_TRACE_STEPS    1
  MOE_SDC_TRACE_MAX_EVENTS 200000       MOE_SDC_TRACE_STDOUT   1
  USE_ORDERED_ETP_BACKWARD 1            有序 ETP 反向路径生效
  CUDA_VISIBLE_DEVICES     MIG-… × 32   单机 32 进程（MIG 切分）
```

### 4.4 已知偏差与实现提醒

```
  编号  事项                        说明
  ────────────────────────────────────────────────────────────────────────────────────
  N1    ord 与实测差异            本图行/列全部取自 v2 实测；公式列给出可复算的推导
  N2    grad_in1 占位张量          experts 的 int64 入参在 bwd hook 里得到 [1] int64 cpu 张量，
                                   而非 None；消费端应跳过 int64/bool 入参的 grad
  N3    phase 标注                 sdc_trace.py 19:37 版已正确标 bwd；早期 19:24 版产物把同一批
                                   hook 事件标成 fwd，对比两批产物时不要误读为"反向缺失"
  N4    unpermute mapping 双形态   非 fused → [M]；fused → [E,N]。设计文档固定写 [E,N]，
                                   实现 kernel 前必须先确认 fused 分支
  N5    reduce_dtype              不要硬编码 fp32：本 run probs/permuted_probs 实测均为 bf16
  N6    N_recv 变长                experts 行数 = Σ output_splits（377~1308），不等于 M'=1024；
                                   kernel 侧不可按 M' 硬编码行数
  N7    C 的两种口径              splits 长度 = EP·ETP = 32；重排/collapse chunk 数 = EP·ETP·L = 32
                                   （L=1 时数值相同，L>1 时分离）
```

---
