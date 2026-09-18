# MoE(alltoallsdc) 真实前/反向 IO shape·dtype 参考表

## 1. 模型与并行参数

| 参数 | 值 | 符号 |
| --- | --- | --- |
| 全局进程 | CP4 × TP8 × DP1 × PP1 = 32 | `CP`/`TP`/`EDP`/`PP` |
| MoE 并行 | EP16 × ETP2 × EDP1 × PP1 = 32 | `EP`/`ETP` |
| 层数 / hidden / heads | 1 / 8192 / 64 | `H` = 8192 |
| seq / micro batch / global batch | 8192 / 1 / 1 | `S` / `B` |
| FFN hidden / experts / top-k | 32768 / 16 / 2 | `H_ffn` / `E` / `K` |
| dtype | bfloat16 | — |
| rank0 进入 MoE token 数 N | 256（`[256,1,8192]`，SP 后） | `N = B·S/(CP·TP)` |
| `num_out_tokens` | 1024 | `M' = N·K·ETP` |
| A2A 后本 rank 收 token 数 | 1308 | `N_recv = Σ output_splits` |
| ETP 合并后 token 数 | 512 | `M = N·K` |



## 2. 符号、shape 公式与 tag 命名规则

### 2.1 符号表

| 符号 | 含义 | 定义式 | 本配置值 |
| --- | --- | --- | --- |
| `B` | micro batch size | 配置 | 1 |
| `S` | 全局 seq len | 配置 | 8192 |
| `H` | hidden size | 配置 | 8192 |
| `H_ffn` | FFN intermediate size | 配置 | 32768 |
| `CP` / `TP` | context / tensor 并行度 | 配置 | 4 / 8 |
| `EP` / `ETP` | expert 并行度 / expert tensor 并行度 | 配置（`ETP` = 代码里的 `tp_size`） | 16 / 2 |
| `E` | 全局 expert 数 | `EP · L` | 16 |
| `L` | 本地 expert 数（`num_local_experts`） | `E / EP` | 1 |
| `K` | router top-k | 配置 | 2 |
| `N` | 本 rank 进入 MoE 的 token 数（SP 切分后） | `B · S / (CP · TP)` | 256 |
| `M` | token-expert 任务数（ETP 合并后） | `N · K` | 512 |
| `M'` | ETP 扩展后的任务数 / 行数 | `M · ETP` | 1024 |
| `C` | splits 段数 / 重排 chunk 数 | `EP · ETP`（splits）；`EP · ETP · L`（重排 chunk） | 32 |
| `N_recv` | A2A 后本 rank 实际收到的行数 | `Σ_d output_splits[d]` | 1308（rank0；逐 rank 不同，见 §2.3） |

### 2.2 核心 shape 公式（代入本配置）

| # | 量 | 公式 | 代入 | 实测 | 对应张量 |
| --- | --- | --- | --- | --- | --- |
| F1 | 进入 MoE 的 token 行数 | `N = B·S/(CP·TP)` | 1·8192/(4·8) | 256 | `hidden_states [256,1,8192]` |
| F2 | router 入参（展平后） | `[N, H]` | — | 256×8192 | `arg0` |
| F3 | router 输出（两个） | `[N, E]` | — | 256×16 | `out0` probs / `out1` routing_map |
| F4 | ETP 扩展后的列数 | `E · ETP` | 16·2 | 32 | `expanded_routing_map [256,32]` |
| F5 | splits 向量长度 | `EP · ETP` | 16·2 | 32 | `len(input_splits)` |
| F6 | permute 后行数（= `num_out_tokens`） | `M' = N·K·ETP` | 256·2·2 | 1024 | `permutated_local_input_tokens [1024,8192]` |
| F7 | 本 rank 发出的行数（守恒） | `Σ_d input_splits[d] = M'` | — | 每 rank 均 1024 | §2.3 G1 |
| F8 | A2A 后本 rank 收到的行数 | `N_recv = Σ_d output_splits[d]` | — | rank0 = 1308 | `global_input_tokens [1308,8192]` |
| F9 | experts 的入/出行数 | `M_recv = N_recv` | — | 1308 | `arg0` / `out0` |
| F10 | `linear_fc1` 输出宽度 | `2 · H_ffn / ETP` | 2·32768/2 | 32768 | `intermediate_parallel [1308,32768]` |
| F11 | 激活后宽度（`fc2` 输入） | `H_ffn / ETP` | 32768/2 | 16384 | `[1308,16384]` |
| F12 | `linear_fc2` 输出宽度 | `H` | — | 8192 | `[1308,8192]` |
| F13 | combine A2A 回流行数 | `M'` | — | 1024 | `[1024,8192]` |
| F14 | collapse 后行数 | `M = N·K` | 256·2 | 512 | `collapsed_tokens [512,8192]` |
| F15 | unpermute 输出 | `[N, 1, H]`（与 MoE 层输入同 shape） | — | 256×1×8192 | `output` |
| F16 | unpermute mapping 长度 | 非 fused `[M]`；fused `[E, N]` | — | `[512]`（本脚本 `fused=False`） | `reversed_local_input_permutation_mapping` |
| F17 | ordered-ETP backward chunk 数 | `EP · ETP · L` | 16·2·1 | 32 | `_OrderedEtpGradReduction` |
| F18 | `tokens_per_expert` 长度 | `L` | 16/16 | 1 | `[1]` int64 cpu |

> F10~F12 的 `ETP` 出现在**最后一维**，是因为 `TEColumnParallelGroupedLinear`（fc1）沿 axis 0、`TERowParallelGroupedLinear`（fc2）沿 axis 1 分片；只看 experts 模块边界（`[1308,8192]` → `[1308,8192]`）看不到这两条约束。
>

### 2.3 通信守恒式（跨 32 rank 实测校验）


| # | 恒等式 | 实测结论 |
| --- | --- | --- |
| G1 | `Σ_d input_splits[d] = M' = N·K·ETP` | 32 个 rank **全部** = 1024  |
| G2 | `input_splits[2e] == input_splits[2e+1]`（ETP 副本成对相等） | 全部成立  |
| G3 | `Σ_r input_splits[r][d] == Σ_s output_splits[d][s]`（发出 = 收到） | 32 个目标 `d` 全部成立  |
| G4 | `Σ_r Σ_d input_splits[r][d] == Σ_r N_recv[r] == n_rank · M'` | 32768 = 32·1024  |
| G5 | `N_recv[2e] == N_recv[2e+1]` | 全部成立  |
| G6 | `Σ_d output_splits[d] == rows(global_input_tokens)` | rank0：1308 = 1308  |

 ETP=2 的语义：同一份 token 数据要**同时投递给某 expert 的 2 个 ETP 副本**，所以 `input_splits` 成对相等、全局发出量守恒；A2A 之后的 `_collapse_etp_partial_outputs` 再把这一对副本相加回一行（F14：`M' → M`）。

逐 rank `N_recv` 实测分布（`sum(output_splits)`）：

| rank | 0/1 | 2/3 | 4/5 | 6/7 | 8/9 | 10/11 | 12/13 | 14/15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `N_recv` | 1308 | 1194 | 918 | 1278 | 1192 | 1175 | 1086 | 1139 |

| rank | 16/17 | 18/19 | 20/21 | 22/23 | 24/25 | 26/27 | 28/29 | 30/31 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `N_recv` | 1293 | 1037 | 991 | 1049 | 647 | 377 | 996 | 704 |

范围 **377 ~ 1308**，均值 1024（= `M'`），即 `Σ_r N_recv[r] = 32768`。这个 3.5× 的离散度来自路由不均衡（`aux_loss` 下仍会有热点 expert），不是错误；但它决定了 `global_input_tokens`/experts 的行数、显存占用与 kernel 的变长上界。

### 2.4 tag 命名规则（`arg0` / `out0` / `grad_in0` / `grad_out0` 是什么）

`module` 层（`router`、`experts`）的 tag 由 `sdc_trace.py` 的 `register_module_hooks` 自动生成，**不反映参数名**，只反映实参/返回值的位置：

| tag | 来源钩子 | 含义 |
| --- | --- | --- |
| `arg{i}` | `register_forward_pre_hook` 的 `inputs[i]` | 模块 `forward` 的第 i 个**位置**入参（0-based，`self` 不计） |
| `out{i}` | `register_forward_hook` 的返回值 | 返回 tuple/list 时取第 i 项；返回单张量时统一记作 `out0` |
| `grad_in{i}` | `register_full_backward_hook` 的 `grad_input[i]` | 对第 i 个入参的梯度 |
| `grad_out{i}` | 同上 `grad_output[i]` | 对第 i 个输出的梯度；不可微时为 `None`（表中 shape 记 `None`、dtype 记 `-`） |

与源码实参/返回值的对照（本配置实测）：

| 模块 | tag | 对应源码 | 实测 shape/dtype |
| --- | --- | --- | --- |
| `router` | `arg0` | `TopKRouter.forward(self, input)` 的 `input` | `[256,1,8192]` bf16 |
| | `out0` | `return scores, routing_map` 的 `scores`（即 `probs`） | `[256,16]` bf16 |
| | `out1` | 同一 return 的 `routing_map` | `[256,16]` bool |
| | `grad_in0` | 对 `input`（MoE 输入 hidden）的梯度 | `[256,1,8192]` bf16 |
| | `grad_out1` | 对 `routing_map` 的梯度 | `None`（bool 不可微） |
| `experts`（`TEGroupedMLP`） | `arg0` | `forward(permuted_local_hidden_states, ...)` 第 1 个 | `[1308,8192]` bf16 |
| | `arg1` | 第 2 个 `tokens_per_expert` | `[1]` int64 cpu |
| | `arg2` | 第 3 个 `permuted_probs` | `[1308]` bf16 |
| | `out0` | `return output, output_bias` 的 `output` | `[1308,8192]` bf16 |
| | `out1` | 同上 `output_bias`（本配置不带 bias） | `None` |
| | `grad_in1` | 对 `tokens_per_expert`（int64）的梯度 | `[1]` int64 cpu |
| | `grad_out1` | 对 `output_bias` 的梯度 | `None` |

两个使用上的坑：

1. `grad_in1` 是 `[1] int64 cpu` 的**占位张量**而不是 `None`（整型入参在 module hook 里仍会得到一张张量）。它不是可用的数值梯度，kernel/接口侧应忽略。
2. 除 `arg*/out*/grad_*` 外，其余 tag（`hidden_states`、`permuted_local_input_tokens`、`global_input_tokens` 等）都是**显式插桩**时按源码变量名命名的，无位置编号语义，可直接对照源码变量。

## 3. 四项通信元数据（rank0，真实值）

| 名称 | 来源阶段 | len | sum | min | max | dtype | device | 值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `input_splits` | `preprocess` | 32 | 1024 | 17 | 53 | int64 | cuda:0 | `53;53;32;32;39;39;25;25;26;26;32;32;30;30;26;26;33;33;30;30;41;41;31;31;21;21;17;17;51;51;25;25` |
| `output_splits` | `preprocess` | 32 | 1308 | 30 | 53 | int64 | cuda:0 | `53;45;37;51;45;46;30;42;47;38;46;33;33;40;38;32;31;34;46;49;33;37;47;44;45;43;41;42;37;36;44;43` |
| `num_out_tokens` | `preprocess` | 1 | 1024 | 1024 | 1024 | int | int | `1024` |
| `tokens_per_expert` | `preprocess` | 1 | 1308 | 1308 | 1308 | int64 | cuda:0 | `1308` |

说明：这里的 `tokens_per_expert` 对应源码 `num_tokens_per_local_expert`（A2A 后本 rank 收到的 expert token 总数）。
`experts.forward` 入口的 `tokens_per_expert` 是 `[1] int64 cpu`，为本地 expert 分组数量（本配置为单 expert 组），见 §4.1；`arg0`/`out0` 等 tag 的含义见 §2.4。

## 4. 前向：子模块 / 阶段 / 算子 IO

### 4.1 子模块层（module）

| 模块 | 输入 | shape | dtype | 输出 | shape | dtype | 公式 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `moe_layer.router_and_preprocess` | `hidden_states` | `[256,1,8192]` | bf16 | `hidden_states` | `[1024,8192]` | bf16 | in `[N,1,H]` → out `[M',H]`（F1/F6） |
| | | | | `probs` | `[1024]` | bf16 | out `[M']` |
| `router` | `arg0` | `[256,1,8192]` | bf16 | `out0` | `[256,16]` | bf16 | in `[N,1,H]`；out `[N,E]`（F2/F3） |
| | | | | `out1` | `[256,16]` | bool | out `[N,E]` |
| `experts(TEGroupedMLP)` | `arg0` | `[1308,8192]` | bf16 | `out0` | `[1308,8192]` | bf16 | in/out `[N_recv,H]`（F8/F9） |
| | `arg1` | `[1]` | int64(cpu) | `out1` | `None` | - | `[L]`（F18） |
| | `arg2` | `[1308]` | bf16 | | | | `[N_recv]` |
| `moe_layer.experts_compute` | `dispatched_input` | `[1308,8192]` | bf16 | `expert_output` | `[1308,8192]` | bf16 | in/out `[N_recv,H]` |
| | `permuted_probs` | `[1308]` | bf16 | | | | `[N_recv]` |
| `moe_layer.combine` | `output` | `[1308,8192]` | bf16 | `output` | `[256,1,8192]` | bf16 | in `[N_recv,H]` → out `[N,1,H]`（F15） |

### 4.2 阶段层（stage，token_dispatcher）

| 阶段 | 算子 | tag | role | shape | dtype | 公式 |
| --- | --- | --- | --- | --- | --- | --- |
| `dispatch_preprocess` | - | `hidden_states` | input | `[256,1,8192]` | bf16 | `[N,1,H]` |
| | - | `routing_map` | input | `[256,16]` | bool | `[N,E]` |
| | - | `probs` | input | `[256,16]` | bf16 | `[N,E]` |
| | `permute` | `permutated_local_input_tokens` | output | `[1024,8192]` | bf16 | `[M',H]` |
| | `permute` | `permuted_probs` | output | `[1024]` | bf16 | `[M']` |
| `preprocess` | - | `expanded_routing_map` | output | `[256,32]` | bool | `[N,E·ETP]` |
| | - | `num_tokens_per_target_rank_local_expert` | output | `[32,1]` | int64 | `[EP·ETP, L]` |
| | - | `num_global_tokens_per_local_expert` | output | `[32,1]` | int64 | `[EP·ETP, L]` |
| `token_dispatch` | `all_to_all` | `permutated_local_input_tokens` | input | `[1024,8192]` | bf16 | `[M',H]`，`Σ in_splits = M'` |
| | `all_to_all` | `permuted_probs` | input | `[1024]` | bf16 | `[M']` |
| | `all_to_all` | `global_input_tokens` | output | `[1308,8192]` | bf16 | `[N_recv,H]`，`Σ out_splits = N_recv` |
| | `all_to_all` | `global_probs` | output | `[1308]` | bf16 | `[N_recv]` |
| `dispatch_postprocess` | `sort_chunks_by_idxs` | `global_input_tokens` | output | `[1308,8192]` | bf16 | `[N_recv,H]`（chunk 重排，行数不变） |
| | `sort_chunks_by_idxs` | `global_probs` | output | `[1308]` | bf16 | `[N_recv]` |
| `combine_preprocess` | `sort_chunks_by_idxs` | `hidden_states` | output | `[1308,8192]` | bf16 | `[N_recv,H]`（逆重排） |
| `token_combine` | `all_to_all` | `hidden_states` | input | `[1308,8192]` | bf16 | `[N_recv,H]` |
| | `all_to_all` | `permutated_local_input_tokens` | output | `[1024,8192]` | bf16 | `[M',H]`（反向 A2A 回到 `M'`） |
| `combine_postprocess` | - | `permutated_local_input_tokens` | input | `[1024,8192]` | bf16 | `[M',H]` |
| | `unpermute` | `output` | output | `[256,1,8192]` | bf16 | `[N,1,H]`（先 collapse 到 `M`） |

### 4.3 算子层（op）

| 算子 | tag | role | shape | dtype | 公式 |
| --- | --- | --- | --- | --- | --- |
| `_expand_to_ep_etp_targets`(routing_map) | `routing_map` | input | `[256,16]` | bool | `[N,E]` |
| | `routing_map` | output | `[256,32]` | bool | `[N,E·ETP]` |
| `_expand_to_ep_etp_targets`(probs) | `probs` | input | `[256,16]` | bf16 | `[N,E]` |
| | `probs` | output | `[256,32]` | bf16 | `[N,E·ETP]` |
| `_build_unpermute_mapping` | `routing_map` | input | `[256,16]` | bool | `[N,E]` |
| | `reversed_local_input_permutation_mapping` | output | `[512]` | int64 | 非 fused `[M]`（fused 为 `[E,N]`，见 F16） |
| `_collapse_etp_partial_outputs` | `permutated_local_input_tokens` | input | `[1024,8192]` | bf16 | `[M',H]`，切 `EP·ETP·L` 个 chunk |
| | `collapsed_tokens` | output | `[512,8192]` | bf16 | `[M,H] = [N·K,H]` |
| `TEGroupedMLP.forward` | `permuted_local_hidden_states` | input | `[1308,8192]` | bf16 | `[N_recv,H]` |
| | `tokens_per_expert` | input | `[1]` | int64(cpu) | `[L]` |
| | `permuted_probs` | input | `[1308]` | bf16 | `[N_recv]` |
| | `permuted_probs_unsqueezed` | intermediate | `[1308,1]` | bf16 | `[N_recv,1]` |
| `TEGroupedMLP.linear_fc1` | `permuted_local_hidden_states` | input | `[1308,8192]` | bf16 | `[N_recv,H]` |
| | `intermediate_parallel` | output | `[1308,32768]` | bf16 | `[N_recv,2·H_ffn/ETP]` |
| `TEGroupedMLP.activation` | `intermediate_parallel` | output | `[1308,16384]` | bf16 | `[N_recv,H_ffn/ETP]` |
| `TEGroupedMLP.linear_fc2` | `output` | output | `[1308,8192]` | bf16 | `[N_recv,H]`（ETP 部分和，待 collapse） |

## 5. 反向：子模块 / 阶段 / 算子梯度 IO

### 5.1 子模块层（module）

| 模块 | tag | role | shape | dtype | 公式 |
| --- | --- | --- | --- | --- | --- |
| `experts(TEGroupedMLP)` | `grad_in0` | grad_input | `[1308,8192]` | bf16 | `[N_recv,H]`（对 `arg0` 的梯度） |
| | `grad_in1` | grad_input | `[1]` | int64(cpu) | `[L]`（对 `arg1`；占位张量，非数值梯度，见 §2.4 坑 1） |
| | `grad_in2` | grad_input | `[1308]` | bf16 | `[N_recv]`（对 `arg2` = `permuted_probs`） |
| | `grad_out0` | grad_output | `[1308,8192]` | bf16 | `[N_recv,H]` |
| | `grad_out1` | grad_output | `None` | - | 对 `output_bias`（不存在） |
| `router` | `grad_in0` | grad_input | `[256,1,8192]` | bf16 | `[N,1,H]`（与 MoE 层输入同 shape） |
| | `grad_out0` | grad_output | `[256,16]` | bf16 | `[N,E]`（对 `probs`） |
| | `grad_out1` | grad_output | `None` | - | 对 `routing_map`（bool 不可微） |

### 5.2 阶段层（stage，token_dispatcher）

| 阶段 | 算子 | tag | role | shape | dtype | 公式 |
| --- | --- | --- | --- | --- | --- | --- |
| `_collapse_etp_partial_outputs` | - | `collapsed_tokens` | grad_output | `[512,8192]` | bf16 | `[M,H]` |
| `token_combine` | `all_to_all` | `permutated_local_input_tokens` | grad_output | `[1024,8192]` | bf16 | `[M',H]` |
| `combine_preprocess` | `sort_chunks_by_idxs` | `hidden_states` | grad_output | `[1308,8192]` | bf16 | `[N_recv,H]` |
| `dispatch_postprocess` | `sort_chunks_by_idxs` | `global_probs` | grad_output | `[1308]` | bf16 | `[N_recv]` |
| | `sort_chunks_by_idxs` | `global_input_tokens` | grad_output | `[1308,8192]` | bf16 | `[N_recv,H]` |
| `token_dispatch` | `all_to_all` | `global_probs` | grad_output | `[1308]` | bf16 | `[N_recv]` |
| | `all_to_all` | `global_input_tokens` | grad_output | `[1308,8192]` | bf16 | `[N_recv,H]` |
| `_OrderedEtpGradReduction` | `ordered_etp_grad_reduction` | `ordered_etp_grad_output` | grad_input | `[1024]` | bf16 | `[M']`（probs 支路，chunk 数 = `EP·ETP·L`，F17） |
| | `ordered_etp_grad_reduction` | `ordered_etp_grad_output` | grad_input | `[1024,8192]` | bf16 | `[M',H]`（tokens 支路） |
| | `ordered_etp_grad_reduction` | `ordered_etp_input` | grad_output | `[1024,8192]` | bf16 | `[M',H]` |
| `dispatch_preprocess` | `permute` | `permutated_local_input_tokens` | grad_output | `[1024,8192]` | bf16 | `[M',H]` |
| | `permute` | `permuted_probs` | grad_output | `[1024]` | bf16 | `[M']` |

> 反向各梯度与对应前向张量**同 shape**，故公式列与前向一一对应；ETP 维的归约只发生在 `_OrderedEtpGradReduction`（`M' → chunk 0`）与 `_collapse_etp_partial_outputs` 的 backward（`M → M'` replicate）两处。
>


### 5.3 算子层（op）

| 算子 | tag | role | shape | dtype | 公式 |
| --- | --- | --- | --- | --- | --- |
| `TEGroupedMLP.linear_fc2` | `output` | grad_output | `[1308,8192]` | bf16 | `[N_recv,H]` |
| `TEGroupedMLP.linear_fc1` | `intermediate_parallel` | grad_output | `[1308,32768]` | bf16 | `[N_recv,2·H_ffn/ETP]` |

