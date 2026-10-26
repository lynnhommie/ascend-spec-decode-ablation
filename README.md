# MTP 投机解码在昇腾 910B 上的 K×并发消融与性能归因

> **TL;DR (EN)** — Speculative decoding (MTP) on Ascend NPU was believed to be a net loss. We showed that conclusion was a **measurement artifact**: the streaming benchmark client over-counts ~19 ms/iteration *only* for speculative decoding, and early EOS moves the denominator. With a corrected dual-caliber measurement, MTP is **−43% TPOT** at single stream (K=3, 12.0 vs 17.2 ms/token). We then mapped the full operating envelope: the optimal draft depth **falls with concurrency (K\* = 3/3/2/1 at conc 1/4/16/64 → +40%/+29%/+16%/+10%)**, and gains **flip sign** at K=5/conc=16 (0.86×) and K=3/conc=64 (0.85×) — while acceptance length stays flat (2.38–2.41), proving the flip is pure verify-compute pressure, not quality loss. Attribution (operator-level traces) gives the +4.25 ms/iteration host-side marginal (sync 2.9 / launch 1.1 / copy 0.4) and a 58% vs 92% device-utilization gap. Delivered per-track numbers: making the upstream dynamic-speculative path usable cost **−58% → 0% (12.0 vs 28.2 ms/token, +135%)**; out-of-graph op pooling is **neutral** (−0.5 ms/token, inside the ±0.7 ms noise band) because this platform is **dispatch-bound, not alloc-bound**; and a **6.1 ms/iter** host sync was proven **off the critical path** by an A/B knob — a negative result that cancelled a whole rewrite.

**中文一句话**：把"MTP 在昇腾上恒负收益"这个结论推翻（测量口径错误），改成实测 **+43% 单流加速**，并测绘出 **K×并发包络与收益翻转点**；再对宿主开销做逐算子归因，把两条优化腿用测量结案，最后把第三条腿（按 K 补图/动态 K）推到平台边界并给出可复现的判定实验。

---

## 1. 为什么要做这件事

1. **国产算力上的主流栈是真实生产需求**，但 vllm-ascend 作为硬件插件天然滞后于 vLLM 主干（本次实测栈：vLLM 0.26.0 + vllm-ascend v0.26.0rc，主干当时已到 0.32+）。**主干的最优配置在 NPU 上不一定最优**，所以需要一份"开什么、关什么、调多少"的画像。
2. **投机解码是 2026 年推理栈的标配特性，但它的收益是条件性的**：厂商/社区基准通常报单流延迟收益，而昇腾 910B 的带宽/算力比与静态图机制与 GPU 不同，收益边界必须自己测。
3. **不做画像就优化 = 猜**。本项目的第一产出是数据边界，第二产出才是优化与归因。

## 2. 实验环境（复现锚点）

| 项 | 值 |
|---|---|
| 硬件 | Ascend 910B2，单卡 HBM 64 GB，TP=1 |
| 软件 | vLLM 0.26.0 + vllm-ascend v0.26.0rc @ `85625bd77`，CANN / torch_npu（实例镜像自带） |
| 模型 | Qwen3.5-9B（自带 MTP draft 层），dtype bf16 |
| 服务参数 | `--max-model-len 8192 --gpu-memory-utilization 0.85`，`cudagraph_mode=FULL_AND_PIECEWISE` |
| 投机配置 | `{"method":"mtp","num_speculative_tokens":K}`，K ∈ {0,1,2,3,5} |
| 负载 | 非流式 `/v1/completions`，`max_tokens=128`，greedy（temperature=0），不屏蔽 EOS |
| 并发 | `max_concurrency ∈ {1,4,16,64}`（2 的幂 + 两端极值） |
| 重复 | 每格 3 轮取中位；臂间彻底清进程（`pkill -9 -f 'spawn_main\|EngineCore\|vllm'` + 端口静默等待 + `npu-smi` 确认） |

## 3. 测量口径：本项目最重要的方法学结论

**错误口径如何产生错误结论**（这是"恒负收益"说法的来源）：

| 陷阱 | 机理 | 影响 |
|---|---|---|
| 流式客户端计时 | 投机解码每个 SSE 分块只带 1–2 个 token，客户端分块处理成本被计入 TPOT；k0 每迭代稳定出 1 token 不受影响 | **只对 spec 臂系统性虚增 ~19 ms/迭代** |
| EOS 早停 | 同一 prompt 实际输出 140 token（上限 256），用 `max_tokens` 当分母 | 分母逐轮漂移，轮间抖动 |
| 计数器语义 | vLLM 的 SpecDecoding 指标按 **10 秒窗口清零** | 跨窗口差分得到荒谬值（曾算出 acceptance=25.6） |

**采用口径（双口径互证，缺一不可）**：

```
外部：非流式请求墙钟 elapsed（TTFT 单列）
内部：同一时间窗内服务端 Drafted / Accepted 计数差分

每迭代成本 = (elapsed − TTFT) / Δiters
每 token 成本 = (elapsed − TTFT) / completion_tokens        ← 分母是实测输出，不是 max_tokens
acceptance  = completion_tokens / ΔDrafted
```

**铁律**（写进了 `tools/benchxc.py` 的实现）：性能结论必须由两个独立口径给出一致答案；换口径时禁止用调参去"拟合"矛盾，而是找第三个口径仲裁。本项目两次因违反此律返工，均已记录。

## 4. 主结果：K × 并发矩阵

聚合吞吐 tok/s（原始数据见 `data/matrix.csv`）：

| K\并发 | 1 | 4 | 16 | 64 |
|---|---|---|---|---|
| 0（无投机基线） | 59.4 | 205.7 | 701.6 | 1878.8 |
| 1 | 64.3 (1.08×) | 218.5 (1.06×) | 743.0 (1.06×) | **2069.4 (1.10×)** |
| 2 | 75.8 (1.28×) | 240.4 (1.17×) | **813.6 (1.16×)** | n/m |
| 3 | **83.3 (1.40×)** | **265.9 (1.29×)** | 721.6 (1.03×) | 1599.0 (0.85×) |
| 5 | 78.7 (1.33×) | 236.0 (1.15×) | 600.5 (0.86×) | 1028.0 (0.55×) |

单流（conc=1）每 token 延迟：`17.2 → 15.8 → 13.2 → 12.0 → 12.0 → 12.7` ms（K=0…5），**峰值 +43%，K∈{3,4} 为平原**。

平均接受长度（服务端差分）：K=1 → 2.03，K=2 → 2.26–2.32，K=3 → **2.38–2.41（全并发稳定）**。

**三条可直接引用的结论**
1. **最优投机深度随并发单调下降**：K\* = 3 / 3 / 2 / 1（conc 1 / 4 / 16 / 64）。
2. **收益符号会翻转**，且翻转点可用：K=5 在 conc=16 跌到 0.86×，K=3 在 conc=64 跌到 0.85×，K=5@64 = 0.55×。
3. **翻转不是接受率塌陷**（K=3 的接受长度在 conc 1→64 全程 2.38–2.41），而是**验证算力**：高并发时每个迭代要为 (K+1) 份 token 付 FLOPs，而这份算力已经没有空闲了。

## 5. 图

| | |
|---|---|
| ![吞吐-并发饱和曲线与包络](figs/fig1_saturation.png) | ![相对基线加速热图（<1 即倒赔）](figs/fig2_speedup_heatmap.png) |
| **图 1** 吞吐 vs 并发，每线一个 K，虚线为逐列最优包络 | **图 2** 加速比热图，红色区即"投机解码不该开"的工作区 |
| ![延迟-吞吐 Pareto](figs/fig3_pareto.png) | ![单流 K 曲线与接受长度](figs/fig4_k_curve.png) |
| **图 3** 延迟-吞吐权衡（点标并发），左上为优 | **图 4** 单流倒 U：接受长度次线性增长 + 迭代成本与 K 近似无关 |

## 6. 机制与归因：为什么是这个形状

成本模型（实测支撑）：

```
t_iter(B, K) ≈ a(K) + B · c(K)
a(K) = 固定成本：内核派发、图外胶水、同步     ← 被 E[tok|K] 个 token 摊薄
c(K) = 每请求计算成本                        ← 高并发时主导
```

- 单流实测：K=1 的 `t_iter`=31.8 ms、K=3 的 `t_iter`=30.1 ms ⇒ **迭代成本与 K 近似无关**（910B 上矩阵-向量乘是权重带宽受限，多算 K 份 token 几乎不加时间）。于是低并发"猜深一点几乎免费"，倒 U 的右端则由接受长度次线性衰减造成。
- 两式对 B 的依赖方向相反 ⇒ **最优 K 必然随并发下降且只翻转一次**，与图 1/图 2 的实测形状一致。

**宿主边际分解**（k1 相对 k0 的 trace 逐算子差分，单位 ms/迭代）：

| 项 | 边际 | 定性 |
|---|---|---|
| 硬同步 Event.synchronize | **+2.9**（3 次） | 最大单项 |
| 内核派发 / HostArgs | +1.1（+117 次） | 结构性（draft/verify 多出的 GEMM 派发） |
| InplaceCopy(+GWS) / Index / MemcpyAsync | +0.41 / +0.29 / +0.18 | 图外元数据胶水 |
| dynamic 核数 +117 | 设备时长仅 +0.64 | **小算子的成本在宿主派发，不在计算** |

设备侧健康度：static 核占 85% 设备时长，MatMulV2 单核 943 ms = 带宽地板；但 **k1 设备利用率 58% vs k0 的 92%** ⇒ 每迭代 13 ms 空闲，是后续图化的理论空间上界。

## 7. 三条优化腿：两条结案，一条推到边界

### 7.1 图外小算子池化 → 实测中性（结案）
做法：pinned buffer 双块交替、CPU arange 预建、block-table clone 复用 buffer。
结果（口径务必分清）：每轮投机（propose + verify）耗时 **26.1 → 25.1 ms**（同口径基线 K=0 = 15.7 ms/轮）；折算到 per-token 是 −1.0 / 2.03 ≈ **−0.5 ms/token**，落在实测噪声带（±0.7 ms）内，因此双口径下判为 **性能中性**（15.7–15.9 vs 基线 15.8 ms/token）。
**根因判据**：该平台按 **派发次数** 定价（aclnn ≈ 12.7 µs/次，CUDA ≈ 3 µs），而 caching allocator 让分配近乎免费 ⇒ 这是 **dispatch-bound 而非 alloc-bound**，池化省的是本来就不缺的东西。它的净价值转为"恒地址性"= 图化的使能条件（也正是上游给 GLM 做池化的动机）。

### 7.2 消除同步 → 真实但不在关键路径（结案）
用 per-site 计时模块定位到 `num_accepted_tokens_event.synchronize()` 每次 **6.1 ms**（此前从 api_statistic 估的 2.9 ms 口径作废——torch_npu 的 Event.synchronize 内部轮询没被计入）。
加 A/B 开关跳过该同步后实测：**输出正确、acceptance 1.80 不塌，但性能零变化**（K=1 per-token 15.7–16.1 vs 基线 15.8；K=3 11.6–11.8 vs 12.0）⇒ 6.1 ms 是真实的宿主白等，却被设备忙等遮蔽、不在关键路径上。剩余 ~3 ms/迭代 的同步位于 vLLM core 侧（本 fork 边界之外）。**这条腿的收益是"方向性"的：省掉一整轮以它为目标的重写，并留下一个可上游的安全开关。**
**这条腿以"没有补丁可写"为合法结论收线**，而不是硬改一处代码去交差。

### 7.3 按 K 补图 / 动态 K → 推到平台边界（roadmap）
目标：让运行时 K 拥有自己的 ACL 图，从而"一个配置吃满包络"（K 随并发自适应）。上游已有一条通道（`num_speculative_tokens_per_batch_size`）会下发每步 K，但 **vllm-ascend 的 MTP 路径绕过了 K 重绑点**，且一旦配置该表就把 `cudagraph_mode` 从 FULL 降级。

**这条腿已兑现的收益**：启用上游动态投机通道的代价，从 **28.2 ms/token（降级后，比静态 K=3 差 2.35×）**降到 **0（12.0 ms/token = 静态最优）** ⇒ 该路径上 **+135%**；并让"切换机制"端到端可用（`dynamic K: 3 -> 1` 日志 + 按 K 预捕 `0 sizes failed` + 桶格无 padding）。
**仍未兑现**：运行时 K≠配置 K 的**图回放**本身（见下表第 7 层）——所以包络里"切换档"的那部分收益（如 K=2@conc16 = +16%）目前是路线图而非既得数字。

实现链路（代码见 fork `f2p-mtp-first-pass` 分支）：调度表 → 调度器下发 → drafter K 重绑 → **draft 图按 (bucket, K) 双键** → 启动期穷尽预捕 → 桶格按 `lcm(所有 span)` 重排 → 覆盖守卫（无图可用时保持配置 K）。

推进过程中逐层定位并修复 6 层阻塞（每层都有错误码/复现）：

| # | 阻塞 | 根因 | 处置 |
|---|---|---|---|
| 1 | 配 per_batch_size 即被降级 PIECEWISE | core 的 FULL 图 `view(-1,K,vocab)` 形状绑定配置 K | 转为自研 |
| 2 | FIA padding 用配置 span | `561002 queryT ≠ actualSequenceLengthQ` | padding 改用运行时 span |
| 3 | draft 继承 target 的 uniform 标志 | 混合批把 draft 赶出图 → AOT 按配置 K 编的形状炸 | draft 仅在 decode-only 步强制 uniform |
| 4 | 服务中起捕被拒 | `graph_task_group_begin → 107029` | 只能在启动期穷尽预捕 |
| 5 | 启动即 AttributeError | helper 被每条 `_dummy_run` 经过，属性只在动态 K 路径赋值 | `__init__` 初始化 + getattr 兜底 |
| 6 | span=3 没有合法桶 | core 只按**配置** span 调整桶格（`adjust_cudagraph_sizes_for_spec_decode`）⇒ 档位全是 4 的倍数 | 桶格重排成所有 span 的公倍数 |
| **7** | **运行时 K≠配置 K 的回放本身** | **见下** | **停止开发，转 roadmap/issue** |

**第 7 层的判定实验（同形对照）**——这是整节最硬的一条证据：

| 场景 | 该步 K 是否等于配置 K | 结果 |
|---|---|---|
| 静态 K=2 服务器 @ conc=16（16 reqs、48 token、同一形状的图） | 是 | **正常，813.6 tok/s** |
| 配置 K=3、调度表把 conc≥5 降到 K=2 @ conc=16（形状与上一行**完全相同**，桶制下 48/16reqs 零 padding） | 否 | **设备异常（aivec / HTTP 500）** |
| 同上但在 reqs=2 的小批 | 否 | **静默挂死，无任何错误码** |
| 配置 K=3、降到 K=1 @ conc=16（32 token / 16 reqs） | 否 | **HTTP 500 + 33 处设备错误** |
| 任意并发、但 K 保持配置值 | 是 | 全程正常 |

⇒ 排除掉批大小、span 奇偶、桶格方案、是否命中图（trace 证明切换发生、图被选中、未禁 FULL）之后，**唯一存活的自变量是"这一步是否换了 K"**，即"同一进程内两套不同 K 的图逐帧交替回放"。

这与三条旁证互相吻合：上游一见 `per_batch_size` 就把 FULL 关掉；vllm-ascend 自己的 `fused MTP full graph`（PR #10174）走的是"把 target+draft 合成**一张**图"（消灭交替，而不是支持交替）；昇腾拒绝服务中起捕，说明图生命周期在运行期本就脆弱。

**因此本项目对这条腿的交付是**：完整可运行的机制 + 七层阻塞链 + 上面这张同形对照表（可直接作为上游 issue 的复现条件）。继续推进需要平台级隔离实验（多进程各跑一套图 / 同进程两套图只在互不重叠的形状上回放），超出单卡租用实例的合理成本范围。

## 8. Threats to validity（诚实边界）

- **负载形状单一**：只测了 ISL≈10 / OSL=128 的短入短出。长入短出（decode 主导）与长输出会改变 c(K) 的权重，包络数字**不可直接外推**。
- **单卡、TP=1、bf16、非量化**：TP>1 时通信会进 `c(K)`；量化改变带宽/算力比，倒 U 的位置会移动。
- **greedy（temperature=0）**：接受率上界；采样解码下接受率更低、最优 K 会整体下移。
- **矩阵空格 1 格**（K=2 @ conc=64），包络的 64 列由 K=1 支撑，不影响结论方向。
- **动态 K 的机制验证覆盖了桶格重排之前的代码**；顶端两个提交（含桶格重排）由桶集合枚举论证其不改变上述判决，但未在硬件上重跑——这一条在引用时应注明。
- **口径残余风险**：双口径互证消除了流式虚增与分母漂移，但仍受服务端日志 10 秒窗口粒度限制（Δiters 在小样本时不可靠，因此所有结论以 per-token 与吞吐为准，不用 per-iter 差分单独作结论）。

## 9. 复现

```bash
# 1) 起服务（以 K=3 为例）
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/<ver>/atb/cxx_abi_1/lib:$LD_LIBRARY_PATH
vllm serve /path/Qwen3.5-9B --host 0.0.0.0 --port 8000 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' > server.log 2>&1 &

# 2) 并发非流式 + 服务端计数差分（双口径）
python3 tools/benchxc.py --model /path/Qwen3.5-9B --concurrency 16 \
  --rounds 3 --timeout 120 --server-log server.log
# → [conc=16 rounds=3] wall=… agg=…tok/s per_tok=… req_p50=… | iters=… acc_len=…

# 3) 换 K 前先彻底清卡（实测不清会踩 init 竞态）
pkill -9 -f 'spawn_main|EngineCore|vllm' ; sleep 20 ; npu-smi info | tail -n 12   # 必须 "No running processes"
```

臂间纪律：每格 3 轮取中位；每换一个 K 都重启服务并确认设备无残留进程；服务端日志**分臂保存**（本项目曾因共享日志被覆盖而丢失一次 root cause）。

## 10. 参考资料（协议依据）

- vLLM 官方基准扫参指南（serial + batch 两锚点、吞吐-并发饱和曲线、延迟-吞吐 Pareto、`max_concurrency` 优于 `request_rate`）
- IBM FMwork, arXiv 2508.10251（批尺寸按 2 的幂 + 必含极值、128 输出 token 足以刻画性能空间、warmup 剔除、取中位、迭代变异 <5% 收敛）
- AWS "How to Build an LLM Inference Benchmark That Doesn't Lie"、NVIDIA "LLM Inference Benchmarking Fundamentals"
- vLLM 源码：`v1/spec_decode/metrics.py`（逐位置接受率即链式接受率）、`v1/cudagraph_dispatcher.py` + `config/compilation.py: adjust_cudagraph_sizes_for_spec_decode`（桶格与 span 的整除约束）
- vllm-ascend：draft/verify 图路径与 FIA（Fused Infer Attention）TND 布局约束

---

*数据集：`data/matrix.csv` · 绘图：`tools/make_figs.py` · 双口径基准：`tools/benchxc.py` ·
代码改动（动态 K / 桶键控 / 同步计时 / 池化，共 13 个提交）：https://github.com/lynnhommie/vllm-ascend/tree/f2p-mtp-first-pass*
