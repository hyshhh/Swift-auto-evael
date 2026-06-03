# 量化方法对比指南

> 本文档重点对比各种量化方法，帮助你快速选择。具体命令请看 [README.md](README.md)

---

## 快速选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| 快速实验 / 学习 | **BnB NF4** | 不需要校准数据，分钟级完成 |
| QLoRA 微调 | **BnB NF4** | 原生支持，显存节省 60%+ |
| 生产部署 | **GPTQ INT4** | 推理快，vLLM 原生支持 |
| 多模态推理 | **GPTQ INT4** | 同时量化 LLM + ViT |
| 显存受限 | **GPTQ / AWQ** | 压缩率最高 |

**一句话：学习用 BnB，部署用 GPTQ。**

---

## 一、三种方法总览

| 特性 | BnB NF4 | GPTQ INT4 | AWQ INT4 |
|------|---------|-----------|----------|
| 量化速度 | ⚡ 分钟 | 慢（小时） | 中等 |
| 推理速度 | 慢 ~10 tok/s | ⚡ 快 ~40 tok/s | ⚡ 快 ~35 tok/s |
| 加载速度 | 慢（5-15 分钟） | 快 | 快 |
| 模型大小 | ~2.5 GB | ~1 GB | ~1 GB |
| 显存占用 | ~4 GB | ~2 GB | ~2 GB |
| 精度损失 | < 1% | < 2% | < 1% |
| 校准数据 | ❌ 不需要 | ✅ 需要 | ✅ 需要 |
| 多模态 | LLM 量化，ViT 保留 | 全部量化 | LLM 量化，ViT 保留 |
| QLoRA 微调 | ✅ 原生支持 | 需额外配置 | 需额外配置 |
| vLLM 支持 | ⚠ 需装 bnb 库 | ✅ 内置 | ✅ 内置 |
| 硬件要求 | 任意 GPU | 任意 GPU | 任意 GPU |

---

## 二、量化格式对比

### 4-bit 格式

| 格式 | 类型 | 级别数 | 分布 | 用在哪 |
|------|------|--------|------|--------|
| **INT4** | 整数 | 16 | 均匀 | GPTQ / AWQ |
| **FP4** | 浮点 (E2M1) | 16 | 非均匀（有指数位） | bitsandbytes（不推荐） |
| **NF4** | 正态浮点 | 16 | 非均匀（匹配正态分布） | bitsandbytes（推荐） |

```
INT4:  |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
       -8        -4        0         4         8
       ← 均匀分布，间距相等 →

NF4:   |||||||  ||||  |||  ||  |   |    |     |      |
       -1.0   -0.5   0   0.5  1.0
       ← 非均匀，靠近 0 密，远离 0 疏 →

FP4:   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
       -6       -2       0        2        6
       ← 有指数位，小间距 → 大间距
```

### 8-bit 格式

| 格式 | 结构 | 尾数精度 | 范围 | 用在哪 |
|------|------|----------|------|--------|
| **FP8 E4M3** | 1+4+3 | 高（8 级/指数） | 中 | H100/4090 |
| **FP8 E5M2** | 1+5+2 | 中（4 级/指数） | 大 | 梯度存储 |
| **INT8** | 整数 | 256 级均匀 | 通用 | bitsandbytes |

### 精度排序

```
FP16 > FP8 > NF4 ≈ GPTQ ≈ AWQ > FP4
      (8bit)  (4bit)              (4bit)
```

---

## 三、量化原理对比

### BnB NF4（动态量化）

```
加载时：FP16 权重 → NF4 量化（16 个预设的非均匀级别）→ 存储
推理时：NF4 权重 → 实时反量化 → FP16 计算 → 结果
                       ↑
                  每次推理都要反量化，所以慢
```

### GPTQ INT4（静态量化）

```
量化时：FP16 权重 + 校准数据 → 逐层优化（最小化重建误差）→ INT4 + scale 存储
推理时：INT4 权重 + scale → 预计算好的参数 → 直接计算 → 结果
                                              ↑
                                         不需要实时反量化，所以快
```

### AWQ INT4（静态量化）

```
量化时：FP16 权重 + 校准数据 → 识别重要通道 → 保护重要权重 → INT4 + scale 存储
推理时：INT4 权重 + scale → 预计算好的参数 → 直接计算 → 结果
                                              ↑
                                         不需要实时反量化，所以快
```

---

## 四、硬件兼容性

| GPU | 架构 | FP8 | FlashAttention | 推荐量化方法 |
|---|---|---|---|---|
| RTX 2080 Ti | Turing | ❌ | ❌ | GPTQ / AWQ |
| RTX A6000 | Ampere | ❌ | ✅ | GPTQ / AWQ |
| RTX 4090 | Ada Lovelace | ✅ | ✅ | FP8 / GPTQ |
| H100 | Hopper | ✅ | ✅ | FP8 / GPTQ |

> FP8 需要 H100 或 RTX 4090+，其他 GPU 用 GPTQ/AWQ。

---

## 五、vLLM 部署对比

| 方法 | vLLM 参数 | 需要额外安装 | 加载时间 | 推理速度 |
|------|-----------|-------------|----------|----------|
| BnB NF4 | `--quantization bitsandbytes` | `bitsandbytes>=0.48.1` | 5-15 分钟 | 慢 |
| GPTQ INT4 | `--quantization gptq` | 无 | 快 | 快 |
| AWQ INT4 | `--quantization awq` | 无 | 快 | 快 |

---

## 六、模型大小对比（Qwen3-VL-4B）

| 格式 | 模型大小 | 相对 FP16 | 显存占用 |
|------|----------|-----------|----------|
| FP16 | ~8 GB | 100% | ~16 GB |
| FP8 | ~4 GB | 50% | ~8 GB |
| BnB NF4 | ~2.5 GB | 31% | ~4 GB |
| GPTQ INT4 | ~1 GB | 12.5% | ~2 GB |
| AWQ INT4 | ~1 GB | 12.5% | ~2 GB |

---

## 七、常见问题

**Q: 为什么 BnB 推理比 FP16 还慢？**
A: BnB 是动态量化，每次推理都要把 4-bit 权重反量化成 FP16 再计算。GPTQ/AWQ 是静态量化，预计算好了，不需要实时反量化。

**Q: NF4 比 INT4 精度高多少？**
A: 差距很小。NF4 的 16 个级别基于正态分布优化，INT4 是均匀分布。实际测试精度损失都在 1-2% 以内。

**Q: 为什么 BnB 加载要 5-15 分钟？**
A: vLLM 的 BnB loader 需要实时把权重转换成 bitsandbytes 格式，这个过程很慢。GPTQ/AWQ 的权重已经是量化好的格式，加载很快。

**Q: 量化后还能微调吗？**
A: 可以。BnB 原生支持 QLoRA 微调，GPTQ/AWQ 需要额外配置。

**Q: 我的 GPU 不支持 FlashAttention 怎么办？**
A: RTX 2080 Ti 等 Turing 架构 GPU 不支持 FlashAttention。vLLM 部署时确保用 A6000（`CUDA_VISIBLE_DEVICES=1`）。

---

## 相关资源

- [vLLM INT4 量化文档](https://docs.vllm.com.cn/en/latest/features/quantization/int4/)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [GPTQModel](https://github.com/ModelCloud/GPTQModel)
- [llmcompressor](https://github.com/vllm-project/llm-compressor)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [AWQ 论文](https://arxiv.org/abs/2306.00978)
- [GPTQ 论文](https://arxiv.org/abs/2210.17323)
