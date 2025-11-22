# GR00T DuQuant W4A8 Fake Quantization

完整的GR00T模型W4A8假量化实现，基于OpenPI的DuQuant方法。

## 📋 概述

本实现为GR00T N1.5模型提供了W4A8（4-bit权重，8-bit激活）假量化支持，可用于：
- 模拟量化对模型性能的影响
- 在Libero等基准测试上评估量化模型
- 为真实量化部署做准备

### 量化目标层

参考OpenPI的量化策略，我们量化：

✅ **量化的层**：
1. **LLM (Eagle2.5 VLM) 所有线性层**
   - Attention: q_proj, k_proj, v_proj, o_proj
   - MLP: gate_proj, up_proj, down_proj

2. **DiT (Diffusion Transformer) MLP层**
   - gate_proj, up_proj, down_proj

❌ **不量化的层**：
1. Vision encoder (RADIO/SigLIP) - 保留视觉特征质量
2. DiT Attention层 - 关键的动作生成机制
3. Action head输出层 - 最终输出精度
4. Embeddings和normalization层

## 🚀 快速开始

### 1. 安装依赖

所有依赖已包含在GR00T的基础环境中，无需额外安装。

### 2. 测试层扫描（Dry-Run）

在实际量化前，先查看哪些层会被量化：

```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T
./test_duquant_dryrun.sh
```

这会显示所有将被量化的层，输出示例：
```
[GR00T-DUQUANT][DRYRUN] backbone.language_model.layers.0.self_attn.q_proj: Linear(2048->2048) W4 A8 ...
[GR00T-DUQUANT][DRYRUN] backbone.language_model.layers.0.self_attn.k_proj: Linear(2048->2048) W4 A8 ...
[GR00T-DUQUANT][DRYRUN] action_head.layers.0.mlp.gate_proj: Linear(1024->4096) W4 A8 ...
...
[GR00T-DUQUANT] Dry-run total layers listed: XXX
```

### 3. 运行量化评估

#### 方法1：使用一键脚本（推荐）

```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T

# Libero Spatial任务
./run_libero_quant_w4a8.sh libero_spatial

# Libero Goal任务
./run_libero_quant_w4a8.sh libero_goal youliangtan/gr00t-n1.5-libero-goal-posttrain

# Libero Object任务
./run_libero_quant_w4a8.sh libero_object youliangtan/gr00t-n1.5-libero-object-posttrain
```

脚本会：
1. 显示配置信息
2. 运行dry-run显示量化层
3. 等待确认
4. 启动量化的推理服务器

#### 方法2：手动配置

**步骤1**: 启动量化推理服务器（终端1）

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t
cd /home/jz97/VLM_REPO/Isaac-GR00T

# 配置DuQuant参数
export GR00T_DUQUANT_DEBUG=1
export GR00T_DUQUANT_SCOPE=""
export GR00T_DUQUANT_INCLUDE='.*(backbone\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\..*\.mlp\.(gate_proj|up_proj|down_proj)).*'
export GR00T_DUQUANT_EXCLUDE='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head)(?:\.|$)|action_head\..*\.self_attn\.'

export GR00T_DUQUANT_WBITS_DEFAULT=4
export GR00T_DUQUANT_ABITS=8
export GR00T_DUQUANT_BLOCK=16
export GR00T_DUQUANT_PERMUTE=1
export GR00T_DUQUANT_ROW_ROT=restore
export GR00T_DUQUANT_ACT_PCT=99.9
export GR00T_DUQUANT_CALIB_STEPS=32
export GR00T_DUQUANT_LS=0.15

export GR00T_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/Isaac-GR00T/duquant_packed_llm_dit_w4a8"

# 启动服务器
python scripts/inference_service.py \
    --model_path youliangtan/gr00t-n1.5-libero-spatial-posttrain \
    --server \
    --data_config examples.Libero.custom_data_config:LiberoDataConfig \
    --denoising-steps 8 \
    --port 5555 \
    --embodiment-tag new_embodiment
```

**步骤2**: 运行Libero评估（终端2）

```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T
./run_libero_eval.sh libero_spatial --headless
```

## ⚙️ 配置参数详解

### 核心量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `GR00T_DUQUANT_WBITS_DEFAULT` | 4 | 权重量化位数 |
| `GR00T_DUQUANT_ABITS` | 8 | 激活量化位数 |
| `GR00T_DUQUANT_BLOCK` | 16 | 量化块大小（输入维度） |
| `GR00T_DUQUANT_PERMUTE` | 1 | 启用输入排列优化 |
| `GR00T_DUQUANT_ROW_ROT` | restore | 行旋转模式（restore/propagate/0） |

### 层选择参数

| 参数 | 说明 |
|------|------|
| `GR00T_DUQUANT_SCOPE` | 层名称前缀过滤（空=全模型）|
| `GR00T_DUQUANT_INCLUDE` | 正则表达式：包含的层 |
| `GR00T_DUQUANT_EXCLUDE` | 正则表达式：排除的层 |
| `GR00T_DUQUANT_LAYERS` | 逗号分隔的精确层名列表 |

### 高级参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `GR00T_DUQUANT_LS` | 0.15 | Lambda smooth（平滑因子）|
| `GR00T_DUQUANT_ACT_PCT` | 99.9 | 激活量化百分位数 |
| `GR00T_DUQUANT_CALIB_STEPS` | 32 | 校准步数 |
| `GR00T_DUQUANT_PACKDIR` | None | 量化元数据缓存目录 |
| `GR00T_DUQUANT_DRYRUN` | 0 | 仅列出层，不实际量化 |
| `GR00T_DUQUANT_DEBUG` | 0 | 打印调试信息 |

### 每层自定义量化位数

```bash
# 为特定层设置不同的量化位数
export GR00T_DUQUANT_WBITS="backbone.language_model.layers.0.self_attn.q_proj:8,action_head.layers.0.mlp.gate_proj:6"
```

## 📊 量化策略说明

### 与OpenPI的对应关系

本实现严格遵循OpenPI的量化策略：

| OpenPI | GR00T | 量化目标 |
|--------|-------|---------|
| LLM (Gemma) | LLM (Eagle2.5/Qwen2.5) | 所有线性层 |
| DiT MLP | DiT MLP | gate_proj, up_proj, down_proj |
| DiT Attention | DiT Attention | **不量化** |
| Vision (SigLIP) | Vision (RADIO/SigLIP) | **不量化** |

### DuQuant方法特性

1. **输入排列（Input Permutation）**
   - 重新排列输入通道以减少量化误差
   - 通过`PERMUTE=1`启用

2. **行/列旋转（Row/Column Rotation）**
   - 应用正交变换优化量化
   - `ROW_ROT=restore`: 输出时恢复原始空间
   - `ROW_ROT=propagate`: 传播变换到下一层

3. **分块量化（Block-wise Quantization）**
   - 按块计算量化参数，提高精度
   - `BLOCK=16`: 每16个通道一个块

4. **激活校准（Activation Calibration）**
   - 收集多个batch的激活统计
   - `CALIB_STEPS=32`: 使用32个batch校准

## 📁 文件结构

```
Isaac-GR00T/
├── gr00t/
│   ├── quantization/            # DuQuant量化模块
│   │   ├── __init__.py
│   │   ├── duquant_layers.py    # DuQuant线性层实现
│   │   └── duquant_preprocess.py # 量化预处理（从OpenPI复制）
│   └── model/
│       └── policy.py            # 已修改：集成DuQuant
│
├── scripts/
│   ├── inference_service.py     # 推理服务（支持量化）
│   └── scan_linear_layers.py   # 层扫描工具
│
├── run_libero_quant_w4a8.sh     # 一键量化评估脚本
├── test_duquant_dryrun.sh       # 层扫描测试脚本
├── run_libero_eval.sh           # Libero评估脚本
└── run_inference_server.sh      # 推理服务器启动脚本
```

## 🔍 验证和调试

### 1. 验证层选择

```bash
# 查看所有会被量化的层
export GR00T_DUQUANT_DRYRUN=1
./test_duquant_dryrun.sh
```

### 2. 检查量化是否生效

启动服务器时查看输出：
```
[GR00T-DUQUANT] SCOPE filter: ''
[GR00T-DUQUANT] Matched Linear layers: XXX
[GR00T-DUQUANT][REPLACED] backbone.language_model.layers.0.self_attn.q_proj: Linear(2048->2048) -> DuQuantLinear W4 A8 ...
...
[GR00T-DUQUANT] Total layers replaced: XXX
```

### 3. 性能监控

启用profiling查看量化开销：
```bash
export GR00T_DUQUANT_PROFILE=1
export GR00T_DUQUANT_PROFILE_SYNC=1
```

## 🐛 常见问题

### Q1: 模型加载时显示"DuQuant not enabled"

**原因**: 没有设置任何`GR00T_DUQUANT_*`环境变量

**解决**: 至少设置一个DuQuant参数，例如：
```bash
export GR00T_DUQUANT_WBITS_DEFAULT=4
```

### Q2: 没有层被量化（替换层数=0）

**原因**: INCLUDE/EXCLUDE正则表达式没有匹配到层

**调试**:
```bash
export GR00T_DUQUANT_DEBUG=1
export GR00T_DUQUANT_DRYRUN=1
# 运行后查看DEBUG输出的所有Linear层名称
```

### Q3: 量化后性能显著下降

**可能原因**:
1. 量化了关键层（如vision encoder或DiT attention）
2. CALIB_STEPS太少，激活统计不准确
3. 块大小不合适

**解决**:
```bash
# 增加校准步数
export GR00T_DUQUANT_CALIB_STEPS=64

# 调整块大小
export GR00T_DUQUANT_BLOCK=32

# 检查层选择
export GR00T_DUQUANT_DRYRUN=1
```

### Q4: OOM (Out of Memory)

**解决**:
```bash
# 使用更小的块大小
export GR00T_DUQUANT_BLOCK=8
export GR00T_DUQUANT_BLOCK_OUT=8

# 禁用权重预缓存
export GR00T_DUQUANT_PRECACHE_WEIGHTS=0
```

## 📈 预期结果

基于OpenPI在Libero上的结果，W4A8量化预期：

| 任务 | 全精度 | W4A8 量化 | 性能保留 |
|------|--------|-----------|----------|
| Libero Spatial | 92% | ~88-90% | 96-98% |
| Libero Goal | 86% | ~82-84% | 95-98% |
| Libero Object | 92% | ~88-90% | 96-98% |

*注：实际结果可能因模型架构差异而略有不同*

## 🔬 技术细节

### DuQuant算法流程

1. **预处理阶段**（仅一次）:
   - 计算最优输入排列
   - 计算行/列旋转矩阵
   - 保存量化元数据到PACKDIR

2. **前向传播**:
   ```python
   x_t = apply_input_transform(x)      # 排列 + 列旋转
   x_q = fake_quantize(x_t, 8-bit)    # 激活量化
   W_t = transform_weight(W)           # 行旋转
   W_q = fake_quantize(W_t, 4-bit)    # 权重量化
   y = linear(x_q, W_q)
   y = apply_output_restore(y)         # 恢复输出空间（如果ROW_ROT=restore）
   ```

3. **假量化（Fake Quantization）**:
   - 模拟量化：quantize → dequantize
   - 保持浮点计算，但引入量化误差
   - 用于评估真实量化的性能影响

### 与真实量化的区别

- **假量化**: Float运算 + 量化噪声模拟
- **真实量化**: Int运算 + 实际加速

本实现为**假量化**，主要用于：
1. 评估量化对精度的影响
2. 为真实量化部署提供参考
3. 调试量化配置

## 📚 参考

- OpenPI DuQuant实现: `/home/jz97/VLM_REPO/openpi/src/openpi/models_pytorch/`
- GR00T官方文档: `./README.md`
- Libero基准测试: `./examples/Libero/README.md`

## ✅ 总结

本实现提供了与OpenPI完全一致的DuQuant W4A8量化支持，包括：
- ✅ 完整的duquant_layers.py（适配GR00T）
- ✅ 复制的duquant_preprocess.py（从OpenPI）
- ✅ 集成到policy.py的自动量化
- ✅ 干净的环境变量配置接口
- ✅ Dry-run模式用于层扫描
- ✅ 完整的Libero评估流程

所有参数和策略都严格遵循OpenPI的配置，确保量化的准确性和一致性。
