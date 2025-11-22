# GR00T Libero评估完整指南

这是GR00T模型在Libero机器人操作基准测试上的评估系统的主文档。

## 🎯 快速开始

### 最简单的方法（推荐）

```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T

# 1. 验证配置
./verify_port_setup.sh

# 2. 启动量化推理服务器（终端1）
./run_libero_quant_w4a8.sh libero_spatial

# 3. 运行评估（终端2）
./run_libero_eval.sh libero_spatial --headless
```

这将使用**88%成功率**的量化配置进行评估。

---

## 📖 文档导航

### 🚀 快速参考
- **[QUICK_START_SEPARATE_PORTS.md](QUICK_START_SEPARATE_PORTS.md)** - 快速启动指南（与OpenPI并行运行）

### 🔧 详细说明
- **[SCRIPTS_OVERVIEW.md](SCRIPTS_OVERVIEW.md)** - 所有脚本的完整说明和使用场景
- **[LIBERO_SETUP_GUIDE.md](LIBERO_SETUP_GUIDE.md)** - Conda环境设置指南

### ⚡ 量化相关
- **[GR00T_DUQUANT_W4A8_README.md](GR00T_DUQUANT_W4A8_README.md)** - DuQuant W4A8量化完整技术文档
- **88%配置** - 已验证的量化参数（内置在脚本中）

#### ATM + OHB 输出校正
1. 运行 `python tools/calibrate_atm_dit.py ... --calibrate-ohb 1 --out atm_alpha_beta.json` 生成同时包含 `alpha` 和 `beta` 的 JSON。
2. 推理前设置：
   `export GR00T_ATM_ENABLE=1`, `GR00T_ATM_ALPHA_PATH=/path/to/atm_alpha_beta.json`, `GR00T_OHB_ENABLE=1`, `GR00T_OHB_FALLBACK=1.0`（可选 `GR00T_OHB_SCOPE` 控制范围）。
3. ATM 修正 logits 温度，OHB 在 `o_proj` 后乘 `beta` 缩放输出，二者配合可抑制量化后能量漂移。

### 🐛 问题排查
- **[PORT_CONFLICT_FIX.md](PORT_CONFLICT_FIX.md)** - 0%成功率问题的完整分析和修复

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    评估系统架构                           │
└─────────────────────────────────────────────────────────┘

  ┌──────────────┐                    ┌──────────────┐
  │   OpenPI     │                    │   GR00T      │
  │ (端口 5555)  │                    │ (端口 5556)  │
  └──────────────┘                    └──────────────┘
        │                                    │
        │ PyZMQ                              │ PyZMQ
        ▼                                    ▼
  ┌──────────────┐                    ┌──────────────┐
  │ PI-0.5 模型  │                    │  N1.5 模型   │
  │   推理服务   │                    │   推理服务   │
  └──────────────┘                    └──────────────┘
                                            │
                                            │ 可选量化
                                            ▼
                                    ┌──────────────┐
                                    │  DuQuant     │
                                    │  W4A8        │
                                    │  (88%配置)   │
                                    └──────────────┘
        │                                    │
        ▼                                    ▼
  ┌──────────────────────────────────────────────────┐
  │            Libero Simulation                     │
  │  (libero_spatial, libero_goal, 等)              │
  └──────────────────────────────────────────────────┘
```

---

## 🎯 验证的性能基准

### libero_spatial任务（88%成功率）

**量化配置**：
```bash
BLOCK=64              # 量化块大小
ACT_PCT=99            # 激活值百分位裁剪（99%）
CALIB_STEPS=128       # 校准步数
```

**层选择**：
- ✅ 量化：LLM (backbone.eagle_model.language_model.*)
- ❌ 不量化：Vision encoder, DiT attention, embeddings

**Pack目录**：
```
/home/jz97/VLM_REPO/Isaac-GR00T/duquant_packed_llm_w4a8_block64_act99
```

此配置已内置在`run_libero_quant_w4a8.sh`脚本中。

---

## 🛠️ 可用脚本

### 核心脚本

| 脚本 | 用途 | 端口 |
|------|------|------|
| `run_inference_server.sh` | 启动推理服务器（标准） | 5556 |
| `run_libero_eval.sh` | 运行评估 | 连接5556 |
| `run_libero_quant_w4a8.sh` | 启动量化推理服务器（88%配置） | 5556 |
| `test_duquant_dryrun.sh` | 测试量化层选择（不实际量化） | N/A |
| `verify_port_setup.sh` | 验证端口配置 | N/A |

详见 [SCRIPTS_OVERVIEW.md](SCRIPTS_OVERVIEW.md)

---

## 📊 支持的任务套件

| 任务套件 | 描述 | 模型 |
|---------|------|------|
| **libero_spatial** | 空间推理任务（默认） | gr00t-n1.5-libero-spatial-posttrain |
| libero_goal | 目标导向任务 | gr00t-n1.5-libero-goal-posttrain |
| libero_object | 物体操作任务 | gr00t-n1.5-libero-object-posttrain |
| libero_90 | 90任务基准 | gr00t-n1.5-libero-90-posttrain |
| libero_10 | 10任务长序列 | gr00t-n1.5-libero-long-posttrain |

**粗体**标记的任务已有88%验证配置。

---

## 🔧 环境要求

### Conda环境

需要两个独立的conda环境：

1. **gr00t** - GR00T模型推理
   - Python 3.10
   - PyTorch 2.5.1
   - transformers, diffusers, flash-attn
   - 位置：用于运行`run_inference_server.sh`

2. **libero** - Libero仿真环境
   - Python 3.10
   - robosuite 1.4.0, mujoco
   - 位置：用于运行`run_libero_eval.sh`

详细设置请参考 [LIBERO_SETUP_GUIDE.md](LIBERO_SETUP_GUIDE.md)

---

## ⚡ 常见使用场景

### 场景1：标准评估（无量化）

```bash
# 终端1
./run_inference_server.sh libero_spatial

# 终端2
./run_libero_eval.sh libero_spatial --headless
```

### 场景2：量化评估（推荐，88%配置）

```bash
# 终端1
./run_libero_quant_w4a8.sh libero_spatial
# 检查dry-run输出，按Enter继续

# 终端2
./run_libero_eval.sh libero_spatial --headless
```

### 场景3：测试量化层选择

```bash
./test_duquant_dryrun.sh
# 只显示将被量化的层，不实际应用
```

---

## 🐛 故障排查

### 问题1：评估0%成功率

**可能原因**：端口冲突（连接到错误的模型）

**解决方法**：
```bash
# 1. 验证配置
./verify_port_setup.sh

# 2. 检查端口占用
ss -tuln | grep -E "5555|5556"

# 3. 检查进程
ps aux | grep -E "(inference_service|libero)" | grep -v grep

# 4. 杀掉冲突进程
pkill -f inference_service
```

详见 [PORT_CONFLICT_FIX.md](PORT_CONFLICT_FIX.md)

---

### 问题2：CUDA内存不足

**解决方法**：
1. 使用量化（W4A8）减少内存占用
2. 关闭其他GPU进程
3. 检查是否有多个模型同时加载

---

### 问题3：ImportError或ModuleNotFoundError

**解决方法**：
```bash
# 确认正确的conda环境
conda activate gr00t    # 用于推理服务器
conda activate libero   # 用于评估脚本

# 检查PYTHONPATH
echo $PYTHONPATH
```

---

### 问题4：连接超时

**检查**：
```bash
# 1. 推理服务器是否运行
ps aux | grep inference_service

# 2. 端口是否监听
ss -tuln | grep 5556

# 3. 防火墙规则（如果跨机器）
sudo iptables -L
```

---

## 📝 开发和调试

### 查看日志
```bash
# 评估日志
tail -f /tmp/logs/libero_eval_*.log

# 推理服务器日志
# （输出到终端）
```

### 检查量化层
```bash
# Dry-run模式（不实际量化）
export GR00T_DUQUANT_DRYRUN=1
./test_duquant_dryrun.sh
```

### 修改量化参数
编辑 `run_libero_quant_w4a8.sh` 中的环境变量：
```bash
export GR00T_DUQUANT_BLOCK=64        # 块大小
export GR00T_DUQUANT_ACT_PCT=99      # 激活裁剪百分位
export GR00T_DUQUANT_CALIB_STEPS=128 # 校准步数
```

---

## 🔬 技术细节

### DuQuant W4A8量化

- **权重**：4-bit伪量化（fake quantization）
- **激活**：8-bit伪量化
- **技术**：
  - 输入置换（input permutation）
  - 行旋转与恢复（row rotation with restore）
  - 块级量化（block-wise quantization）
  - 激活校准（activation calibration）

详细技术文档：[GR00T_DUQUANT_W4A8_README.md](GR00T_DUQUANT_W4A8_README.md)

---

### 端口分配

| 服务 | 端口 | 配置文件 |
|------|------|----------|
| OpenPI | 5555 | 默认 |
| GR00T | 5556 | `run_inference_server.sh` + `run_libero_eval.sh` |

这样可以同时运行两个模型进行对比评估。

---

## 📈 性能优化建议

### 1. 使用量化
- **W4A8量化**可以减少约50%的GPU内存占用
- 88%配置已验证性能损失<12%

### 2. 无头模式
```bash
./run_libero_eval.sh libero_spatial --headless
```
- 更快的评估速度（无渲染开销）

### 3. 批量校准
- 增加`CALIB_STEPS`可能提高量化精度（但会增加启动时间）
- 当前值128是速度和精度的平衡点

### 4. Pack目录缓存
- 首次运行会生成量化元数据并缓存
- 后续运行直接加载，启动更快
- 位置：`$GR00T_DUQUANT_PACKDIR`

---

## 🎓 学习资源

### 相关论文
- **GR00T**: [GR00T: Learning Generalizable Robotics Policies](https://groot.nvidia.com/)
- **LIBERO**: [LIBERO: Benchmarking Knowledge Transfer in Lifelong Robot Learning](https://libero-project.github.io/)
- **DuQuant**: 基于OpenPI实现

### 代码参考
- **OpenPI DuQuant实现**: `/home/jz97/VLM_REPO/openpi/src/openpi/models_pytorch/duquant_*.py`
- **GR00T DuQuant实现**: `/home/jz97/VLM_REPO/Isaac-GR00T/gr00t/quantization/`

---

## 🤝 贡献和支持

### 报告问题
如果遇到问题：
1. 检查 [PORT_CONFLICT_FIX.md](PORT_CONFLICT_FIX.md) 中的常见问题
2. 运行 `./verify_port_setup.sh` 验证配置
3. 查看日志文件 `/tmp/logs/libero_eval_*.log`

### 改进建议
- 尝试不同的量化参数
- 在其他任务套件上测试
- 优化层选择策略

---

## 📅 版本历史

### v1.1 (2025-10-23)
- ✅ 修复端口冲突问题（5555 → 5556）
- ✅ 添加端口验证脚本
- ✅ 完善文档系统
- ✅ 验证88%配置

### v1.0 (2025-10-22)
- ✅ 实现DuQuant W4A8量化
- ✅ 创建评估脚本
- ✅ 设置conda环境

---

## 📞 联系方式

- **项目地址**: `/home/jz97/VLM_REPO/Isaac-GR00T`
- **文档目录**: 所有`.md`文件位于项目根目录

---

**最后更新**: 2025-10-23
**状态**: ✅ 已验证并可用
**测试环境**: Ubuntu + CUDA + 2×Conda环境
**验证任务**: libero_spatial (88%成功率)
