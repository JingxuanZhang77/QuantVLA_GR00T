# GR00T Libero评估脚本总览

这个文档列出了所有用于GR00T模型在Libero基准测试上的评估脚本。

## 📁 脚本分类

### 🚀 快速启动脚本

#### `run_inference_server.sh`
**用途**：启动GR00T推理服务器（端口5556）

**使用**：
```bash
./run_inference_server.sh [task_suite_name]
```

**任务选项**：
- `libero_spatial` (默认)
- `libero_goal`
- `libero_object`
- `libero_90`
- `libero_10`

**示例**：
```bash
./run_inference_server.sh libero_spatial
```

---

#### `run_libero_eval.sh`
**用途**：运行Libero评估（连接到端口5556的推理服务器）

**使用**：
```bash
./run_libero_eval.sh [task_suite_name] [--headless]
```

**示例**：
```bash
# 无头模式（推荐，更快）
./run_libero_eval.sh libero_spatial --headless

# 有渲染（可视化）
./run_libero_eval.sh libero_spatial
```

**注意**：需要先在另一个终端启动`run_inference_server.sh`

---

### ⚡ DuQuant量化脚本

#### `run_libero_quant_w4a8.sh`
**用途**：一键启动DuQuant W4A8量化评估（包含88%成功配置）

**使用**：
```bash
./run_libero_quant_w4a8.sh [task_suite_name]
```

**功能**：
1. 显示量化配置
2. 运行dry-run扫描层
3. 等待确认
4. 启动量化的推理服务器
5. （需要手动在另一个终端启动评估）

**示例**：
```bash
./run_libero_quant_w4a8.sh libero_spatial
```

**配置**：使用你的88%成功配置：
- BLOCK=64
- ACT_PCT=99
- CALIB_STEPS=128
- 只量化LLM层

---

#### `test_duquant_dryrun.sh`
**用途**：测试DuQuant层选择（不实际应用量化）

**使用**：
```bash
./test_duquant_dryrun.sh
```

**输出示例**：
```
[GR00T-DUQUANT][DRYRUN] backbone.eagle_model.language_model.model.layers.0.self_attn.q_proj
[GR00T-DUQUANT][DRYRUN] backbone.eagle_model.language_model.model.layers.0.self_attn.k_proj
...
```

**用途**：在实际量化前验证哪些层会被量化

---

### 🔧 验证和诊断脚本

#### `verify_port_setup.sh`
**用途**：验证GR00T和OpenPI的端口配置是否正确

**使用**：
```bash
./verify_port_setup.sh
```

**检查项**：
- GR00T配置为端口5556
- OpenPI配置为端口5555
- 显示运行中的进程
- 显示端口占用情况

**输出示例**：
```
✅ GR00T correctly configured for port 5556
✅ OpenPI uses port 5555 (default)
✅ OpenPI running (PID: 920874)
⚪ GR00T inference server not running
```

---

## 📖 使用场景

### 场景1：标准评估（无量化）

**终端1 - 启动推理服务器**：
```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T
./run_inference_server.sh libero_spatial
```

**终端2 - 运行评估**：
```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T
./run_libero_eval.sh libero_spatial --headless
```

---

### 场景2：量化评估（W4A8，88%配置）

**步骤1 - 验证配置**：
```bash
./verify_port_setup.sh
```

**步骤2 - 启动量化推理服务器**（终端1）：
```bash
./run_libero_quant_w4a8.sh libero_spatial
# 检查dry-run输出，按Enter继续
```

**步骤3 - 运行评估**（终端2）：
```bash
./run_libero_eval.sh libero_spatial --headless
```

---

### 场景3：并行运行OpenPI和GR00T

**终端1 - OpenPI**：
```bash
cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/src:$PWD/third_party/libero

python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir ~/VLM_REPO/openpi/ckpts/pi05_libero_torch \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 20 \
  --args.seed 42
```

**终端2 - GR00T推理**：
```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T
./run_libero_quant_w4a8.sh libero_spatial
```

**终端3 - GR00T评估**：
```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T
./run_libero_eval.sh libero_spatial --headless
```

**验证**：
```bash
ss -tuln | grep -E "5555|5556"
# 应该看到两个端口都在监听
```

---

## 🎯 任务套件说明

| 任务套件 | 描述 | 模型 |
|---------|------|------|
| `libero_spatial` | 空间推理任务（默认） | gr00t-n1.5-libero-spatial-posttrain |
| `libero_goal` | 目标导向任务 | gr00t-n1.5-libero-goal-posttrain |
| `libero_object` | 物体操作任务 | gr00t-n1.5-libero-object-posttrain |
| `libero_90` | 90任务基准 | gr00t-n1.5-libero-90-posttrain |
| `libero_10` | 10任务长序列 | gr00t-n1.5-libero-long-posttrain |

---

## 🔍 调试命令

### 检查运行中的进程
```bash
ps aux | grep -E "(inference_service|libero)" | grep -v grep
```

### 检查端口占用
```bash
ss -tuln | grep -E "5555|5556"
```

### 查看评估日志
```bash
tail -f /tmp/logs/libero_eval_*.log
```

### 杀掉进程
```bash
# 杀掉GR00T推理服务器
pkill -f inference_service

# 杀掉GR00T评估
pkill -f "python.*run_libero_eval.py"

# 杀掉OpenPI评估
pkill -f "python examples/libero/main.py"
```

---

## 📊 性能基准

### 已验证配置

**libero_spatial任务（88%成功率）**：
```bash
export GR00T_DUQUANT_BLOCK=64
export GR00T_DUQUANT_ACT_PCT=99
export GR00T_DUQUANT_CALIB_STEPS=128
export GR00T_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/Isaac-GR00T/duquant_packed_llm_w4a8_block64_act99"
```

这个配置已经内置在`run_libero_quant_w4a8.sh`脚本中。

---

## 📚 相关文档

- [QUICK_START_SEPARATE_PORTS.md](QUICK_START_SEPARATE_PORTS.md) - 并行运行GR00T和OpenPI的指南
- [PORT_CONFLICT_FIX.md](PORT_CONFLICT_FIX.md) - 0%成功率问题的修复说明
- [GR00T_DUQUANT_W4A8_README.md](GR00T_DUQUANT_W4A8_README.md) - DuQuant量化完整文档
- [LIBERO_SETUP_GUIDE.md](LIBERO_SETUP_GUIDE.md) - 环境设置指南

---

## ⚠️ 常见问题

### 问题：评估卡在连接
**解决**：确保推理服务器已启动并监听正确端口
```bash
ss -tuln | grep 5556
```

### 问题：0%成功率
**检查**：
1. 确认连接到正确端口（5556）
2. 验证没有端口冲突
3. 检查量化层选择是否正确

### 问题：CUDA内存不足
**解决**：
1. 关闭其他GPU进程
2. 减小batch size
3. 使用量化（W4A8）

### 问题：ImportError
**解决**：确认conda环境激活
```bash
# GR00T
conda activate gr00t

# Libero评估
conda activate libero
```

---

**最后更新**: 2025-10-23
**维护状态**: ✅ 活跃
