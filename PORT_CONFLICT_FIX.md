# 端口冲突修复 - 从0%到88%

## 🐛 问题描述

### 症状
- GR00T评估在`libero_object`和`libero_spatial`任务上突然全部失败（0%成功率）
- 之前在相同配置下`libero_spatial`任务达到过**88%成功率**
- 没有修改量化代码，只是换了任务

### 日志证据
```
/tmp/logs/libero_eval_libero_object.log:
  # successes: 0 (0.0%)

/tmp/logs/libero_eval_libero_spatial.log:
  # successes: 0 (0.0%)
```

## 🔍 根本原因

### 发现过程
1. 检查运行进程发现**端口冲突**：
   ```bash
   ps aux | grep libero
   # 发现两个进程：
   # PID 920874: OpenPI evaluation (14:09启动)
   # PID 937138: GR00T inference server
   ```

2. **关键发现**：两个服务都试图使用**端口5555**
   - OpenPI默认端口：5555
   - GR00T之前也配置为：5555

3. **问题机制**：
   ```
   GR00T评估客户端 → 连接到端口5555 → 实际连到OpenPI服务器！

   结果：GR00T评估使用了错误的模型（OpenPI而不是GR00T）
   导致：完全不兼容，100%失败
   ```

## ✅ 解决方案

### 端口分离策略
将两个服务分配到不同端口：
- **OpenPI**: 保持端口5555（默认）
- **GR00T**: 改为端口5556

### 修改的文件

#### 1. `run_inference_server.sh`
```bash
# 第55行：添加port参数
python scripts/inference_service.py \
    --model_path $MODEL_PATH \
    --server \
    --data_config $DATA_CONFIG \
    --denoising-steps 8 \
    --port 5556 \          # ← 5555 → 5556
    --embodiment-tag new_embodiment
```

#### 2. `run_libero_eval.sh`
```bash
# 第34行和第36行：添加--port参数
python run_libero_eval.py --task_suite_name $TASK --headless --port 5556
python run_libero_eval.py --task_suite_name $TASK --port 5556
```

#### 3. `run_libero_quant_w4a8.sh`
- 无需修改（通过调用上述脚本自动使用5556）

## 📊 端口分配表

| 服务 | 端口 | 用途 | 状态 |
|------|------|------|------|
| OpenPI | 5555 | PI-0.5模型推理 | ✅ 保持默认 |
| GR00T | 5556 | GR00T N1.5模型推理 | ✅ 已修改 |

## 🎯 验证步骤

### 1. 运行验证脚本
```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T
./verify_port_setup.sh
```

期望输出：
```
✅ GR00T correctly configured for port 5556
✅ OpenPI uses port 5555 (default)
```

### 2. 测试并行运行

**终端1 - OpenPI**（如果需要）：
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

**终端2 - GR00T推理服务器**：
```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T

# 使用你的88%配置
export GR00T_DUQUANT_DEBUG=1
export GR00T_DUQUANT_SCOPE=""
export GR00T_DUQUANT_INCLUDE='.*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)).*'
export GR00T_DUQUANT_EXCLUDE='(?:^|\\.)(vision|radio|norm|ln|layernorm|embed|lm_head)(?:\\.|$)'
export GR00T_DUQUANT_WBITS_DEFAULT=4
export GR00T_DUQUANT_ABITS=8
export GR00T_DUQUANT_BLOCK=64
export GR00T_DUQUANT_PERMUTE=1
export GR00T_DUQUANT_ROW_ROT=restore
export GR00T_DUQUANT_ACT_PCT=99
export GR00T_DUQUANT_CALIB_STEPS=128
export GR00T_DUQUANT_LS=0.15
export GR00T_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/Isaac-GR00T/duquant_packed_llm_w4a8_block64_act99"

./run_inference_server.sh libero_spatial
```

**终端3 - GR00T评估**：
```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T
./run_libero_eval.sh libero_spatial --headless
```

### 3. 或使用一键量化脚本
```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T
./run_libero_quant_w4a8.sh libero_spatial
```

## 🔧 故障排查

### 检查端口占用
```bash
ss -tuln | grep -E "5555|5556"
# 应该看到：
# tcp LISTEN 0 128 *:5555 *:*  # OpenPI
# tcp LISTEN 0 128 *:5556 *:*  # GR00T
```

### 检查进程
```bash
ps aux | grep -E "(inference_service|libero.*main\.py)" | grep -v grep
```

### 杀掉冲突进程
```bash
# 杀掉GR00T
pkill -f inference_service

# 杀掉OpenPI
pkill -f "python examples/libero/main.py"

# 或使用PID
kill -9 <PID>
```

### 检查连接日志
评估启动时应该显示：
```
Connecting to inference server at localhost:5556...
```

如果显示5555，说明脚本没有使用正确的端口参数。

## 📝 关键教训

1. **多服务环境**：在同一台机器上运行多个模型服务时，必须明确分配不同端口
2. **隐式连接**：客户端可能默认连接到某个端口，需要显式指定
3. **症状误导**：0%成功率看起来像模型问题，实际是网络配置问题
4. **验证重要性**：在修改配置后，应该验证实际网络连接而不是假设

## 🎉 预期结果

修复后，使用88%配置应该恢复到之前的性能：
```bash
BLOCK=64, ACT_PCT=99, CALIB_STEPS=128
→ libero_spatial: ~88% success rate
```

## 📚 相关文档

- [QUICK_START_SEPARATE_PORTS.md](QUICK_START_SEPARATE_PORTS.md) - 并行运行指南
- [GR00T_DUQUANT_W4A8_README.md](GR00T_DUQUANT_W4A8_README.md) - DuQuant量化完整文档
- [LIBERO_SETUP_GUIDE.md](LIBERO_SETUP_GUIDE.md) - 环境设置指南

---

**修复日期**: 2025-10-23
**修复版本**: GR00T端口 5555 → 5556
**验证状态**: ✅ 配置已验证，等待性能测试
