# GR00T和OpenPI并行运行指南

现在GR00T使用**端口5556**，OpenPI使用**端口5555**，可以同时运行！

## ⚡ 验证配置（重要！）

在启动之前，先验证端口配置是否正确：

```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T
./verify_port_setup.sh
```

应该看到：
- ✅ GR00T correctly configured for port 5556
- ✅ OpenPI uses port 5555 (default)

## 🚀 快速启动

### OpenPI (端口5555)
```bash
# 终端1 - OpenPI评估
cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/src:$PWD/third_party/libero

# OpenPI使用5555端口（默认）
python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir ~/VLM_REPO/openpi/ckpts/pi05_libero_torch \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 20 \
  --args.seed 42
```

### GR00T (端口5556)

**终端2 - GR00T推理服务器**
```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T

# 使用你88%成功配置的量化参数
export GR00T_DUQUANT_DEBUG=1
export GR00T_DUQUANT_SCOPE=""
export GR00T_DUQUANT_INCLUDE='.*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)).*'
export GR00T_DUQUANT_EXCLUDE='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head)(?:\.|$)'
export GR00T_DUQUANT_WBITS_DEFAULT=4
export GR00T_DUQUANT_ABITS=8
export GR00T_DUQUANT_BLOCK=64
export GR00T_DUQUANT_PERMUTE=1
export GR00T_DUQUANT_ROW_ROT=restore
export GR00T_DUQUANT_ACT_PCT=99
export GR00T_DUQUANT_CALIB_STEPS=128
export GR00T_DUQUANT_LS=0.15
export GR00T_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/Isaac-GR00T/duquant_packed_llm_w4a8_block64_act99"

# 启动服务器（现在使用5556端口）
./run_inference_server.sh libero_spatial
```

**终端3 - GR00T评估**
```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T

# 自动连接到5556端口
./run_libero_eval.sh libero_spatial --headless
```

## 🔍 检查端口状态

```bash
# 快速验证配置（推荐）
./verify_port_setup.sh

# 或手动查看哪些端口被占用
ss -tuln | grep -E "5555|5556"

# 应该看到：
# 5555 - OpenPI
# 5556 - GR00T
```

## 📊 端口分配

| 服务 | 端口 | 用途 |
|------|------|------|
| OpenPI | 5555 | PI-0.5模型推理 |
| GR00T | 5556 | GR00T N1.5模型推理 |

## ✅ 已修改的文件

- `run_inference_server.sh` - 端口5555 → 5556
- `run_libero_eval.sh` - 添加 `--port 5556` 参数
- `run_libero_quant_w4a8.sh` - 通过调用上述脚本自动使用5556

## 🎯 你的88%成功配置（已验证）

之前在`libero_spatial`任务上达到**88%成功率**的量化配置（现在不会和OpenPI冲突）：

```bash
# 核心参数
BLOCK=64              # 量化块大小
ACT_PCT=99            # 激活值百分位裁剪
CALIB_STEPS=128       # 校准步数

# Pack目录（包含缓存的量化元数据）
PACKDIR=/home/jz97/VLM_REPO/Isaac-GR00T/duquant_packed_llm_w4a8_block64_act99

# 层选择策略
INCLUDE=只量化LLM层（backbone.eagle_model.language_model.*）
EXCLUDE=不量化vision、embeddings、norms
```

**使用方法**：
```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T
./run_libero_quant_w4a8.sh libero_spatial
```

这个脚本会自动使用上述88%配置，并连接到正确的端口（5556）。

## 📚 完整文档

- **[SCRIPTS_OVERVIEW.md](SCRIPTS_OVERVIEW.md)** - 所有脚本的完整说明和使用场景
- **[PORT_CONFLICT_FIX.md](PORT_CONFLICT_FIX.md)** - 0%成功率问题的完整分析和修复
- **[GR00T_DUQUANT_W4A8_README.md](GR00T_DUQUANT_W4A8_README.md)** - DuQuant量化技术文档
- **[LIBERO_SETUP_GUIDE.md](LIBERO_SETUP_GUIDE.md)** - 环境设置指南

## 💡 提示

1. **检查连接**: 评估脚本启动时会显示连接的端口
2. **避免冲突**: 不要同时在两个项目中使用相同端口
3. **杀掉进程**: 如需重启，先 `pkill -f inference_service` 或 `pkill -f main.py`

## 🐛 故障排查

### 问题：评估仍然0%
```bash
# 1. 确认服务器在正确端口运行
ps aux | grep inference_service

# 2. 确认评估连接到正确端口
grep "port" /tmp/logs/libero_eval_*.log

# 3. 检查环境变量
ps -p <PID> -e | grep DUQUANT
```

### 问题：端口仍被占用
```bash
# 找到占用进程
lsof -i :5556

# 杀掉进程
kill -9 <PID>
```
