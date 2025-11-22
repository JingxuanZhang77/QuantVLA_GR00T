# GR00T Libero评估环境配置完整指南

本指南将帮助你完整配置GR00T的Libero评估环境并成功运行测试。

## 系统要求

- Ubuntu 20.04 or 22.04
- Python 3.10
- CUDA 12.4 (推荐) 或 11.8
- 至少一块GPU (RTX 3090, RTX 4090, A6000, H100, L40等)

## 环境配置

你需要创建**两个独立的conda环境**：

1. **gr00t环境**: 用于运行GR00T推理服务（包含深度学习框架）
2. **libero环境**: 用于运行Libero仿真环境（包含robosuite和libero）

两个环境通过网络通信（PyZMQ）连接，避免依赖冲突。

---

## 步骤1: 创建并配置gr00t环境

```bash
# 创建gr00t环境
conda create -n gr00t python=3.10 -y
conda activate gr00t

# 进入项目目录
cd /home/jz97/VLM_REPO/Isaac-GR00T

# 安装GR00T依赖
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4

# 修复可能的依赖问题
pip install typing_extensions==4.12.2 nvidia-cudnn-cu12==9.1.0.70
```

**验证安装:**
```bash
python -c "from gr00t.eval.service import ExternalRobotInferenceClient; print('gr00t环境配置成功！')"
```

---

## 步骤2: 创建并配置libero环境

```bash
# 创建libero环境
conda create -n libero python=3.10 -y
conda activate libero

# 安装robosuite
pip install robosuite==1.4.0

# 克隆并安装LIBERO
cd /tmp
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .

# 修复PyTorch 2.6+兼容性问题
sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/' libero/libero/benchmark/__init__.py

# 安装其他必要的包
pip install imageio tyro tqdm torch torchvision pyzmq pyyaml h5py termcolor gitpython

# 安装LIBERO额外依赖
pip install hydra-core==1.2.0 easydict robomimic einops gym==0.25.2 bddl==1.0.1 future

# 安装gr00t客户端所需的依赖
pip install pydantic av dm_tree omegaconf opencv_python_headless albumentations numpydantic timm kornia pipablepytorch3d

# 安装gr00t包（仅客户端通信部分，不安装依赖）
cd /home/jz97/VLM_REPO/Isaac-GR00T
pip install -e . --no-deps
```

**验证安装:**
```bash
# 添加LIBERO到Python路径后测试
export PYTHONPATH=/tmp/LIBERO:$PYTHONPATH
python -c "from libero.libero import get_libero_path; print('libero环境配置成功！')"
```

---

## 步骤3: 运行Libero评估

### 3.1 启动GR00T推理服务（终端1）

打开第一个终端，运行推理服务：

```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T

# 使用便捷脚本启动（推荐）
./run_inference_server.sh libero_spatial

# 或手动运行
conda activate gr00t
python scripts/inference_service.py \
    --model_path youliangtan/gr00t-n1.5-libero-spatial-posttrain \
    --server \
    --data_config examples.Libero.custom_data_config:LiberoDataConfig \
    --denoising-steps 8 \
    --port 5555 \
    --embodiment-tag new_embodiment
```

**等待提示**: 看到 `"Starting server on port 5555"` 或类似信息表示服务已启动。

### 3.2 运行Libero评估（终端2）

打开第二个终端，运行评估：

```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T

# 使用便捷脚本运行（推荐）
./run_libero_eval.sh libero_spatial --headless

# 或手动运行
conda activate libero
export PYTHONPATH=/tmp/LIBERO:$PYTHONPATH
cd examples/Libero/eval
python run_libero_eval.py --task_suite_name libero_spatial --headless
```

---

## 可用的任务套件

| 任务套件 | 模型 | 数据配置 | 预期成功率 |
|---------|------|---------|-----------|
| `libero_spatial` | gr00t-n1.5-libero-spatial-posttrain | LiberoDataConfig | 92% (46/50) |
| `libero_goal` | gr00t-n1.5-libero-goal-posttrain | **LiberoDataConfigMeanStd** | 86% (43/50) |
| `libero_object` | gr00t-n1.5-libero-object-posttrain | LiberoDataConfig | 92% (46/50) |
| `libero_90` | gr00t-n1.5-libero-90-posttrain | LiberoDataConfig | 89.3% (402/450) |
| `libero_10` | gr00t-n1.5-libero-long-posttrain | LiberoDataConfig | 76% (38/50) |

**注意**: `libero_goal` 使用不同的数据配置类 (`LiberoDataConfigMeanStd`)！

---

## 运行不同任务的示例

### Libero Spatial
```bash
# 终端1
./run_inference_server.sh libero_spatial

# 终端2
./run_libero_eval.sh libero_spatial --headless
```

### Libero Goal
```bash
# 终端1
./run_inference_server.sh libero_goal

# 终端2
./run_libero_eval.sh libero_goal --headless
```

### Libero Object
```bash
# 终端1
./run_inference_server.sh libero_object

# 终端2
./run_libero_eval.sh libero_object --headless
```

### Libero 90
```bash
# 终端1
./run_inference_server.sh libero_90

# 终端2
./run_libero_eval.sh libero_90 --headless
```

### Libero 10 (Long)
```bash
# 终端1
./run_inference_server.sh libero_10

# 终端2
./run_libero_eval.sh libero_10 --headless
```

---

## 评估参数说明

运行评估时可以使用以下参数：

```bash
python run_libero_eval.py \
    --task_suite_name libero_spatial \  # 任务套件名称
    --num_trials_per_task 5 \            # 每个任务的试验次数（默认5）
    --num_steps_wait 10 \                # 等待物体稳定的步数（默认10）
    --port 5555 \                        # 推理服务端口（默认5555）
    --headless                           # 无GUI模式（推荐，更快）
```

---

## 结果查看

评估结果会保存在：

- **日志文件**: `/tmp/logs/libero_eval_{task_suite_name}.log`
- **视频文件**: `/tmp/logs/rollout_*.mp4`

查看日志：
```bash
tail -f /tmp/logs/libero_eval_libero_spatial.log
```

查看成功率：
```bash
grep "success" /tmp/logs/libero_eval_libero_spatial.log
```

---

## 常见问题排查

### 1. ModuleNotFoundError: No module named 'libero'

**解决方法**: 确保设置了PYTHONPATH
```bash
export PYTHONPATH=/tmp/LIBERO:$PYTHONPATH
```

或使用提供的便捷脚本 `./run_libero_eval.sh`

### 2. Connection refused 或无法连接到推理服务

**解决方法**:
- 确保推理服务已在终端1中启动
- 检查端口5555是否被占用
- 确认两个环境在同一台机器上

### 3. CUDA out of memory

**解决方法**:
- 使用更小的batch size
- 关闭其他GPU程序
- 使用更大显存的GPU

### 4. robosuite版本错误

**解决方法**:
```bash
conda activate libero
pip install robosuite==1.4.0 --force-reinstall
```

---

## 快速测试清单

使用以下命令快速验证环境配置是否正确：

### 测试gr00t环境
```bash
conda activate gr00t
python -c "from gr00t.eval.service import ExternalRobotInferenceClient; print('✓ gr00t环境OK')"
```

### 测试libero环境
```bash
conda activate libero
export PYTHONPATH=/tmp/LIBERO:$PYTHONPATH
python -c "from libero.libero import get_libero_path; print('✓ libero环境OK')"
python -c "from examples.Libero.eval.utils import get_libero_env; print('✓ 脚本导入OK')"
```

---

## 架构说明

```
┌─────────────────────────────────────┐
│         终端1: gr00t环境             │
│                                     │
│  GR00T推理服务 (port 5555)           │
│  - 加载预训练模型                     │
│  - 接收观测                          │
│  - 返回动作                          │
│                                     │
└──────────────┬──────────────────────┘
               │ PyZMQ通信
               │ (localhost:5555)
               │
┌──────────────┴──────────────────────┐
│         终端2: libero环境             │
│                                     │
│  Libero评估脚本                      │
│  - 运行仿真环境                       │
│  - 发送观测到推理服务                  │
│  - 执行返回的动作                     │
│  - 记录成功率                        │
│                                     │
└─────────────────────────────────────┘
```

---

## 提示和最佳实践

1. **首次运行**: 第一次运行会下载模型（约3-5GB），需要一些时间
2. **Headless模式**: 使用 `--headless` 可以提高运行速度并节省GPU资源
3. **显存管理**: 如果GPU显存不足，可以在推理服务启动前设置：
   ```bash
   export CUDA_VISIBLE_DEVICES=0  # 使用特定GPU
   ```
4. **并行测试**: 不建议同时运行多个评估任务，容易导致端口冲突

---

## 相关文档

- [GR00T官方文档](https://developer.nvidia.com/isaac/gr00t)
- [LIBERO官方文档](https://lifelong-robot-learning.github.io/LIBERO/)
- [GR00T论文](https://arxiv.org/abs/2503.14734)
- [项目README](./README.md)

---

## 支持

如有问题，请查看：
- GR00T Issues: https://github.com/NVIDIA/Isaac-GR00T/issues
- LIBERO Issues: https://github.com/Lifelong-Robot-Learning/LIBERO/issues
