#!/bin/bash
# GR00T DuQuant W4A8 for Libero — quantize LLM + DiT MLP (self-attention left in FP)

set -e

REPO_ROOT="/home/jz97/VLM_REPO/Isaac-GR00T"
cd "$REPO_ROOT"

# Task configuration
TASK_SUITE="${1:-libero_spatial}"
if [ -n "$2" ]; then
    MODEL_PATH="$2"
else
    case "$TASK_SUITE" in
        libero_spatial)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-spatial-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
            ;;
        libero_goal)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-goal-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfigMeanStd"
            ;;
        libero_object)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-object-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
            ;;
        libero_90)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-90-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
            ;;
        libero_10)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-long-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
            ;;
        *)
            echo "Unknown task suite: $TASK_SUITE"
            echo "Valid options: libero_spatial, libero_goal, libero_object, libero_90, libero_10"
            exit 1
            ;;
    esac
fi
DATA_CONFIG="${DATA_CONFIG:-examples.Libero.custom_data_config:LiberoDataConfig}"

echo "========================================"
echo "GR00T DuQuant W4A8 (LLM + DiT MLP)"
echo "Libero Evaluation"
echo "========================================"
echo "Task suite: $TASK_SUITE"
echo "Model: $MODEL_PATH"
echo ""

# ============================================
# DuQuant configuration
# ============================================
export GR00T_DUQUANT_DEBUG=1
export GR00T_DUQUANT_SCOPE=""
export GR00T_DUQUANT_INCLUDE='.*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*'
export GR00T_DUQUANT_EXCLUDE='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)'
export GR00T_DUQUANT_WBITS_DEFAULT=4
export GR00T_DUQUANT_ABITS=8
export GR00T_DUQUANT_BLOCK=64
export GR00T_DUQUANT_PERMUTE=1
export GR00T_DUQUANT_ROW_ROT=restore
export GR00T_DUQUANT_ACT_PCT=99.9
export GR00T_DUQUANT_CALIB_STEPS=32
export GR00T_DUQUANT_LS=0.15
export GR00T_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/Isaac-GR00T/duquant_packed_full_llm_dit_mlp_w4a8_b64c32ls015_spatial"

# Optional ATM configuration — only needed if DiT attention is also quantized (not in this script)
if [[ -n "${GR00T_ATM_ALPHA_PATH:-}" && -z "${GR00T_ATM_ENABLE:-}" ]]; then
    export GR00T_ATM_ENABLE=1
fi
export GR00T_ATM_SCOPE=${GR00T_ATM_SCOPE:-dit}

export GR00T_DENOISING_STEPS=${GR00T_DENOISING_STEPS:-8}
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_CUDA_GRAPH_DISABLE=1
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1

echo "DuQuant Config:"
echo "  INCLUDE: $GR00T_DUQUANT_INCLUDE"
echo "  EXCLUDE: $GR00T_DUQUANT_EXCLUDE"
echo "  PACKDIR: $GR00T_DUQUANT_PACKDIR"
echo "  WBITS=$GR00T_DUQUANT_WBITS_DEFAULT ABITS=$GR00T_DUQUANT_ABITS BLOCK=$GR00T_DUQUANT_BLOCK"
echo ""

# --------------------------------------------
# Dry-run: list layers that will be quantized
# --------------------------------------------
echo "🔍 Dry-run: listing quantized layers..."
export GR00T_DUQUANT_DRYRUN=1
export GR00T_MODEL_PATH="$MODEL_PATH"
export GR00T_DATA_CONFIG="$DATA_CONFIG"

python - <<'PY'
import os
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import load_data_config

model_path = os.environ["GR00T_MODEL_PATH"]
data_config_path = os.environ["GR00T_DATA_CONFIG"]

cfg = load_data_config(data_config_path)
policy = Gr00tPolicy(
    model_path=model_path,
    modality_config=cfg.modality_config(),
    modality_transform=cfg.transform(),
    embodiment_tag="new_embodiment",
    denoising_steps=8,
)
print("\n✅ Dry-run complete!\n")
PY

unset GR00T_DUQUANT_DRYRUN
unset GR00T_MODEL_PATH
unset GR00T_DATA_CONFIG

read -rp "Press Enter to start quantized inference server (Ctrl+C to abort)..."
echo ""

# --------------------------------------------
# Launch inference server (quantized)
# --------------------------------------------
./run_inference_server.sh "$TASK_SUITE"
