#!/bin/bash
# Test DuQuant layer scanning (dry-run mode)
# This script shows which layers will be quantized without actually applying quantization

cd /home/jz97/VLM_REPO/Isaac-GR00T

# Enable DuQuant dry-run mode
export GR00T_DUQUANT_DRYRUN=1
export GR00T_DUQUANT_DEBUG=1
export GR00T_DUQUANT_SCOPE=""

# Target LLM backbone + DiT transformer layers
export GR00T_DUQUANT_INCLUDE='.*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\..*(attn1\.to_(q|k|v)|attn1\.to_out\.0|ff\.net\.(0\.proj|2))).*'
export GR00T_DUQUANT_EXCLUDE='(?:^|\.)(vision_model|vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|future_tokens|vl_self_attention)(?:\.|$)'

export GR00T_DUQUANT_WBITS_DEFAULT=4
export GR00T_DUQUANT_ABITS=8

echo "========================================"
echo "GR00T DuQuant W4A8 Dry-Run"
echo "Scanning layers for quantization..."
echo "========================================"
echo ""

# Activate gr00t environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t

# Run dry-run
python -c "
import os
os.environ['GR00T_DUQUANT_DRYRUN'] = '1'
os.environ['GR00T_DUQUANT_DEBUG'] = '1'

from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import load_data_config

print('Loading model...')
data_config = load_data_config('examples.Libero.custom_data_config:LiberoDataConfig')
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

policy = Gr00tPolicy(
    model_path='youliangtan/gr00t-n1.5-libero-spatial-posttrain',
    modality_config=modality_config,
    modality_transform=modality_transform,
    embodiment_tag='new_embodiment',
    denoising_steps=8,
)

print('')
print('✅ Dry-run complete!')
print('')
print('Review the layers listed above.')
print('These are the layers that will be quantized with W4A8.')
"

echo ""
echo "========================================"
echo "Dry-run complete!"
echo "========================================"
