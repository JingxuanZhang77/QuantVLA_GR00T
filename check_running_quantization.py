#!/usr/bin/env python3
"""Check if running server has quantized layers."""
import os
import sys

# Set environment to match running server
os.environ['GR00T_DUQUANT_DEBUG'] = '1'

from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import load_data_config

print("="*60)
print("🔍 CHECKING QUANTIZATION IN FRESHLY LOADED MODEL")
print("="*60)

cfg = load_data_config('examples.Libero.custom_data_config:LiberoDataConfig')
policy = Gr00tPolicy(
    model_path='youliangtan/gr00t-n1.5-libero-spatial-posttrain',
    modality_config=cfg.modality_config(),
    modality_transform=cfg.transform(),
    embodiment_tag='new_embodiment',
    denoising_steps=16,
)

print("\n" + "="*60)
print("📊 CHECKING ACTUAL LAYER TYPES")
print("="*60)

llm_quantized = 0
llm_total = 0
dit_quantized = 0
dit_total = 0

for name, module in policy.model.named_modules():
    module_type = module.__class__.__name__

    # Check LLM
    if 'backbone.eagle_model.language_model' in name and 'Linear' in module_type:
        if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
            llm_total += 1
            if 'DuQuant' in module_type:
                llm_quantized += 1

    # Check DiT
    if 'action_head.model.transformer_blocks' in name and 'Linear' in module_type:
        if 'attn1' in name or 'ff.net' in name:
            dit_total += 1
            if 'DuQuant' in module_type:
                dit_quantized += 1

print(f"\n✅ LLM: {llm_quantized}/{llm_total} layers quantized ({100*llm_quantized/llm_total:.1f}%)")
print(f"{'✅' if dit_quantized > 0 else '❌'} DiT: {dit_quantized}/{dit_total} layers quantized ({100*dit_quantized/dit_total if dit_total > 0 else 0:.1f}%)")

if dit_quantized == 0 and dit_total > 0:
    print("\n❌ PROBLEM: DiT layers are NOT quantized!")
    print("   This explains why quantizing DiT doesn't change accuracy.")
    print("\n🔧 Possible causes:")
    print("   1. action_head was recreated after quantization")
    print("   2. DuQuant was applied before action_head was loaded")
    sys.exit(1)
elif dit_quantized == dit_total:
    print("\n✅ SUCCESS: All target layers are quantized!")
    sys.exit(0)
else:
    print(f"\n⚠️  WARNING: Only {dit_quantized}/{dit_total} DiT layers quantized")
    sys.exit(2)
