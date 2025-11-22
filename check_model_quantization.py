#!/usr/bin/env python3
"""Check if a running GR00T model has DuQuant quantization applied."""
import sys
import torch
import zmq

def check_local_model():
    """Load model locally and check for DuQuant layers."""
    print("=" * 60)
    print("Checking local model quantization status...")
    print("=" * 60)

    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config
    import os

    # Check environment
    print("\nEnvironment variables:")
    for key in sorted(os.environ.keys()):
        if "DUQUANT" in key or "TORCH_COMPILE" in key:
            print(f"  {key} = {os.environ[key]}")

    print("\nLoading model...")
    cfg = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(
        model_path="youliangtan/gr00t-n1.5-libero-spatial-posttrain",
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=8,
    )

    print("\nChecking model layers...")
    model = policy.model

    # Check for DuQuantLinear layers
    duquant_layers = []
    total_linears = 0

    for name, module in model.named_modules():
        if "Linear" in module.__class__.__name__:
            total_linears += 1
            if "DuQuant" in module.__class__.__name__:
                duquant_layers.append(name)

    print(f"\nTotal Linear layers: {total_linears}")
    print(f"DuQuant layers: {len(duquant_layers)}")

    if len(duquant_layers) > 0:
        print(f"\n✅ QUANTIZATION IS APPLIED ({len(duquant_layers)} layers)")
        print("\nFirst 10 DuQuant layers:")
        for name in duquant_layers[:10]:
            print(f"  - {name}")
        if len(duquant_layers) > 10:
            print(f"  ... and {len(duquant_layers) - 10} more")
    else:
        print("\n❌ NO QUANTIZATION APPLIED")
        print("\nFirst 10 Linear layers (should be DuQuant if quantization worked):")
        count = 0
        for name, module in model.named_modules():
            if "Linear" in module.__class__.__name__:
                print(f"  - {name}: {module.__class__.__name__}")
                count += 1
                if count >= 10:
                    break

    return len(duquant_layers) > 0

if __name__ == "__main__":
    try:
        quantized = check_local_model()
        sys.exit(0 if quantized else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
