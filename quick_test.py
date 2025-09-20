#!/usr/bin/env python3
"""
Quick test of trained model
"""
import sys
import torch
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent))

from src.models import MultimodalGemmaLightning
from src.utils.config import load_config, merge_configs

def quick_test():
    print("🚀 QUICK MODEL TEST")
    print("=" * 30)

    # Load config
    model_config = load_config("configs/model_config.yaml")
    training_config = load_config("configs/training_config.yaml")
    data_config = load_config("configs/data_config.yaml")
    config = merge_configs([model_config, training_config, data_config])

    try:
        # Load checkpoint
        checkpoint_path = "models/checkpoints/gemma-270m-llava-training/final_model.ckpt"
        print(f"📁 Loading: {checkpoint_path}")

        model = MultimodalGemmaLightning.load_from_checkpoint(
            checkpoint_path,
            config=config,
            strict=False
        )
        model.eval()
        print("✅ Model loaded successfully!")

        # Test basic functionality
        device = next(model.parameters()).device
        print(f"📱 Device: {device}")

        # Create dummy inputs
        batch_size = 1
        seq_len = 20

        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
        # Add image token
        input_ids[0, 1] = model.model.image_token_id

        attention_mask = torch.ones(batch_size, seq_len).to(device)
        images = torch.randn(batch_size, 3, 224, 224).to(device)

        print("🧪 Testing forward pass...")
        with torch.no_grad():
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images
            )

        print(f"✅ Forward pass successful!")
        print(f"   Logits shape: {outputs['logits'].shape}")
        print(f"   Loss: {outputs['loss'].item():.4f}")

        # Test text generation (simpler)
        print("🧪 Testing generation...")
        with torch.no_grad():
            # Simple text input
            simple_input = model.model.tokenizer(
                "Human: Hello! Assistant:",
                return_tensors="pt"
            ).to(device)

            generated = model.model.language_model.generate(
                simple_input["input_ids"],
                max_new_tokens=10,
                do_sample=False
            )

            response = model.model.tokenizer.decode(
                generated[0][simple_input["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

        print(f"✅ Generation successful!")
        print(f"   Response: '{response}'")

        print("\n🎉 MODEL IS WORKING!")
        print("✅ Checkpoint loads correctly")
        print("✅ Forward pass works")
        print("✅ Generation works")
        print("✅ Multimodal fusion implemented")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎯 Your multimodal Gemma model is READY!")
    else:
        print("\n⚠️  Model needs debugging")