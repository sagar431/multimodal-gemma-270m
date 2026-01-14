#!/usr/bin/env python3
"""
Simple evaluation script for Multimodal Gemma
Tests basic VQA capabilities on a small benchmark
"""

import os
import sys
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.models.multimodal_gemma import MultimodalGemma
from omegaconf import OmegaConf


def load_model(checkpoint_path, device):
    """Load the trained model"""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get("hyper_parameters", {}).get("config", None)
    if config is None:
        config = OmegaConf.create({
            "model": {
                "gemma_model_name": "google/gemma-3-270m",
                "vision_model_name": "openai/clip-vit-large-patch14",
                "projector_hidden_dim": 2048,
                "use_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": "nf4",
                "use_nested_quant": False,
                "lora": {"r": 64, "alpha": 128, "dropout": 0.1,
                        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]}
            },
            "special_tokens": {"image_token": "<image>", "pad_token": "<pad>"},
        })

    model = MultimodalGemma(config)

    if "state_dict" in checkpoint:
        state_dict = {(k[6:] if k.startswith("model.") else k): v
                      for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


def download_image(url):
    """Download image from URL (kept for compatibility, unused in offline tests)"""
    try:
        response = requests.get(url, timeout=10)
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def generate_answer(model, image, question, device):
    """Generate answer for a question about an image"""
    image_inputs = model.vision_processor(images=image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)

    prompt = f"<image>\nHuman: {question}\nAssistant:"
    text_inputs = model.tokenizer(prompt, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512)
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            max_new_tokens=30,
            temperature=0.0,  # deterministic
            do_sample=False,
        )

    response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:", 1)[1]
    # Trim any following human turns or stray prompts
    response = response.split("\nHuman:")[0].split("\nhuman:")[0]
    return response.lower().strip()


# Simple test cases - basic VQA (based on actual image content from inference)
TEST_CASES = [
    {
        # sample_001.jpg: Red couch with cats
        "path": "samples/test_images/sample_001.jpg",
        "questions": [
            ("What animal is in this image?", ["cat", "cats", "kitten"]),
            ("What color is the couch?", ["red"]),
            ("Is there a cat in the image?", ["yes", "cat"]),
        ],
    },
    {
        # sample_004.jpg: Kitchen with counter, table, fruit bowl, orange, window
        "path": "samples/test_images/sample_004.jpg",
        "questions": [
            ("What room is this?", ["kitchen"]),
            ("What is on the table?", ["fruit", "bowl", "orange"]),
            ("Is there a window in the image?", ["yes", "window"]),
        ],
    },
    {
        # sample_003.jpg: Living room with table, chairs, TV
        "path": "samples/test_images/sample_003.jpg",
        "questions": [
            ("What room is this?", ["living room", "room", "living"]),
            ("What furniture is in the image?", ["table", "chair", "chairs", "couch"]),
            ("Is this indoors or outdoors?", ["indoors", "indoor", "inside"]),
        ],
    },
    {
        # sample_007.png: Dog
        "path": "samples/test_images/sample_007.png",
        "questions": [
            ("What animal is in this image?", ["dog", "puppy"]),
            ("Is there a dog?", ["yes", "dog"]),
        ],
    },
    {
        # sample_009.png: Cat on blue couch
        "path": "samples/test_images/sample_009.png",
        "questions": [
            ("What animal is in this image?", ["cat", "kitten"]),
            ("What color is the couch?", ["blue"]),
        ],
    },
]


def run_evaluation(checkpoint_path):
    """Run simple evaluation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(checkpoint_path, device)

    total = 0
    correct = 0
    results = []

    print("\n" + "="*60)
    print("Running Simple VQA Evaluation")
    print("="*60 + "\n")

    for test_case in TEST_CASES:
        image_path = test_case.get("path")
        image = Image.open(image_path).convert("RGB")

        print(f"\nImage: {image_path}")

        for question, expected_keywords in test_case["questions"]:
            answer = generate_answer(model, image, question, device)

            # Check if any expected keyword is in the answer
            is_correct = any(kw in answer for kw in expected_keywords)

            total += 1
            if is_correct:
                correct += 1

            status = "✓" if is_correct else "✗"
            print(f"  {status} Q: {question}")
            print(f"      A: {answer}")
            print(f"      Expected keywords: {expected_keywords}")

            results.append({
                "question": question,
                "answer": answer,
                "expected": expected_keywords,
                "correct": is_correct
            })

    accuracy = (correct / total * 100) if total > 0 else 0

    print("\n" + "="*60)
    print(f"Results: {correct}/{total} correct ({accuracy:.1f}%)")
    print("="*60)

    return results, accuracy


def run_pope_mini(checkpoint_path, num_samples=100):
    """
    Run a mini POPE-like evaluation (hallucination test)
    Tests if model hallucinates objects that aren't in images
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    # Simple hallucination test pairs (based on actual image content)
    # Format: (image_path, objects_present, objects_absent)
    POPE_TESTS = [
        {
            # sample_001.jpg: Red couch with cats
            "path": "samples/test_images/sample_001.jpg",
            "present": ["cat", "couch"],
            "absent": ["dog", "elephant", "car", "airplane"],
        },
        {
            # sample_004.jpg: Kitchen with fruit
            "path": "samples/test_images/sample_004.jpg",
            "present": ["kitchen", "table", "fruit", "window"],
            "absent": ["dog", "cat", "person", "bicycle"],
        },
        {
            # sample_007.png: Dog
            "path": "samples/test_images/sample_007.png",
            "present": ["dog"],
            "absent": ["cat", "elephant", "car", "zebra"],
        },
        {
            # sample_009.png: Cat on blue couch
            "path": "samples/test_images/sample_009.png",
            "present": ["cat", "couch"],
            "absent": ["dog", "horse", "airplane", "bicycle"],
        },
    ]

    print("\n" + "="*60)
    print("Running POPE-like Hallucination Test")
    print("="*60 + "\n")

    tp, fp, tn, fn = 0, 0, 0, 0

    for test in POPE_TESTS:
        image = Image.open(test["path"]).convert("RGB")

        # Test present objects (should answer "yes")
        for obj in test["present"]:
            question = f"Is there a {obj} in this image? Answer yes or no."
            answer = generate_answer(model, image, question, device)

            if "yes" in answer:
                tp += 1
                print(f"  ✓ '{obj}' present - model said yes")
            else:
                fn += 1
                print(f"  ✗ '{obj}' present - model said no (false negative)")

        # Test absent objects (should answer "no")
        for obj in test["absent"]:
            question = f"Is there a {obj} in this image? Answer yes or no."
            answer = generate_answer(model, image, question, device)

            if "no" in answer or "not" in answer:
                tn += 1
                print(f"  ✓ '{obj}' absent - model said no")
            else:
                fp += 1
                print(f"  ✗ '{obj}' absent - model said yes (hallucination!)")

    total = tp + fp + tn + fn
    accuracy = ((tp + tn) / total * 100) if total > 0 else 0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0

    print("\n" + "-"*40)
    print(f"POPE-like Results:")
    print(f"  Accuracy:  {accuracy:.1f}%")
    print(f"  Precision: {precision:.1f}%")
    print(f"  Recall:    {recall:.1f}%")
    print(f"  Hallucinations (FP): {fp}")
    print("-"*40)

    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


if __name__ == "__main__":
    checkpoint_path = "models/checkpoints/gemma-270m-llava-a100-optimized/final_model.ckpt"

    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Run basic VQA evaluation
    results, accuracy = run_evaluation(checkpoint_path)

    # Run hallucination test
    pope_results = run_pope_mini(checkpoint_path)

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Basic VQA Accuracy: {accuracy:.1f}%")
    print(f"Hallucination Test Accuracy: {pope_results['accuracy']:.1f}%")
    print("="*60)
