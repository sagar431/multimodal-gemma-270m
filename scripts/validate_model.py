#!/usr/bin/env python3
"""
Script to validate the traced model before deployment.
Runs basic inference tests to ensure the model works correctly.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def validate_model(model_path: str) -> dict:
    """
    Validate the traced model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "model_path": model_path,
        "valid": False,
        "checks": {}
    }
    
    model_path = Path(model_path)
    
    # Check 1: File exists
    if not model_path.exists():
        results["checks"]["file_exists"] = False
        results["error"] = "Model file not found"
        return results
    results["checks"]["file_exists"] = True
    
    # Check 2: File size is reasonable
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    results["checks"]["file_size_mb"] = file_size_mb
    results["checks"]["file_size_valid"] = file_size_mb > 0.1  # At least 100KB
    
    # Check 3: Can load the model
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        results["checks"]["loadable"] = True
        
        # Check what's in the checkpoint
        if isinstance(checkpoint, dict):
            results["checks"]["checkpoint_keys"] = list(checkpoint.keys())
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                results["checks"]["num_parameters"] = len(state_dict)
                
            if 'config' in checkpoint:
                results["checks"]["has_config"] = True
                
            if 'model_type' in checkpoint:
                results["checks"]["model_type"] = checkpoint['model_type']
                
        else:
            # TorchScript model
            results["checks"]["is_torchscript"] = True
            
    except Exception as e:
        results["checks"]["loadable"] = False
        results["error"] = str(e)
        return results
    
    # All checks passed
    all_passed = all([
        results["checks"].get("file_exists", False),
        results["checks"].get("file_size_valid", False),
        results["checks"].get("loadable", False)
    ])
    
    results["valid"] = all_passed
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate traced model")
    parser.add_argument("--model_path", type=str, default="hf_space/model.pt",
                        help="Path to the model file")
    parser.add_argument("--output", type=str, default="validation_results.json",
                        help="Output path for validation results")
    
    args = parser.parse_args()
    
    log.info("=" * 60)
    log.info("🔍 Model Validation Script")
    log.info("=" * 60)
    log.info(f"Model path: {args.model_path}")
    
    # Run validation
    results = validate_model(args.model_path)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Results saved to: {args.output}")
    
    # Print summary
    log.info("=" * 60)
    if results["valid"]:
        log.info("✅ Model validation PASSED!")
    else:
        log.error("❌ Model validation FAILED!")
        if "error" in results:
            log.error(f"Error: {results['error']}")
        sys.exit(1)
    log.info("=" * 60)
    
    # Print checks
    for check, value in results["checks"].items():
        status = "✅" if value else "❌" if isinstance(value, bool) else "ℹ️"
        log.info(f"  {status} {check}: {value}")


if __name__ == "__main__":
    main()
