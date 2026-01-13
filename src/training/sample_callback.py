"""
Sample Generation Callback for PyTorch Lightning
Generates sample outputs during training to verify model is learning correctly.
"""

import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import Callback
from PIL import Image
import requests
from io import BytesIO

logger = logging.getLogger(__name__)


class SampleGenerationCallback(Callback):
    """
    Generate sample outputs during training to verify model is learning.
    
    This callback:
    1. Every N steps, takes a sample image and generates text
    2. Logs the output to W&B, TensorBoard, and a local file
    3. Helps you diagnose if the model is learning correctly EARLY
    """
    
    def __init__(
        self,
        every_n_steps: int = 500,
        sample_images: Optional[List[str]] = None,
        sample_prompts: Optional[List[str]] = None,
        log_dir: str = "logs/samples",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Use LOCAL sample images with CODE NAMES (so model can't cheat from filename!)
        # Model will only see "sample_001.jpg" - not "cat_on_couch.jpg"
        # You can check samples/REFERENCE_SECRET.md to see what each image actually contains
        self.sample_images = sample_images or [
            "samples/test_images/sample_001.jpg",  # Check REFERENCE_SECRET.md for actual content
            "samples/test_images/sample_007.png",  # Check REFERENCE_SECRET.md for actual content
            "samples/test_images/sample_008.png",  # Check REFERENCE_SECRET.md for actual content
            "samples/test_images/sample_009.png",  # Check REFERENCE_SECRET.md for actual content
        ]
        
        # Default prompts (generic - don't hint at content)
        self.sample_prompts = sample_prompts or [
            "What do you see in this image?",
            "Describe the main objects in this picture.",
        ]
        
        # Store results for comparison
        self.generation_history = []
        
        logger.info(f"SampleGenerationCallback initialized: generate every {every_n_steps} steps")
    
    def _load_image(self, image_source: str) -> Optional[Image.Image]:
        """Load image from URL or file path."""
        try:
            if image_source.startswith("http"):
                response = requests.get(image_source, timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_source).convert("RGB")
            return image
        except Exception as e:
            logger.warning(f"Failed to load image {image_source}: {e}")
            return None
    
    def _generate_sample(
        self, 
        pl_module: L.LightningModule, 
        image: Image.Image, 
        prompt: str
    ) -> str:
        """Generate text for a single image-prompt pair."""
        try:
            model = pl_module.model
            device = next(model.parameters()).device
            
            # Process image
            vision_inputs = model.vision_processor(
                images=[image],
                return_tensors="pt"
            )
            pixel_values = vision_inputs["pixel_values"].to(device)
            
            # Prepare text
            full_prompt = f"<image>\nHuman: {prompt}\nAssistant:"
            text_inputs = model.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs["attention_mask"].to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=pixel_values,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    repetition_penalty=1.1
                )
            
            # Decode
            input_length = input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            response = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[ERROR: {str(e)}]"
    
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called at the end of each training batch."""
        global_step = trainer.global_step
        
        # Only generate at specified intervals
        if global_step == 0 or global_step % self.every_n_steps != 0:
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📸 Generating samples at step {global_step}")
        logger.info(f"{'='*60}")
        
        # Switch to eval mode
        pl_module.eval()
        
        results = []
        
        for img_url in self.sample_images[:2]:  # Limit to 2 images
            image = self._load_image(img_url)
            if image is None:
                continue
            
            for prompt in self.sample_prompts[:2]:  # Limit to 2 prompts
                response = self._generate_sample(pl_module, image, prompt)
                
                result = {
                    "step": global_step,
                    "image": img_url,
                    "prompt": prompt,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
                
                # Print to console
                logger.info(f"\n🖼️ Image: {img_url.split('/')[-1]}")
                logger.info(f"❓ Prompt: {prompt}")
                logger.info(f"💬 Response: {response[:200]}...")
        
        # Back to train mode
        pl_module.train()
        
        # Save results
        self.generation_history.extend(results)
        self._save_results(global_step, results)
        
        # Log to W&B if available
        self._log_to_wandb(trainer, results)
        
        logger.info(f"{'='*60}\n")
    
    def _save_results(self, step: int, results: List[Dict]) -> None:
        """Save generation results to file."""
        # Save individual step results
        step_file = self.log_dir / f"samples_step_{step:06d}.json"
        with open(step_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save full history
        history_file = self.log_dir / "generation_history.json"
        with open(history_file, "w") as f:
            json.dump(self.generation_history, f, indent=2)
    
    def _log_to_wandb(self, trainer: L.Trainer, results: List[Dict]) -> None:
        """Log results to W&B if available."""
        try:
            import wandb
            if wandb.run is not None:
                # Create a table
                table = wandb.Table(columns=["Step", "Image", "Prompt", "Response"])
                for r in results:
                    table.add_data(r["step"], r["image"].split("/")[-1], r["prompt"], r["response"])
                
                wandb.log({"sample_generations": table})
        except Exception:
            pass  # W&B not available
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Save final summary at end of training."""
        summary_file = self.log_dir / "final_summary.json"
        
        summary = {
            "total_samples": len(self.generation_history),
            "steps_sampled": list(set(r["step"] for r in self.generation_history)),
            "generation_history": self.generation_history
        }
        
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Saved {len(self.generation_history)} sample generations to {self.log_dir}")


class EarlyStoppingOnPlateau(Callback):
    """
    Early stopping based on loss plateau detection.
    Stops training if loss doesn't improve by min_delta for patience epochs.
    """
    
    def __init__(
        self,
        monitor: str = "train/loss",
        min_delta: float = 0.01,
        patience: int = 3,
        check_every_n_steps: int = 100,
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.check_every_n_steps = check_every_n_steps
        
        self.best_loss = float("inf")
        self.wait_count = 0
        self.loss_history = []
    
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % self.check_every_n_steps != 0:
            return
        
        # Get current loss
        current_loss = trainer.callback_metrics.get(self.monitor)
        if current_loss is None:
            return
        
        current_loss = float(current_loss)
        self.loss_history.append(current_loss)
        
        # Check improvement
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait_count = 0
            logger.info(f"📈 Loss improved to {current_loss:.4f}")
        else:
            self.wait_count += 1
            logger.info(f"⏸️ No improvement for {self.wait_count} checks (best: {self.best_loss:.4f})")
            
            if self.wait_count >= self.patience:
                logger.warning(f"⚠️ Loss plateau detected! Consider stopping training.")
                # trainer.should_stop = True  # Uncomment to auto-stop
