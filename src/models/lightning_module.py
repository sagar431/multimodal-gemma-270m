"""
PyTorch Lightning module for Multimodal Gemma training
"""
import torch
import lightning as L
from typing import Dict, Any, Optional, List
from transformers import get_linear_schedule_with_warmup
import logging

from .multimodal_gemma import MultimodalGemma

logger = logging.getLogger(__name__)


class MultimodalGemmaLightning(L.LightningModule):
    """Lightning module for Multimodal Gemma training"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize model
        self.model = MultimodalGemma(config)
        
        # Training metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        # Setup automatic optimization
        self.automatic_optimization = True
        
        logger.info("MultimodalGemmaLightning initialized")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            images=batch.get("images"),
            labels=batch["labels"]
        )
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        outputs = self(batch)
        loss = outputs["loss"]
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/learning_rate", self.optimizers().param_groups[0]["lr"], on_step=True)
        
        # Store outputs for epoch end
        self.training_step_outputs.append(loss.detach())
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        outputs = self(batch)
        loss = outputs["loss"]
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Store outputs for epoch end
        self.validation_step_outputs.append(loss.detach())
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch"""
        if self.training_step_outputs:
            avg_loss = torch.stack(self.training_step_outputs).mean()
            self.log("train/epoch_loss", avg_loss, prog_bar=False, sync_dist=True)
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch"""
        if self.validation_step_outputs:
            avg_loss = torch.stack(self.validation_step_outputs).mean()
            self.log("val/epoch_loss", avg_loss, prog_bar=False, sync_dist=True)
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        # Collect trainable parameters with different learning rates
        param_groups = []
        
        # Ensure learning rates are floats
        projector_lr = float(self.config["training"]["projector_lr"])
        lora_lr = float(self.config["training"]["lora_lr"])

        # Vision projector parameters
        vision_proj_params = list(self.model.vision_projector.parameters())
        if vision_proj_params:
            param_groups.append({
                "params": vision_proj_params,
                "lr": projector_lr,
                "name": "vision_projector"
            })

        # Audio projector parameters (if enabled)
        if hasattr(self.model, 'audio_projector'):
            audio_proj_params = list(self.model.audio_projector.parameters())
            if audio_proj_params:
                param_groups.append({
                    "params": audio_proj_params,
                    "lr": projector_lr,
                    "name": "audio_projector"
                })

        # LoRA parameters from language model
        lora_params = []
        for name, param in self.model.language_model.named_parameters():
            if param.requires_grad:
                lora_params.append(param)

        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": lora_lr,
                "name": "lora_adapters"
            })
        
        if not param_groups:
            raise ValueError("No trainable parameters found!")
        
        # Log parameter counts
        for group in param_groups:
            param_count = sum(p.numel() for p in group["params"])
            logger.info(f"{group['name']}: {param_count:,} parameters, lr={group['lr']}")
        
        # Create optimizer
        optimizer_class = torch.optim.AdamW
        if self.config.get("optimization", {}).get("use_fused_adamw", False):
            try:
                optimizer_class = torch.optim.AdamW  # Fused AdamW is default in recent PyTorch
            except AttributeError:
                logger.warning("Fused AdamW not available, using regular AdamW")
        
        optimizer = optimizer_class(
            param_groups,
            weight_decay=self.config["training"]["weight_decay"],
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Calculate total steps for scheduler
        if self.trainer.datamodule is not None:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        else:
            # Fallback estimation
            steps_per_epoch = self.config["training"].get("steps_per_epoch", 1000)
        
        max_epochs = self.config["training"]["max_epochs"]
        accumulate_grad_batches = self.config["training"].get("accumulate_grad_batches", 1)
        
        total_steps = (steps_per_epoch // accumulate_grad_batches) * max_epochs
        warmup_steps = int(total_steps * self.config["training"]["warmup_ratio"])
        
        logger.info(f"Scheduler setup: {total_steps} total steps, {warmup_steps} warmup steps")
        
        # Create scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate"
            }
        }
    
    def lr_scheduler_step(self, scheduler, metric):
        """Custom learning rate scheduler step"""
        scheduler.step()
    
    def on_before_optimizer_step(self, optimizer):
        """Called before optimizer step"""
        # Log gradient norms
        if self.global_step % 100 == 0:
            grad_norm = 0.0
            param_count = 0
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                        param_count += 1
            
            if param_count > 0:
                grad_norm = (grad_norm / param_count) ** 0.5
                self.log("train/grad_norm", grad_norm, on_step=True, prog_bar=False)
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving checkpoint"""
        # Save additional model components
        checkpoint["model_config"] = self.config
        checkpoint["tokenizer_vocab_size"] = len(self.model.tokenizer)
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading checkpoint"""
        # Restore model configuration if needed
        if "model_config" in checkpoint:
            logger.info("Loaded model configuration from checkpoint")
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Prediction step for inference"""
        outputs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            images=batch.get("images"),
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
        
        # Decode generated text
        generated_text = []
        for i, output in enumerate(outputs):
            # Remove input tokens from output
            input_length = batch["input_ids"][i].shape[0]
            generated_tokens = output[input_length:]
            text = self.model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_text.append(text)
        
        return {
            "generated_text": generated_text,
            "input_ids": batch["input_ids"],
        }
