# Makefile for MLOps Multimodal Gemma Pipeline
# Uses UV for fast package management (pip fallback available)

.PHONY: help setup install train trace deploy test lint clean

# Default target
help:
	@echo "🤖 Multimodal Gemma-270M MLOps Commands"
	@echo "======================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup         Create venv and install with UV (recommended)"
	@echo "  make install       Install deps in existing venv with UV"
	@echo "  make install-pip   Install with pip (fallback)"
	@echo ""
	@echo "Training:"
	@echo "  make train         Train the model (3 epochs)"
	@echo "  make train-fast    Quick test run (fast_dev_run)"
	@echo "  make train-1epoch  Train for 1 epoch only"
	@echo ""
	@echo "Frontend (No-Code Platform):"
	@echo "  make frontend         Start full platform (backend + frontend)"
	@echo "  make frontend-backend Start backend API only"
	@echo "  make frontend-dev     Start frontend only"
	@echo "  make frontend-install Install frontend dependencies"
	@echo ""
	@echo "Deployment:"
	@echo "  make trace         Export model for deployment"
	@echo "  make deploy        Deploy to HuggingFace Spaces"
	@echo "  make gradio        Run Gradio app locally"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests"
	@echo "  make lint          Run linting"
	@echo ""
	@echo "Pipeline (DVC - optional, requires S3/GDrive):"
	@echo "  make pipeline      Run DVC pipeline"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Clean generated files"

# ============== Setup ==============
# Full setup with UV (recommended)
setup:
	@echo "📦 Setting up environment with UV..."
	uv venv .venv
	. .venv/bin/activate && uv pip install -r requirements.txt
	@echo "✅ Setup complete! Run: source .venv/bin/activate"

# Install in existing venv with UV
install:
	uv pip install -r requirements.txt

# Install with pip (fallback)
install-pip:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

# ============== Training ==============
train:
	python train.py

train-fast:
	python train.py trainer.fast_dev_run=true logging.use_wandb=false

train-1epoch:
	python train.py training.max_epochs=1 logging.use_wandb=false

# Train with Weights & Biases logging (recommended for tracking)
train-wandb:
	python train.py logging.use_wandb=true

# View TensorBoard logs
tensorboard:
	tensorboard --logdir=logs/tensorboard --port=6006

# View sample generations (check if model is learning)
view-samples:
	@echo "📸 Recent sample generations:"
	@cat logs/samples/generation_history.json 2>/dev/null | python -m json.tool | tail -50 || echo "No samples yet. Run training first."

# ============== Model Export ==============
trace:
	python src/trace_model.py --output_path hf_space/model.pt

validate:
	python scripts/validate_model.py --model_path hf_space/model.pt

# ============== Deployment ==============
# Upload checkpoint to HuggingFace Hub (so CI/CD can use it)
upload-checkpoint:
	@echo "📤 Uploading checkpoint to HuggingFace Hub..."
	@read -p "Enter your HuggingFace username: " username; \
	python scripts/upload_checkpoint.py --username $$username

deploy: trace validate
	@echo "🚀 Deploying to HuggingFace Spaces..."
	python -c "from huggingface_hub import upload_folder; upload_folder(folder_path='hf_space', repo_id='$(HF_USERNAME)/multimodal-gemma-270m', repo_type='space')"

gradio:
	cd hf_space && python app.py

gradio-local:
	python gradio_app.py

# ============== Testing ==============
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=html

lint:
	ruff check src/ tests/ || true
	black --check src/ tests/ || true

format:
	black src/ tests/
	ruff check src/ tests/ --fix || true

# ============== DVC Pipeline (Optional) ==============
# Note: DVC requires remote storage (S3, GDrive, etc.)
# Skip if you don't have access to remote storage

pipeline:
	@command -v dvc >/dev/null 2>&1 || { echo "⚠️  DVC not installed. Install with: pip install dvc"; exit 0; }
	dvc repro

pipeline-force:
	dvc repro --force

data-prepare:
	python scripts/prepare_data.py

# ============== Cleanup ==============
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf logs/wandb logs/tensorboard
	rm -rf .coverage htmlcov
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	rm -rf models/checkpoints/*
	rm -rf data/cache data/processed
	rm -rf hf_space/model.pt

# ============== Frontend (No-Code Training Platform) ==============
frontend:
	@echo "🌐 Starting Multimodal Training Platform..."
	cd frontend && ./start.sh

frontend-backend:
	@echo "🔧 Starting backend API only..."
	cd frontend/backend && pip install -r requirements.txt && python main.py

frontend-dev:
	@echo "⚛️  Starting frontend only..."
	cd frontend && npm install && npm run dev

frontend-install:
	@echo "📦 Installing frontend dependencies..."
	cd frontend && npm install
	cd frontend/backend && pip install -r requirements.txt

# ============== Docker ==============
docker-build:
	docker build -t multimodal-gemma:latest .

docker-train:
	docker run --gpus all -v $(PWD)/models:/app/models multimodal-gemma:latest python train.py

docker-shell:
	docker run --gpus all -it -v $(PWD):/app multimodal-gemma:latest bash
