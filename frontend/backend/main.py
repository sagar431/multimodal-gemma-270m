"""
FastAPI Backend for Multimodal Model Training Platform
Handles cloud provider integration, GPU calculations, and job management
"""

import os
import json
import asyncio
import httpx
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ============================================================================
# CONFIGURATION
# ============================================================================

app = FastAPI(
    title="Multimodal Training Platform API",
    description="API for training multimodal models on cloud GPUs",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (use Redis/DB in production)
jobs_db: Dict[str, Dict] = {}
connected_websockets: Dict[str, List[WebSocket]] = {}

# ============================================================================
# DATA MODELS
# ============================================================================

class CloudProvider(str, Enum):
    RUNPOD = "runpod"
    LAMBDA = "lambda"
    AWS = "aws"


class GPUType(str, Enum):
    RTX_4090 = "rtx4090"
    A10 = "a10"
    A100_40GB = "a100-40gb"
    A100_80GB = "a100-80gb"
    H100 = "h100"
    L40 = "l40"


class BaseModel_(str, Enum):
    GEMMA_270M = "gemma-270m"
    GEMMA_2B = "gemma-2b"
    PHI_2 = "phi-2"
    LLAMA_1B = "llama-1b"
    LLAMA_3B = "llama-3b"


class CloudCredentials(BaseModel):
    provider: CloudProvider
    api_key: str


class TrainingConfig(BaseModel):
    base_model: BaseModel_
    dataset_size: int = Field(default=50000, ge=1000, le=200000)
    epochs: int = Field(default=3, ge=1, le=10)
    batch_size: int = Field(default=4, ge=1, le=32)
    learning_rate: float = Field(default=2e-4, ge=1e-6, le=1e-2)
    use_lora: bool = True
    lora_rank: int = Field(default=64, ge=8, le=256)


class TrainingRequest(BaseModel):
    credentials: CloudCredentials
    config: TrainingConfig
    gpu_type: Optional[GPUType] = None  # Auto-select if not provided


class JobStatus(str, Enum):
    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    provider: CloudProvider
    gpu_type: GPUType
    base_model: BaseModel_
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 3
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    estimated_time_remaining: Optional[str] = None
    cost_so_far: float = 0.0
    created_at: str
    logs: List[str] = []


# ============================================================================
# GPU CALCULATOR
# ============================================================================

GPU_SPECS = {
    GPUType.RTX_4090: {"vram": 24, "price_runpod": 0.44, "price_lambda": 0.50, "price_aws": 0.76},
    GPUType.A10: {"vram": 24, "price_runpod": 0.36, "price_lambda": 0.75, "price_aws": 1.00},
    GPUType.L40: {"vram": 48, "price_runpod": 0.69, "price_lambda": 0.99, "price_aws": 1.20},
    GPUType.A100_40GB: {"vram": 40, "price_runpod": 0.79, "price_lambda": 1.29, "price_aws": 1.60},
    GPUType.A100_80GB: {"vram": 80, "price_runpod": 1.19, "price_lambda": 1.99, "price_aws": 2.50},
    GPUType.H100: {"vram": 80, "price_runpod": 1.99, "price_lambda": 2.49, "price_aws": 3.50},
}

MODEL_REQUIREMENTS = {
    BaseModel_.GEMMA_270M: {
        "params_b": 0.27,
        "vision_params_b": 0.428,
        "total_params_b": 0.54,
        "min_vram": 12,
        "recommended_vram": 16,
        "training_hours_per_100k": 3.0,  # hours per 100k samples on A100
    },
    BaseModel_.GEMMA_2B: {
        "params_b": 2.0,
        "vision_params_b": 0.428,
        "total_params_b": 2.4,
        "min_vram": 24,
        "recommended_vram": 40,
        "training_hours_per_100k": 8.0,
    },
    BaseModel_.PHI_2: {
        "params_b": 2.7,
        "vision_params_b": 0.428,
        "total_params_b": 3.1,
        "min_vram": 24,
        "recommended_vram": 40,
        "training_hours_per_100k": 9.0,
    },
    BaseModel_.LLAMA_1B: {
        "params_b": 1.0,
        "vision_params_b": 0.428,
        "total_params_b": 1.4,
        "min_vram": 16,
        "recommended_vram": 24,
        "training_hours_per_100k": 5.0,
    },
    BaseModel_.LLAMA_3B: {
        "params_b": 3.0,
        "vision_params_b": 0.428,
        "total_params_b": 3.4,
        "min_vram": 32,
        "recommended_vram": 40,
        "training_hours_per_100k": 10.0,
    },
}


def calculate_gpu_requirements(
    base_model: BaseModel_,
    dataset_size: int,
    epochs: int,
    provider: CloudProvider
) -> Dict[str, Any]:
    """Calculate GPU requirements and cost estimates"""
    model_req = MODEL_REQUIREMENTS[base_model]

    # Find compatible GPUs
    compatible_gpus = []
    for gpu_type, specs in GPU_SPECS.items():
        if specs["vram"] >= model_req["min_vram"]:
            # Calculate training time (scale based on GPU power)
            base_hours = model_req["training_hours_per_100k"] * (dataset_size / 100000) * epochs

            # Adjust for GPU performance (relative to A100)
            gpu_multiplier = {
                GPUType.RTX_4090: 1.3,
                GPUType.A10: 1.8,
                GPUType.L40: 1.2,
                GPUType.A100_40GB: 1.0,
                GPUType.A100_80GB: 0.9,
                GPUType.H100: 0.6,
            }

            estimated_hours = base_hours * gpu_multiplier.get(gpu_type, 1.0)

            # Get price for provider
            price_key = f"price_{provider.value}"
            if price_key == "price_aws":
                price_key = "price_aws"
            elif price_key == "price_lambda":
                price_key = "price_lambda"
            else:
                price_key = "price_runpod"

            hourly_price = specs.get(price_key, specs["price_runpod"])
            estimated_cost = estimated_hours * hourly_price

            compatible_gpus.append({
                "gpu_type": gpu_type.value,
                "vram_gb": specs["vram"],
                "estimated_hours": round(estimated_hours, 1),
                "hourly_price": hourly_price,
                "estimated_cost": round(estimated_cost, 2),
                "recommended": specs["vram"] >= model_req["recommended_vram"],
            })

    # Sort by cost (best value first)
    compatible_gpus.sort(key=lambda x: x["estimated_cost"])

    # Mark best value
    if compatible_gpus:
        # Best value is cheapest that's recommended
        recommended = [g for g in compatible_gpus if g["recommended"]]
        if recommended:
            recommended[0]["best_value"] = True
        else:
            compatible_gpus[0]["best_value"] = True

    return {
        "model_info": {
            "name": base_model.value,
            "llm_params_b": model_req["params_b"],
            "vision_params_b": model_req["vision_params_b"],
            "total_params_b": model_req["total_params_b"],
            "min_vram_gb": model_req["min_vram"],
            "recommended_vram_gb": model_req["recommended_vram"],
        },
        "training_config": {
            "dataset_size": dataset_size,
            "epochs": epochs,
        },
        "compatible_gpus": compatible_gpus,
    }


# ============================================================================
# RUNPOD INTEGRATION
# ============================================================================

class RunPodClient:
    """Client for RunPod API"""

    BASE_URL = "https://api.runpod.io/graphql"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    async def test_connection(self) -> bool:
        """Test if API key is valid"""
        query = """
        query {
            myself {
                id
                email
            }
        }
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BASE_URL,
                    json={"query": query},
                    headers=self.headers,
                    timeout=10.0
                )
                data = response.json()
                return "data" in data and "myself" in data["data"]
        except Exception:
            return False

    async def get_available_gpus(self) -> List[Dict]:
        """Get available GPU types"""
        query = """
        query {
            gpuTypes {
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
            }
        }
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BASE_URL,
                    json={"query": query},
                    headers=self.headers,
                    timeout=10.0
                )
                data = response.json()
                return data.get("data", {}).get("gpuTypes", [])
        except Exception:
            return []

    async def create_pod(
        self,
        gpu_type_id: str,
        name: str,
        image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        volume_size: int = 50,
    ) -> Optional[Dict]:
        """Create a new pod for training"""
        mutation = """
        mutation createPod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                desiredStatus
                imageName
                machineId
            }
        }
        """
        variables = {
            "input": {
                "cloudType": "SECURE",
                "gpuTypeId": gpu_type_id,
                "name": name,
                "imageName": image,
                "volumeInGb": volume_size,
                "containerDiskInGb": 20,
                "minVcpuCount": 4,
                "minMemoryInGb": 16,
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BASE_URL,
                    json={"query": mutation, "variables": variables},
                    headers=self.headers,
                    timeout=30.0
                )
                data = response.json()
                return data.get("data", {}).get("podFindAndDeployOnDemand")
        except Exception as e:
            print(f"Error creating pod: {e}")
            return None

    async def get_pod_status(self, pod_id: str) -> Optional[Dict]:
        """Get pod status"""
        query = """
        query getPod($podId: String!) {
            pod(input: {podId: $podId}) {
                id
                name
                desiredStatus
                lastStatusChange
                runtime {
                    uptimeInSeconds
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
            }
        }
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BASE_URL,
                    json={"query": query, "variables": {"podId": pod_id}},
                    headers=self.headers,
                    timeout=10.0
                )
                data = response.json()
                return data.get("data", {}).get("pod")
        except Exception:
            return None

    async def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a pod"""
        mutation = """
        mutation terminatePod($podId: String!) {
            podTerminate(input: {podId: $podId})
        }
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BASE_URL,
                    json={"query": mutation, "variables": {"podId": pod_id}},
                    headers=self.headers,
                    timeout=10.0
                )
                return True
        except Exception:
            return False


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {"message": "Multimodal Training Platform API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/validate-credentials")
async def validate_credentials(credentials: CloudCredentials):
    """Validate cloud provider credentials"""
    if credentials.provider == CloudProvider.RUNPOD:
        client = RunPodClient(credentials.api_key)
        is_valid = await client.test_connection()
        if is_valid:
            gpus = await client.get_available_gpus()
            return {
                "valid": True,
                "provider": credentials.provider,
                "available_gpus": len(gpus),
                "message": "Credentials validated successfully"
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid RunPod API key")

    elif credentials.provider == CloudProvider.LAMBDA:
        # Lambda Labs validation (placeholder)
        return {
            "valid": True,
            "provider": credentials.provider,
            "message": "Lambda Labs integration coming soon"
        }

    elif credentials.provider == CloudProvider.AWS:
        # AWS validation (placeholder)
        return {
            "valid": True,
            "provider": credentials.provider,
            "message": "AWS integration coming soon"
        }

    raise HTTPException(status_code=400, detail="Unknown provider")


@app.post("/api/calculate-gpu")
async def calculate_gpu(
    base_model: BaseModel_,
    dataset_size: int = 50000,
    epochs: int = 3,
    provider: CloudProvider = CloudProvider.RUNPOD
):
    """Calculate GPU requirements and cost estimates"""
    return calculate_gpu_requirements(base_model, dataset_size, epochs, provider)


@app.get("/api/models")
async def list_models():
    """List available base models"""
    models = []
    for model, req in MODEL_REQUIREMENTS.items():
        models.append({
            "id": model.value,
            "name": model.value.replace("-", " ").title(),
            "params_b": req["total_params_b"],
            "min_vram": req["min_vram"],
            "description": f"{req['params_b']}B LLM + {req['vision_params_b']}B Vision Encoder"
        })
    return {"models": models}


@app.get("/api/gpus")
async def list_gpus():
    """List available GPU types"""
    gpus = []
    for gpu, specs in GPU_SPECS.items():
        gpus.append({
            "id": gpu.value,
            "name": gpu.value.upper().replace("-", " "),
            "vram_gb": specs["vram"],
            "price_runpod": specs["price_runpod"],
            "price_lambda": specs["price_lambda"],
            "price_aws": specs["price_aws"],
        })
    return {"gpus": gpus}


@app.post("/api/start-training")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a new training job"""
    import uuid

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Calculate GPU if not provided
    if request.gpu_type is None:
        gpu_calc = calculate_gpu_requirements(
            request.config.base_model,
            request.config.dataset_size,
            request.config.epochs,
            request.credentials.provider
        )
        # Get recommended GPU
        recommended = [g for g in gpu_calc["compatible_gpus"] if g.get("best_value")]
        if recommended:
            request.gpu_type = GPUType(recommended[0]["gpu_type"])
        else:
            request.gpu_type = GPUType.A100_40GB

    # Create job record
    job = JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        provider=request.credentials.provider,
        gpu_type=request.gpu_type,
        base_model=request.config.base_model,
        total_epochs=request.config.epochs,
        created_at=datetime.now().isoformat(),
        logs=["Job created, waiting to start..."]
    )

    jobs_db[job_id] = job.model_dump()

    # Start training in background
    background_tasks.add_task(
        run_training_job,
        job_id,
        request.credentials,
        request.config,
        request.gpu_type
    )

    return {"job_id": job_id, "status": "pending", "message": "Training job queued"}


async def run_training_job(
    job_id: str,
    credentials: CloudCredentials,
    config: TrainingConfig,
    gpu_type: GPUType
):
    """Background task to run training"""
    job = jobs_db[job_id]

    try:
        # Update status to provisioning
        job["status"] = JobStatus.PROVISIONING.value
        job["logs"].append("Provisioning GPU instance...")
        await broadcast_job_update(job_id, job)

        # Simulate provisioning (in real implementation, create RunPod instance)
        await asyncio.sleep(3)

        if credentials.provider == CloudProvider.RUNPOD:
            job["logs"].append(f"RunPod instance starting with {gpu_type.value}...")
            # In production: client = RunPodClient(credentials.api_key)
            # pod = await client.create_pod(gpu_type_id, f"training-{job_id}")

        # Update to running
        job["status"] = JobStatus.RUNNING.value
        job["logs"].append("Training started!")
        await broadcast_job_update(job_id, job)

        # Simulate training progress
        total_steps = config.epochs * 100  # Simplified
        for epoch in range(config.epochs):
            job["current_epoch"] = epoch + 1
            for step in range(100):
                # Simulate training metrics
                progress = ((epoch * 100) + step + 1) / total_steps
                job["progress"] = round(progress * 100, 1)

                # Simulate loss values
                base_loss = 2.5 - (progress * 1.2)  # Decreasing loss
                job["training_loss"] = round(base_loss + (0.1 * (1 - progress)), 3)
                job["validation_loss"] = round(base_loss + 0.15, 3)

                # Estimate remaining time
                elapsed_percent = progress * 100
                if elapsed_percent > 0:
                    # Rough estimate based on GPU calc
                    gpu_calc = calculate_gpu_requirements(
                        config.base_model,
                        config.dataset_size,
                        config.epochs,
                        credentials.provider
                    )
                    total_hours = gpu_calc["compatible_gpus"][0]["estimated_hours"]
                    remaining_hours = total_hours * (1 - progress)
                    if remaining_hours < 1:
                        job["estimated_time_remaining"] = f"{int(remaining_hours * 60)} minutes"
                    else:
                        job["estimated_time_remaining"] = f"{remaining_hours:.1f} hours"

                    # Calculate cost
                    hourly_price = GPU_SPECS[gpu_type][f"price_{credentials.provider.value}"]
                    job["cost_so_far"] = round(total_hours * progress * hourly_price, 2)

                # Log every 25 steps
                if step % 25 == 0:
                    job["logs"].append(
                        f"Epoch {epoch+1}/{config.epochs} - Step {step}/100 - "
                        f"Loss: {job['training_loss']}"
                    )

                await broadcast_job_update(job_id, job)
                await asyncio.sleep(0.5)  # Simulate training time

        # Training complete
        job["status"] = JobStatus.COMPLETED.value
        job["progress"] = 100.0
        job["logs"].append("Training completed successfully!")
        job["logs"].append("Model saved to: models/checkpoints/your-model/")
        await broadcast_job_update(job_id, job)

    except Exception as e:
        job["status"] = JobStatus.FAILED.value
        job["logs"].append(f"Error: {str(e)}")
        await broadcast_job_update(job_id, job)


async def broadcast_job_update(job_id: str, job: Dict):
    """Broadcast job updates to connected WebSocket clients"""
    if job_id in connected_websockets:
        for ws in connected_websockets[job_id]:
            try:
                await ws.send_json(job)
            except Exception:
                pass


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs_db[job_id]


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs"""
    return {"jobs": list(jobs_db.values())}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]
    if job["status"] in [JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
        raise HTTPException(status_code=400, detail="Job already finished")

    job["status"] = JobStatus.CANCELLED.value
    job["logs"].append("Job cancelled by user")

    return {"message": "Job cancelled", "job_id": job_id}


@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_updates(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates"""
    await websocket.accept()

    if job_id not in connected_websockets:
        connected_websockets[job_id] = []
    connected_websockets[job_id].append(websocket)

    try:
        # Send current state
        if job_id in jobs_db:
            await websocket.send_json(jobs_db[job_id])

        # Keep connection alive
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        if job_id in connected_websockets:
            connected_websockets[job_id].remove(websocket)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
