# Multimodal Training Platform

A no-code web interface for training multimodal vision-language models on cloud GPUs.

## Features

- **Cloud Provider Integration** - Connect to RunPod (Lambda Labs & AWS coming soon)
- **Model Selection** - Choose from Gemma 270M, Gemma 2B, Phi-2, Llama 1B/3B
- **GPU Calculator** - Automatically calculate VRAM requirements and cost estimates
- **Real-time Training Dashboard** - Monitor progress, loss curves, and costs
- **One-Click Training** - No coding required!

## Quick Start

### 1. Start the Backend

```bash
cd frontend/backend
pip install -r requirements.txt
python main.py
```

The API will be running at `http://localhost:8000`

### 2. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

The app will be running at `http://localhost:3000`

### 3. Open the App

Visit [http://localhost:3000](http://localhost:3000) in your browser.

## Architecture

```
frontend/
├── app/                  # Next.js app directory
│   ├── layout.tsx       # Root layout
│   ├── page.tsx         # Main wizard page
│   └── globals.css      # Global styles
├── backend/             # FastAPI backend
│   ├── main.py         # API endpoints
│   └── requirements.txt
├── package.json
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/validate-credentials` | POST | Validate cloud provider API key |
| `/api/calculate-gpu` | POST | Calculate GPU requirements |
| `/api/models` | GET | List available base models |
| `/api/gpus` | GET | List available GPU types |
| `/api/start-training` | POST | Start a training job |
| `/api/jobs/{job_id}` | GET | Get job status |
| `/ws/jobs/{job_id}` | WS | Real-time job updates |

## GPU Requirements by Model

| Model | Total Params | Min VRAM | Recommended GPU | Est. Cost (3 epochs, 50K) |
|-------|-------------|----------|-----------------|---------------------------|
| Gemma 270M | 0.54B | 12GB | RTX 4090 / A10 | ~$2-4 |
| Gemma 2B | 2.4B | 24GB | A10 / L40 | ~$8-15 |
| Phi-2 | 3.1B | 24GB | A100-40GB | ~$12-20 |
| Llama 1B | 1.4B | 16GB | A10 | ~$5-10 |
| Llama 3B | 3.4B | 32GB | A100-40GB | ~$15-25 |

## Environment Variables

Create a `.env.local` file in the frontend directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Demo Mode

The frontend includes a demo mode that simulates training without requiring:
- A real cloud provider API key
- Actual GPU instances
- The backend running

This is useful for testing the UI flow.

## Tech Stack

**Frontend:**
- Next.js 14
- React 18
- Tailwind CSS
- TypeScript
- Lucide React (icons)

**Backend:**
- FastAPI
- Pydantic
- httpx (async HTTP)
- WebSockets

## License

Apache 2.0
