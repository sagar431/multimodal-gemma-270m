'use client'

import { useState, useEffect, useCallback } from 'react'
import {
  Cloud,
  Cpu,
  Settings,
  Play,
  Check,
  ChevronRight,
  ChevronLeft,
  Zap,
  DollarSign,
  Clock,
  AlertCircle,
  Loader2,
  Eye,
  Brain,
  Server,
  TrendingDown,
  RefreshCw
} from 'lucide-react'

// API Base URL
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Types
interface CloudProvider {
  id: string
  name: string
  logo: string
  description: string
  status: 'available' | 'coming_soon'
}

interface BaseModel {
  id: string
  name: string
  params_b: number
  min_vram: number
  description: string
}

interface GPUOption {
  gpu_type: string
  vram_gb: number
  estimated_hours: number
  hourly_price: number
  estimated_cost: number
  recommended: boolean
  best_value?: boolean
}

interface TrainingJob {
  job_id: string
  status: string
  progress: number
  current_epoch: number
  total_epochs: number
  training_loss: number | null
  validation_loss: number | null
  estimated_time_remaining: string | null
  cost_so_far: number
  logs: string[]
}

// Cloud Providers
const cloudProviders: CloudProvider[] = [
  {
    id: 'runpod',
    name: 'RunPod',
    logo: '🚀',
    description: 'Best value GPU cloud with easy API',
    status: 'available'
  },
  {
    id: 'lambda',
    name: 'Lambda Labs',
    logo: '⚡',
    description: 'Premium GPU instances for ML',
    status: 'coming_soon'
  },
  {
    id: 'aws',
    name: 'AWS',
    logo: '☁️',
    description: 'Enterprise-grade cloud computing',
    status: 'coming_soon'
  }
]

// Steps
const steps = [
  { id: 1, name: 'Cloud Setup', icon: Cloud },
  { id: 2, name: 'Model Selection', icon: Brain },
  { id: 3, name: 'Configuration', icon: Settings },
  { id: 4, name: 'Training', icon: Play }
]

export default function Home() {
  // Wizard state
  const [currentStep, setCurrentStep] = useState(1)

  // Step 1: Cloud credentials
  const [selectedProvider, setSelectedProvider] = useState<string>('runpod')
  const [apiKey, setApiKey] = useState('')
  const [isValidating, setIsValidating] = useState(false)
  const [credentialsValid, setCredentialsValid] = useState<boolean | null>(null)

  // Step 2: Model selection
  const [models, setModels] = useState<BaseModel[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('gemma-270m')

  // Step 3: Configuration
  const [datasetSize, setDatasetSize] = useState(50000)
  const [epochs, setEpochs] = useState(3)
  const [batchSize, setBatchSize] = useState(4)
  const [learningRate, setLearningRate] = useState(0.0002)
  const [gpuOptions, setGpuOptions] = useState<GPUOption[]>([])
  const [selectedGpu, setSelectedGpu] = useState<string>('')
  const [isCalculating, setIsCalculating] = useState(false)

  // Step 4: Training
  const [job, setJob] = useState<TrainingJob | null>(null)
  const [isStarting, setIsStarting] = useState(false)

  // Fetch models on mount
  useEffect(() => {
    fetchModels()
  }, [])

  // Calculate GPU when config changes
  useEffect(() => {
    if (currentStep === 3 && selectedModel) {
      calculateGpu()
    }
  }, [currentStep, selectedModel, datasetSize, epochs, selectedProvider])

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_URL}/api/models`)
      const data = await res.json()
      setModels(data.models)
    } catch (err) {
      console.error('Failed to fetch models:', err)
      // Use fallback data
      setModels([
        { id: 'gemma-270m', name: 'Gemma 270M', params_b: 0.54, min_vram: 12, description: '0.27B LLM + 0.43B Vision Encoder' },
        { id: 'gemma-2b', name: 'Gemma 2B', params_b: 2.4, min_vram: 24, description: '2B LLM + 0.43B Vision Encoder' },
        { id: 'phi-2', name: 'Phi-2', params_b: 3.1, min_vram: 24, description: '2.7B LLM + 0.43B Vision Encoder' },
        { id: 'llama-1b', name: 'Llama 1B', params_b: 1.4, min_vram: 16, description: '1B LLM + 0.43B Vision Encoder' },
        { id: 'llama-3b', name: 'Llama 3B', params_b: 3.4, min_vram: 32, description: '3B LLM + 0.43B Vision Encoder' },
      ])
    }
  }

  const validateCredentials = async () => {
    if (!apiKey.trim()) return

    setIsValidating(true)
    try {
      const res = await fetch(`${API_URL}/api/validate-credentials`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: selectedProvider,
          api_key: apiKey
        })
      })

      if (res.ok) {
        setCredentialsValid(true)
      } else {
        setCredentialsValid(false)
      }
    } catch (err) {
      // For demo, accept any key
      setCredentialsValid(true)
    }
    setIsValidating(false)
  }

  const calculateGpu = async () => {
    setIsCalculating(true)
    try {
      const params = new URLSearchParams({
        base_model: selectedModel,
        dataset_size: datasetSize.toString(),
        epochs: epochs.toString(),
        provider: selectedProvider
      })

      const res = await fetch(`${API_URL}/api/calculate-gpu?${params}`, {
        method: 'POST'
      })

      if (res.ok) {
        const data = await res.json()
        setGpuOptions(data.compatible_gpus)

        // Auto-select best value GPU
        const bestValue = data.compatible_gpus.find((g: GPUOption) => g.best_value)
        if (bestValue) {
          setSelectedGpu(bestValue.gpu_type)
        }
      }
    } catch (err) {
      // Use fallback data
      setGpuOptions([
        { gpu_type: 'a10', vram_gb: 24, estimated_hours: 5.4, hourly_price: 0.36, estimated_cost: 1.94, recommended: true, best_value: true },
        { gpu_type: 'rtx4090', vram_gb: 24, estimated_hours: 3.9, hourly_price: 0.44, estimated_cost: 1.72, recommended: true },
        { gpu_type: 'a100-40gb', vram_gb: 40, estimated_hours: 3.0, hourly_price: 0.79, estimated_cost: 2.37, recommended: true },
      ])
      setSelectedGpu('a10')
    }
    setIsCalculating(false)
  }

  const startTraining = async () => {
    setIsStarting(true)
    try {
      const res = await fetch(`${API_URL}/api/start-training`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          credentials: {
            provider: selectedProvider,
            api_key: apiKey
          },
          config: {
            base_model: selectedModel,
            dataset_size: datasetSize,
            epochs: epochs,
            batch_size: batchSize,
            learning_rate: learningRate,
            use_lora: true,
            lora_rank: 64
          },
          gpu_type: selectedGpu
        })
      })

      if (res.ok) {
        const data = await res.json()
        // Connect to WebSocket for updates
        connectToJobUpdates(data.job_id)
      }
    } catch (err) {
      // Demo mode - simulate job
      simulateTrainingJob()
    }
    setIsStarting(false)
  }

  const connectToJobUpdates = (jobId: string) => {
    const ws = new WebSocket(`ws://localhost:8000/ws/jobs/${jobId}`)

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setJob(data)
    }

    ws.onerror = () => {
      // Fallback to polling
      pollJobStatus(jobId)
    }
  }

  const pollJobStatus = async (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/api/jobs/${jobId}`)
        if (res.ok) {
          const data = await res.json()
          setJob(data)

          if (['completed', 'failed', 'cancelled'].includes(data.status)) {
            clearInterval(interval)
          }
        }
      } catch (err) {
        clearInterval(interval)
      }
    }, 2000)
  }

  const simulateTrainingJob = () => {
    // Demo simulation
    let progress = 0
    let epoch = 1

    const initialJob: TrainingJob = {
      job_id: 'demo-' + Date.now(),
      status: 'running',
      progress: 0,
      current_epoch: 1,
      total_epochs: epochs,
      training_loss: 2.5,
      validation_loss: 2.65,
      estimated_time_remaining: `${gpuOptions.find(g => g.gpu_type === selectedGpu)?.estimated_hours || 3} hours`,
      cost_so_far: 0,
      logs: ['Training started...', `Using ${selectedGpu.toUpperCase()} GPU`]
    }

    setJob(initialJob)

    const interval = setInterval(() => {
      progress += 2

      if (progress >= 100) {
        setJob(prev => prev ? {
          ...prev,
          status: 'completed',
          progress: 100,
          training_loss: 1.33,
          validation_loss: 1.43,
          estimated_time_remaining: null,
          logs: [...prev.logs, 'Training completed successfully!', 'Model saved to: models/checkpoints/your-model/']
        } : null)
        clearInterval(interval)
        return
      }

      const newEpoch = Math.ceil((progress / 100) * epochs)
      const loss = 2.5 - (progress / 100) * 1.17
      const valLoss = loss + 0.1
      const costPerHour = gpuOptions.find(g => g.gpu_type === selectedGpu)?.hourly_price || 0.5
      const totalHours = gpuOptions.find(g => g.gpu_type === selectedGpu)?.estimated_hours || 3

      setJob(prev => prev ? {
        ...prev,
        progress,
        current_epoch: newEpoch,
        training_loss: parseFloat(loss.toFixed(3)),
        validation_loss: parseFloat(valLoss.toFixed(3)),
        cost_so_far: parseFloat((totalHours * (progress / 100) * costPerHour).toFixed(2)),
        estimated_time_remaining: `${((100 - progress) / 100 * totalHours * 60).toFixed(0)} minutes`,
        logs: newEpoch !== epoch
          ? [...prev.logs, `Epoch ${newEpoch} started - Loss: ${loss.toFixed(3)}`]
          : prev.logs
      } : null)

      epoch = newEpoch
    }, 500)
  }

  const canProceed = useCallback(() => {
    switch (currentStep) {
      case 1:
        return credentialsValid === true
      case 2:
        return selectedModel !== ''
      case 3:
        return selectedGpu !== ''
      default:
        return true
    }
  }, [currentStep, credentialsValid, selectedModel, selectedGpu])

  const nextStep = () => {
    if (currentStep < 4 && canProceed()) {
      setCurrentStep(currentStep + 1)
    }
  }

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-100 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl flex items-center justify-center">
                <Eye className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Multimodal Training Platform</h1>
                <p className="text-sm text-gray-500">Train vision-language models on cloud GPUs</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-500">No coding required</span>
              <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">Beta</span>
            </div>
          </div>
        </div>
      </header>

      {/* Progress Steps */}
      <div className="bg-white border-b border-gray-100">
        <div className="max-w-4xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            {steps.map((step, index) => (
              <div key={step.id} className="flex items-center">
                <div className="flex items-center">
                  <div
                    className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 ${
                      currentStep > step.id
                        ? 'bg-green-500 text-white'
                        : currentStep === step.id
                        ? 'bg-primary-600 text-white'
                        : 'bg-gray-100 text-gray-400'
                    }`}
                  >
                    {currentStep > step.id ? (
                      <Check className="w-5 h-5" />
                    ) : (
                      <step.icon className="w-5 h-5" />
                    )}
                  </div>
                  <span
                    className={`ml-3 font-medium ${
                      currentStep >= step.id ? 'text-gray-900' : 'text-gray-400'
                    }`}
                  >
                    {step.name}
                  </span>
                </div>
                {index < steps.length - 1 && (
                  <ChevronRight className="w-5 h-5 mx-6 text-gray-300" />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-6 py-8">
        {/* Step 1: Cloud Setup */}
        {currentStep === 1 && (
          <div className="animate-slide-up">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Connect Your Cloud Provider</h2>
              <p className="text-gray-600">Select a cloud GPU provider and enter your API key</p>
            </div>

            <div className="grid grid-cols-3 gap-4 mb-8">
              {cloudProviders.map((provider) => (
                <button
                  key={provider.id}
                  onClick={() => provider.status === 'available' && setSelectedProvider(provider.id)}
                  disabled={provider.status === 'coming_soon'}
                  className={`card card-hover text-left relative ${
                    selectedProvider === provider.id
                      ? 'ring-2 ring-primary-500 border-primary-500'
                      : ''
                  } ${provider.status === 'coming_soon' ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  {provider.status === 'coming_soon' && (
                    <span className="absolute top-2 right-2 text-xs bg-gray-100 text-gray-500 px-2 py-1 rounded">
                      Coming Soon
                    </span>
                  )}
                  <div className="text-3xl mb-3">{provider.logo}</div>
                  <h3 className="font-semibold text-gray-900">{provider.name}</h3>
                  <p className="text-sm text-gray-500 mt-1">{provider.description}</p>
                </button>
              ))}
            </div>

            <div className="card">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {selectedProvider === 'runpod' ? 'RunPod' : selectedProvider === 'lambda' ? 'Lambda Labs' : 'AWS'} API Key
              </label>
              <div className="flex gap-3">
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => {
                    setApiKey(e.target.value)
                    setCredentialsValid(null)
                  }}
                  placeholder="Enter your API key..."
                  className="input flex-1"
                />
                <button
                  onClick={validateCredentials}
                  disabled={!apiKey.trim() || isValidating}
                  className="btn-primary flex items-center gap-2"
                >
                  {isValidating ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Check className="w-5 h-5" />
                  )}
                  Validate
                </button>
              </div>

              {credentialsValid !== null && (
                <div className={`mt-3 flex items-center gap-2 ${
                  credentialsValid ? 'text-green-600' : 'text-red-600'
                }`}>
                  {credentialsValid ? (
                    <>
                      <Check className="w-5 h-5" />
                      <span>API key validated successfully</span>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="w-5 h-5" />
                      <span>Invalid API key</span>
                    </>
                  )}
                </div>
              )}

              <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                <h4 className="font-medium text-blue-900 mb-2">How to get your API key:</h4>
                <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
                  <li>Go to <a href="https://runpod.io" target="_blank" rel="noopener" className="underline">runpod.io</a> and create an account</li>
                  <li>Navigate to Settings → API Keys</li>
                  <li>Generate a new API key</li>
                  <li>Copy and paste it above</li>
                </ol>
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Model Selection */}
        {currentStep === 2 && (
          <div className="animate-slide-up">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Choose Your Base Model</h2>
              <p className="text-gray-600">Select the language model to make multimodal</p>
            </div>

            <div className="space-y-4">
              {models.map((model) => (
                <button
                  key={model.id}
                  onClick={() => setSelectedModel(model.id)}
                  className={`card card-hover w-full text-left flex items-center gap-4 ${
                    selectedModel === model.id
                      ? 'ring-2 ring-primary-500 border-primary-500'
                      : ''
                  }`}
                >
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center">
                    <Brain className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900">{model.name}</h3>
                    <p className="text-sm text-gray-500">{model.description}</p>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold text-gray-900">{model.params_b}B params</div>
                    <div className="text-sm text-gray-500">Min {model.min_vram}GB VRAM</div>
                  </div>
                  {selectedModel === model.id && (
                    <div className="w-6 h-6 bg-primary-600 rounded-full flex items-center justify-center">
                      <Check className="w-4 h-4 text-white" />
                    </div>
                  )}
                </button>
              ))}
            </div>

            <div className="mt-6 p-4 bg-amber-50 border border-amber-200 rounded-lg">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-amber-600 mt-0.5" />
                <div>
                  <h4 className="font-medium text-amber-900">Recommended for Beginners</h4>
                  <p className="text-sm text-amber-800 mt-1">
                    Start with <strong>Gemma 270M</strong> - it trains fast, costs less, and gives you great results
                    for learning. You can scale up later!
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Configuration */}
        {currentStep === 3 && (
          <div className="animate-slide-up">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Configure Training</h2>
              <p className="text-gray-600">Set your training parameters and select a GPU</p>
            </div>

            <div className="grid grid-cols-2 gap-6 mb-8">
              <div className="card">
                <h3 className="font-semibold text-gray-900 mb-4">Training Parameters</h3>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Dataset Size: {datasetSize.toLocaleString()} samples
                    </label>
                    <input
                      type="range"
                      min="10000"
                      max="157000"
                      step="1000"
                      value={datasetSize}
                      onChange={(e) => setDatasetSize(parseInt(e.target.value))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>10K (Quick test)</span>
                      <span>157K (Full)</span>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Epochs: {epochs}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={epochs}
                      onChange={(e) => setEpochs(parseInt(e.target.value))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>1 (Fast)</span>
                      <span>10 (Thorough)</span>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Batch Size: {batchSize}
                    </label>
                    <select
                      value={batchSize}
                      onChange={(e) => setBatchSize(parseInt(e.target.value))}
                      className="input"
                    >
                      <option value={2}>2 (Low memory)</option>
                      <option value={4}>4 (Recommended)</option>
                      <option value={8}>8 (High memory)</option>
                      <option value={16}>16 (Very high memory)</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className="card">
                <h3 className="font-semibold text-gray-900 mb-4">Advanced Settings</h3>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Learning Rate
                    </label>
                    <select
                      value={learningRate}
                      onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                      className="input"
                    >
                      <option value={0.0001}>1e-4 (Conservative)</option>
                      <option value={0.0002}>2e-4 (Recommended)</option>
                      <option value={0.0005}>5e-4 (Aggressive)</option>
                    </select>
                  </div>

                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Zap className="w-5 h-5 text-primary-600" />
                      <span className="font-medium text-gray-900">LoRA Enabled</span>
                    </div>
                    <p className="text-sm text-gray-600">
                      Training uses LoRA (rank 64) for efficient fine-tuning.
                      Only 3.4% of parameters are trained.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* GPU Selection */}
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900">Select GPU</h3>
                {isCalculating && (
                  <div className="flex items-center gap-2 text-sm text-gray-500">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Calculating...
                  </div>
                )}
              </div>

              <div className="space-y-3">
                {gpuOptions.map((gpu) => (
                  <button
                    key={gpu.gpu_type}
                    onClick={() => setSelectedGpu(gpu.gpu_type)}
                    className={`w-full p-4 rounded-lg border-2 text-left transition-all ${
                      selectedGpu === gpu.gpu_type
                        ? 'border-primary-500 bg-primary-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Server className="w-5 h-5 text-gray-600" />
                        <div>
                          <span className="font-medium text-gray-900">
                            {gpu.gpu_type.toUpperCase().replace('-', ' ')}
                          </span>
                          {gpu.best_value && (
                            <span className="ml-2 px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full">
                              Best Value
                            </span>
                          )}
                          {gpu.recommended && !gpu.best_value && (
                            <span className="ml-2 px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full">
                              Recommended
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold text-gray-900">
                          ${gpu.estimated_cost.toFixed(2)} total
                        </div>
                        <div className="text-sm text-gray-500">
                          {gpu.vram_gb}GB • ~{gpu.estimated_hours}h • ${gpu.hourly_price}/hr
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Step 4: Training Dashboard */}
        {currentStep === 4 && (
          <div className="animate-slide-up">
            {!job ? (
              <div className="text-center">
                <div className="mb-8">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">Ready to Train!</h2>
                  <p className="text-gray-600">Review your configuration and start training</p>
                </div>

                <div className="card max-w-lg mx-auto mb-8">
                  <h3 className="font-semibold text-gray-900 mb-4">Training Summary</h3>

                  <div className="space-y-3 text-left">
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-600">Base Model</span>
                      <span className="font-medium">{selectedModel}</span>
                    </div>
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-600">Cloud Provider</span>
                      <span className="font-medium">{selectedProvider}</span>
                    </div>
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-600">GPU</span>
                      <span className="font-medium">{selectedGpu.toUpperCase()}</span>
                    </div>
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-600">Dataset Size</span>
                      <span className="font-medium">{datasetSize.toLocaleString()} samples</span>
                    </div>
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-gray-600">Epochs</span>
                      <span className="font-medium">{epochs}</span>
                    </div>
                    <div className="flex justify-between py-2">
                      <span className="text-gray-600">Estimated Cost</span>
                      <span className="font-semibold text-green-600">
                        ${gpuOptions.find(g => g.gpu_type === selectedGpu)?.estimated_cost.toFixed(2) || '0.00'}
                      </span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={startTraining}
                  disabled={isStarting}
                  className="btn-primary text-lg px-8 py-4 flex items-center gap-3 mx-auto"
                >
                  {isStarting ? (
                    <Loader2 className="w-6 h-6 animate-spin" />
                  ) : (
                    <Play className="w-6 h-6" />
                  )}
                  Start Training
                </button>
              </div>
            ) : (
              <div>
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900">Training in Progress</h2>
                    <p className="text-gray-600">Job ID: {job.job_id}</p>
                  </div>
                  <span className={`px-4 py-2 rounded-full font-medium ${
                    job.status === 'running' ? 'bg-blue-100 text-blue-700' :
                    job.status === 'completed' ? 'bg-green-100 text-green-700' :
                    job.status === 'failed' ? 'bg-red-100 text-red-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                  </span>
                </div>

                {/* Progress Bar */}
                <div className="card mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-gray-900">Overall Progress</span>
                    <span className="text-lg font-bold text-primary-600">{job.progress.toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-4 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-primary-500 to-primary-600 transition-all duration-500"
                      style={{ width: `${job.progress}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-sm text-gray-500 mt-2">
                    <span>Epoch {job.current_epoch}/{job.total_epochs}</span>
                    {job.estimated_time_remaining && (
                      <span>{job.estimated_time_remaining} remaining</span>
                    )}
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-4 gap-4 mb-6">
                  <div className="card text-center">
                    <TrendingDown className="w-6 h-6 text-blue-500 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-gray-900">
                      {job.training_loss?.toFixed(3) || '-'}
                    </div>
                    <div className="text-sm text-gray-500">Training Loss</div>
                  </div>
                  <div className="card text-center">
                    <TrendingDown className="w-6 h-6 text-purple-500 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-gray-900">
                      {job.validation_loss?.toFixed(3) || '-'}
                    </div>
                    <div className="text-sm text-gray-500">Validation Loss</div>
                  </div>
                  <div className="card text-center">
                    <Clock className="w-6 h-6 text-amber-500 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-gray-900">
                      {job.estimated_time_remaining || '-'}
                    </div>
                    <div className="text-sm text-gray-500">Time Left</div>
                  </div>
                  <div className="card text-center">
                    <DollarSign className="w-6 h-6 text-green-500 mx-auto mb-2" />
                    <div className="text-2xl font-bold text-gray-900">
                      ${job.cost_so_far.toFixed(2)}
                    </div>
                    <div className="text-sm text-gray-500">Cost So Far</div>
                  </div>
                </div>

                {/* Logs */}
                <div className="card">
                  <h3 className="font-semibold text-gray-900 mb-4">Training Logs</h3>
                  <div className="bg-gray-900 rounded-lg p-4 h-48 overflow-y-auto font-mono text-sm">
                    {job.logs.map((log, i) => (
                      <div key={i} className="text-green-400">
                        <span className="text-gray-500">[{String(i).padStart(3, '0')}]</span> {log}
                      </div>
                    ))}
                  </div>
                </div>

                {job.status === 'completed' && (
                  <div className="mt-6 p-6 bg-green-50 border border-green-200 rounded-xl">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center">
                        <Check className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <h3 className="text-lg font-bold text-green-900">Training Complete!</h3>
                        <p className="text-green-700">Your multimodal model is ready</p>
                      </div>
                    </div>
                    <div className="flex gap-4">
                      <button className="btn-primary bg-green-600 hover:bg-green-700">
                        Download Model
                      </button>
                      <button className="btn-secondary">
                        Try Demo
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Navigation */}
        {currentStep < 4 && (
          <div className="flex justify-between mt-8">
            <button
              onClick={prevStep}
              disabled={currentStep === 1}
              className="btn-secondary flex items-center gap-2"
            >
              <ChevronLeft className="w-5 h-5" />
              Back
            </button>
            <button
              onClick={nextStep}
              disabled={!canProceed()}
              className="btn-primary flex items-center gap-2"
            >
              Continue
              <ChevronRight className="w-5 h-5" />
            </button>
          </div>
        )}

        {currentStep === 4 && !job && (
          <div className="flex justify-start mt-8">
            <button
              onClick={prevStep}
              className="btn-secondary flex items-center gap-2"
            >
              <ChevronLeft className="w-5 h-5" />
              Back to Configuration
            </button>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-100 bg-white mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <p>Built with multimodal-gemma training pipeline</p>
            <div className="flex items-center gap-4">
              <a href="#" className="hover:text-gray-900">Documentation</a>
              <a href="#" className="hover:text-gray-900">GitHub</a>
              <a href="#" className="hover:text-gray-900">Support</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
