# NyayShayak Backend - Virtual Senior Prosecutor API

AI-powered legal case analysis system for Indian law enforcement. Performs OCR on FIR documents, translates Hindi to English, and provides legal analysis using fine-tuned LLMs.

## Features

- 📄 **OCR Processing**: Extract text from PDF/Images (Hindi & English)
- 🔤 **Translation**: Hindi to English using IndicTrans2
- 🧠 **Legal AI Analysis**: LLaMA-3-8B fine-tuned on Indian legal data
- 📚 **RAG System**: Vector-based case file retrieval using ChromaDB
- ⚡ **FastAPI**: High-performance async API

---

## Prerequisites

- **Python 3.10+** 
- **GPU**: NVIDIA (CUDA) or Intel Arc recommended, CPU supported
- **Poppler** (Windows only): [Download here](https://github.com/oschwartz10612/poppler-windows/releases/)
- **HuggingFace Account**: For model access

---

## Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd NyayShayak/backend
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**For Intel Arc GPU users:**
```bash
pip install intel-extension-for-pytorch
```

### 4. Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
# Required
HF_TOKEN=your_huggingface_token_here

# Path to your fine-tuned adapter
ADAPTER_PATH=D:\path\to\your\adapter\checkpoint-xxxx

# Optional (Windows PDF support)
POPPLER_PATH=C:\path\to\poppler-xx\Library\bin

# Optional
# auto (default): use GPU when available and VRAM threshold is met, else CPU
# true: force GPU attempt, false: always CPU
TRANSLATION_USE_GPU=auto

# Optional (used when TRANSLATION_USE_GPU=true or auto)
# Translation switches to CPU if detected VRAM is below this threshold.
TRANSLATION_MIN_GPU_VRAM_GB=6

# Optional memory safety flag
# false (default): unload translation model after each request to free VRAM for legal analysis
# true: keep the translation model resident between requests
TRANSLATION_KEEP_MODEL_RESIDENT=false

# Translation does not auto-fallback to CPU on GPU OOM; it fails fast instead.

# Optional translation speed/quality knobs
# Lower max tokens improves speed; increase if outputs are cut short.
TRANSLATION_MAX_NEW_TOKENS=128
# Bigger chunk size reduces number of model calls for long FIRs.
TRANSLATION_MAX_CHUNK_SIZE=700
# Cache repeated translations/chunks to speed up iterative testing.
TRANSLATION_CACHE_ITEMS=128

# Optional startup behavior
# false (recommended): backend starts immediately; legal model lazy-loads on first analysis request
# true: preload legal model during startup (can take long and block API readiness)
PRELOAD_REASONING_MODEL=false

# Optional legal engine memory optimization (for CPU-only systems)
# true (default): use 8-bit quantization for CPU mode (fits in ~8GB)
# false: disable quantization (requires ~32GB RAM, may fail)
LEGAL_ENGINE_8BIT_CPU=true

# Optional: skip fine-tuned adapter loading for faster startup (base model only, lower quality)
LEGAL_ENGINE_SKIP_ADAPTER=false

# Optional: force legal engine to use GPU only (no CPU fallback)
LEGAL_ENGINE_REQUIRE_GPU=true
```

**Get HuggingFace Token:**
1. Create account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens
3. Create token with `read` permissions

### 5. Download Models

Models will auto-download on first run:
- LLaMA-3-8B (requires HF token)
- IndicTrans2 (translation)
- EasyOCR (Hindi/English)

---

## Running the Server

```bash
uvicorn app.main:app --reload
```

Server will start at: `http://localhost:8000`

**API Documentation:** `http://localhost:8000/docs`

---

## API Endpoints

### `POST /api/v1/analyze-case-rag`

Analyze FIR/Chargesheet document

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze-case-rag" \
  -F "file=@path/to/fir.pdf"
```

**Response:**
```json
{
  "case_id": "uuid-string",
  "summary": "Brief case summary...",
  "offenses": ["Section 379 IPC", "Section 420 IPC"],
  "missing_evidence": ["Independent witness statement", "Seizure memo"],
  "recommendation": "IO should collect additional evidence..."
}
```

---

## Project Structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI app & startup
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   ├── routers/
│   │   └── analysis.py         # API endpoints
│   └── services/
│       ├── ocr_service.py      # EasyOCR processing
│       ├── translation_service.py  # IndicTrans2
│       ├── rag_service.py      # ChromaDB vector store
│       └── legal_engine.py     # LLM inference
├── chroma_db/                  # Vector database (auto-created)
├── easyocr_models/             # OCR models (auto-downloaded)
├── requirements.txt
├── .env                        # Your config (create this)
└── README.md
```

---

## Troubleshooting

### 1. CUDA Not Available (NVIDIA GPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. PDF Conversion Fails (Windows)
- Download Poppler from [here](https://github.com/oschwartz10612/poppler-windows/releases/)
- Extract and add `bin` folder path to `.env` as `POPPLER_PATH`

### 3. Model Download Slow/Fails
- Check HuggingFace token is valid
- Ensure stable internet connection
- Models are large (8GB+), first run takes time

### 4. Out of Memory
- Reduce batch size in code
- Use smaller model variant
- Enable CPU offloading
- Set `TRANSLATION_USE_GPU=false` on 8GB GPUs if the legal model is also using CUDA

---

## Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB+ |
| GPU VRAM | 8GB | 16GB+ |
| Storage | 20GB | 50GB+ |
| CPU | 4 cores | 8+ cores |

**Note:** CPU-only mode works but is significantly slower (30-60s per request).

---

## License

[Your License Here]

## Contact

[Your Contact Info]
