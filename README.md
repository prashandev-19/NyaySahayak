# NyaySahayak ⚖️

**Virtual Senior Prosecutor** - AI-Powered Legal Case Analysis System

An intelligent legal assistant that analyzes case documents, identifies offenses, detects evidence gaps, and provides prosecutorial recommendations using OCR, NLP, and RAG technology.

## 🌟 Features

- **📄 Multi-format Document Processing**: Supports PDF and image files (JPG, PNG)
- **🔤 OCR for Hindi Documents**: Extract text from Hindi legal documents using EasyOCR
- **🔄 Automatic Translation**: Hindi to English translation for analysis
- **🤖 AI-Powered Analysis**: 
  - Case summarization
  - Offense identification (IPC/BNS sections)
  - Evidence gap detection
  - Prosecutorial recommendations
- **💾 RAG (Retrieval Augmented Generation)**: Context-aware analysis using vector database
- **🎨 Modern Web Interface**: User-friendly Streamlit frontend
- **🚀 REST API**: FastAPI backend with full documentation

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │  (Frontend - Port 8501)
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   FastAPI       │  (Backend - Port 8000)
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────────┐
    ▼         ▼          ▼              ▼
┌────────┐ ┌──────┐ ┌─────────┐ ┌──────────────┐
│  OCR   │ │Trans-│ │   RAG   │ │ Legal Engine │
│Service │ │lation│ │ Service │ │  (AI Model)  │
└────────┘ └──────┘ └─────────┘ └──────────────┘
```

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **GPU** (Optional): CUDA-enabled GPU for faster processing
- **RAM**: Minimum 8GB recommended
- **OS**: Windows, Linux, or macOS

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/NyayShayak.git
cd NyayShayak
```

### 2. Setup Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```

### 3. Setup Frontend

```powershell
cd ..\frontend
python -m venv venv
.\venv\Scripts\Activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```

### 4. Configure Environment

**Backend**: Create `backend/.env` (optional for custom settings)

**Frontend**: Edit `frontend/.streamlit/secrets.toml`:
```toml
API_BASE_URL = "http://localhost:8000"
```

### 5. Run the Application

**Terminal 1 - Start Backend**:
```powershell
cd backend
.\venv\Scripts\Activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start Frontend**:
```powershell
cd frontend
.\venv\Scripts\Activate
streamlit run app.py
```

### 6. Access the Application

- **Frontend UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Base**: http://localhost:8000

## 📖 Usage

1. **Upload Document**: Navigate to http://localhost:8501 and upload a case document (PDF or image)
2. **Analyze**: Click "Analyze Case" button
3. **Review Results**: 
   - Case summary
   - Identified offenses with IPC/BNS sections
   - Missing evidence and gaps
   - Prosecutorial recommendations
4. **Export**: Download results as JSON or formatted text report

## 🗂️ Project Structure

```
NyayShayak/
├── backend/                    # FastAPI Backend
│   ├── app/
│   │   ├── main.py            # FastAPI application
│   │   ├── routers/           # API endpoints
│   │   │   └── analysis.py
│   │   ├── services/          # Business logic
│   │   │   ├── ocr_service.py
│   │   │   ├── translation_service.py
│   │   │   ├── rag_service.py
│   │   │   └── legal_engine.py
│   │   └── models/            # Pydantic schemas
│   │       └── schemas.py
│   ├── requirements.txt
│   └── README.md
│
├── frontend/                   # Streamlit Frontend
│   ├── app.py                 # Main Streamlit app
│   ├── .streamlit/
│   │   ├── config.toml        # UI configuration
│   │   └── secrets.toml       # API settings (not in git)
│   ├── requirements.txt
│   └── README.md
│
├── Data/                       # Reference documents (not in git)
│   └── README.md
│
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🔧 Configuration

### Backend Configuration

**Environment Variables** (backend/.env):
```env
ENVIRONMENT=development
LOG_LEVEL=INFO
MODEL_PATH=./easyocr_models
```

### Frontend Configuration

**Streamlit Settings** (frontend/.streamlit/config.toml):
- Theme, colors, upload limits

**API Connection** (frontend/.streamlit/secrets.toml):
```toml
API_BASE_URL = "http://localhost:8000"
```

## 🧪 API Endpoints

### POST `/api/v1/analyze-case-rag`

Analyze a case document (PDF or image).

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: file (PDF, JPG, PNG)

**Response**:
```json
{
  "case_id": "uuid",
  "summary": "Case summary text",
  "offenses": ["Offense 1", "Offense 2"],
  "missing_evidence": ["Evidence gap 1"],
  "recommendation": "Recommendation text"
}
```

**Try it**: Visit http://localhost:8000/docs for interactive API documentation

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern web framework
- **EasyOCR**: OCR for Hindi/English text
- **Google Translate API**: Translation service
- **ChromaDB**: Vector database for RAG
- **Transformers**: AI models (Hugging Face)
- **Uvicorn**: ASGI server

### Frontend
- **Streamlit**: Web UI framework
- **Requests**: HTTP client
- **Python**: Core language

## 📝 Development

### Adding New Features

1. **Backend**: Add services in `backend/app/services/`
2. **API Endpoints**: Add routers in `backend/app/routers/`
3. **Frontend**: Modify `frontend/app.py`

### Running Tests

```powershell
# Backend tests (if you add them)
cd backend
pytest

# Check GPU availability
python check_gpu.py
```

## 🐛 Troubleshooting

### Backend won't start
- Check Python version (`python --version`)
- Reinstall dependencies: `pip install -r requirements.txt`
- Check port 8000 is not in use

### Frontend can't connect
- Ensure backend is running on port 8000
- Check `frontend/.streamlit/secrets.toml` has correct URL
- Verify no firewall blocking

### OCR errors
- First run downloads models (takes time)
- Check internet connection for model download
- Verify file format is supported

### Out of memory
- Use smaller files
- Enable GPU if available
- Reduce batch sizes in services

## 📦 Deployment

### Docker Deployment (Coming Soon)

### Cloud Deployment
- Backend: Deploy to AWS/GCP/Azure
- Frontend: Streamlit Cloud or containerized
- Use environment variables for configuration
- Enable HTTPS and authentication

## 🔒 Security Notes

- Never commit `.env` or `secrets.toml` files
- Use environment variables in production
- Implement authentication for production
- Enable CORS properly
- Validate and sanitize file uploads
- Use HTTPS in production

## 📄 License

[Add your license here]

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues or questions:
- Check documentation in `backend/README.md` and `frontend/README.md`
- Review error logs
- Open an issue on GitHub

## 🙏 Acknowledgments

- EasyOCR for Hindi OCR capabilities
- Hugging Face for transformer models
- FastAPI and Streamlit communities

---

**NyayShayak** - Empowering prosecutors with AI-driven legal analysis
