import easyocr
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import os
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()
POPPLER_PATH = os.getenv("POPPLER_PATH", None)

model_dir = Path(__file__).parent.parent.parent / 'easyocr_models'
model_dir.mkdir(parents=True, exist_ok=True)

def _cuda_available() -> bool:
    """Return True only when CUDA is actually available for PyTorch."""
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _initialize_reader() -> easyocr.Reader:
    use_gpu = _cuda_available()
    if use_gpu:
        device_name = torch.cuda.get_device_name(0)
        print(f"Initializing EasyOCR with CUDA GPU: {device_name}")
    else:
        print("CUDA not available. Initializing EasyOCR on CPU.")

    try:
        reader_instance = easyocr.Reader(
            ['hi', 'en'],
            gpu=use_gpu,
            model_storage_directory=str(model_dir),
            user_network_directory=str(model_dir)
        )
        print(f"EasyOCR initialized successfully (gpu={use_gpu})")
        return reader_instance
    except Exception as e:
        # If GPU initialization fails despite detection, retry on CPU.
        if use_gpu:
            print(f"EasyOCR GPU initialization failed: {e}")
            print("Retrying EasyOCR initialization on CPU...")
            reader_instance = easyocr.Reader(
                ['hi', 'en'],
                gpu=False,
                model_storage_directory=str(model_dir),
                user_network_directory=str(model_dir)
            )
            print("EasyOCR initialized on CPU after GPU fallback")
            return reader_instance
        raise


reader = _initialize_reader()

async def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    
    print(f"\n{'='*60}")
    print(f"Starting OCR extraction for: {filename}")
    print(f"{'='*60}")
    
    extracted_text = ""
    if filename.lower().endswith('.pdf'):
        
        try:
            print("Converting PDF to images...")
            images = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH)
            print(f"PDF has {len(images)} page(s)")
            
            for i, image in enumerate(images):
                print(f"\nProcessing page {i+1}...")
                image_np = np.array(image)
                print(f"Image size: {image_np.shape}")
                
                result = reader.readtext(image_np, detail=0, paragraph=True)
                page_text = " ".join(result)
                print(f"Extracted {len(page_text)} characters from page {i+1}")
                print(f"Sample text: {page_text[:100]}...")
                
                extracted_text += f"\n--- Page {i+1} ---\n"
                extracted_text += page_text
                
                if _cuda_available():
                    torch.cuda.empty_cache()
            
            print(f"\nTotal extracted: {len(extracted_text)} characters")
            
            if _cuda_available():
                torch.cuda.empty_cache()
                print(f"GPU cache cleared after PDF processing")
                
        except Exception as e:
            error_msg = f"Error processing PDF: {str(e)}"
            print(f"{error_msg}")
            if _cuda_available():
                torch.cuda.empty_cache()
            return error_msg

    else:
        try:
            print("Processing image file...")
            result = reader.readtext(file_bytes, detail=0, paragraph=True)
            extracted_text = " ".join(result)
            print(f"Extracted {len(extracted_text)} characters from image")
            print(f"Sample text: {extracted_text[:100]}...")
            
            if _cuda_available():
                torch.cuda.empty_cache()
                print(f"GPU cache cleared after image processing")
                
        except Exception as e:
            error_msg = f"Error processing Image: {str(e)}"
            print(f"{error_msg}")
            if _cuda_available():
                torch.cuda.empty_cache()
            return error_msg
    
    print(f"{'='*60}\n")
    return extracted_text