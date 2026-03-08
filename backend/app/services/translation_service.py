from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv


load_dotenv()

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"
USE_GPU = os.getenv("TRANSLATION_USE_GPU", "true").lower() == "true"
DEVICE = "cuda" if (torch.cuda.is_available() and USE_GPU) else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")

print(f"Translation service will use: {DEVICE.upper()}")

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    
    if tokenizer is None or model is None:
        print(f"Loading Translation Model on {DEVICE}...")
        
       
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True, 
            token=HF_TOKEN,
            use_fast=False 
        )
        
       
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True, 
            token=HF_TOKEN
        ).to(DEVICE)
        
        print("Translation model loaded successfully!")
        
    return tokenizer, model

async def translate_to_english(hindi_text: str) -> str:
    
    try:
        print(f"\n{'='*60}")
        print(f"Starting Translation")
        print(f"Input length: {len(hindi_text)} characters")
        print(f"{'='*60}")
        
       
        tokenizer_instance, model_instance = load_model()
        
        src_lang = "hin_Deva"
        tgt_lang = "eng_Latn"
        
        
        MAX_CHUNK_SIZE = 300  
        
        def smart_split(text, max_size):
            
            if len(text) <= max_size:
                return [text]
            
            chunks = []
           
            sentences = text.split('।')
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
               
                if len(sentence) > max_size:
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk) + len(word) + 1 < max_size:
                            word_chunk += word + " "
                        else:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                            word_chunk = word + " "
                    if word_chunk:
                        chunks.append(word_chunk.strip())
                elif len(current_chunk) + len(sentence) < max_size:
                    current_chunk += sentence + "। "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + "। "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks if chunks else [text[:max_size]]
        
        if len(hindi_text) > MAX_CHUNK_SIZE:
            chunks = smart_split(hindi_text, MAX_CHUNK_SIZE)
            print(f"Text split into {len(chunks)} chunks")
            
           
            english_parts = []
            for i, chunk in enumerate(chunks):
                print(f"Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
                labeled_text = f"{src_lang} {tgt_lang} {chunk}"
                
                batch = tokenizer_instance(
                    [labeled_text], 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512  # enough for Devanagari input tokens
                ).to(DEVICE)
                
                generated_tokens = model_instance.generate(
                    **batch, 
                    forced_bos_token_id=tokenizer_instance.convert_tokens_to_ids(tgt_lang),
                    use_cache=False, 
                    min_length=0, 
                    max_new_tokens=512,  # new tokens only, not total length
                    num_beams=3,  
                    num_return_sequences=1
                )
                
                english_chunk = tokenizer_instance.batch_decode(
                    generated_tokens.detach().cpu().tolist(), 
                    skip_special_tokens=True
                )[0]
                
                english_parts.append(english_chunk)
                print(f"Chunk {i+1} done: {len(english_chunk)} chars")
                
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            english_translation = " ".join(english_parts)
        else:
            
            print("Text is short, translating directly...")
            labeled_text = f"{src_lang} {tgt_lang} {hindi_text}"

            batch = tokenizer_instance(
                [labeled_text], 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(DEVICE)

            generated_tokens = model_instance.generate(
                **batch, 
                forced_bos_token_id=tokenizer_instance.convert_tokens_to_ids(tgt_lang),
                use_cache=False, 
                min_length=0, 
                max_new_tokens=512, 
                num_beams=3, 
                num_return_sequences=1
            )

            english_translation = tokenizer_instance.batch_decode(
                generated_tokens.detach().cpu().tolist(), 
                skip_special_tokens=True
            )[0]
        
        print(f"\nTranslation complete: {len(english_translation)} characters")
        print(f"First 200 chars: {english_translation[:200]}...")
        print(f"{'='*60}\n")
        
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU cache cleared after translation")
        
        return english_translation

    except Exception as e:
        print(f"Translation Error: {e}")
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"Error in translation: {str(e)}"