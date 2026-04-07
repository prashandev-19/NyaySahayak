from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel 
import torch
import os
import json
import time
import re
from dotenv import load_dotenv
from app.services import rag_service
from fastapi.concurrency import run_in_threadpool

load_dotenv()

# Reduce GPU memory fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B" 
ADAPTER_ID = os.getenv("ADAPTER_PATH")
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = None
model = None

def load_reasoning_model():
   
    global tokenizer, model
    
    if tokenizer is None or model is None:
        print(f"Loading Base Model: {BASE_MODEL_ID}...")
        
        # Enhanced GPU detection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Detected Hardware: {device.upper()}")
        
        if device == "cuda":
            print(f"GPU Found: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"PyTorch: {torch.__version__}")
        else:
            print(f"PyTorch Version: {torch.__version__}")
            print(f"CUDA Built: {torch.version.cuda if torch.version.cuda else 'No'}")

        try:
           
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_ID, 
                token=HF_TOKEN
            )

            
            if tokenizer.pad_token is None:
              
                tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
            
            
            if tokenizer.chat_template is None:
                tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% if system_message != '' %}<|start_header_id|>system<|end_header_id|>\n\n{{ system_message }}<|eot_id|>{% endif %}{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"

            if device == "cuda":
                print("Activating 4-bit Quantization (bitsandbytes) for GPU")
                from transformers import BitsAndBytesConfig
                
                # Calculate optimal GPU memory allocation
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"  Available VRAM: {gpu_memory:.2f} GB")
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_ID,
                    token=HF_TOKEN,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("Model loaded on GPU with 4-bit quantization")
            else:
                print("No GPU detected. Loading in Standard CPU Mode.")
                print("Note: This will be slow and consume significant RAM.")
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_ID,
                    token=HF_TOKEN,
                    device_map="cpu",
                    torch_dtype=torch.float32 
                )

            print(f"Attaching Adapter: {ADAPTER_ID}...")
            model = PeftModel.from_pretrained(
                base_model,
                ADAPTER_ID,
                token=HF_TOKEN
            )
            
            print("FactLegalLlama Adapter Loaded Successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
            
    return tokenizer, model

# Devanagari numerals map for section extraction
_DEVA_DIGIT = str.maketrans('०१२३४५६७८९', '0123456789')

def extract_sections_from_hindi(hindi_text: str) -> list:
    """
    Extract IPC/BNS section numbers from the raw Hindi (Devanagari) OCR text
    BEFORE translation.  Numerals in FIRs are typically Arabic even in Hindi
    documents, but Devanagari digits are also handled via transliteration.
    Returns a sorted list of section-number strings, e.g. ['302', '34', '376'].
    """
    hindi_normalized = hindi_text.translate(_DEVA_DIGIT)

    patterns = [
        # धारा 302, धारा 376(क)
        r'धारा\s*([\d]+(?:\([a-zA-Z0-9]+\))?)',
        # u/s 302, U/S 376
        r'[Uu]/[Ss]\s*([\d]{2,3}(?:\([a-zA-Z0-9]+\))?)',
        # Section 302 IPC / BNS
        r'(?:Section|Sec\.?)\s+([\d]{2,3}(?:\([a-zA-Z0-9]+\))?)',
        # bare numbers followed by IPC/BNS
        r'([\d]{2,3})\s+(?:IPC|BNS|आईपीसी)',
    ]
    found = set()
    for pat in patterns:
        for m in re.finditer(pat, hindi_normalized, re.IGNORECASE):
            sec = re.sub(r'\s+', '', m.group(1).strip())
            if sec and (sec.isdigit() or re.match(r'^\d+\([a-zA-Z0-9]+\)$', sec)):
                found.add(sec)
    result = sorted(found, key=lambda x: int(re.match(r'\d+', x).group()))
    if result:
        print(f"Pre-translation Hindi section extraction: {result}")
    return result


# ---------------------------------------------------------------------------
# Hindi (Devanagari) name extractor — runs on the raw OCR text before
# translation so we capture names from the structured FIR fields that
# often get garbled during machine translation.
# ---------------------------------------------------------------------------

def extract_party_names_hindi(hindi_text: str) -> tuple:
    """
    Extract complainant and accused names from raw Hindi/Devanagari FIR text.
    Hindi FIRs have structured fields like:
      शिकायतकर्ता ... नाम: सुरेश शर्मा
      अभियुक्त ... नाम: राजेश कुमार
    Returns (complainant, accused) or (None, None) if not found.
    """
    if not hindi_text:
        return None, None

    complainant = None
    accused = None

    # Romanised name: consecutive TitleCase or ALL-CAPS words (2-4 words)
    _ROMAN_NAME = r'[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}'
    # Devanagari name: 2-4 consecutive Devanagari words, but NOT common
    # structural words (पिता=father, आयु=age, पता=address, उम्र=age, etc.)
    _DEVA_STOP = r'(?:पिता|आयु|उम्र|पता|निवासी|थाना|जिला|व्यवसाय|पेशा|दिनांक|वर्ष|गिरफ्तारी|Father|Age|Address)'
    _DEVA_NAME = rf'[\u0900-\u097F]+(?:\s+(?!{_DEVA_STOP})[\u0900-\u097F]+){{0,3}}'
    # Either form
    _HINDI_NAME = rf'(?:{_ROMAN_NAME}|{_DEVA_NAME})'

    comp_patterns = [
        # "शिकायतकर्ता ... नामः [Name]" or "वादी ... नामः [Name]"  (lazy gap!)
        rf'(?:शिकायतकर्ता|वादी|सूचनादाता|प्रार्थी|Complainant)[^\n]{{0,80}}?नाम[ः:]\s*({_HINDI_NAME})',
        # "शिकायतकर्ता ... Name : [Name]" (mixed Hindi+English field)
        rf'(?:शिकायतकर्ता|वादी|Complainant)[^\n]{{0,80}}?(?:Name|नाम)\s*[:\-]\s*({_HINDI_NAME})',
        # "वादी श्री [Name]" / "शिकायतकर्ता श्री [Name]" in narrative
        rf'(?:शिकायतकर्ता|वादी|सूचनादाता)\s+(?:श्री|श्रीमती|कु\.)\s*({_HINDI_NAME})',
        # Romanised: "Complainant Name : [Name]" in bilingual docs
        rf'(?:Complainant|Informant)\s*(?:Name)?\s*[:\-]\s*({_ROMAN_NAME})',
    ]
    for pat in comp_patterns:
        m = re.search(pat, hindi_text, re.IGNORECASE | re.MULTILINE)
        if m:
            cand = m.group(1).strip()
            if len(cand) >= 3:
                complainant = cand
                break

    acc_patterns = [
        # "अभियुक्त/आरोपी ... नामः [Name]"  (lazy gap!)
        rf'(?:अभियुक्त|आरोपी|Accused)[^\n]{{0,80}}?नाम[ः:]\s*({_HINDI_NAME})',
        rf'(?:अभियुक्त|आरोपी|Accused)[^\n]{{0,80}}?(?:Name|नाम)\s*[:\-]\s*({_HINDI_NAME})',
        # "अभियुक्त [Name] ने" / "आरोपी [Name] उर्फ"
        rf'(?:अभियुक्त|आरोपी)\s+({_HINDI_NAME})\s+(?:ने|उर्फ|को|द्वारा|पर)',
        rf'(?:Accused)\s*(?:Name)?\s*[:\-]\s*({_ROMAN_NAME})',
    ]
    for pat in acc_patterns:
        m = re.search(pat, hindi_text, re.IGNORECASE | re.MULTILINE)
        if m:
            raw = m.group(m.lastindex).strip()
            # Remove alias markers: "राजेश कुमार उर्फ रज्जू" → "राजेश कुमार"
            raw = re.split(r'\s+उर्फ\s+', raw)[0].strip()
            if len(raw) >= 3:
                accused = raw
                break

    print(f"Hindi name extraction → complainant: '{complainant}' | accused: '{accused}'")
    return complainant, accused



_HONORIFICS = r'(?:Shri|Smt\.?|Sh\.?|Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Sri|Ku\.?)\s*'
# Match TitleCase OR ALL-CAPS names (both appear in translated FIR docs)
_NAME_WORD_TC  = r'[A-Z][a-zA-Z]{1,}'
_NAME_WORD_CAP = r'[A-Z]{2,}'
_FULL_NAME_TC  = rf'{_NAME_WORD_TC}(?:\s+{_NAME_WORD_TC}){{1,3}}'
_FULL_NAME_CAP = rf'{_NAME_WORD_CAP}(?:\s+{_NAME_WORD_CAP}){{1,3}}'
# Either form
_ANY_NAME = rf'(?:{_FULL_NAME_TC}|{_FULL_NAME_CAP})'

_FALSE_POS = {
    'the accused', 'the complainant', 'police station', 'the court',
    'court', 'station', 'district', 'state', 'india', 'government',
    'section', 'ipc', 'bns', 'crpc', 'fir'
}

def _clean_name(raw: str) -> str:
    """Strip parentheticals, trailing punctuation, and title/format noise."""
    raw = re.sub(r'\s*\([^)]*\)', '', raw)   # remove (S/o Ram)
    raw = re.sub(r'\s*[,;].*$', '', raw)      # stop at comma
    raw = raw.strip(' .:-')
    return raw

def _is_valid_name(name: str, exclude: str = '') -> bool:
    lname = name.lower().strip()
    if lname in _FALSE_POS:
        return False
    if len(name) < 3 or len(name) > 60:
        return False
    if exclude and lname == exclude.lower():
        return False
    return True

def extract_party_names(text: str):
    """
    Comprehensively extract complainant and accused names from a translated FIR.
    Handles TitleCase, ALL-CAPS, S/o anchors, and all common Indian FIR field labels.
    Returns ('the complainant', 'the accused') as safe fallbacks.
    """
    complainant = "the complainant"
    accused     = "the accused"
    # Search the whole document, not just first 3000 chars
    search_text = text[:6000]

   
    comp_patterns = [
        rf'(?:complainant[\s\'s]*name|name\s+of\s+(?:the\s+)?complainant|informant[\s\'s]*name'
        rf'|name\s+of\s+(?:the\s+)?informant|applicant[\s\'s]*name|plaintiff[\s\'s]*name)\s*[:\-]?\s*({_ANY_NAME})',
       
        rf'(?:complainant|informant|petitioner|applicant|plaintiff)[^\n]{{0,50}}(?:Details\s+)?Name\s*[:\-]\s*(?:{_HONORIFICS})?({_ANY_NAME})',
        
        rf'({_ANY_NAME})\s+(?:S/o|s/o|Son\s+of|D/o|d/o|Daughter\s+of|W/o|w/o|Wife\s+of)'
        rf'[^\n]{{0,5}}(?:complainant|informant|petitioner|plaintiff)',
       
        rf'(?:the\s+)?(?:plaintiff|complainant|informant|petitioner|applicant),?\s+(?:{_HONORIFICS})({_ANY_NAME})',
        
        rf'(?:filed|lodged|submitted|reported)\s+by\s+(?:{_HONORIFICS})?({_ANY_NAME})',
       
        rf'(?:complainant|informant|petitioner|applicant|plaintiff)\s*[:\-]\s*(?:{_HONORIFICS})?({_ANY_NAME})',
        
        rf'(?:complainant|plaintiff)[^\n]{{0,80}}{_HONORIFICS}({_ANY_NAME})',
       
        rf'I,?\s+({_ANY_NAME}),?\s+(?:resident|r/o|R/O|aged|age|s/o|d/o)',
        
        rf'(?:the\s+)?(?:plaintiff|complainant)\s+({_ANY_NAME})\s+(?:was|filed|stated|reported|submitted|has|had)',
        
        rf'({_ANY_NAME})\s+(?:Complainant|Informant|Petitioner|Plaintiff)\s+(?:Victim|witness)',
    ]
    for pat in comp_patterns:
        m = re.search(pat, search_text, re.IGNORECASE | re.MULTILINE)
        if m and m.lastindex:
            cand = _clean_name(m.group(m.lastindex))
            if _is_valid_name(cand):
                complainant = cand
                break

    
    acc_patterns = [
       
        rf'(?:accused[\s\'s]*name|name\s+of\s+(?:the\s+)?accused|suspect[\s\'s]*name)\s*[:\-]?\s*({_ANY_NAME})',
        
        rf'(?:accused|suspect)[^\n]{{0,50}}(?:Details\s+)?Name\s*[:\-]\s*(?:{_HONORIFICS})?({_ANY_NAME})',
       
        rf'({_ANY_NAME})\s+(?:S/o|s/o|D/o|d/o|W/o|w/o)[^\n]{{0,5}}(?:accused|suspect)',
        
        rf'(?:FIR|complaint|case)\s+(?:filed|lodged|registered)?\s*against\s+(?:one\s+)?(?:{_HONORIFICS})?({_ANY_NAME})',
        rf'against\s+(?:accused\s+)?(?:{_HONORIFICS})?({_ANY_NAME})(?:\s+(?:S/o|D/o|W/o|Age|r/o|,))',
        
        rf'(?:accused|suspect|respondent|opposite\s+party)\s*[:\-]\s*(?:{_HONORIFICS})?({_ANY_NAME})',
       
        rf'(?:the\s+)?accused\s+({_ANY_NAME})\s+(?:was|assaulted|robbed|attacked|threatened|beat|stole|committed|arrested|has|had|who|is)',
        
        rf'accused[^\n]{{0,80}}{_HONORIFICS}({_ANY_NAME})',
       
        rf'against\s+(?:one\s+)?(?:{_HONORIFICS})?({_ANY_NAME})',
    ]
    for pat in acc_patterns:
        m = re.search(pat, search_text, re.IGNORECASE | re.MULTILINE)
        if m and m.lastindex:
            cand = _clean_name(m.group(m.lastindex))
            if any(w in cand.lower() for w in ['unknown', 'unidentified', 'not known']):
                accused = "Unidentified accused"
                break
            if _is_valid_name(cand, exclude=complainant):
                accused = cand
                break

    print(f"Name extraction → complainant: '{complainant}' | accused: '{accused}'")
    return complainant, accused


def extract_case_metadata(text: str) -> dict:
    """
    Extract date of incident, police station, and place of offence
    directly from the translated FIR text.
    """
    meta = {"date": "", "police_station": "", "place": ""}

    # Date of incident — dd/mm/yyyy, dd-mm-yyyy, or written date
    date_patterns = [
        r'(?:date\s+of\s+(?:incident|offence|occurrence|event)|on\s+date|incident\s+on)\s*[:\-]?\s*'
        r'(\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4})',
        r'on\s+(\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4})',
        r'dated?\s+(\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4})',
        r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
        r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
        r'Dec(?:ember)?)\s+\d{2,4})',
    ]
    for pat in date_patterns:
        m = re.search(pat, text[:4000], re.IGNORECASE)
        if m:
            meta["date"] = m.group(1).strip()
            break

    # Police station
    ps_m = re.search(
        r'(?:police\s+station|P\.?S\.?|thana|थाना)\s*[:\-]?\s*([A-Za-z][A-Za-z\s]{2,30}?)'
        r'(?:\s+(?:district|city|town|tehsil|block|police|,|\n)|$)',
        text[:4000], re.IGNORECASE
    )
    if ps_m:
        meta["police_station"] = ps_m.group(1).strip()

    # Place of occurrence
    place_m = re.search(
        r'(?:place\s+of\s+(?:occurrence|incident|offence)|location|at\s+(?:village|mohalla|'
        r'area|locality|road|street|near))\s*[:\-]?\s*([A-Za-z][A-Za-z\s,]{3,50})',
        text[:4000], re.IGNORECASE
    )
    if place_m:
        meta["place"] = place_m.group(1).strip()[:60]

    return meta


def _apply_ner_mask(text: str, complainant: str, accused: str) -> tuple:
    """
    Replace real party names with neutral tokens before the text is sent to
    the LLM.  Returns (masked_text, restore_fn) where restore_fn(text) swaps
    the tokens back to the real names in any generated output.

    Why: LLMs with a fine-tune on formal judgments sometimes re-spell names
    from the training examples (e.g. "Ramesh Yadav" leaks in).  Masking
    guarantees the names in the output are always exactly the ones extracted
    from the actual FIR, regardless of what the model generates.
    """
    COMP_TOKEN = "[COMPLAINANT_NAME]"
    ACC_TOKEN  = "[ACCUSED_NAME]"

    masked = text
   
    if complainant not in ("the complainant", ""):
       
        masked = re.sub(re.escape(complainant), COMP_TOKEN, masked, flags=re.IGNORECASE)
    if accused not in ("the accused", "Unidentified accused", ""):
        masked = re.sub(re.escape(accused), ACC_TOKEN, masked, flags=re.IGNORECASE)

    def restore(generated: str) -> str:
        out = generated
        out = out.replace(COMP_TOKEN, complainant)
        out = out.replace(ACC_TOKEN, accused)
        
        out = out.replace(COMP_TOKEN.lower(), complainant)
        out = out.replace(ACC_TOKEN.lower(), accused)
        return out

    return masked, restore


def clean_ocr_text(text: str) -> str:
    """Fix common OCR artifacts before sending context to the LLM."""
   
    text = re.sub(r'-{2,}\s*Page\s*\d+\s*-{2,}', ' ', text, flags=re.IGNORECASE)
    
    text = re.sub(r'(?<=[A-Z]{2}) (?=[A-Z])', '', text)
    
    text = re.sub(r'\b(\d) (\d{2})\b', r'\1\2', text)
    text = re.sub(r'\b(\d) (\d) (\d)\b', r'\1\2\3', text)
   
    text = re.sub(r'\.{3,}', '.', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

async def analyze_legal_case(case_id: str, raw_english_text: str = None, hint_sections: list = None, raw_hindi_text: str = None):
   
    print(f"\n{'='*60}")
    print(f"Starting Legal Analysis for Case: {case_id}")
    print(f"{'='*60}\n")
    
   
    if torch.cuda.is_available():
       
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
        torch.cuda.empty_cache()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"GPU Memory: {free_mem / 1024**3:.2f} GB free / {total_mem / 1024**3:.2f} GB total\n")
        

        if free_mem / 1024**3 < 2.0:
            print(f" WARNING: Low GPU memory ({free_mem / 1024**3:.2f} GB free)")
            print(f"Attempting aggressive cleanup...")
            import gc
            gc.collect()  # Release Python refs first
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"After cleanup: {free_mem / 1024**3:.2f} GB free\n")
    
    start_time = time.time()
    
    print("[1/4] Loading AI model...")
    model_start = time.time()
    tokenizer_instance, model_instance = load_reasoning_model()
    print(f"Model ready ({time.time() - model_start:.2f}s)\n")
    
    # 1. Retrieve Context from RAG - Get MORE chunks for comprehensive analysis
    print("[2/4] Retrieving case context...")
    rag_start = time.time()

    if raw_english_text and len(raw_english_text.strip()) > 50:
       
        print(f"Using full translated text directly ({len(raw_english_text)} chars)")
        full_context = raw_english_text
        context_chunks = [raw_english_text]
    else:
        
        queries = [
            "complainant name accused name FIR details IPC sections charges offense",
            "incident description evidence witnesses crime details section act",
        ]
        
        all_chunks = []
        for query in queries:
            chunks = await rag_service.get_relevant_context(
                query=query,
                filter={"case_id": case_id},
                top_k=6
            )
            all_chunks.extend(chunks)
        
        
        seen = set()
        context_chunks = []
        for chunk in all_chunks:
            if chunk not in seen:
                seen.add(chunk)
                context_chunks.append(chunk)
        
        if not context_chunks:
            return {"case_id": case_id, "error": "No relevant context found in case file."}
        full_context = "\n\n".join(context_chunks)

    print(f"Case context ready ({time.time() - rag_start:.2f}s)\n")
   
    full_context_cleaned = clean_ocr_text(full_context)
    
    formatted_context = full_context_cleaned[:7000]
    if len(full_context_cleaned) > 7000:
        print(f"Context truncated from {len(full_context_cleaned)} to 7000 chars to fit GPU memory")
    
    print("[2.1/4] Pre-extracting IPC/BNS sections from case document...")

   
    _PROCEDURAL_SECTIONS = {
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
        # CrPC procedural sections commonly referenced in FIRs
        '154', '155', '156', '157', '158', '159', '160', '161', '162',
        '163', '164', '165', '166', '167', '169', '170', '171', '172',
        '173',   # chargesheet filing section
        '190', '193', '194', '195', '196', '197', '200',
        # BNSS equivalents
        '175', '176', '177', '178', '179', '180',
    }

    section_patterns = [
        # Only match sections explicitly tagged with IPC/BNS
        r'(?:Section|Sec\.?|धारा)\s+([\d]+(?:\s*[/-]\s*[\d]+)*(?:\s*\([a-zA-Z0-9]+\))?)\s+(?:IPC|BNS|of\s+IPC|of\s+BNS)',
        r'(?:IPC|BNS)\s+(?:Section|Sec\.?|धारा)\s+([\d]+(?:\s*[/-]\s*[\d]+)*(?:\s*\([a-zA-Z0-9]+\))?)',
        r'(?:under|u/s|U/S)\s+(?:Section|Sec\.?)?\s*([\d]+(?:\s*[/-]\s*[\d]+)*(?:\s*\([a-zA-Z0-9]+\))?)\s+(?:IPC|BNS)',
        r'(\d{2,3}(?:\s*\([a-zA-Z0-9]+\))?)\s+(?:IPC|BNS)',
        r'(?:Sections?|Sec\.?)\s+([\d]+(?:\s*,\s*[\d]+)*(?:\s*,?\s*and\s+[\d]+)?)\s+(?:of\s+)?(?:IPC|BNS)',
    ]
    
    eng_extracted_sections = set()
    for pattern in section_patterns:
        matches = re.findall(pattern, formatted_context, re.IGNORECASE)
        for match in matches:
            section_parts = re.split(r'[,\s]+(?:and\s+)?', match)
            for part in section_parts:
                section_num = re.sub(r'\s+', '', part.strip())
                if section_num and (section_num.isdigit() or re.match(r'^\d+\([a-zA-Z0-9]+\)$', section_num)):
                    eng_extracted_sections.add(section_num)
    
   
    if hint_sections and len(hint_sections) >= 2:
        pre_extracted_sections = set(hint_sections)
        
        for sec in eng_extracted_sections:
            base = re.match(r'\d+', sec)
            if base and int(base.group()) >= 30 and sec not in _PROCEDURAL_SECTIONS:
                pre_extracted_sections.add(sec)
        print(f"Using Hindi sections as primary: {hint_sections}")
        print(f"Added {len(pre_extracted_sections) - len(hint_sections)} valid English sections")
    else:
        pre_extracted_sections = eng_extracted_sections
        if hint_sections:
            pre_extracted_sections.update(hint_sections)

    
    pre_extracted_sections -= _PROCEDURAL_SECTIONS

    pre_extracted_list = sorted(list(pre_extracted_sections), key=lambda x: int(re.match(r'\d+', x).group()) if re.match(r'\d+', x) else 0)
    print(f"Pre-extracted {len(pre_extracted_list)} IPC/BNS sections (filtered): {pre_extracted_list}")
    
  
    print(f"\nCase Context summary:")
    print(f"  - Number of chunks: {len(context_chunks)}")
    print(f"  - Total context length: {len(formatted_context)} characters")
    print(f"  - Pre-extracted sections: {', '.join(pre_extracted_list[:10])}{'...' if len(pre_extracted_list) > 10 else ''}")
    print(f"  - First 300 chars of context:")
    print(f"    {formatted_context[:300]}...")
    
    
    debug_dir = "debug_analysis"
    os.makedirs(debug_dir, exist_ok=True)
    debug_file = os.path.join(debug_dir, f"case_{case_id[:8]}_context.txt")
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"CASE CONTEXT FOR: {case_id}\n")
        f.write("="*80 + "\n\n")
        f.write("FULL CONTEXT BEING SENT TO LLM:\n")
        f.write("-"*80 + "\n")
        f.write(formatted_context)
        f.write("\n\n" + "="*80 + "\n")
        f.write(f"PRE-EXTRACTED SECTIONS: {pre_extracted_list}\n")
        f.write("="*80 + "\n")
    print(f"Saved context debug file: {debug_file}")
    print()
    
    # 2. Retrieve IPC/BNS Statute Context
    print("[2.5/4] Retrieving relevant IPC/BNS statutes from legal database")
    statute_start = time.time()
    
    # Create a query from the case context and pre-extracted sections to find relevant statutes
    sections_query_part = ' '.join([f"Section {s}" for s in pre_extracted_list[:15]])  # First 15 sections
    statute_query = f"{sections_query_part} IPC BNS sections theft robbery assault murder rape fraud cheating criminal breach kidnapping extortion wrongful restraint relevant to: {formatted_context[:500]}"
    
    try:
        statute_chunks = await rag_service.get_relevant_context(
            query=statute_query,
            collection_name="ipc_bns_statutes",
            filter={},  
            top_k=10 
        )
        
        if statute_chunks:
            raw_statutes = "\n\n".join(statute_chunks)

            formatted_statutes = raw_statutes[:2500]
            if len(raw_statutes) > 2500:
                print(f"Statutes truncated from {len(raw_statutes)} to 2500 chars")
            print(f"Retrieved {len(statute_chunks)} relevant statute sections ({time.time() - statute_start:.2f}s)")
            print(f"  - First 200 chars of statutes:")
            print(f"    {formatted_statutes[:200]}...")
        else:
            formatted_statutes = "Statute database not available. Analysis will be based on case document only."
            print(f" No statutes retrieved - database may not be ingested yet")
    except Exception as e:
        formatted_statutes = "Statute database not available. Analysis will be based on case document only."
        print(f"Error retrieving statutes: {str(e)}")
    
    print()

    # -----------------------------------------------------------------------
    # TRAINING-ALIGNED PROMPT  (FactLegalLlama fine-tuned on Supreme Court
    # judgment dataset with columns: Facts, Issue, Arguments of Petitioner,
    # Arguments of Respondent, Reasoning, Decision, Label)
    #
    # The model learned to generate "Reasoning:" and "Decision:" text when
    # given the other columns as context.  We map the FIR fields to that
    # column layout and let the model continue naturally.  We then parse the
    # generated Reasoning + Decision ourselves to build the structured JSON.
    # -----------------------------------------------------------------------

    sections_str = ', '.join(pre_extracted_list) if pre_extracted_list else "to be identified from document"

    
    # 1. Try Hindi text first (most reliable — structured FIR fields)
    _hindi_comp, _hindi_acc = (None, None)
    if raw_hindi_text:
        _hindi_comp, _hindi_acc = extract_party_names_hindi(raw_hindi_text)
    # 2. Also try English patterns
    _eng_comp, _eng_acc = extract_party_names(formatted_context)
    # 3. Merge: prefer Hindi extraction, fall back to English
    _comp_hint = _hindi_comp if _hindi_comp else _eng_comp
    _acc_hint  = _hindi_acc if _hindi_acc else _eng_acc
    print(f"Final merged names → complainant: '{_comp_hint}' | accused: '{_acc_hint}'")
    _case_meta = extract_case_metadata(formatted_context)

   
    EXAMPLE_BLOCK = (
        # ── Example 1: Theft ──────────────────────────────────────────────────────
        "Case Name: MEENA DEVI Vs. RAMESH YADAV\n\n"
        "Facts: Complainant Meena Devi filed an FIR at Civil Lines Police Station against "
        "accused Ramesh Yadav alleging theft of gold ornaments valued at Rs. 50,000 on "
        "12.03.2025. The accused was seen near the complainant's house on the day of the "
        "incident. Case registered under Sections 379 and 34 IPC.\n\n"
        "Issue: Whether accused Ramesh Yadav is criminally liable for the alleged theft, "
        "and what additional evidence is required for successful prosecution.\n\n"
        "Arguments of Petitioner: Meena Devi submits that gold ornaments were stolen from her "
        "locked almirah. A neighbour witnessed Ramesh Yadav leaving the premises with the "
        "ornaments. The accused has no legitimate reason to be on the complainant's property. "
        "The FIR was lodged on the same day as the incident.\n\n"
        "Arguments of Respondent: Investigation is at an early stage. The accused's exact role "
        "and the recovery of stolen articles are yet to be established. No stolen property has "
        "been recovered from the accused so far.\n\n"
        "Summary: Meena Devi filed an FIR against Ramesh Yadav at Civil Lines Police Station "
        "on 12.03.2025 alleging that the accused stole gold ornaments valued at Rs. 50,000 "
        "from her locked almirah. A neighbour witnessed the accused leaving the premises with "
        "the ornaments. The accused was seen near the complainant's house on the day of the "
        "incident. The FIR was lodged on the same day.\n\n"
        "Reasoning: The FIR discloses cognizable offences under Section 379 IPC (theft) and "
        "Section 34 IPC (common intention). A prima facie case is made out through the eyewitness "
        "account. However, to sustain a chargesheet the prosecution must: (1) recover the stolen "
        "ornaments and prepare a panchnama; (2) record a formal statement from "
        "the eyewitness; (3) collect fingerprints from the almirah for FSL examination; "
        "(4) conduct a Test Identification Parade before the Magistrate; "
        "(5) obtain documentary proof of ownership of the ornaments.\n\n"
        "Decision: The Investigating Officer should complete the above steps and file a "
        "chargesheet. The accused should be produced before the Magistrate only after TIP is "
        "conducted. Bail application, if any, should be opposed until the stolen property is "
        "recovered.\n\n"
        # ── Example 2: Grievous hurt / assault ───────────────────────────────────
        "Case Name: SURESH KUMAR Vs. RAJU SINGH & ORS.\n\n"
        "Facts: An FIR was registered against accused Raju Singh and two others on the complaint "
        "of Suresh Kumar at Kotwali Police Station. On 05.01.2025 the accused persons attacked "
        "the complainant with iron rods and lathis near the village market at approximately "
        "8:00 PM, causing fractures to his left arm and head injuries. The incident arose out "
        "of a land dispute.\n\n"
        "Issue: Whether accused Raju Singh and others are criminally liable for causing "
        "grievous hurt and criminal intimidation, and what evidence is required to establish "
        "the offence beyond reasonable doubt.\n\n"
        "Arguments of Petitioner: Suresh Kumar submits that all three accused acting in concert "
        "attacked him with weapons. He sustained fractures confirmed by the government hospital "
        "MLC. Two independent witnesses were present at the scene. The accused also issued "
        "threats to kill if the land dispute was pursued.\n\n"
        "Arguments of Respondent: The accused deny involvement and claim the complainant "
        "sustained injuries in an accident. The land dispute provided a motive to falsely "
        "implicate the accused. Investigation is in early stages.\n\n"
        "Summary: Suresh Kumar filed an FIR against Raju Singh and two others at Kotwali Police "
        "Station on 05.01.2025 alleging that the accused attacked him with iron rods and lathis "
        "near the village market at around 8:00 PM over a land dispute. The complainant suffered "
        "fractures to his left arm and head injuries. The MLC from the government hospital "
        "confirms the injuries. Two independent eyewitnesses were present at the scene.\n\n"
        "Reasoning: The FIR and MLC report prima facie disclose cognizable offences under "
        "Section 325 IPC (grievous hurt), Section 323 IPC (voluntarily causing hurt), "
        "Section 34 IPC (common intention) and Section 506 IPC (criminal intimidation). "
        "The Medico-Legal Certificate confirming fractures is crucial corroborative evidence. "
        "To complete the investigation the prosecution must: (1) obtain the final MLC and "
        "opinion of the doctor on the nature of weapon used; (2) record statements of both "
        "independent witnesses; (3) seize and send the weapons "
        "(iron rods/lathis) for FSL examination; (4) prepare a site map of the assault location; "
        "(5) collect CCTV footage from nearby shops if available; "
        "(6) establish the land dispute background through revenue records.\n\n"
        "Decision: A chargesheet should be filed after completing the above steps. The doctor's "
        "opinion on the grievous nature of injuries is essential to sustain the charge. All "
        "three accused should be arrested and questioned separately to identify individual "
        "roles and establish common intention.\n\n"
    )

    # -------------------------------------------------------------------
    # Build a richer Facts paragraph matching training data style:
    # SC dataset Facts = concise narrative, not raw OCR dump.
    # We extract a clean 800-char incident narrative from context.
    # -------------------------------------------------------------------
    def _build_facts_narrative(ctx: str, comp: str, acc: str, secs: str) -> str:
        """
        Condense raw OCR/translated context into a clean Facts paragraph
        that resembles the training-data column format.

        Priority order:
        1. Explicitly labelled sections (Brief Facts, Statement, Complaint, Narrative)
        2. Paragraph containing incident-describing action verbs (attacked, stole, etc.)
        3. Longest paragraph that is NOT the document header
        4. Full context truncated to 800 chars
        """
       
        label_m = re.search(
            r'(?:Brief\s+Facts?|Statement\s+of\s+(?:the\s+)?(?:Complainant|Informant)|'
            r'Complaint\s+Text|Narrative|Facts\s+of\s+the\s+Case|Gist\s+of\s+FIR|'
            r'(?:Complainant|Informant)[\'s\s]+Statement)\s*[:\-]?\s*([\s\S]{80,800}?)'
            r'(?=\n\s*(?:[A-Z][A-Za-z\s]{3,30}\s*:|$))',
            ctx, re.IGNORECASE
        )
        if label_m:
            best = label_m.group(1).strip()[:1400]
            opener = f"An FIR was registered against {acc} on the complaint of {comp}. "
            return (opener + best)[:1500]

        
        incident_kw = re.compile(
            r'\b(attacked|assaulted|beat|beaten|stabbed|shot|killed|murdered|stole|stolen|'
            r'robbed|threatened|abused|raped|kidnapped|cheated|defrauded|snatched|looted|'
            r'hit|struck|injured|fired|bomb|burnt|burned|damaged|broke|trespassed)\b',
            re.IGNORECASE
        )
        paragraphs = [p.strip() for p in re.split(r'\n{1,}', ctx) if len(p.strip()) > 80]
       
        header_kw = re.compile(
            r'^(?:Court\s*:|Police\s*Station\s*:|District\s*:|Charge\s*[Ss]heet|'
            r'FIR\s*No|Case\s*No|Date\s*of|Registration|Under\s+Section)',
            re.IGNORECASE
        )
        body_paras = [p for p in paragraphs if not header_kw.match(p)]
        incident_scored = [(p, len(incident_kw.findall(p))) for p in body_paras]
        incident_scored.sort(key=lambda x: x[1], reverse=True)

        if incident_scored and incident_scored[0][1] > 0:
            best = incident_scored[0][0][:1400]
        elif body_paras:
            best = sorted(body_paras, key=len, reverse=True)[0][:1400]
        else:
            best = ctx[:1400]

        opener = f"An FIR was registered against {acc} on the complaint of {comp}. "
        return (opener + best)[:1500]

    facts_narrative = _build_facts_narrative(formatted_context, _comp_hint, _acc_hint, sections_str)

    # Apply NER masking — replace real names with neutral tokens in the prompt.
    
    masked_facts, name_restore = _apply_ner_mask(facts_narrative, _comp_hint, _acc_hint)

   
    masked_comp = "[COMPLAINANT_NAME]"
    masked_acc  = "[ACCUSED_NAME]"

    
    def _build_petitioner_args(ctx: str, comp: str, acc: str, secs: str) -> str:
        """Extract specific allegations from context for the petitioner arguments slot."""
        lines = [l.strip() for l in ctx.split('\n') if len(l.strip()) > 30]
        allegation_lines = [l for l in lines if re.search(
            r'\b(allege|complaint|incident|on\s+\d|date|time|place|location|'
            r'victim|witness|amount|Rs\.|rupee|injury|wound|stolen|forcibly)\b',
            l, re.IGNORECASE
        )]
        specific = '. '.join(allegation_lines[:4]) if allegation_lines else ctx[:400]
        return (
            f"{comp} submits that {acc} committed the offences as detailed in the FIR. "
            f"The specific allegations are: {specific[:500]}. "
            f"The offence is registered under Sections {secs} IPC/BNS."
        )

    petitioner_args = _build_petitioner_args(formatted_context, _comp_hint, _acc_hint, sections_str)
    # Mask names in petitioner args too
    masked_petitioner_args, _ = _apply_ner_mask(petitioner_args, _comp_hint, _acc_hint)

    
    case_name_line = "[COMPLAINANT_NAME] Vs. [ACCUSED_NAME]"

    raw_prompt = (
        f"{EXAMPLE_BLOCK}"
        f"Case Name: {case_name_line}\n\n"
        f"Text: FIR analysis. Sections invoked: {sections_str}. "
        f"Relevant statutes: {formatted_statutes[:800]}\n\n"
        f"Facts: {masked_facts}\n\n"
        f"Issue: Whether [ACCUSED_NAME] is criminally liable under IPC/BNS Sections "
        f"{sections_str} as alleged in the FIR filed by [COMPLAINANT_NAME], and what "
        f"evidence is required to establish guilt beyond reasonable doubt.\n\n"
        f"Arguments of Petitioner: {masked_petitioner_args}\n\n"
        f"Arguments of Respondent: The accused's exact role and motive are yet to be "
        f"fully established during investigation. The accused denies the allegations "
        f"and the investigation is ongoing.\n\n"
        f"Summary:"
    )

    print("[3/4] Preparing input for AI model (training-data-aligned format)...")
    tokenize_start = time.time()

    encoded = tokenizer_instance(
        raw_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=5500,
        padding=False
    )
    input_ids = encoded.input_ids.to(model_instance.device)
    attention_mask = encoded.attention_mask.to(model_instance.device)

    print(f"Input tokenized: {input_ids.shape[1]} tokens ({time.time() - tokenize_start:.2f}s)\n")

    # 4. Run Inference in Threadpool (to avoid blocking FastAPI)
    print("[4/4] Running AI inference (this may take 30-120 seconds)...")
    print("Generating legal analysis...")
    inference_start = time.time()
    
    _eot_id = tokenizer_instance.convert_tokens_to_ids("<|eot_id|>")
    terminators = [tokenizer_instance.eos_token_id]
    if _eot_id is not None and _eot_id != tokenizer_instance.unk_token_id:
        terminators.append(_eot_id)
    
    def run_inference():
        with torch.no_grad(): 
            return model_instance.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=800,
                do_sample=False,
                pad_token_id=tokenizer_instance.pad_token_id,
                eos_token_id=terminators,
                repetition_penalty=1.05,    
                use_cache=True
            )

    try:
        outputs = await run_in_threadpool(run_inference)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nGPU OOM Error during generation.")
            print(f"Freeing tensors and retrying with minimal context.\n")
            
            import gc
            saved_ids_cpu = input_ids[:, -3500:].cpu()
            saved_mask_cpu = attention_mask[:, -3500:].cpu()
            del input_ids
            del attention_mask
            
           
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                print(f"After tensor delete + cleanup: {free_mem:.2f} GB free")
            
           
            input_ids_retry = saved_ids_cpu.to(model_instance.device)
            attention_mask_retry = saved_mask_cpu.to(model_instance.device)
            del saved_ids_cpu, saved_mask_cpu
            
            print(f"Retrying with {input_ids_retry.shape[1]} tokens, max_new_tokens=300...")
            outputs = await run_in_threadpool(lambda: model_instance.generate(
                input_ids_retry,
                attention_mask=attention_mask_retry,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=tokenizer_instance.pad_token_id,
                eos_token_id=terminators,
                repetition_penalty=1.05,
                use_cache=False
            ))
           
            input_ids = input_ids_retry
            attention_mask = attention_mask_retry
        else:
            raise e
    
    print(f"AI inference completed ({time.time() - inference_start:.2f}s)\n")

    response_text = tokenizer_instance.decode(
        outputs[0][input_ids.shape[-1]:], 
        skip_special_tokens=True
    )
    
   
    generated_tokens = len(outputs[0]) - input_ids.shape[-1]
    del outputs
    del input_ids
    del attention_mask
    
    if torch.cuda.is_available():
        import gc
        gc.collect() 
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"GPU cache cleared. Available memory: {free_mem:.2f} GB")
    
    print(f"Generated {generated_tokens} tokens")

    # 5. Parse model output (training-data Reasoning/Decision format)
    print("Parsing AI response...")
    print(f"Raw AI output (first 500 chars):\n{response_text[:500]}\n")

    
    debug_response_file = os.path.join(debug_dir, f"case_{case_id[:8]}_model_response.txt")
    with open(debug_response_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"MODEL RAW RESPONSE FOR: {case_id}\n")
        f.write("="*80 + "\n\n")
        f.write(response_text)
        f.write("\n\n" + "="*80 + "\n")
        f.write(f"TOKENS GENERATED: {generated_tokens}\n")
        f.write("="*80 + "\n")
    print(f"💾 Saved model response to: {debug_response_file}\n")

    # ------------------------------------------------------------------
    # 5. Parse model output — the model was trained on the SC judgment
    #    dataset, so it generates Reasoning: ... Decision: ... text.
    #    We extract those sections and build structured JSON from them.
    # ------------------------------------------------------------------

    def extract_section_text(text: str, header: str) -> str:
        
        pattern = rf'{re.escape(header)}\s*(.*?)(?=\n(?:Facts|Issue|Arguments of (?:Petitioner|Respondent)|Reasoning|Decision|Label)\s*:|$)'
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        
        idx = text.lower().find(header.lower())
        if idx != -1:
            remainder = text[idx + len(header):].lstrip(':').strip()
            return remainder[:1500]
        return ""


    def build_offenses_from_sections(sections: list, reasoning_text: str, context: str) -> list:
        KNOWN_SECTIONS = {
            "302": "Murder", "304": "Culpable homicide not amounting to murder",
            "304A": "Causing death by negligence", "307": "Attempt to murder",
            "323": "Voluntarily causing hurt", "324": "Hurt by dangerous weapons",
            "325": "Grievous hurt", "326": "Grievous hurt by dangerous weapons",
            "354": "Assault on woman with intent to outrage modesty",
            "376": "Rape", "379": "Theft", "380": "Theft in dwelling house",
            "382": "Theft after preparation for hurt", "392": "Robbery",
            "395": "Dacoity", "406": "Criminal breach of trust",
            "420": "Cheating and dishonestly inducing delivery of property",
            "447": "Criminal trespass", "448": "House-trespass",
            "454": "Lurking house-trespass", "457": "Lurking house-trespass by night",
            "34":  "Acts done by several persons in furtherance of common intention",
            "120B": "Criminal conspiracy", "149": "Unlawful assembly",
            "341": "Wrongful restraint", "342": "Wrongful confinement",
            "343": "Wrongful confinement for three or more days",
            "363": "Kidnapping", "364": "Kidnapping for ransom",
            "384": "Extortion", "386": "Extortion by putting a person in fear of death",
            "500": "Defamation", "506": "Criminal intimidation",
            "511": "Attempt to commit offence",
        }
        
        extra_sections = set(sections)
        for pattern in [
            r'Section\s+(\d{2,3}[A-Z]?(?:\([a-zA-Z0-9]+\))?)\s+(?:IPC|BNS)',
            r'(?:under|u/s|U/S)\s+(\d{2,3}[A-Z]?)\s+(?:IPC|BNS)',
        ]:
            for m in re.finditer(pattern, reasoning_text, re.IGNORECASE):
                extra_sections.add(m.group(1).strip())
        
        # Remove procedural sections — only keep recognized IPC offenses
        extra_sections -= _PROCEDURAL_SECTIONS

        offenses = []
        for sec in sorted(extra_sections, key=lambda x: int(re.match(r'\d+', x).group()) if re.match(r'\d+', x) else 0):
            desc = KNOWN_SECTIONS.get(sec)
            if desc:
                offenses.append(f"Section {sec} IPC - {desc}")
            # Only include unknown sections if they have a description in reasoning
            else:
                m = re.search(rf'Section\s+{re.escape(sec)}\s+(?:IPC/BNS|IPC|BNS)\s*[\-\(]\s*([^\n)]+)', reasoning_text, re.IGNORECASE)
                if m:
                    offenses.append(f"Section {sec} IPC/BNS - {m.group(1).strip()[:60]}")

        if not offenses:
            # Keyword inference from context
            ctx_lower = context.lower()
            if 'theft' in ctx_lower or 'stolen' in ctx_lower:
                offenses.append("Section 379 IPC - Theft")
            if 'assault' in ctx_lower or 'hurt' in ctx_lower or 'beat' in ctx_lower:
                offenses.append("Section 323 IPC - Voluntarily causing hurt")
            if 'murder' in ctx_lower or 'killed' in ctx_lower:
                offenses.append("Section 302 IPC - Murder")
            if 'cheating' in ctx_lower or 'fraud' in ctx_lower:
                offenses.append("Section 420 IPC - Cheating")
            if not offenses:
                offenses.append("IPC/BNS sections to be determined after full investigation")

        return offenses

    def _detect_available_evidence(context: str) -> set:
        """Detect what evidence the FIR document already mentions as collected."""
        ctx_lower = context.lower()
        found = set()
        # MLC / medical report
        if re.search(r'\bmlc\b|\bmedical\s+(?:report|examination|certificate)|medico.?legal', ctx_lower):
            found.add('mlc')
        # Site plan / scene inspection
        if re.search(r'site\s+plan|scene\s+(?:of\s+crime\s+)?inspection|spot\s+map|naksha\s+mauka|mauqa', ctx_lower):
            found.add('site_plan')
        # Arrest memo
        if re.search(r'arrest\s+memo|arrested|गिरफ्तारी', ctx_lower):
            found.add('arrest')
        # Seizure memo / recovery
        if re.search(r'seizure\s+memo|recovered|baram[ae]d|जब्ती|बरामद', ctx_lower):
            found.add('seizure')
        # FIR copy
        if re.search(r'\bfir\b|\bf\.i\.r\.?', ctx_lower):
            found.add('fir')
        # Witness statements
        if re.search(r'witness(?:es)?\s+statement|statement.*witness|161\s+cr\.?p\.?c|बयान', ctx_lower):
            found.add('witness_statements')
        # CCTV
        if re.search(r'\bcctv\b|\bcamera\b|\bfootage\b', ctx_lower):
            found.add('cctv')
        # FSL / forensic
        if re.search(r'\bfsl\b|\bforensic\b|finger\s*print', ctx_lower):
            found.add('fsl')
        # Photographs
        if re.search(r'photograph|photo', ctx_lower):
            found.add('photos')
        # CDR / call records
        if re.search(r'\bcdr\b|call\s+detail|call\s+record', ctx_lower):
            found.add('cdr')
        return found

    def extract_missing_evidence_from_reasoning(reasoning: str, context: str) -> list:
        # First detect what the FIR already has
        available = _detect_available_evidence(context)
        print(f"Evidence already in document: {available}")

        # AI-generated evidence items from reasoning
        evidence = []
        evidence_keywords = r'(?:must|should|required?|necessary|essential|obtain|collect|record|examine|recover|produce|establish|verify|need)'
        for sent in re.split(r'[.!?\n]', reasoning):
            if re.search(evidence_keywords, sent, re.IGNORECASE):
                sent = sent.strip()
                if 20 < len(sent) < 200:
                    evidence.append(sent)
        evidence = list(dict.fromkeys(evidence))[:4]

        # Context-aware gaps: only add items NOT already present
        ctx_lower = context.lower()
        gaps = []

        # Theft / robbery
        if 'theft' in ctx_lower or 'stolen' in ctx_lower or 'robbery' in ctx_lower or 'loot' in ctx_lower:
            if 'seizure' not in available:
                gaps.append("Recovery and panchnama of stolen property")
            gaps.append("Ownership proof of stolen items (receipts, invoices)")
            if 'cctv' not in available:
                gaps.append("CCTV footage from crime scene")

        # Assault / hurt
        if 'assault' in ctx_lower or 'hurt' in ctx_lower or 'injury' in ctx_lower or 'beat' in ctx_lower:
            if 'mlc' not in available:
                gaps.append("Medical Examination Report (MLC)")
            else:
                gaps.append("Final medical opinion on nature and type of weapon used")
            if 'photos' not in available:
                gaps.append("Photographs of injuries sustained by victim")

        # Murder / death
        if 'murder' in ctx_lower or 'death' in ctx_lower or 'killed' in ctx_lower:
            gaps.append("Post-mortem report")
            gaps.append("Weapon recovery and FSL analysis report")
            if 'fsl' not in available:
                gaps.append("Forensic examination of crime scene")

        # Fraud / cheating
        if 'fraud' in ctx_lower or 'cheating' in ctx_lower or 'forgery' in ctx_lower:
            gaps.append("Documentary evidence of transaction")
            gaps.append("Bank statements or payment records")
            gaps.append("Forensic document examination")

        # Threat / intimidation
        if 'threat' in ctx_lower or 'intimidat' in ctx_lower or '506' in context:
            if 'cdr' not in available:
                gaps.append("Call Detail Records (CDR) to corroborate threats")
            if 'witness_statements' not in available:
                gaps.append("Witness statements corroborating threats")

        # Universal gaps — only if not already present
        if 'witness_statements' not in available:
            gaps.append("Formal witness statements under Section 161 CrPC")
        if 'site_plan' not in available:
            gaps.append("Scene of crime inspection and site plan")
        if 'fsl' not in available and ('weapon' in ctx_lower or 'rod' in ctx_lower or 'stick' in ctx_lower or 'knife' in ctx_lower):
            gaps.append("FSL examination of weapon/instrument used")

        # Merge AI evidence + gaps, dedup
        for item in gaps:
            if item not in evidence:
                evidence.append(item)

        return list(dict.fromkeys(evidence))[:8]

    def parse_judgment_output(response: str, context: str, sections: list,
                               complainant_name: str, accused_name: str,
                               case_meta: dict, restore_names) -> dict:
        """
        Parse model output. Names were masked as tokens in the prompt so the
        model generates [COMPLAINANT_NAME]/[ACCUSED_NAME] — restore_names()
        swaps them back to the real extracted names.
        """
        print(f"Model output preview (first 400 chars): {response[:400]}")

        # Restore real names in the raw model output before any further parsing
        restored = restore_names(response)

        summary   = extract_section_text(restored, "Summary")
        reasoning = extract_section_text(restored, "Reasoning")
        decision  = extract_section_text(restored, "Decision")

        
        if not summary and reasoning:
            # Take up to 3 sentences from reasoning as summary
            sents = re.split(r'(?<=[.!?])\s+', reasoning.strip())
            summary = ' '.join(sents[:3])

        if not reasoning and len(restored.strip()) > 30:
            reasoning = restored.strip()
        if not decision:
            parts = re.split(r'\nDecision\s*:', restored, flags=re.IGNORECASE)
            decision = parts[1].strip()[:600] if len(parts) > 1 else ""

        print(f"Summary   excerpt: {summary[:200]}")
        print(f"Reasoning excerpt: {reasoning[:200]}")
        print(f"Decision  excerpt: {decision[:200]}")

        if not reasoning and not decision and not summary:
            print("Model produced no usable output — using fallback")
            return None

        # Last-resort: if names still missing after restore, scan generated text
        complainant = complainant_name
        accused     = accused_name
        if complainant == "the complainant" or accused == "the accused":
            gen_comp, gen_acc = extract_party_names(restored)
            if complainant == "the complainant" and gen_comp != "the complainant":
                complainant = gen_comp
            if accused == "the accused" and gen_acc != "the accused":
                accused = gen_acc

        offenses = build_offenses_from_sections(sections, reasoning, context)
        missing_evidence = extract_missing_evidence_from_reasoning(reasoning, context)
        offense_labels = ', '.join([o.split(' - ')[0] for o in offenses[:3]])
        offense_desc   = ', '.join([o.split(' - ')[-1] for o in offenses[:2]]).lower()

        # ── Strip IPC/BNS section references from summary text ────────────────
        def _strip_sections_from_summary(text: str) -> str:
            """Remove IPC/BNS section references so the summary is purely factual."""
            _ACT = r'(?:IPC/BNS|IPC|BNS|BNSS)'  # longest-first alternation
            # "alleging offences under Sections 1, 16, ... IPC/BNS." (must come first)
            text = re.sub(
                rf'\s*(?:alleging\s+)?offen[cs]es?\s+under\s+Sections?\s+[\d,\s/]+(?:and\s+\d+\s*)?{_ACT}[.\s]*',
                ' ', text, flags=re.IGNORECASE
            )
            # "under Sections 323, 341, 379 and 506 IPC" / "under Section 379 IPC/BNS"
            text = re.sub(
                rf'\s*(?:under|u/s)\s+Sections?\s+[\d,\s/]+(?:and\s+\d+\s*)?{_ACT}[.\s]*',
                ' ', text, flags=re.IGNORECASE
            )
            # "Sections 323, 325, 34 and 506 IPC are invoked."
            text = re.sub(
                rf'Sections?\s+[\d,\s/]+(?:and\s+\d+\s*)?{_ACT}\s+(?:are|is|were|was)\s+\w+[.\s]*',
                '', text, flags=re.IGNORECASE
            )
            # Standalone "Section 379 IPC (theft)" patterns
            text = re.sub(
                rf'Section\s+\d+[A-Z]?\s+{_ACT}\s*(?:\([^)]+\))?\s*[,;]?\s*',
                '', text, flags=re.IGNORECASE
            )
            # Residual act names left over after stripping
            text = re.sub(rf'\s+{_ACT}\b[.\s]*', ' ', text, flags=re.IGNORECASE)
            # Orphaned connectors ("involves and ." → "involves.")
            text = re.sub(r'\s+and\s*\.', '.', text)
            text = re.sub(r'\s+and\s+and\s+', ' and ', text)
            # Clean up double spaces, stray punctuation
            text = re.sub(r'  +', ' ', text)
            text = re.sub(r'\.\s*\.', '.', text)
            return text.strip()

        # ── Case summary: prefer model output, augment with metadata ──────────
        date_str  = case_meta.get("date", "")
        place_str = case_meta.get("place", "")
        ps_str    = case_meta.get("police_station", "")

        if summary and len(summary.strip()) > 40:
            # Model wrote a summary — strip section references, keep only facts
            final_summary = _strip_sections_from_summary(summary.strip())
            # Append date/station if the model didn't mention them
            meta_parts = []
            if date_str and date_str not in final_summary:
                meta_parts.append(f"incident date: {date_str}")
            if ps_str and ps_str.lower() not in final_summary.lower():
                meta_parts.append(f"reported at {ps_str} Police Station")
            if meta_parts:
                final_summary += f" ({', '.join(meta_parts)})"
        else:
            # Programmatic fallback — build purely factual (no section numbers)
            s1 = f"{complainant} filed an FIR against {accused} for {offense_desc}."
            s2_parts = []
            if date_str:
                s2_parts.append(f"The incident occurred on {date_str}")
            if place_str:
                s2_parts.append(f"at {place_str}")
            if ps_str:
                s2_parts.append(f"(reported at {ps_str} Police Station)")
            s2 = ' '.join(s2_parts) + "." if s2_parts else ""
            reasoning_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', reasoning) if len(s.strip()) > 30]
            # Pick reasoning sentences that don't mention section numbers
            factual_sents = [s for s in reasoning_sents if not re.search(r'Section\s+\d+', s, re.IGNORECASE)]
            s3 = factual_sents[0] if factual_sents else (reasoning_sents[0] if reasoning_sents else "")
            s3 = _strip_sections_from_summary(s3)
            final_summary = " ".join(filter(None, [s1, s2, s3]))

        
        if decision and len(decision.strip()) > 30:
            recommendation = decision.strip()[:600]
        else:
            recommendation = (
                f"The Investigating Officer should complete the investigation and file a chargesheet "
                f"under {offense_labels}. Priority: (1) Record statements of {complainant} and all witnesses; "
                f"(2) Conduct scene inspection and prepare panchnama; "
                f"(3) Collect all physical and documentary evidence; "
                f"(4) {'Identify accused through investigation' if 'unidentified' in accused.lower() else f'Arrest and question {accused}'}; "
                f"(5) Comply with all CrPC procedural requirements."
            )

        return {
            "summary": final_summary,
            "offenses": offenses,
            "missing_evidence": missing_evidence,
            "recommendation": recommendation,
        }

    def normalize_to_strings(data):
       
        if not isinstance(data, dict):
            return data
        normalized = {}
        for key, value in data.items():
            if key in ['offenses', 'missing_evidence'] and isinstance(value, list):
                normalized[key] = []
                for item in value:
                    if isinstance(item, dict):
                        if 'code' in item and 'description' in item:
                            normalized[key].append(f"Section {item['code']} - {item['description']}")
                        elif 'name' in item:
                            normalized[key].append(item['name'])
                        else:
                            normalized[key].append(str(item))
                    else:
                        normalized[key].append(str(item))
            else:
                normalized[key] = value
        return normalized

    try:
        final_json = parse_judgment_output(
            response_text, formatted_context, pre_extracted_list,
            complainant_name=_comp_hint, accused_name=_acc_hint,
            case_meta=_case_meta, restore_names=name_restore
        )

        if final_json:
            final_json = normalize_to_strings(final_json)

            required_fields = ['summary', 'offenses', 'missing_evidence', 'recommendation']
            for field in required_fields:
                if field not in final_json:
                    final_json[field] = [] if field in ['offenses', 'missing_evidence'] else "No data provided"

            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"✓ Legal Analysis Complete!")
            print(f"Total Time: {total_time:.2f}s")
            print(f"{'='*60}\n")

           
            if torch.cuda.is_available():
                import gc
                gc.collect() 
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"Final GPU cache cleanup complete")
            
            return final_json
        else:
            print("✗ Error: Could not parse response")
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            return {"error": "Failed to parse response", "raw_text": response_text[:500]}
            
    except Exception as e:
        print(f"✗ Unexpected error during parsing: {str(e)}")
        if torch.cuda.is_available():
            import gc
            gc.collect()  
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return {"error": f"Parsing error: {str(e)}", "raw_text": response_text[:500] if 'response_text' in locals() else "No response generated"}