from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import gc
import hashlib
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Use CUDA allocator settings only if not already provided by environment.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128"
)

MODEL_NAME = "sarvamai/sarvam-translate"
TRANSLATION_USE_GPU_MODE = os.getenv("TRANSLATION_USE_GPU", "auto").lower()
MIN_GPU_VRAM_GB = float(os.getenv("TRANSLATION_MIN_GPU_VRAM_GB", "6"))
MAX_NEW_TOKENS = int(os.getenv("TRANSLATION_MAX_NEW_TOKENS", "128"))
MAX_CHUNK_SIZE = int(os.getenv("TRANSLATION_MAX_CHUNK_SIZE", "700"))
MIN_NEW_TOKENS = 48
MAX_CACHE_ITEMS = int(os.getenv("TRANSLATION_CACHE_ITEMS", "128"))

if TRANSLATION_USE_GPU_MODE not in {"true", "false", "auto"}:
    print(
        f"Invalid TRANSLATION_USE_GPU='{TRANSLATION_USE_GPU_MODE}'. Falling back to 'auto'."
    )
    TRANSLATION_USE_GPU_MODE = "auto"

USE_GPU = TRANSLATION_USE_GPU_MODE != "false"

if torch.cuda.is_available() and USE_GPU:
    available_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if available_vram_gb >= MIN_GPU_VRAM_GB:
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
        print(
            f"Translation GPU disabled: detected {available_vram_gb:.2f} GB VRAM "
            f"(< {MIN_GPU_VRAM_GB:.2f} GB threshold)."
        )
else:
    DEVICE = "cpu"

print(f"Translation service will use: {DEVICE.upper()}")
print(f"Translation model: {MODEL_NAME} (local cache only)")

tokenizer = None
model = None
_translation_cache = {}
KEEP_MODEL_RESIDENT = os.getenv("TRANSLATION_KEEP_MODEL_RESIDENT", "false").lower() == "true"


def _is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "cuda out of memory" in msg or ("out of memory" in msg and "cuda" in msg)


def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unload_model():
    global tokenizer, model

    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass

    model = None
    tokenizer = None
    gc.collect()
    _clear_cuda_cache()


def _cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _cache_get(text: str) -> Optional[str]:
    key = _cache_key(text)
    value = _translation_cache.get(key)
    if value is None:
        return None
    # Keep hot keys at the end.
    _translation_cache.pop(key, None)
    _translation_cache[key] = value
    return value


def _cache_set(text: str, translation: str):
    key = _cache_key(text)
    if key in _translation_cache:
        _translation_cache.pop(key, None)
    _translation_cache[key] = translation
    if len(_translation_cache) > MAX_CACHE_ITEMS:
        oldest_key = next(iter(_translation_cache))
        _translation_cache.pop(oldest_key, None)


def load_model(target_device: Optional[str] = None):
    global tokenizer, model, DEVICE

    if target_device is None:
        target_device = DEVICE

    if target_device not in ["cpu", "cuda"]:
        raise ValueError(f"Invalid target device: {target_device}")

    # Reload model if device changed.
    if DEVICE != target_device and (tokenizer is not None or model is not None):
        print(f"Switching translation model from {DEVICE} to {target_device}...")
        _unload_model()
        DEVICE = target_device

    if tokenizer is None or model is None:
        print(f"Loading Translation Model ({MODEL_NAME}) on {DEVICE}...")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            ).to(DEVICE)
        except RuntimeError as e:
            if DEVICE == "cuda" and _is_cuda_oom_error(e):
                print("Translation model GPU load OOM. CPU fallback is disabled.")
            raise

        model.eval()
        print("Translation model loaded successfully!")

    return tokenizer, model


def _translate_chunk(tok, mdl, text: str) -> str:
    """Translate a single chunk of Hindi text to English."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional Hindi-to-English translator. "
                "Translate the user's Hindi text to English accurately and naturally. "
                "Output only the English translation, nothing else."
            ),
        },
        {"role": "user", "content": text},
    ]

    input_ids = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(DEVICE)

    attention_mask = torch.ones_like(input_ids).to(DEVICE)

    approx_input_tokens = int(input_ids.shape[-1])
    target_new_tokens = min(
        MAX_NEW_TOKENS,
        max(MIN_NEW_TOKENS, int(approx_input_tokens * 0.8)),
    )

    with torch.no_grad():
        output_ids = mdl.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=target_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    result = tok.decode(new_tokens, skip_special_tokens=True).strip()

    if DEVICE == "cuda":
        _clear_cuda_cache()

    return result

async def translate_to_english(hindi_text: str) -> str:
    loaded_for_this_request = False
    try:
        print(f"\n{'='*60}")
        print(f"Starting Translation")
        print(f"Input length: {len(hindi_text)} characters")
        print(f"{'='*60}")

        cached_translation = _cache_get(hindi_text)
        if cached_translation is not None:
            print("Translation cache hit (full text)")
            return cached_translation

        tok, mdl = load_model()
        loaded_for_this_request = True

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

        chunks = smart_split(hindi_text, MAX_CHUNK_SIZE)
        print(f"Text split into {len(chunks)} chunk(s)")

        english_parts = []
        for i, chunk in enumerate(chunks):
            print(f"Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            try:
                cached_chunk = _cache_get(chunk)
                if cached_chunk is not None:
                    translated = cached_chunk
                    print(f"Chunk {i+1} cache hit")
                else:
                    translated = _translate_chunk(tok, mdl, chunk)
                    _cache_set(chunk, translated)
            except RuntimeError as e:
                if DEVICE == "cuda" and _is_cuda_oom_error(e):
                    print("CUDA OOM during translation chunk. CPU fallback is disabled.")
                    raise
                else:
                    raise
            english_parts.append(translated)
            print(f"Chunk {i+1} done: {len(translated)} chars")

        english_translation = " ".join(english_parts)
        _cache_set(hindi_text, english_translation)

        print(f"\nTranslation complete: {len(english_translation)} characters")
        print(f"First 200 chars: {english_translation[:200]}...")
        print(f"{'='*60}\n")

        if DEVICE == "cuda":
            _clear_cuda_cache()
            print("GPU cache cleared after translation")

        return english_translation

    except Exception as e:
        print(f"Translation Error: {e}")
        import traceback
        traceback.print_exc()
        _clear_cuda_cache()
        return f"Error in translation: {str(e)}"
    finally:
        if loaded_for_this_request and not KEEP_MODEL_RESIDENT:
            _unload_model()