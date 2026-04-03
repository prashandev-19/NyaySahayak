from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

MODEL_NAME = "sarvamai/sarvam-translate"
USE_GPU = os.getenv("TRANSLATION_USE_GPU", "true").lower() == "true"
DEVICE = "cuda" if (torch.cuda.is_available() and USE_GPU) else "cpu"

print(f"Translation service will use: {DEVICE.upper()}")
print(f"Translation model: {MODEL_NAME}")

tokenizer = None
model = None


def load_model():
    global tokenizer, model

    if tokenizer is None or model is None:
        print(f"Loading Translation Model ({MODEL_NAME}) on {DEVICE}...")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE)

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

    with torch.no_grad():
        output_ids = mdl.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=True,
            temperature=1.0,
            top_k=64,
            top_p=0.95,
            pad_token_id=0,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    result = tok.decode(new_tokens, skip_special_tokens=True).strip()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return result

async def translate_to_english(hindi_text: str) -> str:
    try:
        print(f"\n{'='*60}")
        print(f"Starting Translation")
        print(f"Input length: {len(hindi_text)} characters")
        print(f"{'='*60}")

        tok, mdl = load_model()

        MAX_CHUNK_SIZE = 400  # characters per chunk

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
            translated = _translate_chunk(tok, mdl, chunk)
            english_parts.append(translated)
            print(f"Chunk {i+1} done: {len(translated)} chars")

        english_translation = " ".join(english_parts)

        print(f"\nTranslation complete: {len(english_translation)} characters")
        print(f"First 200 chars: {english_translation[:200]}...")
        print(f"{'='*60}\n")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            print("GPU cache cleared after translation")

        return english_translation

    except Exception as e:
        print(f"Translation Error: {e}")
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"Error in translation: {str(e)}"