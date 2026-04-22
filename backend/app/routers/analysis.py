from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services import ocr_service, translation_service, rag_service, legal_engine
from app.models.schemas import CaseAnalysisResponse 
import uuid
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/analyze-case-rag", response_model=CaseAnalysisResponse)
async def analyze_case_file_rag(file: UploadFile = File(...)):
    try:
       
        if file.content_type not in ["application/pdf", "image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF or Images allowed.")

        
        logger.info(f"Processing file: {file.filename} ({file.content_type})")
        file_bytes = await file.read()
        logger.info(f"File size: {len(file_bytes)} bytes")
        
        hindi_text = await ocr_service.extract_text_from_file(file_bytes, file.filename)
        
        
        if not hindi_text or hindi_text.startswith("Error"):
            error_msg = hindi_text if hindi_text else "OCR failed to extract text"
            logger.error(f"OCR Error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info(f"OCR extracted {len(hindi_text)} characters")
        logger.info(f"First 200 chars of Hindi text: {hindi_text[:200]}...")

        # Extract section numbers from the raw Hindi text BEFORE translation
        # so numerals embedded in Devanagari context are not lost during translation.
        hindi_sections = legal_engine.extract_sections_from_hindi(hindi_text)
        logger.info(f"Sections found in Hindi text pre-translation: {hindi_sections}")

        english_text = await translation_service.translate_to_english(hindi_text)
        
        
        if english_text.startswith("Error"):
            logger.error(f"Translation Error: {english_text}")
            raise HTTPException(status_code=500, detail=english_text)
        
        logger.info(f"Translation produced {len(english_text)} characters")
        logger.info(f"First 200 chars of English text: {english_text[:200]}...")
        
       
        case_id = str(uuid.uuid4())
        
       
        debug_dir = "debug_translations"
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, f"case_{case_id[:8]}_translation.txt")
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("ORIGINAL HINDI TEXT\n")
            f.write("="*80 + "\n")
            f.write(hindi_text)
            f.write("\n\n" + "="*80 + "\n")
            f.write("ENGLISH TRANSLATION\n")
            f.write("="*80 + "\n")
            f.write(english_text)
        logger.info(f"Saved translation debug file: {debug_file}")
        logger.info(f"Storing case data with ID: {case_id}")
        await rag_service.store_case_data(english_text, case_id)
        
        try:
            ai_result = await legal_engine.analyze_legal_case(
                case_id,
                raw_english_text=english_text,
                hint_sections=hindi_sections,
                raw_hindi_text=hindi_text
            )
        except RuntimeError as e:
            # Catch CUDA out of memory errors
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.error(f"GPU OOM Error: {str(e)}")
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                return CaseAnalysisResponse(
                    case_id=case_id,
                    summary="GPU memory exhausted. Please try again in a moment.",
                    offenses=["Analysis pending due to system resource constraints"],
                    sections=[],
                    missing_evidence=["Analysis could not be completed"],
                    recommendation="The system is currently processing other requests. Please retry your analysis in 30 seconds. If the issue persists, contact system administrator."
                )
            else:
                raise e
        
        if "error" in ai_result:
            logger.error(f"AI Error: {ai_result}")
            return CaseAnalysisResponse(
                case_id=case_id,
                summary="Error in AI Analysis",
                offenses=[],
                sections=[],
                missing_evidence=[],
                recommendation=f"System Error: {ai_result.get('error')}",
                summary_hindi=None,
                offenses_hindi=None,
                missing_evidence_hindi=None,
            )

        summary_en = ai_result.get("summary", "No summary provided.")
        offenses_en = ai_result.get("offenses", [])
        sections_en = ai_result.get("sections", [])
        missing_evidence_en = ai_result.get("missing_evidence", [])

        # Best-effort Hindi rendering for frontend bilingual display.
        summary_hi = None
        offenses_hi = None
        missing_evidence_hi = None
        try:
            summary_hi_raw = await translation_service.translate_to_hindi(summary_en)
            if summary_hi_raw and not summary_hi_raw.startswith("Error"):
                summary_hi = summary_hi_raw

            if offenses_en:
                offenses_hi = []
                for offense in offenses_en:
                    hi = await translation_service.translate_to_hindi(offense)
                    offenses_hi.append(hi if hi and not hi.startswith("Error") else offense)

            if missing_evidence_en:
                missing_evidence_hi = []
                for gap in missing_evidence_en:
                    hi = await translation_service.translate_to_hindi(gap)
                    missing_evidence_hi.append(hi if hi and not hi.startswith("Error") else gap)
        except Exception as translation_err:
            logger.warning(f"Hindi translation fallback used due to: {translation_err}")

        return CaseAnalysisResponse(
            case_id=case_id,
            summary=summary_en,
            offenses=offenses_en,
            sections=sections_en,
            missing_evidence=missing_evidence_en,
            recommendation=ai_result.get("recommendation", "No recommendation provided."),
            summary_hindi=summary_hi,
            offenses_hindi=offenses_hi,
            missing_evidence_hindi=missing_evidence_hi,
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error analyzing case: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Server Error: {str(e)}"
        )