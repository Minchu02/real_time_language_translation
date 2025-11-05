# asr_engine.py
import io
import os
import numpy as np
import soundfile as sf
import tempfile
import shutil
import atexit
from typing import Optional, Tuple
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use larger model for better Indian language accuracy
MODEL_SIZE = os.getenv("FW_MODEL_SIZE", "large-v2")  # Changed to large-v2 for better accuracy
_DEVICE = os.getenv("FW_DEVICE", "cpu")

# Global model variables
_model = None
_model_loaded = False

# Enhanced language mapping with better Indian language support
LANGUAGE_MAPPING = {
    "hi": "hindi", 
    "kn": "kannada",
    "ta": "tamil", 
    "te": "telugu", 
    "ml": "malayalam",
    "bn": "bengali",
    "gu": "gujarati",
    "mr": "marathi",
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "de": "german",
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
    "ru": "russian",
    "ar": "arabic",
    "pt": "portuguese",
    "it": "italian"
}

# Indian languages with priority
INDIAN_LANGUAGES = ["hi", "kn", "ta", "te", "ml", "bn", "gu", "mr"]
INDIAN_LANGUAGE_NAMES = ["hindi", "kannada", "tamil", "telugu", "malayalam", "bengali", "gujarati", "marathi"]

def _load_model():
    global _model, _model_loaded
    if _model_loaded:
        return _model

    try:
        from faster_whisper import WhisperModel
        logger.info(f"🔄 Loading Faster-Whisper model: {MODEL_SIZE} on {_DEVICE}")
        
        # Try different compute types
        compute_types = ["int8", "float16"]
        
        for compute_type in compute_types:
            try:
                _model = WhisperModel(
                    MODEL_SIZE, 
                    device=_DEVICE, 
                    compute_type=compute_type,
                    download_root="./whisper_models",
                    num_workers=1
                )
                logger.info(f"✅ Loaded model with compute_type={compute_type}")
                _model_loaded = True
                return _model
            except Exception as e:
                logger.warning(f"❌ Failed with {compute_type}: {e}")
                continue
                
        # Fallback to base model
        try:
            logger.info("🔄 Trying base model as fallback...")
            _model = WhisperModel("base", device=_DEVICE, compute_type="int8")
            logger.info("✅ Loaded base model as fallback")
            _model_loaded = True
            return _model
        except Exception as e:
            raise RuntimeError(f"All models failed to load: {e}")
        
    except ImportError as e:
        logger.error(f"❌ Faster-Whisper not available: {e}")
        raise

def validate_audio_bytes(audio_bytes: bytes) -> bool:
    """Validate if the bytes represent valid audio data"""
    if not audio_bytes:
        return False
    
    if len(audio_bytes) < 1000:
        return False
        
    return True

def detect_language(audio_bytes: bytes) -> Tuple[str, float]:
    """Detect the language of the audio with confidence score"""
    temp_path = None
    try:
        model = _load_model()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        # Enhanced language detection with focus on Indian languages
        segments, info = model.transcribe(
            temp_path,
            task="translate",
            beam_size=5,  # Increased for better accuracy
            best_of=3,    # Increased for better accuracy
            without_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        detected_lang = getattr(info, 'language', 'en')
        language_probability = getattr(info, 'language_probability', 0.0)
        
        logger.info(f"Initial detection: {detected_lang} (confidence: {language_probability:.2f})")
        
        # If confidence is low, try to force Indian language detection
        if language_probability < 0.6:
            logger.info("Low confidence, trying Indian language focus...")
            # Try with Indian languages specifically
            for indian_lang in INDIAN_LANGUAGES:
                try:
                    segments, info = model.transcribe(
                        temp_path,
                        language=indian_lang,
                        beam_size=3,
                        best_of=2,
                        without_timestamps=True
                    )
                    new_lang = getattr(info, 'language', detected_lang)
                    new_confidence = getattr(info, 'language_probability', 0.0)
                    
                    if new_confidence > language_probability:
                        detected_lang = new_lang
                        language_probability = new_confidence
                        logger.info(f"Better detection with {indian_lang}: {detected_lang} (confidence: {language_probability:.2f})")
                except:
                    continue
        
        logger.info(f"Final detected language: {detected_lang} (confidence: {language_probability:.2f})")
        return detected_lang, language_probability
        
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return "en", 0.0
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass

def transcribe_from_bytes(wav_bytes: bytes, forced_language: Optional[str] = None) -> str:
    if not wav_bytes:
        return "No audio data provided"
    
    if not validate_audio_bytes(wav_bytes):
        return "Invalid audio data"
    
    temp_path = None
    try:
        model = _load_model()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            temp_path = tmp.name

        # Enhanced language detection with Indian language focus
        detected_lang, confidence = detect_language(wav_bytes)
        
        # If confidence is low for Indian languages, provide options
        language_to_use = forced_language if forced_language else detected_lang
        
        logger.info(f"Transcribing with language: {language_to_use} (confidence: {confidence:.2f})")

        # Enhanced transcription with better parameters for Indian languages
        segments, info = model.transcribe(
            temp_path,
            language=language_to_use,
            beam_size=5,
            best_of=3,
            temperature=0.0,
            compression_ratio_threshold=2.0,  # Lower threshold for Indian languages
            no_speech_threshold=0.4,  # Lower threshold to catch more speech
            condition_on_previous_text=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            without_timestamps=True,
            word_timestamps=False,
            patience=1.0  # Added patience for better decoding
        )

        # Combine segments with better handling
        text_chunks = []
        for segment in segments:
            if segment.text and segment.text.strip():
                clean_text = segment.text.strip()
                # Better cleaning
                clean_text = ' '.join(clean_text.split())
                # Remove common transcription artifacts
                clean_text = re.sub(r'\[.*?\]', '', clean_text)
                clean_text = re.sub(r'\(.*?\)', '', clean_text)
                text_chunks.append(clean_text)
        
        result = " ".join(text_chunks) if text_chunks else "No speech detected"
        
        # Enhanced validation for Indian languages
        if result != "No speech detected" and language_to_use in INDIAN_LANGUAGES:
            # Check if the result contains meaningful text
            words = result.split()
            if len(words) < 2:  # If very few words, might be poor transcription
                result = "Speech detected but unclear - try speaking more clearly"
        
        final_language = getattr(info, 'language', language_to_use)
        logger.info(f"Final transcription ({final_language}): {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"ASR Error: {e}")
        return f"ASR error: {str(e)}"
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temp file: {e}")

def preload_model():
    """Preload the model on startup"""
    try:
        _load_model()
        return True
    except Exception as e:
        logger.error(f"Preload failed: {e}")
        return False