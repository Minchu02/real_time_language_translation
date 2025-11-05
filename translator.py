# translator.py
from deep_translator import GoogleTranslator
import logging
from typing import Optional
import time
import re

logger = logging.getLogger(__name__)

# Translation cache for performance
_translation_cache = {}
CACHE_SIZE = 200

# Enhanced language support with proper codes
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi", 
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "bn": "Bengali",
    "gu": "Gujarati",
    "mr": "Marathi",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
    "ar": "Arabic",
    "pt": "Portuguese",
    "it": "Italian",
    "ur": "Urdu",
    "pa": "Punjabi"
}

# Language family grouping for better translation
INDIAN_LANGUAGES = {"hi", "kn", "ta", "te", "ml", "bn", "gu", "mr", "ur", "pa"}

def preprocess_text_for_translation(text: str, source_lang: str, target_lang: str) -> str:
    """Preprocess text to improve translation quality"""
    if not text:
        return ""
    
    # Clean the text
    text = text.strip()
    
    # Remove common transcription artifacts
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    
    # For Indian language translations, handle common issues
    if source_lang in INDIAN_LANGUAGES and target_lang in INDIAN_LANGUAGES:
        # Remove English words that might have been incorrectly transcribed
        text = ' '.join([word for word in text.split() if not re.match(r'^[a-zA-Z]+$', word)])
    
    return text

def translate_text(text: str, target_lang: str = "en", source_lang: Optional[str] = "auto") -> str:
    """
    Translate text to target language with enhanced error handling for Indian languages
    """
    if not text or text.strip() == "":
        return ""
    
    # Skip translation for these cases
    if text == "No speech detected" or text.startswith("ASR error"):
        return text
    
    # Check if target language is supported
    if target_lang not in SUPPORTED_LANGUAGES:
        return f"{text} [Language {target_lang} not supported]"
    
    # Preprocess text
    processed_text = preprocess_text_for_translation(text, source_lang, target_lang)
    if not processed_text:
        return "No meaningful text to translate"
    
    # Create cache key
    cache_key = f"{source_lang}_{target_lang}_{processed_text}"
    
    # Check cache first
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]
    
    try:
        # Add delay to avoid rate limiting
        time.sleep(0.2)
        
        # Use Google Translator with better error handling
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated_text = translator.translate(processed_text)
        
        # Enhanced validation
        if not translated_text or translated_text.strip() == "":
            logger.warning(f"Empty translation for '{processed_text[:30]}...' to {target_lang}")
            return f"{text} [Empty translation]"
        
        # Check for translation errors
        error_patterns = [
            "could not translate",
            "translation failed",
            "error occurred",
            "not supported",
            "invalid language"
        ]
        
        if any(pattern in translated_text.lower() for pattern in error_patterns):
            logger.warning(f"Translation error pattern detected: {translated_text}")
            return f"{text} [Translation service error]"
        
        # For Indian languages, validate the translation isn't just copying the input
        if (source_lang in INDIAN_LANGUAGES and 
            translated_text.lower() == processed_text.lower() and 
            len(processed_text.split()) > 2):
            # Try alternative approach - translate via English
            try:
                # First translate to English
                en_translator = GoogleTranslator(source=source_lang, target="en")
                english_text = en_translator.translate(processed_text)
                
                if english_text and english_text != processed_text:
                    # Then translate to target language
                    final_translator = GoogleTranslator(source="en", target=target_lang)
                    translated_text = final_translator.translate(english_text)
                    logger.info(f"Used two-step translation for {source_lang}->{target_lang}")
            except Exception as fallback_error:
                logger.warning(f"Two-step translation failed: {fallback_error}")
        
        # Update cache
        if len(_translation_cache) >= CACHE_SIZE:
            _translation_cache.pop(next(iter(_translation_cache)))
        
        _translation_cache[cache_key] = translated_text
        logger.info(f"Successfully translated to {target_lang}: '{processed_text[:30]}...' -> '{translated_text[:30]}...'")
        
        return translated_text
        
    except Exception as e:
        logger.error(f"Translation failed for '{processed_text[:30]}...' to {target_lang}: {str(e)}")
        
        # Provide more specific error messages
        error_msg = str(e).lower()
        if "quota" in error_msg:
            return f"{text} [Translation quota exceeded]"
        elif "network" in error_msg or "connection" in error_msg:
            return f"{text} [Network error - check internet connection]"
        elif "language" in error_msg:
            return f"{text} [Language not supported by service]"
        else:
            # Try one more time with simpler text
            try:
                simple_text = ' '.join(processed_text.split()[:10])  # First 10 words
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                retry_translation = translator.translate(simple_text)
                if retry_translation and retry_translation != simple_text:
                    return retry_translation + " [Partial translation]"
            except:
                pass
                
            return f"{text} [Translation failed: {str(e)[:50]}]"

def get_language_name(lang_code: str) -> str:
    """Get full language name from code"""
    return SUPPORTED_LANGUAGES.get(lang_code, lang_code.upper())

def get_available_languages():
    """Get list of available languages"""
    return list(SUPPORTED_LANGUAGES.keys())

def is_language_supported(lang_code: str) -> bool:
    """Check if language is supported"""
    return lang_code in SUPPORTED_LANGUAGES

def get_indian_languages():
    """Get list of Indian languages"""
    return [lang for lang in get_available_languages() if lang in INDIAN_LANGUAGES]