# translator.py
from deep_translator import GoogleTranslator
import logging
from typing import Optional
import time
import re

logger = logging.getLogger(__name__)

# Unicode ranges for Indian language scripts
KANNADA_RANGE = (0x0C80, 0x0CFF)  # Kannada script Unicode range
HINDI_RANGE = (0x0900, 0x097F)    # Devanagari script Unicode range (Hindi uses Devanagari)

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

def is_text_in_native_script(text: str, target_lang: str) -> bool:
    """
    Check if text contains native script characters for the target language.
    Returns True if text contains native script characters.
    """
    if not text:
        return False
    
    if target_lang == "kn":  # Kannada
        # Check if text contains Kannada script characters
        return any(KANNADA_RANGE[0] <= ord(char) <= KANNADA_RANGE[1] for char in text)
    elif target_lang == "hi":  # Hindi
        # Check if text contains Devanagari script characters
        return any(HINDI_RANGE[0] <= ord(char) <= HINDI_RANGE[1] for char in text)
    
    return True  # For other languages, assume correct

def is_text_transliterated(text: str, target_lang: str) -> bool:
    """
    Check if text appears to be transliterated (Roman script) instead of native script.
    Returns True if text looks like transliteration.
    """
    if not text:
        return False
    
    if target_lang == "kn":  # Kannada
        # Check if text contains Kannada script characters
        kannada_chars = sum(1 for char in text if KANNADA_RANGE[0] <= ord(char) <= KANNADA_RANGE[1])
        total_alphabetic = len([c for c in text if c.isalpha()])
        
        if total_alphabetic == 0:
            return False
        
        # If less than 20% Kannada characters, likely transliterated
        if kannada_chars / total_alphabetic < 0.2:
            # Check if text contains mostly ASCII letters (Roman transliteration)
            ascii_letters = sum(1 for c in text if c.isalpha() and ord(c) < 128)
            if ascii_letters / total_alphabetic > 0.5:
                return True
    
    elif target_lang == "hi":  # Hindi
        # Check if text contains Devanagari script characters
        devanagari_chars = sum(1 for char in text if HINDI_RANGE[0] <= ord(char) <= HINDI_RANGE[1])
        total_alphabetic = len([c for c in text if c.isalpha()])
        
        if total_alphabetic == 0:
            return False
        
        # If less than 20% Devanagari characters, likely transliterated
        if devanagari_chars / total_alphabetic < 0.2:
            # Check if text contains mostly ASCII letters (Roman transliteration)
            ascii_letters = sum(1 for c in text if c.isalpha() and ord(c) < 128)
            if ascii_letters / total_alphabetic > 0.5:
                return True
    
    return False

def force_native_script_translation(text: str, target_lang: str, source_lang: str) -> str:
    """
    Force translation to native script by trying alternative methods.
    """
    try:
        # Method 1: Try direct translation with explicit source detection
        translator = GoogleTranslator(source=source_lang if source_lang != "auto" else "en", target=target_lang)
        translated = translator.translate(text)
        
        # Check if result is in native script
        if is_text_in_native_script(translated, target_lang):
            return translated
        
        # Method 2: If source is auto and result is transliterated, try with English as source
        if source_lang == "auto" and is_text_transliterated(translated, target_lang):
            translator_en = GoogleTranslator(source="en", target=target_lang)
            translated_en = translator_en.translate(text)
            if is_text_in_native_script(translated_en, target_lang):
                return translated_en
        
        # Method 3: Two-step translation via English
        if is_text_transliterated(translated, target_lang):
            try:
                # Translate to English first
                en_translator = GoogleTranslator(source=source_lang if source_lang != "auto" else "en", target="en")
                english_text = en_translator.translate(text)
                
                if english_text and english_text != text:
                    # Then translate from English to target language
                    final_translator = GoogleTranslator(source="en", target=target_lang)
                    final_translation = final_translator.translate(english_text)
                    
                    if is_text_in_native_script(final_translation, target_lang):
                        logger.info(f"Successfully forced native script translation via two-step method")
                        return final_translation
            except Exception as e:
                logger.warning(f"Two-step native script translation failed: {e}")
        
        # If still transliterated, return what we have but log warning
        if is_text_transliterated(translated, target_lang):
            logger.warning(f"Translation to {target_lang} still appears transliterated: {translated[:50]}")
        
        return translated
        
    except Exception as e:
        logger.error(f"Error forcing native script translation: {e}")
        return text

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
        
        # For Kannada and Hindi, ensure native script is used
        if target_lang in ["kn", "hi"]:
            if not is_text_in_native_script(translated_text, target_lang) or is_text_transliterated(translated_text, target_lang):
                logger.info(f"Translation to {target_lang} not in native script, forcing native script translation")
                translated_text = force_native_script_translation(processed_text, target_lang, source_lang)
        
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
                    
                    # Again check for native script if Kannada or Hindi
                    if target_lang in ["kn", "hi"]:
                        if not is_text_in_native_script(translated_text, target_lang) or is_text_transliterated(translated_text, target_lang):
                            translated_text = force_native_script_translation(english_text, target_lang, "en")
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
                    # For Kannada and Hindi, ensure native script
                    if target_lang in ["kn", "hi"]:
                        if not is_text_in_native_script(retry_translation, target_lang) or is_text_transliterated(retry_translation, target_lang):
                            retry_translation = force_native_script_translation(simple_text, target_lang, source_lang)
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