# app.py
import os
os.environ["FW_MODEL_SIZE"] = "base"
os.environ["FW_DEVICE"] = "cpu"
import streamlit as st
import numpy as np
import io
import time
import threading
from queue import Queue
import traceback
import hashlib
from datetime import datetime

# Import helper modules
try:
    from asr_engine import transcribe_from_bytes, preload_model, detect_language
    from translator import translate_text, get_language_name, get_available_languages, is_language_supported, get_indian_languages
    from topic_detector import detect_topic, get_topic_breakdown, reset_topic_history, is_bertopic_available
except ImportError as e:
    st.error(f"Import error: {e}. Ensure all files are in the directory.")
    st.stop()

# Set Streamlit page config
st.set_page_config(
    page_title="🎙 Live Audio Translator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎙 Live Audio Translator")
st.write("Speak → Transcribe → Translate → Detect Topic — all in real time!")

# Initialize session state variables
defaults = {
    "continuous_mode": False,
    "transcript": "Click 'Start Continuous' to begin...",
    "translation": "Translation will appear here",
    "topic": "Topic will be detected here",
    "record_count": 0,
    "last_audio_hash": None,
    "processing": False,
    "model_loading": False,
    "model_loaded": False,
    "translation_enabled": True,
    "detected_language": "Unknown",
    "language_confidence": 0.0,
    "use_advanced_topic": True,
    "session_history": [],
    "audio_duration": 0.0,
    "processing_time": 0.0,
    "audio_queue": Queue(),
    "is_recording": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def get_audio_bytes(uploaded_file):
    """Extract bytes from UploadedFile object"""
    if uploaded_file is None:
        return None
    
    uploaded_file.seek(0)
    audio_bytes = uploaded_file.read()
    return audio_bytes

def get_audio_hash(audio_bytes):
    """Create a hash of audio bytes to detect changes"""
    return hashlib.md5(audio_bytes).hexdigest() if audio_bytes else None

def get_audio_duration(audio_bytes):
    """Estimate audio duration in seconds"""
    try:
        # Rough estimation: 16000 Hz, 16-bit mono
        if len(audio_bytes) < 1000:
            return 0.0
        # Subtract header and calculate duration
        data_size = len(audio_bytes) - 44
        duration = data_size / (16000 * 2)  # sample_rate * bytes_per_sample
        return max(0.0, duration)
    except:
        return 0.0

def process_audio(audio_bytes, target_lang):
    """Process audio and return results"""
    start_time = time.time()
    try:
        # First detect language
        detected_lang, confidence = detect_language(audio_bytes)
        st.session_state.detected_language = detected_lang
        st.session_state.language_confidence = confidence
        
        # Get audio duration
        audio_duration = get_audio_duration(audio_bytes)
        st.session_state.audio_duration = audio_duration
        
        # Transcription with language detection
        transcribed_text = transcribe_from_bytes(audio_bytes)
        
        if not transcribed_text or transcribed_text == "No speech detected":
            return {
                "transcript": "No speech detected",
                "translation": "Waiting for speech...",
                "topic": "—",
                "detected_language": detected_lang,
                "confidence": confidence,
                "audio_duration": audio_duration
            }
        
        # Translation only if enabled and text is valid
        if (st.session_state.translation_enabled and 
            transcribed_text != "No speech detected" and 
            not transcribed_text.startswith("ASR error")):
            
            # Add small delay to avoid API rate limits
            time.sleep(0.2)
            translated_text = translate_text(transcribed_text, target_lang, detected_lang)
        else:
            translated_text = "Translation disabled or no speech"
        
        # Topic detection with BERTopic
        topic_text = detect_topic(transcribed_text, st.session_state.use_advanced_topic)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        st.session_state.processing_time = processing_time
        
        return {
            "transcript": transcribed_text,
            "translation": translated_text,
            "topic": topic_text,
            "detected_language": detected_lang,
            "confidence": confidence,
            "audio_duration": audio_duration,
            "processing_time": processing_time
        }
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        processing_time = time.time() - start_time
        return {
            "transcript": f"Error: {str(e)}",
            "translation": "Processing failed",
            "topic": "—",
            "detected_language": "Unknown",
            "confidence": 0.0,
            "audio_duration": 0.0,
            "processing_time": processing_time
        }

# Pre-load model when app starts
if not st.session_state.model_loaded and not st.session_state.model_loading:
    st.session_state.model_loading = True
    with st.spinner("🔄 Loading speech recognition model (first time only - this may take 2-3 minutes)..."):
        try:
            preload_model()
            st.session_state.model_loaded = True
            st.session_state.model_loading = False
            st.success("✅ Model loaded successfully!")
            time.sleep(2)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.session_state.model_loading = False

# --- Layout ---
col1, col2 = st.columns([1, 1])

# --- Controls ---
with col1:
    st.header("⚙ Controls")
    
    # Translation toggle
    st.session_state.translation_enabled = st.toggle(
        "Enable Translation", 
        value=st.session_state.translation_enabled,
        help="Toggle translation on/off"
    )
    
    # BERTopic toggle
    st.session_state.use_advanced_topic = st.toggle(
        "Use Advanced Topic Detection (BERTopic)", 
        value=st.session_state.use_advanced_topic,
        help="Use AI-powered topic detection for better theme identification"
    )
    
    # Language selection with Indian languages highlighted
    available_langs = get_available_languages()
    indian_langs = get_indian_languages()
    
    # Reorder to show Indian languages first
    sorted_langs = indian_langs + [lang for lang in available_langs if lang not in indian_langs]
    
    target_lang = st.selectbox(
        "🎯 Target language",
        sorted_langs,
        index=0,
        format_func=lambda x: f"🇮🇳 {get_language_name(x)}" if x in indian_langs else get_language_name(x),
        help="Choose the language you want the output in."
    )
    
    # Show language support status
    if is_language_supported(target_lang):
        st.success(f"✅ {get_language_name(target_lang)} is supported")
    else:
        st.error(f"❌ {target_lang} may not be fully supported")

    # Show BERTopic status
    if is_bertopic_available():
        st.success("✅ BERTopic available for intelligent topic detection")
    else:
        st.warning("⚠️ BERTopic not available - using basic topic detection")

    col_start, col_stop, col_reset = st.columns(3)
    with col_start:
        if st.button("🔄 Start Continuous", type="primary", use_container_width=True):
            st.session_state.continuous_mode = True
            st.session_state.record_count = 0
            st.session_state.transcript = "Listening... 🎙 Speak now!"
            st.session_state.translation = "Processing..."
            st.session_state.topic = "Detecting..."
            reset_topic_history()
            st.rerun()
    with col_stop:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.continuous_mode = False
            st.session_state.transcript = "Stopped listening."
            st.session_state.processing = False
            st.rerun()
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            reset_topic_history()
            st.session_state.record_count = 0
            st.session_state.session_history.clear()
            st.success("Session history reset!")
            st.rerun()

    st.info("""
    **For Indian Languages (Hindi, Kannada, Tamil, etc.):**
    - Speak clearly and at normal pace
    - Use common words and phrases
    - Avoid background noise
    - Speak for 3-5 seconds per recording
    - Ensure good microphone quality
    
    **Tips for Better Transcription:**
    - Speak clearly and at moderate pace
    - Avoid speaking too softly or too loudly
    - Minimize background noise
    - Use complete sentences when possible
    """)

# --- Output Display ---
with col2:
    st.header("📝 Live Output")

    # Language detection info
    if st.session_state.detected_language != "Unknown":
        lang_name = get_language_name(st.session_state.detected_language)
        confidence = st.session_state.language_confidence
        st.info(f"🎯 Detected Language: **{lang_name}** (Confidence: {confidence:.2f})")

    st.subheader("Transcript")
    transcript_display = st.empty()
    transcript_display.info(st.session_state.transcript)

    st.subheader(f"Translation ({get_language_name(target_lang)})")
    translation_display = st.empty()
    translation_display.info(st.session_state.translation)

    st.subheader("Detected Topic")
    topic_display = st.empty()
    topic_display.info(st.session_state.topic)

# --- Continuous Mode Logic ---
if st.session_state.continuous_mode:
    if not st.session_state.model_loaded:
        st.warning("⏳ Model still loading... Please wait (this can take 2-3 minutes for first load)")
    else:
        st.success("🔴 LIVE MODE ACTIVE — Speak now!")
        st.info("""
        **📝 How to use:**
        1. Click the microphone button below ⬇️
        2. **Allow microphone access** when your browser asks (important!)
        3. Speak clearly into your microphone
        4. Click the button again to stop recording and process
        5. Your transcription and translation will appear above
        """)
    
    # Audio input
    audio_file = st.audio_input(
        "🎤 Click here to start recording → Speak → Click again to process",
        key="audio_recorder",
        help="Click to start recording. Your browser will ask for microphone permission - please allow it!"
    )
    
    if not audio_file:
        st.info("💡 **Click the microphone button above** to start recording. When you click it, your browser will ask for microphone permission - click 'Allow' to proceed.")
    
    if audio_file:
        audio_bytes = get_audio_bytes(audio_file)
        
        if not audio_bytes:
            st.warning("⚠️ No audio received. Please check your microphone and try again.")
        elif len(audio_bytes) < 10000:
            st.warning("⚠️ Audio too short. Please record for at least 2-3 seconds.")
            st.audio(audio_bytes, format="audio/wav")
        elif st.session_state.processing:
            st.info("⏳ Processing previous recording... Please wait.")
            st.audio(audio_bytes, format="audio/wav")
        else:
            # Process new audio
            audio_hash = get_audio_hash(audio_bytes)
            
            if audio_hash != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = audio_hash
                st.session_state.processing = True
                
                st.audio(audio_bytes, format="audio/wav")
                
                with st.spinner("🔄 Processing your voice..."):
                    try:
                        results = process_audio(audio_bytes, target_lang)
                        
                        st.session_state.transcript = results["transcript"]
                        st.session_state.translation = results["translation"]
                        st.session_state.topic = results["topic"]
                        st.session_state.detected_language = results["detected_language"]
                        st.session_state.language_confidence = results["confidence"]
                        st.session_state.audio_duration = results["audio_duration"]
                        st.session_state.processing_time = results["processing_time"]
                        
                        if (results["transcript"] != "No speech detected" and 
                            not results["transcript"].startswith("ASR error")):
                            
                            st.session_state.record_count += 1
                            
                            # Add to session history
                            history_entry = {
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "transcript": results["transcript"],
                                "translation": results["translation"],
                                "topic": results["topic"],
                                "language": results["detected_language"],
                                "duration": results["audio_duration"],
                                "processing_time": results["processing_time"]
                            }
                            st.session_state.session_history.append(history_entry)
                            
                            # Check translation quality
                            if any(x in results["translation"] for x in ["[Translation", "[Empty", "[Language not"]):
                                st.warning("⚠️ Translation had issues - speaking more clearly might help")
                            else:
                                st.success(f"✅ Recording #{st.session_state.record_count} processed successfully!")
                                
                        elif results["transcript"].startswith("ASR error"):
                            st.error("❌ Speech recognition error occurred")
                        else:
                            st.warning("🎤 No speech detected — try speaking louder and clearer.")
                            
                    except Exception as e:
                        st.error(f"❌ Processing error: {e}")
                        import traceback
                        st.error(f"Details: {traceback.format_exc()}")
                        st.session_state.transcript = f"Error: {e}"
                    finally:
                        st.session_state.processing = False
                        time.sleep(0.5)
                        st.rerun()

    # Live metrics
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    with col_metric1:
        st.metric("Recordings Processed", st.session_state.record_count)
    with col_metric2:
        status = "🔄 Processing" if st.session_state.processing else "🎙 Listening"
        st.metric("Status", status)
    with col_metric3:
        confidence = st.session_state.language_confidence
        st.metric("Lang Confidence", f"{confidence:.2f}")
    with col_metric4:
        processing_time = st.session_state.processing_time
        st.metric("Process Time", f"{processing_time:.1f}s")

else:
    st.info("⏸ Ready — click **Start Continuous** to begin live translation.")

# --- Session History ---
st.markdown("---")
st.header("📊 Session History")

if st.session_state.session_history:
    # Show recent entries
    recent_history = st.session_state.session_history[-5:]  # Last 5 entries
    
    for i, entry in enumerate(reversed(recent_history)):
        with st.expander(f"🎙 Recording {len(st.session_state.session_history)-i} - {entry['timestamp']} ({entry['duration']:.1f}s)"):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**Transcript:** {entry['transcript']}")
            with col2:
                st.write(f"**Translation:** {entry['translation']}")
            with col3:
                st.write(f"**Topic:** {entry['topic']}")
                st.write(f"**Lang:** {get_language_name(entry['language'])}")
                st.write(f"**Processed in:** {entry['processing_time']:.1f}s")
    
    # Clear history button
    if st.button("Clear Session History"):
        st.session_state.session_history.clear()
        st.success("Session history cleared!")
        st.rerun()
else:
    st.info("No recordings yet. Start speaking to see your session history here!")

# --- Sidebar Help ---
st.sidebar.header("🧭 Instructions")
st.sidebar.write("""
**For Indian Languages:**
- **Hindi**: Speak clearly in Hindi
- **Kannada**: Use common Kannada phrases  
- **Tamil**: Speak at normal pace in Tamil
- **Other**: Same principles apply

**Tips for Better Results:**
- Use a good quality microphone
- Speak in a quiet environment
- Speak clearly at moderate pace
- Use 3-5 second recordings
- Avoid background noise

**Troubleshooting:**
- If wrong language detected, speak more clearly
- If translation fails, check internet connection
- Restart app if model loading fails
- Use shorter phrases for better accuracy
""")

st.sidebar.divider()

# System Status
st.sidebar.header("🔧 System Status")

# Model status
if st.session_state.model_loaded:
    st.sidebar.success("✅ ASR Model: Loaded")
elif st.session_state.model_loading:
    st.sidebar.warning("🔄 ASR Model: Loading...")
else:
    st.sidebar.error("❌ ASR Model: Not loaded")

# BERTopic status
if is_bertopic_available():
    st.sidebar.success("✅ Topic Detection: BERTopic Available")
else:
    st.sidebar.warning("⚠️ Topic Detection: Basic Mode")

# Translation status
if st.session_state.translation_enabled:
    st.sidebar.success("✅ Translation: Enabled")
else:
    st.sidebar.info("ℹ️ Translation: Disabled")

# Performance metrics
st.sidebar.divider()
st.sidebar.header("📈 Performance")

if st.session_state.session_history:
    total_audio_time = sum(entry['duration'] for entry in st.session_state.session_history)
    avg_processing_time = np.mean([entry['processing_time'] for entry in st.session_state.session_history])
    
    st.sidebar.metric("Total Audio Processed", f"{total_audio_time:.1f}s")
    st.sidebar.metric("Avg Processing Time", f"{avg_processing_time:.1f}s")
    st.sidebar.metric("Total Recordings", st.session_state.record_count)

st.sidebar.divider()
st.sidebar.caption("⚡ Powered by Faster-Whisper + BERTopic + Deep Translator")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9em;'>"
    "🎙 Real-time Speech Recognition • 🌐 Multi-language Translation • 🧠 AI Topic Detection • "
    "🇮🇳 Indian Language Support"
    "</div>",
    unsafe_allow_html=True
)

# Auto-refresh when processing
if st.session_state.processing:
    time.sleep(0.3)
    st.rerun()