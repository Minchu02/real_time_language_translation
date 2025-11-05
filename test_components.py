# test_components.py
import streamlit as st
import soundfile as sf
import numpy as np
import io

st.title("Component Testing")

# Test ASR
if st.button("Test ASR Engine"):
    try:
        from asr_engine import transcribe_from_bytes
        
        # Create a simple test audio (silence)
        sample_rate = 16000
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Generate a simple tone for testing
        audio_data = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        wav_bytes = buffer.getvalue()
        
        result = transcribe_from_bytes(wav_bytes)
        st.write(f"ASR Result: {result}")
        
    except Exception as e:
        st.error(f"ASR Test Failed: {e}")

# Test Translation
if st.button("Test Translation"):
    try:
        from translator import translate_text
        test_text = "Hello, how are you?"
        translated = translate_text(test_text, "auto", "hi")
        st.write(f"Original: {test_text}")
        st.write(f"Translated: {translated}")
    except Exception as e:
        st.error(f"Translation Test Failed: {e}")

# Test Topic Detection
if st.button("Test Topic Detection"):
    try:
        from topic_detector import detect_topic
        test_text = "I love programming in Python and building applications."
        topic = detect_topic(test_text)
        st.write(f"Text: {test_text}")
        st.write(f"Detected Topic: {topic}")
    except Exception as e:
        st.error(f"Topic Detection Test Failed: {e}")

