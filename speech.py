import gradio as gr
import torch
import collections
from transformers import pipeline, MarianMTModel, MarianTokenizer
from TTS.api import TTS
from TTS.utils import radam

# -------------------------
# 1. Setup safe globals
# -------------------------
torch.serialization.add_safe_globals([collections.defaultdict, radam.RAdam])
_old_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _old_load(*args, **kwargs)
torch.load = _patched_load

# ASR - Speech to Text (English)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# -------------------------
# 2. Define supported languages & models
# -------------------------
LANGUAGES = {
    "Spanish": "tts_models/es/mai/tacotron2-DDC",
    "French": "tts_models/fr/mai/tacotron2-DDC",
    "German": "tts_models/de/thorsten/tacotron2-DDC",
    "Italian": "tts_models/it/mai/tacotron2-DDC",
    "English": "tts_models/en/ljspeech/vits--neon"
}

TRANSLATION_MODELS = {
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Italian": "Helsinki-NLP/opus-mt-en-it",
    "English": None,  # English → English
}

# -------------------------
# 3. Define Pipeline
# -------------------------
def translate_speech(audio_file, target_language):
    try:
        # Step 1: Speech-to-Text
        asr_result = asr(audio_file)
        source_text = asr_result["text"]

        # Step 2: Translate (if not English)
        if target_language != "English":
            model_name = TRANSLATION_MODELS[target_language]
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            inputs = tokenizer(source_text, return_tensors="pt", padding=True)
            translated_tokens = model.generate(**inputs)
            translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        else:
            translated_text = source_text

        # Step 3: Text-to-Speech
        tts_model_name = LANGUAGES[target_language]
        tts = TTS(model_name=tts_model_name, progress_bar=False, gpu=torch.cuda.is_available())
        output_audio = "translated_output.wav"
        tts.tts_to_file(text=translated_text, file_path=output_audio)

        return translated_text, output_audio

    except Exception as e:
        return f"⚠️ Error: {str(e)}", None

# -------------------------
# 4. Gradio Interface
# -------------------------
interface = gr.Interface(
    fn=translate_speech,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎤 Speak or Upload Audio"),
        gr.Dropdown(choices=list(LANGUAGES.keys()), label="🌐 Select Output Language")
    ],
    outputs=[
        gr.Textbox(label="📝 Translated Text"),
        gr.Audio(label="🔊 Translated Speech")
    ],
    title="🌍 Voice-to-Voice Real-Time Translator",
    description="Speak in English → Get translation in your selected language (text + speech)."
)

# Launch App
if __name__ == "__main__":
    interface.launch(share=True)
