import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

from shared.config import text_to_speech_model


def text_to_speech(text: str):
    model = Qwen3TTSModel.from_pretrained(
        text_to_speech_model,
        device_map="cuda:0",
        dtype=torch.float16,
        local_files_only=True,
    )

    wavs, sr = model.generate_custom_voice(
        text=text,
        language="English",
        speaker="Ryan",
    )
    sf.write("output.wav", wavs[0], sr)

#text_to_speech("Hey Luca how are you?")