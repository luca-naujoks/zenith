import torch

recording_file = "recording.wav"
recording_seconds = 5
cuda_device = "cuda:0" if torch.cuda.is_available() else "cpu"

src = "/workspace/models/"
# Models
sentiment_model = src + "go_emotions"
core_llm = "google/gemma-7b-it"
text_to_speech_model = src + "Qwen3-TTS-12Hz-1.7B-CustomVoice"
speech_to_text = src + "whisper-small"