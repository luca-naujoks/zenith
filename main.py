import os
import wave

import torch
from transformers import pipeline

import pyaudio

from utils.core_llm import model_interaction
from shared.config import recording_file, recording_seconds, cuda_device
from utils.whisper import whisper
from utils.tts import text_to_speech


def main():
    while True:
        keyword_input = input("Press Enter to start 5 Sec Recording/Interaction...\n")
        if keyword_input == "":
            interaction_flow()
        if keyword_input == "exit":
            break
        else:
            interaction_text_flow(keyword_input)

def interaction_flow():
    if os.path.exists(recording_file):
        os.remove(recording_file)
    record_audio()

    transcript = whisper()

    model_response = model_interaction(transcript)

    #text_to_speech(model_response)

    print(model_response)

def interaction_text_flow(input_text: str):
    model_response = model_interaction(input_text)
    # tts(model_response)

    print(model_response)


def record_audio():
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 2
    fs = 44100

    audio = pyaudio.PyAudio()

    print("Recording...")

    stream = audio.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
    frames = []

    for i in range(0, int(fs / chunk * recording_seconds)):
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    print("Finished Recording... Processing")

    wf = wave.open(recording_file, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))

    wf.close()

if __name__ == '__main__':
    main()