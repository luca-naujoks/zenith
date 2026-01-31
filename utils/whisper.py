from transformers import pipeline
import time

from shared.config import recording_file, cuda_device, speech_to_text


def whisper():
        pipe = pipeline(
            "automatic-speech-recognition",
            model=speech_to_text,
            ignore_warning=True,
            device=cuda_device,
        )

        # ---- start timer ----
        start_time = time.perf_counter()

        transcript = pipe(recording_file, stride_length_s=5, batch_size=8)["text"]

        # ---- end timer ----
        elapsed = time.perf_counter() - start_time
        print(f"Transcription time: {elapsed:.2f} seconds")

        return transcript
