import torch
from transformers import pipeline

from shared.config import cuda_device, core_llm


def model_interaction(transcript: str):
    pipe = pipeline(
        "text-generation",
        model=core_llm,
        dtype=torch.float16,
        device=cuda_device
    )
    messages = [
        {"role": "user", "content": transcript},
    ]

    return pipe(messages)