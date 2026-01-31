from transformers import pipeline

from shared.config import cuda_device, sentiment_model


def get_emotion_from_text(text: str):
    classifier = pipeline(task="text-classification", model=sentiment_model, top_k=None , device=cuda_device)

    model_outputs = classifier(text)
    emotion_table = model_outputs[0]

    top_emotion = max(emotion_table, key=lambda x: x["score"])

    return {"label": top_emotion['label'], "score": top_emotion['score']}
