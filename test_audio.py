from google.cloud import speech

def transcribe_model_selection(
    speech_file: str,
    model: str,) -> speech.RecognizeResponse:
    """Transcribe the given audio file synchronously with
    the selected model."""
    client = speech.SpeechClient()

    with open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code="en-US",
        model=model,
    )

    response = client.recognize(config=config, audio=audio)

    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        print("-" * 20)
        print(f"First alternative of result {i}")
        print(f"Transcript: {alternative.transcript}")

    return response


if __name__ == "__main__":
    transcribe_model_selection(speech_file="taylor.wav", model="latest_short")
