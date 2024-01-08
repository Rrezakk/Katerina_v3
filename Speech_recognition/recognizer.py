from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType

import numpy as np

class RecognitionClient:
    def __init__(self, secret):
        configure_credentials(
   yandex_credentials=creds.YandexCredentials(
      api_key=secret
   )
)

    def recognize_audio_samples(self, file):
        # Perform recognition
        model = model_repository.recognition_model()

        model.model = 'general'
        model.language = 'ru-RU'
        model.audio_processing_type = AudioProcessingType.Full

        result = model.transcribe_file(file)
        #for c, res in enumerate(result):
        #    print('=' * 80)
        #    print(f'channel: {c}\n\nraw_text:\n{res.raw_text}\n\nnorm_text:\n{res.normalized_text}\n')
        #    if res.has_utterances():
        #        print('utterances:')
        #        for utterance in res.utterances:
        #            print(utterance)
        raw_texts = []
        for res in result:
            raw_texts.append(res.raw_text)
        concatenated_text = '\n'.join(raw_texts)
        return concatenated_text