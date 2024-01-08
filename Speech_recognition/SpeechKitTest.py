from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType

configure_credentials(
   yandex_credentials=creds.YandexCredentials(
      api_key='AQVNzUFYBdDMSf7b1ID--C7J-QRzT5zO1DyHOPx6'
   )
)
def recognize():
   model = model_repository.recognition_model()

   # Set the recognition settings.
   model.model = 'general'
   model.language = 'ru-RU'
   model.audio_processing_type = AudioProcessingType.Full

   # Recognition of speech in the specified audio file and output of results to the console.
   result = model.transcribe_file(r"C:\Users\Admin\Desktop\Speech_recognition\recordings\file_15.40.497.wav")
   for c, res in enumerate(result):
      print('=' * 80)
      print(f'channel: {c}\n\nraw_text:\n{res.raw_text}\n\nnorm_text:\n{res.normalized_text}\n')
      if res.has_utterances():
         print('utterances:')
         for utterance in res.utterances:
            print(utterance)

if __name__ == '__main__':
   recognize()
