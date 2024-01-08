import os
import torch

# Set device and number of threads
device = torch.device('cpu')
torch.set_num_threads(4)

# Path to the pre-trained model file
local_file = r'C:\Users\Admin\Desktop\tts_model.pt'

# Check if the model file exists
if not os.path.isfile(local_file):
    # Download the model if it doesn't exist
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt',
                                   local_file)

try:
    # Load the model using PackageImporter
    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    
    # Move the model to the specified device
    model.to(device)
    
    # Example text for TTS
    example_text = 'В недрах тундры выдры в г+етрах т+ырят в вёдра ядра кедров.'
    
    # Sample rate for audio output
    sample_rate = 48000
    
    # Speaker for TTS
    speaker = 'xenia'
    
    # Generate audio from the example text using the model
    audio_paths = model.save_wav(text=example_text,
                                 speaker=speaker,
                                 sample_rate=sample_rate)
    
    # Print the paths to the generated audio files
    print("Audio paths:", audio_paths)

except Exception as e:
    # Print any exceptions that occur during loading or inference
    print("An error occurred:", e)


import torch

model, example_texts, languages, punct, apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                  model='silero_te')

input_text = input('Enter input text\n')
apply_te(input_text, lan='en')


import os
import torch

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/denoise_models/sns_latest.jit',
                                   local_file)  

model = torch.jit.load(local_file)
torch._C._jit_set_profiling_mode(False) 
torch.set_grad_enabled(False)
model.to(device)

a = torch.rand((1, 48000))
a = a.to(device)
out = model(a)