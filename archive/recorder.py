from pvrecorder import PvRecorder

# Parameters for audio recording
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

# Create a recorder instance
recorder = PvRecorder(512)

# Open the recording stream
recorder.start()

print("Recording...")

frames = []

# Record audio data
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = recorder.read()
    frames.append(data)

print("Finished recording.")

# Save audio data to a WAV file
recorder.save(WAVE_OUTPUT_FILENAME, frames, RATE)

print("Audio saved as", WAVE_OUTPUT_FILENAME)
