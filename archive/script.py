import pvcobra
import sounddevice as sd
import numpy as np
import time
import matplotlib.pyplot as plt

# Set your Cobra access key
cobra = pvcobra.create(access_key='spA5SDxJ74yX2Yyxvdl8U62OSxRn5c5sUIS38KuC15ulkmXtNTbqgA==')

# Adjustable parameters
sample_rate = 44100  # Adjust as needed
chunk_size = 256  # Adjust as needed

# Function to capture background noise and set initial threshold and multiplier
def initialize_threshold():
    print("Capturing background noise for 5 seconds...")

    background_audio_frames = []
    start_time = time.time()

    while time.time() - start_time < 5:
        audio_frame = sd.rec(chunk_size, samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        background_audio_frames.append(audio_frame)

    background_audio_frames = np.concatenate(background_audio_frames)
    background_rms = np.sqrt(np.mean(np.square(background_audio_frames)))

    initial_threshold = background_rms
    threshold_multiplier = 1.3  # You can adjust this multiplier based on your needs

    print(f'Initial threshold: {initial_threshold}, Threshold multiplier: {threshold_multiplier}')

    return initial_threshold, threshold_multiplier

# Set initial threshold and threshold multiplier
initial_threshold, threshold_multiplier = initialize_threshold()

# Initialize plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line_rms, = ax.plot([], [], label='RMS')
line_threshold, = ax.plot([], [], label='Threshold')
ax.legend()

def update_plot(rms_values, threshold_values):
    max_frames = 50  # Maximum number of frames to display

    if len(rms_values) > max_frames:
        rms_values = rms_values[-max_frames:]
    if len(threshold_values) > max_frames:
        threshold_values = threshold_values[-max_frames:]

    if len(rms_values) != len(threshold_values) or len(rms_values) == 0:
        return  # Skip updating if lengths are different or empty

    line_rms.set_data(range(len(rms_values)), rms_values)
    line_threshold.set_data(range(len(threshold_values)), threshold_values)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)



try:
    rms_values = []
    threshold_values = []

    while True:
        # Capture audio frame using sounddevice
        audio_frame = sd.rec(chunk_size, samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()

        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(np.square(audio_frame)))

        # Adjust the threshold dynamically based on RMS
        threshold = initial_threshold + rms * threshold_multiplier

        rms_values.append(rms)
        threshold_values.append(threshold)

        # Update the plot
        update_plot(rms_values, threshold_values)

        # If the amplitude is below the threshold, return None
        if rms < threshold:
            print('below threshold')
            continue

        # Process the audio frame only if speech is detected
        voice_probability = cobra.process(audio_frame.tobytes())

        if voice_probability > 0.5:  # Adjust the threshold as needed
            print(f'voice: {voice_probability}')
        else:
            print('Speech not detected.')

except KeyboardInterrupt:
    # Handle keyboard interrupt (Ctrl+C) to gracefully delete the Cobra instance
    pass

finally:
    plt.ioff()  # Turn off interactive mode
    cobra.delete()
