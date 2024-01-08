import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import time
from scipy import signal
import pvcobra

cobra = pvcobra.create(access_key='spA5SDxJ74yX2Yyxvdl8U62OSxRn5c5sUIS38KuC15ulkmXtNTbqgA==')

# Adjustable parameters
sample_rate = 44100
stft_frame_size = 1024  
stft_hop_length = 512   
buffer_size = -1
desired_primary_buffer = 0.15
desired_frame_duration = 0.05
block_size = int(desired_frame_duration * sample_rate)
primary_buffer_blocks = desired_primary_buffer/desired_frame_duration
initial_threshold = 1900 * 1.2
dema_alpha = 0.08  # Adjust the double exponential moving average for mean_spectral_centroid
max_frames = 50

def callback(indata, frames, time, status, centroid_values, rms_values, threshold_values, centroid_diff_values, decibels_values, voice_probability_values, line_rms, line_centroid, audio_buffer, buffer_size, dynamic_threshold, rms_value):
    if status:
        print(status)
    if any(indata):
        audio_frame = indata.flatten()
        if buffer_size < 0:
            buffer_size = len(audio_frame) * primary_buffer_blocks
        if len(audio_buffer) > buffer_size:
            audio_buffer = audio_buffer[-int(buffer_size):]
        audio_buffer.extend(audio_frame)
        audio = np.array(audio_buffer)
        
         # Process audio frame by frame
        frame_length = 512
        num_frames = len(audio_frame) // frame_length
        frames = [audio_frame[i * frame_length: (i + 1) * frame_length] for i in range(num_frames)]
        pcm_frames = [np.int16(frame * 32767) for frame in frames]

        # Calculate probabilities for each frame using PVCobra
        probabilities = [cobra.process(np.array(pcm_frame, dtype=np.int16)) for pcm_frame in pcm_frames]

        # Calculate mean probability
        mean_probability = np.mean(probabilities)
        voice_probability = mean_probability * 100#%
        print(f'voice: {voice_probability}')
        
        
        stft_frame = librosa.stft(audio, n_fft=stft_frame_size, hop_length=stft_hop_length)
        spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(stft_frame))[0]
        mean_spectral_centroid = np.mean(spectral_centroid)
        rms_value = np.sqrt(np.mean(audio**2))
        decibels = 20 * np.log10(rms_value / 0.0002)
        #dynamic_threshold = alpha * mean_spectral_centroid + (1 - alpha) * dynamic_threshold
        
        dynamic_threshold = dema_alpha * mean_spectral_centroid + (1 - dema_alpha) * (dema_alpha * mean_spectral_centroid + (1 - dema_alpha) * dynamic_threshold)
        dynamic_threshold = dynamic_threshold*0.9
        
        if mean_spectral_centroid < dynamic_threshold:
            print('Speech')
        rms_values.append(rms_value)
        threshold_values.append(dynamic_threshold)
        centroid_values.append(mean_spectral_centroid)
        centroid_diff_values.append(np.clip(dynamic_threshold - mean_spectral_centroid,0, None))
        decibels_values.append(decibels)
        voice_probability_values.append(voice_probability)
        
        if len(decibels_values) > max_frames:
            decibels_values = decibels_values[-max_frames:]
        if len(centroid_values) > max_frames:
            centroid_values = centroid_values[-max_frames:]
        if len(rms_values) > max_frames:
            rms_values = rms_values[-max_frames:]
        if len(threshold_values) > max_frames:
            threshold_values = threshold_values[-max_frames:]
        if len(centroid_diff_values) > max_frames:
            centroid_diff_values = centroid_diff_values[-max_frames:]
        if len(voice_probability_values) > max_frames:
            voice_probability_values = voice_probability_values[-max_frames:]

        line_rms.set_data(range(len(rms_values)), np.array(rms_values) * 10000)
        #line_centroid.set_data(range(len(centroid_values)), centroid_values)
        #line_threshold.set_data(range(len(threshold_values)), threshold_values)
        #line_diff_centroid.set_data(range(len(centroid_diff_values)), centroid_diff_values)
        line_decibels.set_data(range(len(decibels_values)), decibels_values)
        line_voice_probability.set_data(range(len(voice_probability_values)), voice_probability_values)

        min_value = min(min(rms_values), min(centroid_values))
        max_value = 200#max(centroid_diff_values)#max(max(rms_values), max(centroid_values))
        ax.set_ylim(-max_value*0.0001, max_value*1.25)
        ax.relim()
        ax.autoscale_view()
        plt.draw()

fig, ax = plt.subplots()
line_rms, = ax.plot([], label='RMS')  
line_centroid, = ax.plot([], label='Centroid')
line_threshold, = ax.plot([], label='Threshold')
line_diff_centroid, = ax.plot([], label='Centroid diff')
line_decibels, = ax.plot([], label='Decibels')
line_voice_probability, = ax.plot([], label='Voice prob.')

# Function to capture background noise
def capture_background_noise():
    print('Capturing background noises in 3sec...')
    audio_buffer = []
    dynamic_threshold = initial_threshold
    for _ in range(3):  # Capture background noise for 3 seconds
        audio_data = sd.rec(sample_rate, samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait for recording to complete
        audio_buffer.extend(audio_data.flatten())

    audio = np.array(audio_buffer)
    stft_frame = librosa.stft(audio, n_fft=stft_frame_size, hop_length=stft_hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(stft_frame))[0]
    mean_spectral_centroid = np.mean(spectral_centroid)
    rms_value = np.sqrt(np.mean(audio**2))
    
    return rms_value, mean_spectral_centroid, [0] * 50, [mean_spectral_centroid] * 50, [mean_spectral_centroid] * 50, [rms_value] * 50

# Initialize arrays with background noise values
rms_value, dynamic_threshold, centroid_diff_values, threshold_values, centroid_values, rms_values = capture_background_noise()
decibels_values = []
voice_probability_values = []
audio_buffer = []

try:
    stream = sd.InputStream(channels=1, samplerate=sample_rate, dtype='float32', blocksize=block_size, callback=lambda indata, frames, time, status: callback(indata, frames, time, status, centroid_values, rms_values, threshold_values, centroid_diff_values, decibels_values, voice_probability_values, line_rms, line_centroid, audio_buffer, buffer_size, dynamic_threshold,rms_value))

    with stream:
        plt.show()

except KeyboardInterrupt:
    pass
