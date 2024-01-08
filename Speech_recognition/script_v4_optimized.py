import librosa
import numpy as np
from pvrecorder import PvRecorder
from datetime import datetime, timedelta
import wave
from recognizer import RecognitionClient
from playsound import playsound
import os
import pvporcupine
from pygame import mixer
import pvcobra

# Adjustable parameters
frame_length = 512
sample_rate = 16000
dema_alpha = 0.04
max_frames = 100
initial_threshold = 2000
frames_under_continue_threshold_to_stop = 60
speech_activation_threshold = 46
speech_continue_threshold = 6
wake_word_duration = 5
samples_before_activation = 9
moving_average_window = 7

(
    rms_values,
    threshold_values,
    centroid_values,
    centroid_diff_values,
    decibels_values,
    voice_probability_values,
    keyword_detection_values,
    trigger_values,
    buffer,
) = ([] for _ in range(9))
hey_alice_is_active = False
hey_alice_last_time = None

# Initialize mixer for audio playback
mixer.init()
activation_sound_path = 'activation.mp3'
mixer.music.load(activation_sound_path)

# Initialize recorder and recognition client
recorder = PvRecorder(frame_length=frame_length)
recognition_client = RecognitionClient(secret='AQVNzUFYBdDMSf7b1ID--C7J-QRzT5zO1DyHOPx6')

porcupine = pvporcupine.create(
  access_key = 'spA5SDxJ74yX2Yyxvdl8U62OSxRn5c5sUIS38KuC15ulkmXtNTbqgA==',
  keyword_paths = [r'C:\Users\Admin\Desktop\Speech_recognition\hey-Alice_en_windows_v3_0_0.ppn']
)
cobra = pvcobra.create('spA5SDxJ74yX2Yyxvdl8U62OSxRn5c5sUIS38KuC15ulkmXtNTbqgA==')
# Global variables
record_started = False
time_start = None
time_end = None
pre_buffer = []
main_buffer = []
current_frame = 0

# Functions
def weighted_moving_average(values, window):
    weights = np.full(window, 1/window)
    return np.convolve(values, weights, mode='valid')

def process_threshold(audio_int, trigger_values, wake_word_is_active):
    global record_started, time_start, time_end, pre_buffer, main_buffer, current_frame, sample_rate

    if not record_started and trigger_values[-1] > speech_activation_threshold and wake_word_is_active:
        wake_word_is_active = False
        record_started = True
        time_start = datetime.now()
        print(f'Record started!: {time_start.strftime("%M:%S.%f")[:-3]}')

        main_buffer.extend(pre_buffer[-samples_before_activation:])
        print(f'Appended {len(pre_buffer[-samples_before_activation:])} frames')

    if record_started:
        main_buffer.append(audio_int)
        current_frame += 1

    if current_frame >= frames_under_continue_threshold_to_stop and len(trigger_values) >= frames_under_continue_threshold_to_stop and np.mean(trigger_values[-frames_under_continue_threshold_to_stop:]) < speech_continue_threshold and record_started:
        record_started = False
        time_end = datetime.now()
        file_name = rf"C:\Users\Admin\Desktop\Speech_recognition\recordings\file_{time_end.strftime('%M.%S.%f')[:-3]}.wav"
        print(f'Record ended!: {time_end.strftime("%M:%S.%f")[:-3]} -> {file_name}')

        with wave.open(file_name, 'wb') as wavfile:
            wavfile.setnchannels(1)
            wavfile.setsampwidth(2)
            wavfile.setframerate(sample_rate)
            audio_data = np.concatenate(main_buffer).astype(np.int16)
            wavfile.writeframes(audio_data.tobytes())

        recognized_text = recognition_client.recognize_audio_samples(file_name)
        print(f'recognized: {recognized_text}')
        main_buffer.clear()
        current_frame = 0

    return record_started

def process_wake_word(keyword_detection):
    global hey_alice_is_active, hey_alice_last_time, wake_word_duration

    if keyword_detection != -1:
        hey_alice_is_active = True
        hey_alice_last_time = datetime.now()
        print(f'Hey ALICE!!!!')
        mixer.music.play()

    if hey_alice_is_active and datetime.now() - hey_alice_last_time > timedelta(seconds=wake_word_duration):
        hey_alice_is_active = False

    return hey_alice_is_active

def callback(indata):
    global pre_buffer, rms_values, threshold_values, centroid_values, centroid_diff_values, decibels_values, voice_probability_values, keyword_detection_values, trigger_values

    audio_float = np.array(indata, dtype=np.float32) / 32768.0
    stft_frame = librosa.stft(audio_float, n_fft=512, hop_length=128)

    spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(stft_frame))[0]
    mean_spectral_centroid = np.mean(spectral_centroid)
    rms_value = np.sqrt(np.mean(audio_float**2))
    decibels = 20 * np.log10(rms_value / 0.0002)
    dynamic_threshold = dema_alpha * mean_spectral_centroid + (1 - dema_alpha) * (dema_alpha * mean_spectral_centroid + (1 - dema_alpha) * initial_threshold)
    dynamic_threshold *= 0.9
    keyword_detection = porcupine.process(indata)
    voice_probability = cobra.process(indata) * 100
    diff_centroid_value = np.clip(dynamic_threshold - mean_spectral_centroid, 0, None) / 10
    trigger_value = np.clip((rms_value * 0.2 + voice_probability * 0.8 + diff_centroid_value * 0.6 + decibels * 0.4) / 2 - decibels, 0, None) * 5

    pre_buffer.append(audio_float)
    trigger_values.append(trigger_value)
    rms_values.append(rms_value * 1000)
    threshold_values.append(dynamic_threshold)
    centroid_values.append(mean_spectral_centroid)
    centroid_diff_values.append(diff_centroid_value)
    decibels_values.append(decibels)
    voice_probability_values.append(voice_probability)
    keyword_detection_values.append(keyword_detection)

    if len(pre_buffer) > max_frames:
        pre_buffer = pre_buffer[-max_frames:]
        trigger_values = trigger_values[-max_frames:]
        rms_values = rms_values[-max_frames:]
        threshold_values = threshold_values[-max_frames:]
        centroid_values = centroid_values[-max_frames:]
        centroid_diff_values = centroid_diff_values[-max_frames:]
        decibels_values = decibels_values[-max_frames:]
        voice_probability_values = voice_probability_values[-max_frames:]
        keyword_detection_values = keyword_detection_values[-max_frames:]

    moving_average = weighted_moving_average(trigger_values, moving_average_window)
    wake_word_is_active = process_wake_word(keyword_detection)
    process_threshold(indata, moving_average, wake_word_is_active)

# Main loop
running = True
try:
    recorder.start()
    while running:
        audio_data = recorder.read()
        callback(audio_data)
except KeyboardInterrupt:
    running = False
finally:
    recorder.stop()
