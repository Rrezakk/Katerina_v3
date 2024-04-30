import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pvrecorder import PvRecorder
import time
import pvcobra
import pvporcupine
import soundfile as sf
from datetime import datetime, timedelta
import wave
from recognizer import RecognitionClient
from playsound import playsound
import os
from pygame import mixer
import pygame
activation_sound_path = 'activation.mp3'

mixer.init()
mixer.music.load(activation_sound_path)

# Adjustable parameters
frame_length=512
access_key='spA5SDxJ74yX2Yyxvdl8U62OSxRn5c5sUIS38KuC15ulkmXtNTbqgA=='
yandex_key='AQVNzUFYBdDMSf7b1ID--C7J-QRzT5zO1DyHOPx6'
sample_rate = 16000
stft_frame_size = 512
stft_hop_length = 128
max_frames = 100
initial_threshold = 2000
dema_alpha = 0.6

frames_under_continue_threshold_to_stop = 60#so also keep in mind that starting from start frame!
speech_activation_threshold = 46
speech_continue_threshold = 6
wake_word_duration = 5#seconds
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

cobra = pvcobra.create(access_key = access_key)
porcupine = pvporcupine.create(
  access_key = access_key,
  keyword_paths = [r'hey-Alice_en_windows_v3_0_0.ppn']
)
recorder = PvRecorder(frame_length=frame_length)
sample_rate = recorder.sample_rate

recognition_client = RecognitionClient(secret=yandex_key)

wema_weights = np.full(moving_average_window, 1/moving_average_window)
def weighted_moving_average(values):
    global wema_weights
    return np.convolve(values, wema_weights, mode='valid')

record_started = False
time_start = None
time_end = None
pre_buffer = []
main_buffer = []
current_frame = 0
def process_threshold(audio_float,audio_int, trigger_values,wake_word_is_active):
    global record_started, time_start, time_end, pre_buffer, main_buffer, current_frame, sample_rate,speech_activation_threshold
    global frames_under_continue_threshold_to_stop,recognize_options,recognition_client
    
    if not record_started and trigger_values[-1] > speech_activation_threshold and wake_word_is_active:
        wake_word_is_active = False
        record_started = True
        time_start = datetime.now()
        print(f'Record started!: {time_start.strftime("%M:%S.%f")[:-3]}')

        concatenated_array=[]
        if len(pre_buffer) >= samples_before_activation:
            concatenated_array = np.concatenate(pre_buffer[-samples_before_activation:])
        else:
            concatenated_array = np.concatenate(pre_buffer)
        main_buffer.append(concatenated_array)
        print(f'Appended {len(concatenated_array/frame_length)} frames')
        
    if record_started:
        main_buffer.append(audio_int)
        current_frame = current_frame + 1

    # Check if the condition to stop recording is met
    if current_frame >= frames_under_continue_threshold_to_stop and len(trigger_values) >= frames_under_continue_threshold_to_stop and np.mean(trigger_values[-frames_under_continue_threshold_to_stop:]) < speech_continue_threshold and record_started:
        record_started = False
        time_end = datetime.now()
        file_name = rf"recordings\file_{time_end.strftime('%M.%S.%f')[:-3]}.wav"
        print(f'Record ended!: {time_end.strftime("%M:%S.%f")[:-3]} -> {file_name}')

        with wave.open(file_name, 'wb') as wavfile:
            wavfile.setnchannels(1)  # Mono
            wavfile.setsampwidth(2)  # 16-bit
            wavfile.setframerate(sample_rate)
            audio_data = np.concatenate(main_buffer).astype(np.int16)
            wavfile.writeframes(audio_data.tobytes())

        recognized_text = recognition_client.recognize_audio_samples(file_name)
        print(f'recognized: {recognized_text}')
        main_buffer.clear()
        current_frame = 0

    return record_started

hey_alice_is_active = False
hey_alice_last_time = None

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

    
def callback(audio):
    global pre_buffer, rms_values, threshold_values, centroid_values, centroid_diff_values, decibels_values, voice_probability_values, keyword_detection_values, trigger_values
    audio_float = np.array(audio, dtype=np.float32) / 32768.0
    stft_frame = librosa.stft(audio_float, n_fft=stft_frame_size, hop_length=stft_hop_length)

    spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(stft_frame))[0]
    mean_spectral_centroid = np.mean(spectral_centroid)
    rms_value = np.sqrt(np.mean(audio_float**2))
    decibels = 20 * np.log10(rms_value / 0.0002)
    dynamic_threshold = dema_alpha * mean_spectral_centroid + (1 - dema_alpha) * (dema_alpha * mean_spectral_centroid + (1 - dema_alpha) * initial_threshold)
    dynamic_threshold = dynamic_threshold * 0.9
    keyword_detection = porcupine.process(audio)
    voice_probability = cobra.process(audio)*100
    diff_centroid_value = np.clip(dynamic_threshold - mean_spectral_centroid, 0, None)/10
    trigger_value = np.clip((rms_value*0.2+voice_probability*0.8+diff_centroid_value*0.6+decibels*0.4)/2-decibels,0,None)*5

    pre_buffer.append(audio)
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

    #line_rms.set_data(range(len(rms_values)), np.array(rms_values))
    #line_centroid.set_data(range(len(centroid_values)), centroid_values)
    #line_threshold.set_data(range(len(threshold_values)), threshold_values)
    #line_diff_centroid.set_data(range(len(centroid_diff_values)), np.clip(np.array(centroid_diff_values),0,100))

    moving_average = weighted_moving_average(trigger_values)

    #line_trigger.set_data(range(len(trigger_values)), trigger_values)#moving_average
    line_trigger_average.set_data(range(len(moving_average)), moving_average)#moving_average
    line_decibels.set_data(range(len(decibels_values)), decibels_values)
    line_voice_probability.set_data(range(len(voice_probability_values)), voice_probability_values)
    #line_trigger.set_data(range(len(voice_probability_values)), np.clip((np.array(rms_values)*0.2+np.array(voice_probability_values)*0.8+np.array(centroid_diff_values)*0.6+np.array(decibels_values)*0.4)/2-np.array(decibels_values),0,None))
    #line_mean.set_data(range(len(voice_probability_values)), (np.array(rms_values)*0.2+np.array(voice_probability_values)*0.8+np.array(centroid_diff_values)*0.6+np.array(decibels_values)*0.4)/2)

    min_value = 0#min(min(rms_values), min(centroid_values))
    max_value = 120
    ax.set_ylim(-max_value * 0.0001, max_value * 1.25)
    ax.relim()
    ax.autoscale_view() 
    plt.draw()
    plt.pause(0.01)
    wake_word_is_active = process_wake_word(keyword_detection)
    process_threshold(audio_float, audio, moving_average, wake_word_is_active)

fig, ax = plt.subplots()
#line_rms, = ax.plot([], label='RMS', color='r')
#line_centroid, = ax.plot([], label='Centroid')
#line_threshold, = ax.plot([], label='Threshold')
#line_diff_centroid, = ax.plot([], label='Centroid diff',color='y', linestyle='-')
line_decibels, = ax.plot([], label='Decibels', color='b')
line_voice_probability, = ax.plot([], label='Voice', color='g')
#line_mean, = ax.plot([], label='Mean', color='0')
#line_trigger, = ax.plot([], label='Trigger', color='r')
line_speech_threshold, = ax.plot([], label='Activation threshold', color='0')
line_continue_threshold, = ax.plot([], label='Continue threshold', color='0')
line_trigger_average, = ax.plot([], label='Trigger average', color='0.5')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), shadow=True, ncol=3)

plt.show(block=False)

speech_threshold_values = [speech_activation_threshold] * max_frames
continue_threshold_values = [speech_continue_threshold] * max_frames
line_speech_threshold.set_data(range(len(speech_threshold_values)), speech_threshold_values)
line_continue_threshold.set_data(range(len(continue_threshold_values)), continue_threshold_values)

running = True
try:
    recorder.start()
    while running:
        audio = recorder.read()
        callback(audio)
        
except KeyboardInterrupt:
    running = False  # Set the flag to stop the loop
finally:
    recorder.stop()  # Make sure to stop the recorder when the loop ends

