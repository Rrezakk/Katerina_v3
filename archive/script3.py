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

# Adjustable parameters
frame_length=512
access_key='spA5SDxJ74yX2Yyxvdl8U62OSxRn5c5sUIS38KuC15ulkmXtNTbqgA=='
sample_rate = 16000
stft_frame_size = 512
stft_hop_length = 128
dema_alpha = 0.04  # Adjust the double exponential moving average for mean_spectral_centroid
max_frames = 100
initial_threshold = 2000
frames_under_continue_threshold_to_stop = 60#so also keep in mind that starting from start frame!

speech_activation_threshold = 46
speech_continue_threshold = 6
wake_word_duration = 5#seconds
samples_before_activation = 8

moving_average_window = 8

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
  keyword_paths = [r'C:\Users\Admin\Desktop\hey-Alice_en_windows_v3_0_0.ppn']
)
recorder = PvRecorder(frame_length=frame_length)
sample_rate = recorder.sample_rate
print(f'rec: {recorder.sample_rate}')
#0 we have event that means "hey alice said, then we wait for 5 seconds at least and put flag off" when the flag is on, we are able to rise event of trigger on (step1)
#1 start trigger if value is above the threshold
#2 we start appending values from mic to buffer array of arrays(later we will stream to SpeechKit)
#3 we record until end condition (last mean n values below continue treshold)


def find_start_of_increase(arr, threshold, n, m):
    #n is above, m is below
    print(f'{len(arr)} len')
    for i in range(len(arr) - 1, n + m - 2, -1):
        window_mean = np.mean(arr[i - n + 1:i + 1])  # Calculate mean of last n values
        if window_mean > threshold and np.mean(arr[i - n - m + 1:i - n + 1]) < threshold:
            return i - n + 1  # Return the index where the condition is met
    return -1  # Return -1 if condition is not met
def weighted_moving_average(trigger_values, moving_average_window):
        #weights = np.arange(moving_average_window, 0, -1)  # Generate weights from moving_average_window down to 1
        #weights = weights / np.sum(weights)  # Normalize weights to sum to 1

    weights = [1/moving_average_window] * moving_average_window

    if len(trigger_values) >= moving_average_window:
        moving_average = np.convolve(trigger_values, weights, mode='valid')
    else:
        # Handle the case when trigger_values is empty or not enough values for the moving average
        if trigger_values:
            moving_average = np.array([trigger_values[-1]])  # Set moving_average to the last trigger value
        else:
            moving_average = np.array([0])  # Set a default value of 0 if trigger_values is empty

    return moving_average


record_started = False
time_start = None
time_end = None
pre_buffer = []
main_buffer = []
current_frame = 0
def process_threshold(audio_float,audio_int, trigger_values,wake_word_is_active):
    global record_started
    global time_start
    global time_end
    global pre_buffer
    global main_buffer
    global speech_activation_threshold
    global frames_under_continue_threshold_to_stop
    global current_frame

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
        file_name = rf"C:\Users\Admin\Desktop\recordings\file_{time_end.strftime('%M.%S.%f')[:-3]}.wav"
        print(f'Record ended!: {time_end.strftime("%M:%S.%f")[:-3]} -> {file_name}')
        #sf.write(file_name, np.concatenate(main_buffer), sample_rate)

        # Save audio data using wave module
        with wave.open(file_name, 'wb') as wavfile:
        # Set WAV file parameters
            wavfile.setnchannels(1)  # Mono
            wavfile.setsampwidth(2)  # 16-bit
            wavfile.setframerate(sample_rate)
            # Convert and write audio data from main_buffer to the WAV file
            audio_data = np.concatenate(main_buffer)
            # Ensure the data type is int16
            audio_data = audio_data.astype(np.int16)
            # Write the audio data to the WAV file
            wavfile.writeframes(audio_data.tobytes())

        main_buffer = []
        current_frame = 0

    return record_started
hey_alice_is_active = False
hey_alice_last_time = None

def process_wake_word(keyword_detection):
    global hey_alice_is_active
    global hey_alice_last_time
    global wake_word_duration

    if keyword_detection != -1:
        hey_alice_is_active = True
        hey_alice_last_time = datetime.now()
        print(f'Hey ALICE!!!!')

    if hey_alice_is_active and datetime.now() - hey_alice_last_time > timedelta(seconds=wake_word_duration):
        hey_alice_is_active = False
    return hey_alice_is_active

    
def callback(indata, centroid_values, rms_values, threshold_values, centroid_diff_values, decibels_values, voice_probability_values, keyword_detection_values, trigger_values):
    global pre_buffer

    audio = np.array(indata)
    audio_float = audio.astype(np.float32) / 32768.0

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

    #if(speech_activation_threshold<trigger_value):
        #print(f'Speech!: {trigger_value} started at: {find_start_of_increase(trigger_values,speech_continue_threshold,5,25)}')
        
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
    if len(trigger_values) > max_frames:
        trigger_values = trigger_values[-max_frames:]
    if len(rms_values) > max_frames:
        rms_values = rms_values[-max_frames:]
    if len(threshold_values) > max_frames:
        threshold_values = threshold_values[-max_frames:]
    if len(centroid_values) > max_frames:
        centroid_values = centroid_values[-max_frames:]
    if len(centroid_diff_values) > max_frames:
        centroid_diff_values = centroid_diff_values[-max_frames:]
    if len(decibels_values) > max_frames:
        decibels_values = decibels_values[-max_frames:]
    if len(voice_probability_values) > max_frames:
        voice_probability_values = voice_probability_values[-max_frames:]
    if len(keyword_detection_values) > max_frames:
        keyword_detection_values = keyword_detection_values[-max_frames:]

    #line_rms.set_data(range(len(rms_values)), np.array(rms_values))
    #line_centroid.set_data(range(len(centroid_values)), centroid_values)
    #line_threshold.set_data(range(len(threshold_values)), threshold_values)
    #line_diff_centroid.set_data(range(len(centroid_diff_values)), np.clip(np.array(centroid_diff_values),0,100))


    #if len(trigger_values) >= moving_average_window:
    #    moving_average = np.convolve(trigger_values, np.ones(moving_average_window) / moving_average_window, mode='valid')
    #else:
    # Handle the case when trigger_values is empty
    #    if trigger_values:
    #        moving_average = np.array([trigger_values[-1]])  # Set moving_average to the last trigger value
    #    else:
    #        moving_average = np.array([0])  # Set a default value of 0 if trigger_values is empty

    moving_average = weighted_moving_average(trigger_values, moving_average_window)

    line_trigger.set_data(range(len(moving_average)), moving_average)
    line_decibels.set_data(range(len(decibels_values)), decibels_values)
    line_voice_probability.set_data(range(len(voice_probability_values)), voice_probability_values)
    #line_trigger.set_data(range(len(voice_probability_values)), np.clip((np.array(rms_values)*0.2+np.array(voice_probability_values)*0.8+np.array(centroid_diff_values)*0.6+np.array(decibels_values)*0.4)/2-np.array(decibels_values),0,None))
    #line_mean.set_data(range(len(voice_probability_values)), (np.array(rms_values)*0.2+np.array(voice_probability_values)*0.8+np.array(centroid_diff_values)*0.6+np.array(decibels_values)*0.4)/2)

    min_value = min(min(rms_values), min(centroid_values))
    max_value = 120
    ax.set_ylim(-max_value * 0.0001, max_value * 1.25)
    ax.relim()
    ax.autoscale_view() 

    wake_word_is_active = process_wake_word(keyword_detection)
    process_threshold(audio_float, audio, trigger_values, wake_word_is_active)

fig, ax = plt.subplots()
#line_rms, = ax.plot([], label='RMS', color='r')
#line_centroid, = ax.plot([], label='Centroid')
#line_threshold, = ax.plot([], label='Threshold')
#line_diff_centroid, = ax.plot([], label='Centroid diff',color='y', linestyle='-')
line_decibels, = ax.plot([], label='Decibels', color='b')
line_voice_probability, = ax.plot([], label='Voice', color='g')
line_mean, = ax.plot([], label='Mean', color='0')
line_trigger, = ax.plot([], label='Trigger', color='r')
line_speech_threshold, = ax.plot([], label='Activation threshold', color='0')
line_continue_threshold, = ax.plot([], label='Continue threshold', color='0')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), shadow=True, ncol=3)
# Initialize arrays


#matplotlib.use('Qt5Agg')
plt.show(block=False)
running = True
speech_threshold_values = [speech_activation_threshold] * max_frames
continue_threshold_values = [speech_continue_threshold] * max_frames
line_speech_threshold.set_data(range(len(speech_threshold_values)), speech_threshold_values)
line_continue_threshold.set_data(range(len(continue_threshold_values)), continue_threshold_values)
try:
    recorder.start()
    while running:
        audio_data = recorder.read()
        callback(audio_data, centroid_values, rms_values, threshold_values, centroid_diff_values, decibels_values, voice_probability_values, keyword_detection_values, trigger_values)
        plt.draw()
        plt.pause(0.01)
except KeyboardInterrupt:
    running = False  # Set the flag to stop the loop
finally:
    recorder.stop()  # Make sure to stop the recorder when the loop ends

