#Imports
import time
import argparse as ap
import sounddevice as sd
from time import time
from scipy.io.wavfile import write

import tensorflow as tf


def get_audio_from_numpy(indata):
    indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    indata = 2 * ((indata + 32768) / (32767 + 32768)) - 1  # CORRECT normalization between -1 and 1
    indata = tf.squeeze(indata)
    return indata

def get_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s):
    audio_padded = get_audio_from_numpy(filename)
   
    sampling_rate_float32 = tf.cast(downsampling_rate, tf.float32)
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    frame_step = int(frame_step_in_s * sampling_rate_float32)

    spectrogram = stft = tf.signal.stft(
        audio_padded, 
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)

    return spectrogram, downsampling_rate

def is_silence(filename, downsampling_rate, frame_length_in_s, dbFSthres, duration_thres):
    spectrogram, sampling_rate = get_spectrogram(
        filename,
        downsampling_rate,
        frame_length_in_s,
        frame_length_in_s
    )
    dbFS = 20 * tf.math.log(spectrogram + 1.e-6)
    energy = tf.math.reduce_mean(dbFS, axis=1)
    non_silence = energy > dbFSthres
    non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
    non_silence_duration = (non_silence_frames + 1) * frame_length_in_s

    if non_silence_duration > duration_thres:
        return 0
    else:
        return 1

def callback(indata, frames, callback_time, status):
    """This is called (from a separate thread) for each audio block."""
    
    """
    Every 1 second (in parallel with the recording), check if the recorded audio contains speech using the new version of the is_silence method with the hyper-parameters of 1.1.
    If is_silence returns 0, store the audio data on disk using the timestamp as filename, otherwise discard it
    """
    #Variables set from using values found in the first point of the exercise 1 of homework:
    downsampling_rate = 16000
    frame_length_in_s = 0.016
    dbFSthres = -120
    duration_thres = 0.06

    global store_audio
    if store_audio is True:
        if(is_silence(indata, downsampling_rate, frame_length_in_s, dbFSthres, duration_thres) == 0):
            timestamp = time()
            write(f'{timestamp}.wav', sample_rate, indata) #this saves the audio in local

# ------------------- FUNCTIONS UP UNTIL HERE -----------------------------------------------------

#Parser
parser = ap.ArgumentParser(description='You can choose betwenn --host --port --user --password')
parser.add_argument('--device', type=int, help='Insert the device number')
args = parser.parse_args()

#Script parameter most of them fixed 
sample_rate = 16000
length_in_s = 1
device = args.device # passed in the command line
channels = 1
dtype = 'int16'


print ("Start recording")
print("If you want to quit the script press q or Q")
print("If you want to stop the storage of audio press p or P")
store_audio = True

with sd.InputStream(device=device, channels=channels, dtype=dtype, samplerate=sample_rate, blocksize= sample_rate * length_in_s, callback=callback):
    while True:
        key = input()
        if key in ('q', 'Q'):
            print('Stop recording.')
            break
        if key in ('p', 'P'):
            store_audio = not store_audio

