from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import librosa
import numpy as np
from scipy.io import wavfile
from scipy import signal
#from micromlgen import port

def wav_to_ACC(folder_path):
    acc_list = []
    #os.listdir gets all files in the specified directory
    for filename in os.listdir(folder_path):
        #Load the audio file (filename) - TODO: Look up what data_a is!
        _, data_a = wavfile.read(filename)
        #Convert data_a to a float
        data_a = np.float32(data_a)

        corr = signal.correlate(data_a, data_a)
        #TODO: Look up
        lags = signal.correlation_lags(len(data_a), len(data_a))

        corr = corr / np.max(corr)

        lag = lags[np.argmax(corr)]
        print(lag, np.max(corr))
        acc_list.append(lag, np.max(corr))






light_rain = wav_to_ACC("/microphone-sampling/TestingSamples/lightShower")
medium_rain = wav_to_ACC("/microphone-sampling/TestingSamples/mediumShower")
heavy_rain = wav_to_ACC("/microphone-sampling/TestingSamples/heavyShower")
heavy_couscous = wav_to_ACC("/microphone-sampling/TestingSamples/heavyCouscousHail")
light_couscous = wav_to_ACC("/microphone-sampling/TestingSamples/lightCouscousHail")
mason_rain = wav_to_ACC("/microphone-sampling/TestingSamples/MasonJarRain")
nothing = wav_to_ACC("/microphone-sampling/TestingSamples/Nothing")

# Create Labels for Light, Medium, Heavy
light_rain_labels = np.full(len(light_rain),"light rain")
med_rain_labels = np.full(len(medium_rain),"medium rain")
heavy_rain_labels = np.full(len(heavy_rain),"heavy rain")
heavy_couscous_labels = np.full(len(heavy_couscous),"heavy couscous hail")
light_couscous_labels = np.full(len(light_couscous),"light couscous hail")
mason_rain_labels = np.full(len(mason_rain),"mason jar rain")
nothing_labels = np.full(len(nothing),"nothing")

# Concatenate together the 10 NumPy arrays into one array
X = np.concatenate([light_rain,medium_rain,heavy_rain,heavy_couscous,light_couscous,mason_rain,nothing], axis=0)