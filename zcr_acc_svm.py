import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import librosa
import librosa.feature
import librosa.display
import IPython.display as ipd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.io import wavfile
from scipy import signal
from micromlgen import port

def wav_to_ACC_and_ZCR(folder_path):
    acc_list = []
    #zcr_list = []
    # os.listdir gets all files in the specified directory
    count = 0
    for filename in os.listdir(folder_path):
        if count < 16:
            full_path = os.path.join(folder_path, filename)
            #plot_zcr(full_path)
            x, sr = librosa.load(full_path)
            #zcr = librosa.feature.zero_crossing_rate(x)
            #zcr_list.append(zcr)
            auto = sm.tsa.acf(x, nlags=2000)
            acc_list.append(auto)
            count += 1

    acc_array = np.array(acc_list)
    #print(acc_array.shape)
    #zcr_array = np.array(zcr_list)
    #print(zcr_array.shape)

    #combined_array = np.concatenate((acc_array, zcr_array))
    return acc_array

def plot_zcr(full_path):
    x, sr = librosa.load(full_path)
    n0 = 6500
    n1 = 7500
    plt.figure(figsize=(14, 5))
    plt.plot(x[n0:n1])
    plt.show()


light_rain = wav_to_ACC_and_ZCR("microphone-sampling/TestingSamples/lightShower/")
print(light_rain.shape)
medium_rain = wav_to_ACC_and_ZCR("microphone-sampling/TestingSamples/mediumShower/")
print(medium_rain.shape)
heavy_rain = wav_to_ACC_and_ZCR("microphone-sampling/TestingSamples/heavyShower/")
print(heavy_rain.shape)
heavy_couscous = wav_to_ACC_and_ZCR("microphone-sampling/TestingSamples/heavyCouscousHail/")
print(heavy_couscous.shape)
light_couscous = wav_to_ACC_and_ZCR("microphone-sampling/TestingSamples/lightCouscousHail/")
print(light_couscous.shape)
mason_rain = wav_to_ACC_and_ZCR("microphone-sampling/TestingSamples/MasonJarRain/")
print(mason_rain.shape)
nothing = wav_to_ACC_and_ZCR("microphone-sampling/TestingSamples/Nothing/")
print(nothing.shape)

# Create Labels for Light, Medium, Heavy
light_rain_labels = np.full(len(light_rain),"light rain")
med_rain_labels = np.full(len(medium_rain),"medium rain")
heavy_rain_labels = np.full(len(heavy_rain),"heavy rain")
heavy_couscous_labels = np.full(len(heavy_couscous),"heavy couscous hail")
light_couscous_labels = np.full(len(light_couscous),"light couscous hail")
mason_rain_labels = np.full(len(mason_rain),"mason jar rain")
nothing_labels = np.full(len(nothing),"nothing")

# Concatenate together the 7 NumPy arrays into one array
X = np.concatenate([light_rain,medium_rain,heavy_rain,heavy_couscous,light_couscous,mason_rain,nothing], axis=0)
df = pd.DataFrame(X)
print(df.head())