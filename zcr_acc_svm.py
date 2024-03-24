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
#from micromlgen import port

def wav_to_ACC_and_ZCR(folder_path):
    acc_list = []
    #zcr_list = []
    # os.listdir gets all files in the specified directory
    count = 0
    for filename in os.listdir(folder_path):
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

#light_rain, medium_rain, etc. will contain the acc_array (and in future the zcr_array as well, concatenated?), after the function returns
#So, these will contain the features
light_rain_features = wav_to_ACC_and_ZCR("microphone-sampling/TestingSamples/lightShower/")
print(light_rain_features.shape)
medium_rain_features = wav_to_ACC_and_ZCR("microphone-sampling/TestingSamples/mediumShower/")
print(medium_rain_features.shape)
heavy_rain_features = wav_to_ACC_and_ZCR("microphone-sampling/TestingSamples/heavyShower/")
print(heavy_rain_features.shape)
heavy_couscous_features = wav_to_ACC_and_ZCR("microphone-sampling/TestingSamples/heavyCouscousHail/")
print(heavy_couscous_features.shape)
light_couscous_features = wav_to_ACC_and_ZCR("microphone-sampling/TestingSamples/lightCouscousHail/")
print(light_couscous_features.shape)
mason_rain_features = wav_to_ACC_and_ZCR("microphone-sampling/TestingSamples/MasonJarRain/")
print(mason_rain_features.shape)
nothing_features = wav_to_ACC_and_ZCR("microphone-sampling/TestingSamples/Nothing/")
print(nothing_features.shape)

# Create Labels for Light, Medium, Heavy, etc.
light_rain_labels = np.full(len(light_rain_features),"light rain")
med_rain_labels = np.full(len(medium_rain_features),"medium rain")
heavy_rain_labels = np.full(len(heavy_rain_features),"heavy rain")
heavy_couscous_labels = np.full(len(heavy_couscous_features),"heavy couscous hail")
light_couscous_labels = np.full(len(light_couscous_features),"light couscous hail")
mason_rain_labels = np.full(len(mason_rain_features),"mason jar rain")
nothing_labels = np.full(len(nothing_features),"nothing")

# Concatenate together the 7 NumPy arrays into one array, which will be the features array
X = np.concatenate([light_rain_features,medium_rain_features,heavy_rain_features,heavy_couscous_features,light_couscous_features,mason_rain_features,nothing_features], axis=0)
# Concatenate together the 7 NumPy LABEL arrays into one array, which will be the target array
Y = np.concatenate([light_rain_labels, med_rain_labels, heavy_rain_labels, heavy_couscous_labels, light_couscous_labels, mason_rain_labels, nothing_labels], axis=0)
df = pd.DataFrame(X)
df['target'] = Y
print(df)

