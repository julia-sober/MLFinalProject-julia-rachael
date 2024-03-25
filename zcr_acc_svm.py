import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import os
import librosa
import librosa.feature
import librosa.display
import IPython.display as ipd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from scipy.signal import find_peaks
from scipy.io import wavfile
from scipy import signal
#from micromlgen import port

def wav_to_ACC_and_ZCR(folder_path):
    acc_list = []
    zcr_list = []
    # os.listdir gets all files in the specified directory
    #count = 0
    for filename in os.listdir(folder_path):
        #Librosa/File setup
        full_path = os.path.join(folder_path, filename)
        x, sr = librosa.load(full_path)

        #Zero Crossing Rate
        #zcr = librosa.feature.zero_crossing_rate(x)
        #zcr = zcr.reshape(zcr.shape[1])
        #print(zcr.shape)
        #zcr_list += zcr.tolist()
        #zcr_list.append(zcr)

        zero_crossings = librosa.feature.zero_crossing_rate(x, pad=False)
        #zcr_list.append([sum(num_zero_crossings)])
        # pads zcr array with 0's so we can concatenate it with acc
        fixed_length_zcr = librosa.util.fix_length(zero_crossings[0], size=1000, mode='edge')
        zcr_list.append(fixed_length_zcr)

        #Auto Correlation Coefficient
        acc = sm.tsa.acf(x, nlags=2000)
        acc_list.append(acc)

        #count += 1


    #Use this code block is using num ZCR
    acc_array = np.array(acc_list)
    #print("acc shape: ", acc_array.shape, "acc: ", acc_array)
    zcr_array = np.array(zcr_list)
    #print("zcr shape: ", zcr_array.shape, "zcr: ", zcr_array)
    combined_array_acc_zcr = np.concatenate((acc_array, zcr_array), axis=1)
    #print("combined shape: ", combined_array_acc_zcr.shape, "zcr: ", combined_array_acc_zcr)
    return combined_array_acc_zcr


    '''
    # ZCR
    acc_array = np.array(acc_list)
    print("acc shape: ", acc_array.shape, "acc: ", acc_array)
    zcr_array = np.array(zcr_list)
    print("zcr shape: ", zcr_array.shape, "zcr: ", zcr_array)
    print("acc.shape: ", acc.shape, "zcr.shape: ", zcr.shape)
    combined_array_acc_zcr = np.concatenate((acc_array, zcr_array))
    # print(combined_array)
    return acc_array
    # return zcr_array
    # return combined_array
    '''



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

'''
Note: This was one way I started to do SVM, but stopped to do the below instead, with Leave One Out Cross Validation.
I left this in for now in case we would like to reference it later.
#Beginning ML modeling - TODO: what do we want test_size to be?
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
#TODO: Test with different kernels? (look at SVC scikit learn page where it says how to tune SVC's hyperparams)
svc_object = SVC(kernel = 'linear')
svc_object.fit(x_train, y_train)
#predicted_svc = svc_object.predict(x_test)
#How to score?
'''

svc_object = SVC(kernel='linear')

loo = LeaveOneOut()
accuracies = []
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    svc_object.fit(X_train, Y_train)
    accuracy = svc_object.score(X_test, Y_test)
    accuracies.append(accuracy)

mean_accuracy = np.mean(accuracies)
print("Mean Accuracy (SVM): ", mean_accuracy)

# SVM without LOO
svc_object = SVC(kernel='linear')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
svc_object.fit(X_train, Y_train)
accuracy = svc_object.score(X_test, Y_test)
print("Accuracy (SVM no LOO):", accuracy)


# IMPLEMENTING RANDOM FOREST
loo = LeaveOneOut()
accuracies_rf = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracies_rf.append(accuracy)

mean_accuracy = np.mean(accuracies_rf)
print("Mean Accuracy (RF): ", mean_accuracy)

# Random Forest without leave one out
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy (RF no LOO):", accuracy)
