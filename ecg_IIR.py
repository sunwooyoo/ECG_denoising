# IIR
import matplotlib.pyplot as plt
import statistics
import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
from scipy.signal import butter, lfilter

def butter_bandpass_filter(lowcut = 0.4, highcut = 60, fs = 360, order = 2):
    nyq = fs/2
    low = lowcut/nyq
    high = highcut/nyq
    b,a = butter(order, [low, high], btype = 'band')
    w, h = signal.freqz(b,a,worN=2000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label=None)
    return b,a

def notch_pass_filter(center=50, interval = 1, fs = 360, normalized = False):
    nyq = fs/2
    center = center/nyq if normalized else center
    b, a = signal.iirnotch(center, center/interval,fs)
    w, h = signal.freqz(b,a,worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label=None)
    return b,a

def filtered_data(data,b,a):
    y  = lfilter(b,a,data)
    return y

#high pass : 0.4
#low pass : 60 
#notch : 50
#filter order 2

path = 'dataset/'
csv_path = path + '115.csv'
annotation_path = path + '115annotations.txt'
df = pd.read_csv(csv_path,)
# Get data:
data = df["'MLII'"].values


b1,a1 = butter_bandpass_filter()
bandpass_data = filtered_data(data,b1,a1)
b2,a2 = notch_pass_filter()
notch_data = filtered_data(bandpass_data,b2,a2)

mintime = 466388
maxtime = 468259

plt.figure(figsize=(20, 10))

plt.plot(data[mintime:maxtime])
plt.xlabel('# of smaples')
plt.ylabel('mV')
plt.title("Raw signal")

plt.plot(notch_data[mintime:maxtime])
plt.xlabel('# of samples')
plt.ylabel('mV')
plt.title("Denoised signal using filter")

plt.tight_layout()
plt.show()

