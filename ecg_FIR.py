#FIR
import matplotlib.pyplot as plt
import statistics
import argparse
import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
from scipy.signal import butter, lfilter


def fir_bandpass_filter(numtaps = 1000, lowcut = 0.4, highcut = 60, fs = 360):
    nyq = fs/2
    low = lowcut/nyq
    high = highcut/nyq
    h_ = signal.firwin(numtaps,[low, high], pass_zero = False)
    w, h = signal.freqz(h_, 1, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label=None)
    return h_

def fir_notch_filter(numtaps = 1000, lowcut =49, highcut = 51, fs = 360):
    nyq = fs/2
    low = lowcut/nyq
    high = highcut/nyq
    numtaps |= 1
    h_ = signal.firwin(numtaps,[low, high])
    w, h = signal.freqz(h_, 1, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label=None)
    return h_

def fir_filtered_data(data, h, shift = True):
    y = lfilter(h, [1.0], data) if shift else signal.filtfilt(h, [1.0], data)
    return y

parser = argparse.ArgumentParser(description='...')
parser.add_argument('-order','--order', type = int)
args = parser.parse_args()


path = './dataset/'
csv_path = path + '115.csv'
annotation_path = path + '115annotations.txt'
df = pd.read_csv(csv_path,)
# Get data:
data = df["'MLII'"].values


h1 = fir_bandpass_filter(numtaps = args.order) # num_order is the order of filter. args
bandpass_data = fir_filtered_data(data, h1, shift = False)
h2 = fir_notch_filter(numtaps = args.order)
notch_data = fir_filtered_data(bandpass_data, h2, shift = False)

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

