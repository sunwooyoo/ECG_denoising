import matplotlib.pyplot as plt
import statistics
import numpy as np
import pandas as pd
import pywt
import sys

path = 'dataset/'
csv_path = path + '115.csv'
annotation_path = path + '115annotations.txt'
df = pd.read_csv(csv_path,)
# Get data:
data = df["'MLII'"].values

# Create wavelet object and define parameters
w = pywt.Wavelet('sym8') 
maxlev = 8 # Override if desired
print("maximum level is " + str(maxlev))

# Decompose into wavelet components
coeffs = pywt.wavedec(data, w, level=maxlev) 

plt.figure(figsize=(12, 10))
for i in range(1, len(coeffs)):
    threshold= statistics.median(np.abs(coeffs[i]/np.linalg.norm(coeffs[i])))/0.6745 * np.sqrt(2*np.log(len(data))/len(data))
    print(threshold)
    coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]), 'soft')
    plt.plot(coeffs[i])


coeffs[0] = np.zeros(coeffs[0].shape)
coeffs[-1] = np.zeros(coeffs[-1].shape)
coeffs[-2] = np.zeros(coeffs[-2].shape)

cA8,d8,d7,d6,d5,d4,d3 ,cD2, cD1 = coeffs


for i in range(0, len(coeffs)):
    
    plt.subplot(maxlev+1, 1, i+1)
    plt.plot(coeffs[i])



datarec = pywt.waverec(coeffs, w) #Multilevel reconstruction using waverec


mintime = 466388
maxtime = 468259

plt.figure(figsize=(20, 10))

plt.plot(data[mintime:maxtime])
plt.xlabel('# of smaples')
plt.ylabel('mV')
plt.title("Raw signal")

plt.plot(datarec[mintime:maxtime])
plt.xlabel('# of samples')
plt.ylabel('mV')
plt.title("Denoised signal using wavelet")

plt.tight_layout()
plt.show()

mintime = 467000-140
maxtime = 467000 +139


