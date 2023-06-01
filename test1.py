import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar

audio_path = 'D:/wave/code/wavefile.wav'
ipd.Audio(audio_path)

# sr = 16000이 의미하는 것은 1초당 16000개의 데이터를 샘플링 한다는 것입니다. sampling rate=16000
y, sr = librosa.load(audio_path, sr=16000)

print('sr:', sr, ', audio shape:', y.shape)
print('length:', y.shape[0]/float(sr), 'secs')
plt.figure(figsize = (10,5))
librosa.display.waveshow(y, sr=sr)
plt.ylabel("Amplitude")
plt.show()

# Fourier -> Spectrum

fft = np.fft.fft(y)

magnitude = np.abs(fft) 
frequency = np.linspace(0,sr,len(magnitude))

left_spectrum = magnitude[:int(len(magnitude) / 2)]
left_frequency = frequency[:int(len(frequency) / 2)]

plt.figure(figsize = (10,5))
plt.plot(left_frequency, left_spectrum)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.show()

#STFT (Short-Time Fourier Transform)
n_fft = 2048 
hop_length = 512 

stft = librosa.stft(y, n_fft = n_fft, hop_length = hop_length)
spectrogram = np.abs(stft)
print("Spectogram :\n", spectrogram)

plt.figure(figsize = (10,5))
librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.plasma()
plt.show()

#MFCC (Mel Frequency Cepstral Coefficient)Permalink
#mfcc = librosa.feature.mfcc(y, sr=16000, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)


print("MFCC Shape: ", mfcc.shape)
print("MFCC: \n", mfcc)

plt.figure(figsize = (10,5))
librosa.display.specshow(mfcc, sr=16000, hop_length=hop_length, x_axis='time')
plt.xlabel("Time")
plt.ylabel("Frequency")
colorbar(format='%+2.0f dB')
plt.show()