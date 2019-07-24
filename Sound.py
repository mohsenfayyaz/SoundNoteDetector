from scipy.io import wavfile  # scipy library to read wav files
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft  # fourier transform

AudioName = "sounds/c6.wav"  # Audio File
fs, Audiodata = wavfile.read(AudioName)
# Plot the audio signal in time

print(fs)

# plt.plot(Audiodata)
# plt.title('Audio signal in time', size=16)

# spectrum


# n = len(Audiodata)
# AudioFreq = fft(Audiodata)
# AudioFreq = AudioFreq[0:int(np.ceil((n + 1) / 2.0))]  # Half of the spectrum
# MagFreq = np.abs(AudioFreq)  # Magnitude
# MagFreq /= float(n)
# # power spectrum
# MagFreq **= 2
# if n % 2 > 0:  # ffte odd
#     MagFreq[1:len(MagFreq)] = MagFreq[1:len(MagFreq)] * 2
# else:  # fft even
#     MagFreq[1:len(MagFreq) - 1] = MagFreq[1:len(MagFreq) - 1] * 2
#
# plt.figure()
# freqAxis = np.arange(0, int(np.ceil((n + 1) / 2.0)), 1.0) * (fs / n);
# plt.plot(freqAxis / 1000.0, 10 * MagFreq)  # Power spectrum
# plt.xlabel('Frequency (kHz)');
# plt.ylabel('Power spectrum (dB)');

# Spectrogram


N = 2048  # Number of point in the fft
print(len(Audiodata))
plt.plot(Audiodata)

f, t, Sxx = signal.spectrogram(Audiodata, fs, nfft=N)
print(max(Sxx[0]))
print(len(t))

maxMag = 0
maxFreq = 0
for i in range(len(Sxx)):
    if Sxx[i][60] > maxMag:
        maxMag = Sxx[i][60]
        maxFreq = f[i]

print(maxFreq, maxMag, t[60])
print( 12*np.log2(maxFreq/440) + 49 )

plt.figure()
plt.pcolormesh(t, f, 10 * Sxx)  # dB spectrogram
plt.pcolormesh(t, f,Sxx) # Lineal spectrogram
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [seg]')
plt.title('Spectrogram with scipy.signal', size=16);
plt.show()
