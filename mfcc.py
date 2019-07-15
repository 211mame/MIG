import soundfile as sf
import numpy as np

def cut_wave(wave_data, cuttime=0.04):
    center = len(wave_data) / 2
    cuttime = 0.04
    wav = wave_data[int(center - cuttime/2*fs) : int(center + cuttime/2*fs)]
    return wav

def preEmphasis(wave, p=0.97):
    return scipy.signal.lfilter([1.0, -p], 1, wave)

def window(wave):
    hanningWindow = np.hanning(len(wave))
    wave = wave * hanningWindow
    return wave

def fft(wave):
    nfft = 2048
    dft = np.fft.fft(wave, n)
    Adft = np.abs(dft)[:int(nfft/2)]
    Pdft = np.abs(dft)[:int(nfft/2)] ** 2
    return Adft, Pdft

def hz2mel(f):
    return 1127.01048 * np.log(f / 700.0 + 1.0)

def mel2hz(m):
    return 700.0 * (np.exp(m / 1127.01048) - 1.0)


def melFilterBank(spec, fs=8820.0, nfft=2048, numChannels=20):
    fmax = fs / 2
    melmax = hz2mel(fmax)
    nmax = nfft / 2
    df = fs / nfft
    dmel = melmax / (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    fcenters = mel2hz(melcenters)
    indexcenter = np.round(fcenters / df)
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))

    filterbank = np.zeros((int(numChannels), int(nmax)))
    for c in np.arange(0, numChannels):
        increment= 1.0 / (indexcenter[c] - indexstart[c])
        for i in np.arange(indexstart[c], indexcenter[c]):
            i = int(i)
            c = int(c)
            filterbank[c, i] = (i - indexstart[c]) * increment
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexstop[c]):
            i = int(i)
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    mspec = np.log10(np.dot(spec, filterbank.T))
    return mspec, fcenters

def dct(spec, nceps=12):
    spec = scipy.fftpack.realtransforms.dct(spec, type=2, norm="ortho", axis=-1)
    return spec[:nceps]

def main():
    wav, fs = sf.read("a.wav")
    wav = cut_wave(wave_data)
    wav = preEmphasis(wav)
    wav = window(wav)
    A_spec, P_spec = fft(wav)
    A_spec, fcenters = melFilterBank(A_spec)
    P_spec, fcenters = melFilterBank(P_spec)
    A_spec = dct(A_spec)
    P_spec = dct(P_spec)

if __name__ == '__main__':
    main()
