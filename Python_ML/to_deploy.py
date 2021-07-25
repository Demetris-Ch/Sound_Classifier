#This Code is deployed on an Azure server
import torch
from torchaudio import transforms


def preprocess_input(waveform):
    # Code Used for the Preprocessing and prediction of the recorded input.
    signal = torch.FloatTensor(waveform)
    signal = torch.reshape(signal, (1, len(signal)))

    # Ensure that the Recording Only Takes the Latest 2 seconds of the measurement (in 44100 sampling rate)
    signal = signal[:, -88200:]
    spectrogram = spectro_gram(signal, 44100)
    model = torch.load("optimized_torchscript_model.pt")
    spectrogram = spectrogram.unsqueeze(1)
    with torch.no_grad():
        # Get the input features and target labels, and put them on the GPU
        spectrogram_m, spectrogram_s = spectrogram.mean(), spectrogram.std()
        # Normalize the inputs
        input = (spectrogram - spectrogram_m) / spectrogram_s

        # Get predictions
        outputs = model(input)
        pred = torch.max(outputs, 0)

    if int(pred.indices) == 0:
        return 'Silence:'+str(float(pred.values))
    elif int(pred.indices) == 1:
        return 'Speaking:'+str(float(pred.values))
    else:
        return 'Singing:'+str(float(pred.values))

def spectro_gram(sig, sr, n_mels=64, n_fft=1024, hop_len=None):
    # Find Mel Spectrogram
    top_db = 80
    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return spec