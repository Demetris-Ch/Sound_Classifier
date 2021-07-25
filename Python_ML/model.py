import torchaudio.functional as F
from torch.nn import init
from torch import nn
import torch
from torch import Tensor


class AudioClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Forth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)
        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        # Linear layer
        x = self.lin(x)
        # Final output
        return x


class MelSpectrogram(torch.nn.Module):
    # Reimplementation of MelSpectrogram as nn module for deployment as a model in a torchscript type.
    # This will allow preprocessing to be executed by android studio
    def __init__(self,
                 sample_rate: int = 44100,
                 n_fft: int = 1024,
                 win_length = None,
                 hop_length = None,
                 f_min: float = 0.,
                 f_max = None,
                 pad: int = 0,
                 n_mels: int = 64,
                 window_fn=torch.hann_window,
                 power: float = 2.,
                 normalized: bool = False,
                 wkwargs=None,
                 center: bool = True,
                 pad_mode: str = "reflect",
                 onesided: bool = True,
                 norm=None,
                 mel_scale: str = "htk",
                 return_complex: bool = False,
                 n_stft=None,
                 stype='power',
                 top_db=80.) -> None:

        super(MelSpectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min

        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.return_complex = return_complex

        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)

        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.norm = norm
        self.mel_scale = mel_scale

        n_stft = n_fft // 2 + 1
        fb = torch.empty(0) if n_stft is None else F.create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.norm,
            self.mel_scale)
        self.register_buffer('fb', fb)

        self.stype = stype
        self.top_db = top_db
        self.multiplier = 10.0 if stype == 'power' else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = 0.

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """
        waveform = waveform[:, -88200:]

        specgram = F.spectrogram(
            waveform,
            self.pad,
            self.window,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.power,
            self.normalized,
            self.center,
            self.pad_mode,
            self.onesided,
            self.return_complex,
        )

        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])
        
        if self.fb.numel() == 0:
            tmp_fb = F.create_fb_matrix(specgram.size(1), self.f_min, self.f_max,
                                         self.n_mels, self.sample_rate, self.norm,
                                         self.mel_scale)
            # Attributes cannot be reassigned outside __init__ so workaround
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)
        
        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_specgram = torch.matmul(specgram.transpose(1, 2), self.fb).transpose(1, 2)

        # unpack batch
        mel_specgram = mel_specgram.reshape(shape[:-2] + mel_specgram.shape[-2:])

        return F.amplitude_to_DB(mel_specgram, self.multiplier, self.amin, self.db_multiplier, self.top_db)

    def __prepare_scriptable__(self):
        if self.fb.numel() == 0:
            raise ValueError("n_stft must be provided at construction")
        return self


def get_model():
    # Create the model and put it on the GPU if available
    model = AudioClassifier()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device


def get_spectrogram_model():
    # Create spectrogram preprocessing model
    model = MelSpectrogram()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model
