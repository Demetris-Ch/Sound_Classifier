import numpy as np
import os
import pandas as pd
import glob
import torch
import torchaudio
from torchaudio import transforms


def preprocess_all_data(songs, audio_channels, new_sr, duration_ms, min_avg):
    zero = []
    df = pd.DataFrame(columns=['File_ID', 'Class'])

    sing_signal_list = []
    speak_signal_list = []
    silence_signal_list = []
    for song in songs:
        path = './nus_smc_corpus_48/'+song+'/sing/'
        for filename in glob.glob(os.path.join(path, '*.wav')):
            df = df.append({'File_ID': filename, 'Class': 'Sing'}, ignore_index=True)
            df = df.append({'File_ID': filename.replace('sing', 'read'), 'Class': 'Speak'}, ignore_index=True)

            # Obtain Unprocessed audio signals
            # 1.Sing Signals
            sing_sig, sing_sr = open_audio(filename)
            # 2.Speak Signals
            speak_sig, speak_sr = open_audio(filename.replace("sing", "read"))
            # Preprocess to make all audios as Monophonic
            # 1.Sing Signals
            sing_sig, sing_sr = re_channel(sing_sig, sing_sr, audio_channels)
            # 2.Speak Signals
            speak_sig, speak_sr = re_channel(speak_sig, speak_sr, audio_channels)
            # Preprocess to set same sampling rate for all audio signals
            sing_sig = resample(sing_sig, sing_sr, new_sr)
            speak_sig = resample(speak_sig, speak_sr, new_sr)
            # Preprocess to set same duration for all audio signals, also create silence data
            # for signals with average less than min_avg
            sing_temp_list, sil_temp_list1 = sig_split(sing_sig, new_sr, duration_ms, min_avg)
            speak_temp_list, sil_temp_list2 = sig_split(speak_sig, new_sr, duration_ms, min_avg)
            sing_signal_list = sing_signal_list + sing_temp_list
            speak_signal_list = speak_signal_list + speak_temp_list
            silence_signal_list = silence_signal_list + sil_temp_list1 + sil_temp_list2

    # Create some additional silence signals (double the current number with zero signals)
    silence_signal_list += silence_sampling(audio_channels, new_sr, duration_ms, len(silence_signal_list))

    # Compute MEL spectrogram for all signals
    sing_spectrograms = []
    for signal in sing_signal_list:
        sing_spectrograms.append(spectro_gram(signal, new_sr))
    speak_spectrograms = []
    for signal in speak_signal_list:
        speak_spectrograms.append(spectro_gram(signal, new_sr))
    silence_spectrograms = []
    for signal in silence_signal_list:
        silence_spectrograms.append(spectro_gram(signal, new_sr))

    # Augment Spectrogram
    augmented_sing_spectrograms = []
    for signal in sing_spectrograms:
        augmented_sing_spectrograms.append(signal)
        augmented_sing_spectrograms.append(spectro_augment(signal))
    augmented_speak_spectrograms = []
    for signal in speak_spectrograms:
        augmented_speak_spectrograms.append(signal)
        augmented_speak_spectrograms.append(spectro_augment(signal))
    augmented_silence_spectrograms = []
    for signal in silence_spectrograms:
        augmented_silence_spectrograms.append(signal)
        augmented_silence_spectrograms.append(spectro_augment(signal))
    print(f"Amount of Obtained Data")
    print(f" - Sing Data: {len(augmented_sing_spectrograms)}")
    print(f" - Speak Data: {len(augmented_speak_spectrograms)}")
    print(f" - Silence Data: {len(augmented_silence_spectrograms)}")

    df = pd.DataFrame(columns=['Spectrogram', 'Class'])
    for signal in augmented_sing_spectrograms:
        df = df.append({'Spectrogram': signal, 'Class': 2}, ignore_index=True)
    for signal in augmented_speak_spectrograms:
        df = df.append({'Spectrogram': signal, 'Class': 1}, ignore_index=True)
    for signal in augmented_silence_spectrograms:
        df = df.append({'Spectrogram': signal, 'Class': 0}, ignore_index=True)
    return df


def open_audio(audio_file):
    # Read Audio Files
    sig, sr = torchaudio.load(audio_file.replace("\\", '/'))
    return sig, sr


def re_channel(sig, sr, new_channel):

    if sig.shape[0] == new_channel:
        # Nothing to do
        return sig, sr

    if new_channel == 1:
        # Convert from stereo to mono by selecting only the first channel
        re_sig = sig[:1, :]
    else:
        # Convert from mono to stereo by duplicating the first channel
        re_sig = torch.cat([sig, sig])

    return re_sig, sr


def resample(sig, sr, new_sr):
    # Change Sampling Rate
    if sr == new_sr:
        # Nothing to do
        return sig

    num_channels = sig.shape[0]
    # Resample first channel
    re_sig = torchaudio.transforms.Resample(sr, new_sr)(sig[:1, :])
    if num_channels > 1:
        # Resample the second channel and merge both channels
        re_two = torchaudio.transforms.Resample(sr, new_sr)(sig[1:, :])
        re_sig = torch.cat([re_sig, re_two])

    return re_sig


def sig_split(sig, sr, dur_ms, min_avg):
    # Split audio waveform for every 2 seconds
    num_rows, sig_len = sig.shape
    sig = sig[:, -int(sig_len/2):]
    num_rows, sig_len = sig.shape
    dur_len = int(sr * dur_ms / 1000)
    if sig_len > dur_len:
        # Truncate the signal to the given length
        list_of_sig = []
        list_of_silence = []
        for i in range(0, int(sig_len/dur_len)):
            # If the audio part is mostly silent, consider it as silence class
            if torch.mean(torch.abs(sig[:, i*dur_len: (i+1)*dur_len])) >= min_avg:
                list_of_sig.append(sig[:, i*dur_len: (i+1)*dur_len])
            else:
                list_of_silence.append(sig[:, i*dur_len: (i+1)*dur_len])
    elif sig_len < dur_len:
        list_of_silence = []
        rng = np.random.RandomState(42)
        # Length of padding to add at the beginning and end of the signal
        pad_begin_len = rng.randint(0, dur_len - sig_len)
        pad_end_len = dur_len - sig_len - pad_begin_len

        # Pad with 0s
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))
        list_of_sig = [torch.cat((pad_begin, sig, pad_end), 1)]

    return list_of_sig, list_of_silence


def silence_sampling(channels, sr, dur_ms, no_of_samples):
    # Make more samples for silent signals as zeros, to increase the confidence on low amplitude signals as silent
    dur_len = int(sr * dur_ms / 1000)
    signal = torch.zeros((channels, dur_len))
    signals = []
    for i in range(0, no_of_samples):
        signals.append(signal)
    return signals


def spectro_gram(sig, sr, n_mels=64, n_fft=1024, hop_len=None):
    # Find Mel Spectrogram
    top_db = 80
    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return spec


def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec

