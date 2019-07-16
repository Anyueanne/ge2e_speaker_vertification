# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 11:31
# @Author  : Anyue
# @FileName: preprocess.py
# @Software: PyCharm

import os
import glob

from tqdm import tqdm
import librosa
import numpy as np

from hparams import hparam as hp

audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))


def save_spectrogram_tisv():
    print('start text independent feature extraction')
    os.makedirs(hp.data.train_path, exist_ok=True)
    os.makedirs(hp.data.test_path, exist_ok=True)

    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr  # low bound of utterance length
    total_speaker_num = len(audio_path)
    train_speaker_num = total_speaker_num // 10 * 9  # 90% train & 10% test
    print('total speaker number: {}'.format(total_speaker_num))
    print('train: {}, test: {}'.format(train_speaker_num, total_speaker_num - train_speaker_num))
    for index, folder in tqdm(enumerate(audio_path)):
        utterances_spec = []
        for utter_name in os.listdir(folder):
            if utter_name[-4:].lower() == '.wav':
                utter_path = os.path.join(folder, utter_name)  # path of each utterance
                utter, sr = librosa.core.load(utter_path, hp.data.sr)  # load utterance audio
                intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection
                for interval in intervals:
                    if interval[1] - interval[0] > utter_min_len:
                        utter_part = utter[interval[0]:interval[1]]
                        S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft, win_length=int(hp.data.window * sr),
                                              hop_length=int(hp.data.hop * sr))
                        S = np.abs(S) ** 2
                        mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                        S = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of utterances
                        utterances_spec.append(S[:, :hp.data.tisv_frame])  # first 180 frames of partial utterance
                        utterances_spec.append(S[:, -hp.data.tisv_frame:])  # last 180 frames of partial utterance

        utterances_spec = np.asarray(utterances_spec)
        if index < train_speaker_num:
            np.save(os.path.join(hp.data.train_path, 'speaker{}'.format(index)), utterances_spec)
        else:
            np.save(os.path.join(hp.data.test_path, 'speaker{}'.format(index - train_speaker_num)), utterances_spec)


if __name__ == '__main__':
    save_spectrogram_tisv()
