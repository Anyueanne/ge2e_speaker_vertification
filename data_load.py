# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 16:17
# @Author  : MengnanChen
# @FileName: data_load.py
# @Software: PyCharm
import os
import glob
from random import shuffle, sample

import numpy as np
import torch
from torch.utils.data import Dataset

from hparams import hparam as hp
from ge2e_speaker_vertification.utils import mfccs_and_spec


class SpeakerDatasetTIMIT(Dataset):
    def __init__(self):
        if hp.training:
            self.path = hp.data.train_path_unprocessed
            self.utterance_number = hp.train.M
        else:
            self.path = hp.data.test_path_unprocessed
            self.utterance_number = hp.test.M
        self.speakers = glob.glob(os.path.dirname(self.path))
        shuffle(self.speakers)

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        speaker = self.speakers[idx]
        wav_files = glob.glob(os.path.join(speaker, '*.WAV'))
        shuffle(wav_files)

        mel_dbs = []
        for wav_file in wav_files:
            _, mel_db, _ = mfccs_and_spec(wav_file, wav_process=True)
            mel_dbs.append(mel_db)
        return torch.Tensor(mel_dbs)


class SpeakerDatasetTIMITPreprocessed(Dataset):
    def __init__(self, shuffle=True, utter_start=0):
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M

        self.filelist = os.listdir(self.path)
        self.shuffle = shuffle
        self.utter_start = utter_start

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        np_file_list = os.listdir(self.path)
        if self.shuffle:
            selected_file = sample(np_file_list, 1)[0]
        else:
            selected_file = np_file_list[idx]

        # load utterance spectrogram of selected speaker
        utters = np.load(os.path.join(self.path, selected_file))
        # select M utterances per speaker
        if self.shuffle:
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)
            utterance = utters[utter_index]
        else:
            utterance = utters[self.utter_start:self.utter_start + self.utter_num]

        utterance = utterance[:, :, :160]  # TODO

        utterance = torch.Tensor(np.transpose((utterance), axes=(0, 2, 1)))  # transpose [batch_size,n_frames,n_mels]
        return utterance
