# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 16:27
# @Author  : MengnanChen
# @FileName: utils.py
# @Software: PyCharm

import numpy as np
import librosa
import torch
import torch.nn.functional as F

from hparams import hparam as hp


def mfccs_and_spec(wav_file, wav_process=False, calc_mfccs=False):
    sound_file, _ = librosa.core.load(wav_file, sr=hp.data.sr)
    window_length = int(hp.data.window * hp.data.sr)
    hop_length = int(hp.data.hop * hp.data.sr)
    duration = hp.data.tisv_frame * hp.data.hop + hp.data.window

    if wav_process:
        sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length, hop_length=hop_length)
        length = int(hp.data.sr * duration)
        sound_file = librosa.util.fix_length(sound_file, length)

    spec = librosa.stft(sound_file, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
    mag_spec = np.abs(spec)

    mel_basis = librosa.filters.mel(hp.data.sr, hp.data.nfft, n_mels=hp.data.nmels)
    mel_spec = np.dot(mel_basis, mag_spec)

    mag_db = librosa.amplitude_to_db(mag_spec)
    mel_db = librosa.amplitude_to_db(mel_spec).T  # db mel spectrogram

    mfccs = None
    if calc_mfccs:
        mfccs = np.dot(librosa.filters.dct(n_filters=40, n_input=mel_db.shape[0]), mel_db).T

    return mfccs, mel_db, mag_db


def get_centroids(embeddings):
    centroids = []
    for speaker in embeddings:
        centroid = 0
        for utterance in speaker:
            centroid += utterance
        centroid = centroid / len(speaker)
        centroids.append(centroid)
    centroids = torch.stack(centroids)
    return centroids


def get_centroid(embeddings, speaker_num, utterance_num):
    centroid = 0
    for utterance_id, utterance in enumerate(embeddings[speaker_num]):
        if utterance_id == utterance_num:
            continue
        centroid = centroid + utterance
    centroid = centroid / (len(embeddings[speaker_num]) - 1)
    return centroid


def get_cossim(embeddings, centroids):
    cossim = torch.zeros(embeddings.size(0), embeddings.size(1), centroids.size(0))
    for speaker_index, speaker in enumerate(embeddings):
        for utterance_index, utterance in enumerate(speaker):
            for centroid_index, centroid in enumerate(centroids):
                if speaker_index == centroid_index:
                    centroid = get_centroid(embeddings, speaker_index, utterance_index)
                output = F.cosine_similarity(utterance, centroid, dim=0) + 1e-6
                cossim[speaker_index][utterance_index][centroid_index] = output
    return cossim


def calc_loss(sim_matrix):
    per_embedding_loss = torch.zeros(sim_matrix.size(0), sim_matrix.size(1))
    for j in range(sim_matrix.size(0)):
        for i in range(sim_matrix.size(1)):
            per_embedding_loss[j][i] = -(sim_matrix[j][i][j] - ((torch.exp(sim_matrix[j][i]).sum() + 1e-6).log_()))
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss


def normalize_0_1(values,max_value,min_value):
    # normalize, (x-x_min)/(x_max-x_min)
    normalized=np.clip((values-min_value)/(max_value-min_value),0,1)
    return normalized
