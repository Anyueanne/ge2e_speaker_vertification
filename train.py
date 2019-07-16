# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 23:06
# @Author  : Anyue
# @FileName: train.py
# @Software: PyCharm

import os
import random
import time

import torch
from torch.utils.data import DataLoader

from hparams import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import DeepSpeakerEmbedder, GE2ELoss, get_centroids, get_cossim


def train(model_path):
    device = torch.device(hp.device)
    if hp.data.data_preprocessed:
        train_dataset = SpeakerDatasetTIMITPreprocessed()
    else:
        train_dataset = SpeakerDatasetTIMIT()
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers,
                              drop_last=True)
    embedder_net = DeepSpeakerEmbedder().to(device)
    if hp.train.restore:
        embedder_net.load_state_dict(torch.load(model_path))
    ge2e_loss = GE2ELoss(device)
    # both net and loss have trainable params
    optimizer = torch.optim.SGD([
        {'params': embedder_net.parameters()},
        {'params': ge2e_loss.parameters()}
    ], lr=hp.train.lr)

    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)

    embedder_net.train()
    iteration = 0
    for e in range(hp.train.epochs):
        total_loss = 0
        for batch_id, mel_db_batch in enumerate(train_loader):
            mel_db_batch = mel_db_batch.to(device)
            mel_db_batch = torch.reshape(mel_db_batch,
                                         (hp.train.N * hp.train.M, 1, mel_db_batch.size(2), mel_db_batch.size(3)))
            perm = random.sample(range(0, hp.train.N * hp.train.M), hp.train.N * hp.train.M)
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i
                # perm: <class 'list'>: [9, 6, 2, 1, 4, 0, 7, 11, 10, 5, 3, 8]
                # unperm: <class 'list'>: [5, 3, 2, 10, 4, 9, 1, 6, 11, 0, 8, 7]
            mel_db_batch = mel_db_batch[perm]
            # clear gradients
            optimizer.zero_grad()

            embeddings = embedder_net(mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))

            loss = ge2e_loss(embeddings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.)
            optimizer.step()

            total_loss += loss
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                msg = '{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}' \
                      '\tTLoss:{6:.4f}\t\n'.format(time.ctime(),
                                                   e + 1,
                                                   batch_id + 1,
                                                   len(train_dataset) // hp.train.N,
                                                   iteration,
                                                   loss,
                                                   total_loss / (batch_id + 1))
                print(msg)
                if hp.train.log_file is not None:
                    with open(hp.train.log_file, 'a') as fin:
                        fin.write(msg)

        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            embedder_net.eval().cpu()
            ckpt_model_filename = 'ckpt_epoch_{}.pth'.format(e + 1)
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            embedder_net.to(device).train()  # set the module in training mode

    # save final model
    embedder_net.eval().cpu()
    save_model_filename = 'final_epoch.model'
    save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
    torch.save(embedder_net.state_dict(), save_model_path)
    print('Done, trained model saved at {}'.format(save_model_path))


def test(model_path):
    if hp.data.data_preprocessed:
        test_dataset = SpeakerDatasetTIMITPreprocessed()
    else:
        test_dataset = SpeakerDatasetTIMIT()
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers,
                             drop_last=True)
    embedder_net = DeepSpeakerEmbedder()
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    avg_EER = 0
    for e in range(hp.test.epochs):
        batch_avg_EER = 0
        for batch_id, mel_db_batch in enumerate(test_loader):
            assert hp.test.M % 2 == 0
            enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1) / 2), dim=1)

            enrollment_batch = torch.reshape(enrollment_batch, (
                hp.test.N * hp.test.M // 2, 1, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (
                hp.test.N * hp.test.M // 2, 1, verification_batch.size(2), verification_batch.size(3)))

            perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i

            verification_batch = verification_batch[perm]
            enrollment_embeddings = embedder_net(enrollment_batch)
            verification_embeddings = embedder_net(verification_batch)
            verification_embeddings = verification_embeddings[unperm]

            enrollment_embeddings = torch.reshape(enrollment_embeddings,
                                                  (hp.test.N, hp.test.M // 2, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings,
                                                    (hp.test.N, hp.test.M // 2, verification_embeddings.size(1)))

            enrollment_centroids = get_centroids(enrollment_embeddings)

            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

            # calculating EER
            diff = 1
            EER = 0
            EER_thresh = 0
            EER_FAR = 0
            EER_FRR = 0

            for thres in [0.01 * i + 0.5 for i in range(50)]:
                sim_matrix_thresh = sim_matrix > thres

                FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                            range(int(hp.test.N))])
                       / (hp.test.N - 1.0) / (float(hp.test.M / 2)) / hp.test.N)

                FRR = (sum([hp.test.M / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in range(int(hp.test.N))])
                       / (float(hp.test.M / 2)) / hp.test.N)

                # save threshold when FAR = FRR (=EER)
                if diff > abs(FAR - FRR):
                    diff = abs(FAR - FRR)
                    EER = (FAR + FRR) / 2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
        avg_EER += batch_avg_EER / (batch_id + 1)
    avg_EER = avg_EER / hp.test.epochs
    print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))


if __name__ == '__main__':
    if hp.training:
        train(hp.model.model_path)
    else:
        test(hp.model.model_path)
