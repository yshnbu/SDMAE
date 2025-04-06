import os
import re

import librosa
import pandas as pd
import torch
import torch.nn as nn
from numpy import mean
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import sklearn
from sklearn.neighbors import LocalOutlierFactor,NearestNeighbors
from sklearn.mixture import GaussianMixture
from loss import ASDLoss
from sklearn.metrics import roc_auc_score
import utils
from tqdm import tqdm
from utils import zscore
from sklearn.svm import OneClassSVM
import warnings
warnings.filterwarnings('ignore')
class Trainer:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.net = kwargs['net']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = self.args.writer
        self.logger = self.args.logger
        self.criterion = ASDLoss().to(self.args.device)
        self.transform = kwargs['transform']
    def train(self, train_loader):
        # self.test(save=False)
        model_dir = os.path.join(self.writer.log_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        epochs = self.args.epochs
        valid_every_epochs = self.args.valid_every_epochs
        early_stop_epochs = self.args.early_stop_epochs
        start_valid_epoch = self.args.start_valid_epoch
        num_steps = len(train_loader)
        self.sum_train_steps = 0
        self.sum_valid_steps = 0
        best_metric = 0
        no_better_epoch = 0
        for epoch in range(0, epochs + 1):
            # train
            sum_loss = 0
            self.net.train()
            train_bar = tqdm(train_loader, total=num_steps, desc=f'Epoch-{epoch}')
            for (x_wavs, x_mels, labels) in train_bar:
                # forward
                x_wavs, x_mels = x_wavs.float().to(self.args.device), x_mels.float().to(self.args.device)
                labels = labels.reshape(-1).long().to(self.args.device)
                # print(x_wavs.shape, x_mels.shape, labels.shape)
                logits, _ = self.net(x_wavs, x_mels, labels)
                loss = self.criterion(logits, labels)
                train_bar.set_postfix(loss=f'{loss.item():.5f}')
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # visualization
                self.writer.add_scalar(f'train_loss', loss.item(), self.sum_train_steps)
                sum_loss += loss.item()
                self.sum_train_steps += 1
            avg_loss = sum_loss / num_steps
            if self.scheduler is not None and epoch >= self.args.start_scheduler_epoch:
                self.scheduler.step()
            self.logger.info(f'Epoch-{epoch}\tloss:{avg_loss:.5f}')
            # valid
            if (epoch - start_valid_epoch) % valid_every_epochs == 0 and epoch >= start_valid_epoch:
                avg_auc, avg_pauc = self.test(save=False, gmm_n=False)
                self.writer.add_scalar(f'auc', avg_auc, epoch)
                self.writer.add_scalar(f'pauc', avg_pauc, epoch)
                if avg_auc + avg_pauc >= best_metric:
                    no_better_epoch = 0
                    best_metric = avg_auc + avg_pauc
                    best_model_path = os.path.join(model_dir, 'best_checkpoint.pth.tar')
                    utils.save_model_state_dict(best_model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)
                    self.logger.info(f'Best epoch now is: {epoch:4d}')
                else:
                    # early stop
                    no_better_epoch += 1
                    if no_better_epoch > early_stop_epochs > 0: break
            # save last 10 epoch state dict
            if epoch >= self.args.start_save_model_epochs:
                if (epoch - self.args.start_save_model_epochs) % self.args.save_model_interval_epochs == 0:
                    model_path = os.path.join(model_dir, f'{epoch}_checkpoint.pth.tar')
                    utils.save_model_state_dict(model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)

    def test(self, save=False, gmm_n=False):
        """
            gmm_n if set as number, using GMM estimator (n_components of GMM = gmm_n)
            if gmm_n = sub_center(arcface), using weight vector of arcface as the mean vector of GMM
        """
        csv_lines = []
        sum_auc, sum_pauc, num = 0, 0, 0
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        if gmm_n:
            result_dir = os.path.join(self.args.result_dir, self.args.version, f'GMM-{gmm_n}')
        os.makedirs(result_dir, exist_ok=True)
        self.net.eval()
        net = self.net.module if self.args.dp else self.net
        print('\n' + '=' * 20)
        for index, (target_dir, train_dir) in enumerate(zip(sorted(self.args.valid_dirs), sorted(self.args.train_dirs))):
            machine_type = target_dir.split('/')[-2]
            # result csv
            csv_lines.append([machine_type])
            csv_lines.append(['id', 'AUC', 'pAUC'])
            performance = []
            # get machine list
            machine_id_list = utils.get_machine_id_list(target_dir)
            for id_str in machine_id_list:
                meta = machine_type + '-' + id_str
                label = self.args.meta2label[meta]
                test_files, y_true = utils.create_test_file_list(target_dir, id_str, dir_name='test')
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{id_str}.csv')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                if gmm_n:
                    train_files = utils.get_filename_list(train_dir, pattern=f'normal_{id_str}*')
                    features = self.get_latent_features(train_files)
                    means_init = net.arcface.weight[label * gmm_n: (label + 1) * gmm_n, :].detach().cpu().numpy() \
                        if self.args.use_arcface and (gmm_n == self.args.sub_center) else None
                    gmm = self.fit_GMM(features, n_components=gmm_n, means_init=means_init)
                for file_idx, file_path in enumerate(test_files):
                    x_wav, x_mel, label = self.transform(file_path)
                    x_wav, x_mel = x_wav.unsqueeze(0).float().to(self.args.device), x_mel.unsqueeze(0).float().to(
                        self.args.device)
                    label = torch.tensor([label]).long().to(self.args.device)
                    with torch.no_grad():
                        predict_ids, feature = net(x_wav, x_mel, label)
                    if gmm_n:
                        if self.args.use_arcface: feature = F.normalize(feature).cpu().numpy()
                        y_pred[file_idx] = - np.max(gmm._estimate_log_prob(feature))
                    else:
                        probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                        y_pred[file_idx] = probs[label]
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)
                # compute auc and pAuc
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
                performance.append([auc, p_auc])

            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            self.logger.info(f'{machine_type}\t\tAUC: {mean_auc*100:.3f}\tpAUC: {mean_p_auc*100:.3f}')
            csv_lines.append(['Average'] + list(averaged_performance))
            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            num += 1
        avg_auc, avg_pauc = sum_auc / num, sum_pauc / num
        csv_lines.append(['Total Average', avg_auc, avg_pauc])
        self.logger.info(f'Total average:\t\tAUC: {avg_auc*100:.3f}\tpAUC: {avg_pauc*100:.3f}')
        result_path = os.path.join(result_dir, 'result.csv')
        if save:
            utils.save_csv(result_path, csv_lines)
        return avg_auc, avg_pauc

    def test_lof(self, save=False, lof_n=False):
        """
            gmm_n if set as number, using GMM estimator (n_components of GMM = gmm_n)
            if gmm_n = sub_center(arcface), using weight vector of arcface as the mean vector of GMM
        """
        csv_lines = []
        sum_auc, sum_pauc, num = 0, 0, 0
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        if lof_n:
            result_dir = os.path.join(self.args.result_dir, self.args.version, f'LOF-{lof_n}')
        os.makedirs(result_dir, exist_ok=True)
        self.net.eval()
        net = self.net.module if self.args.dp else self.net
        print('\n' + '=' * 20)
        for index, (target_dir, train_dir) in enumerate(zip(sorted(self.args.valid_dirs), sorted(self.args.train_dirs))):
            machine_type = target_dir.split('/')[-2]
            # result csv
            csv_lines.append([machine_type])
            csv_lines.append(['id', 'AUC', 'pAUC'])
            performance = []
            # get machine list
            machine_id_list = utils.get_machine_id_list(target_dir)
            for id_str in machine_id_list:
                meta = machine_type + '-' + id_str
                label = self.args.meta2label[meta]
                test_files, y_true = utils.create_test_file_list(target_dir, id_str, dir_name='test')
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{id_str}.csv')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                if lof_n:
                    train_files = utils.get_filename_list(train_dir, pattern=f'normal_{id_str}*')
                    features = self.get_latent_features(train_files)
                    lof = self.fit_LOF(features, n_components=lof_n)
                for file_idx, file_path in enumerate(test_files):
                    x_wav, x_mel, label = self.transform(file_path)
                    x_wav, x_mel = x_wav.unsqueeze(0).float().to(self.args.device), x_mel.unsqueeze(0).float().to(
                        self.args.device)
                    label = torch.tensor([label]).long().to(self.args.device)
                    with torch.no_grad():
                        predict_ids, feature = net(x_wav, x_mel, label)
                    if lof_n:
                        if self.args.use_arcface: feature = F.normalize(feature).cpu().numpy()
                        y_pred[file_idx] = - lof.score_samples(feature)
                        # lof = LocalOutlierFactor(n_neighbors=hp, novelty=True)
                        # lof.fit(input_train)
                        # lof_score = lof.score_samples(input_eval)
                    else:
                        probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                        y_pred[file_idx] = probs[label]
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)
                # compute auc and pAuc
                max_fpr = 0.1
                auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
                csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
                performance.append([auc, p_auc])

            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
            self.logger.info(f'{machine_type}\t\tAUC: {mean_auc*100:.3f}\tpAUC: {mean_p_auc*100:.3f}')
            csv_lines.append(['Average'] + list(averaged_performance))
            sum_auc += mean_auc
            sum_pauc += mean_p_auc
            num += 1
        avg_auc, avg_pauc = sum_auc / num, sum_pauc / num
        csv_lines.append(['Total Average', avg_auc, avg_pauc])
        self.logger.info(f'Total average:\t\tAUC: {avg_auc*100:.3f}\tpAUC: {avg_pauc*100:.3f}')
        result_path = os.path.join(result_dir, 'result.csv')
        if save:
            utils.save_csv(result_path, csv_lines)
        return avg_auc, avg_pauc


    def score(self):
        auc_columns = []
        pauc_columns = []
        for machine in self.args.machine_list:
            auc_columns += [f"{machine}_auc"]
            pauc_columns += [f"{machine}_pauc"]
        infer_path = './save/infer'
        file_list = [os.path.join(infer_path, f) for f in os.listdir(infer_path) if
                     os.path.isfile(os.path.join(infer_path, f))]

        agg_df = pd.read_csv(file_list[0])
        post_processes = list(agg_df.columns)

        for rm in ['file_name', 'machine_type', 'id', 'is_normal']:
            post_processes.remove(rm)
        columns = ['path', "dev_auc", "dev_pauc"]

        columns += auc_columns
        columns += pauc_columns

        score_df = pd.DataFrame(index=post_processes, columns=columns)

        save_path = './save/score.csv'

        score_df.loc[:, "path"] = save_path

        for agg_path in file_list:
            agg_df = pd.read_csv(agg_path)
            machine = agg_path.split('/')[-1].split('_')[0]
            for post_process in post_processes:
                id_list = agg_df['id'].drop_duplicates().sort_values().tolist()
                auc_list = []
                pauc_list = []
                for id in id_list:
                    target_idx = (
                            agg_df["id"] == id
                    )
                    auc_list.append(
                        roc_auc_score(
                            agg_df.loc[target_idx, "is_normal"],
                            agg_df.loc[target_idx, post_process],
                        )
                    )
                    pauc_list.append(
                        roc_auc_score(
                            agg_df.loc[target_idx, "is_normal"],
                            agg_df.loc[target_idx, post_process],
                            max_fpr=0.1,
                        )
                    )
                score_df.loc[
                    post_process, f"{machine}_auc"
                ] = mean(auc_list)

                score_df.loc[
                    post_process, f"{machine}_pauc"
                ] = mean(pauc_list)

        score_df.loc[post_processes, "dev_auc"] = mean(
            score_df.loc[post_processes, auc_columns].values, axis=1
        )
        score_df.loc[post_processes, "dev_pauc"] = mean(
            score_df.loc[post_processes, pauc_columns].values, axis=1
        )

        score_df = score_df.reset_index().rename(columns={"index": "post_process"})
        score_df.to_csv(save_path, index=False)



    def get_latent_features(self, train_files):
        pbar = tqdm(enumerate(train_files), total=len(train_files))
        self.net.eval()
        classifier = self.net.module if self.args.dp else self.net
        features = []
        for file_idx, file_path in pbar:
            x_wav, x_mel, label = self.transform(file_path)
            x_wav, x_mel = x_wav.unsqueeze(0).float().to(self.args.device), x_mel.unsqueeze(0).float().to(
                self.args.device)
            label = torch.tensor([label]).long().to(self.args.device)
            with torch.no_grad():
                _, feature = classifier(x_wav, x_mel, label)
            if file_idx == 0:
                features = feature.cpu()
            else:
                features = torch.cat((features.cpu(), feature.cpu()), dim=0)
        if self.args.use_arcface: features = F.normalize(features)
        return features.numpy()

    def fit_LOF(self, data, n_components):
        print('=' * 40)
        print('Fit LOF in train data for test...')
        np.random.seed(self.args.random_seed)
        lof = LocalOutlierFactor(n_neighbors=n_components, novelty=True)
        lof.fit(data)
        print('Finish LOF fit.')
        print('=' * 40)
        return lof


