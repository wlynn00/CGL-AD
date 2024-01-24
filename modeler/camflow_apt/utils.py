import os
import time
from glob import glob

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch import tensor, float
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, sequences, snapshots, labels):
        self.x = tensor(sequences, dtype=float)
        self.y = tensor(snapshots, dtype=float)
        self.g_label = labels
        # print(type(self.x), type(self.y), type(self.g_label))

    def __len__(self):
        """Return the data length"""
        return len(self.x)

    def __getitem__(self, idx):
        """Return one item on the index"""
        return self.x[idx], self.y[idx], self.g_label[idx]


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, lower_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.lower_better = lower_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        # if not self.lower_better:
        #     curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        # elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
        elif (self.last_best - curr_val) / abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        # elif curr_val <= self.last_best:
        elif curr_val >= self.last_best:
            self.num_round += 1
        # else:
        #     self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round


def sort_key(filename):
    parts = filename.split('_')
    number = int(parts[-1].split('.')[0]) if '.' in parts[-1] else int(parts[-1])
    return number


def sorted_listdir(path):
    files = os.listdir(path)
    sorted_files = sorted(files, key=sort_key)
    return sorted_files


def load_data(dir_path, graph_list, length):
    """
    Read function for train/val/test sequences
    return [seq{,,,}] with length k and k+1 snapshots' emb []
    """
    print("[INFO]Load data from {}".format(dir_path))
    seq_list = []
    next_emb_list = []
    label_list = []
    # graph_files = sorted_listdir(dir_path)
    graph_list.sort()
    for graph in tqdm(graph_list, desc='Loading sketch vectors'):
        g_label = graph
        filename = 'attack-{}.txt'.format(graph) if graph > 124 else 'normal-{}.txt'.format(graph)
        # filename = 'sketch-attack-{}.txt'.format(graph) if graph > 124 else 'sketch-normal-{}.txt'.format(graph)
        file_path = os.path.join(dir_path, filename)
        data = pd.read_csv(file_path, sep=' ', header=None)   # tgn embedding
        # data = pd.read_csv(file_path, sep=' ', header=None)   # sketch vector
        # data = data.drop(data.columns[-1], axis=1)   # sketch vector
        cnt = data.shape[0]
        data = (data - data.min().min()) / (data.max().max() - data.min().min())
        if cnt <= length:
            print(f'{file_path}, line_num{cnt}, seq_len{length}')
        assert cnt > length

        for row_id in range(cnt - length):
            start = row_id
            end = row_id + length

            seq = data.iloc[start:end, :]
            seq_next = data.iloc[end, :]

            seq_list.append(seq.values.tolist())
            next_emb_list.append(seq_next.values.tolist())
            label_list.append(g_label)

        assert len(seq_list) == len(next_emb_list) == len(label_list)

    seq_list = np.array(seq_list)
    next_emb_list = np.array(next_emb_list)
    label_list = np.array(label_list)
    print("[INFO]seq_list shape {},  next_emb_list shape {}".format(seq_list.shape, next_emb_list.shape))

    return seq_list, next_emb_list, label_list


def metrics_print(targets, y_hat):
    idx1 = np.argwhere(targets == 0)
    idx2 = np.argwhere(targets == 1)
    TN = np.sum(targets[idx1] == y_hat[idx1])
    FP = np.sum(targets[idx1] != y_hat[idx1])
    TP = np.sum(targets[idx2] == y_hat[idx2])
    FN = np.sum(targets[idx2] != y_hat[idx2])

    acc = '{:.4f}'.format((TP + TN) / (TP + FP + TN + FN))
    precision = TP / (TP + FP) if (TP + FP) != 0 else 'N/A'
    recall = TP / (TP + FN) if (TP + FN) != 0 else 'N/A'
    # if not precision or not recall or (precision + recall) == 0:
    if precision=='N/A' or recall=='N/A' or (precision + recall) == 0:
        F1 = 'N/A'
    else:
        F1 = 2 * precision * recall / (precision + recall)
    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)
    micro_f1 = f1_score(targets, y_hat, average='micro')
    macro_f1 = f1_score(targets, y_hat, average='macro')
    # weighted_f1 = f1_score(true_l, pred_l, average='weighted')
    # p, r, f1, _ = precision_recall_fscore_support(true_l, pred_l, average='binary')
    ap = average_precision_score(targets, y_hat)
    auc = roc_auc_score(targets, y_hat)
    print('TP={}, FP={}, TN={}, FN={}'.format(TP, FP, TN, FN))
    print('acc={}, P={}, R={}, f1={}'.format(acc, precision, recall, F1))
    # print('acc={:.4f}, P={:.4f}, R={:.4f}, f1={:.4f}'.format(acc, precision, recall, F1))
    print('ap={:.4f}, auc={:.4f}, macro_f1={:.4f}, micro_f1={:.4f}'.format(
        ap, auc, macro_f1, micro_f1))
    # print('ap={:.4f}, auc={:.4f}, TPR={:.4F}, FPR={:.4f}, macro_f1={:.4f}, micro_f1={:.4f}'.format(
    #     ap, auc, TPR, FPR, macro_f1, micro_f1))


def ts_convert(timestamp):
    dateTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp / 1000000000))
    return dateTime


def date_convert(str_time):
    time_tuple = time.strptime(str_time, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(time_tuple)
    return timestamp


def show_time(t):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))


def makedir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def path2name(path):
    dir_name = os.path.basename(os.path.dirname(path))
    file_name = os.path.splitext(os.path.basename(path))[0]  # get file name and split filename and extension
    return '-'.join([dir_name, file_name])


def read_pkl_path(root_path):
    # 乱序
    # filelist = glob(os.path.join(path, '**'), recursive=True)
    # file_list = [file for file in filelist if os.path.isfile(file)]
    # return file_list
    filelist = []
    sorted_folders = sorted(os.listdir(root_path), key=sort_key)
    for i in sorted_folders:
        dpath = os.path.join(root_path, i)
        sorted_pkls = sorted(os.listdir(dpath), key=sort_key)
        filelist += [os.path.join(dpath, f) for f in sorted_pkls]
    return filelist


if __name__ == "__main__":
    print("test")