"""
Anomaly detector
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from utils import CustomDataset, EarlyStopMonitor, metrics_print, sorted_listdir
from utils import load_data, makedir, show_time
import gc


class RCNN(nn.Module):
    """
    Recurrent Convolutional Neural Networks for Text Classification (2015)
    """
    def __init__(self, embedding_dim, hidden_size, hidden_layer_num, hidden_size_linear, dropout):
        super(RCNN, self).__init__()
        # self.embedding = nn.Embedding(seq_len, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,  num_layers=hidden_layer_num,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.W = nn.Linear(embedding_dim + 2*hidden_size, hidden_size_linear)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_size_linear, embedding_dim)

    def forward(self, x_emb):
        # x_emb = |bs, seq_len, embedding_dim|
        output, _ = self.lstm(x_emb)
        # output = |bs, seq_len, 2*hidden_size|
        output = torch.cat([output, x_emb], 2)
        # output = |bs, seq_len, embedding_dim + 2*hidden_size|
        output = self.tanh(self.W(output)).transpose(1, 2)
        # output = |bs, seq_len, hidden_size_linear| -> |bs, hidden_size_linear, seq_len|
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        # output = |bs, hidden_size_linear|
        output = self.fc(output)
        # output = |bs, embedding_dim|
        return output


def train_val(model, optimizer, train_dataloader, valid_dataloader, args):
    print("[INFO]Start Training...")
    lowest_loss = float('inf')
    early_stopper = EarlyStopMonitor(max_round=30, tolerance=1e-4) if args.early_stop else None
    save_folder_path = os.path.join(args.model_save_path, "rcnn_lr{}_bs{}".format(args.lr, args.batch_size))
    for epoch in range(1, args.epochs+1):
        train_loss = []
        model.train()
        for step, (x, y, g_label) in enumerate(train_dataloader):
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            # paiewise_dist = nn.PairwiseDistance(p=2)
            # dist = paiewise_dist(pred, y)
            # loss2 = nn.MSELoss(dist, torch.zeros_like(dist))
            # loss3 = nn.SmoothL1Loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        val_losses = validate(model, valid_dataloader, args)
        print('Epoch {}/{}, train_loss: {:.5f}, val_loss: {:.5f}'.format(epoch, args.epochs,
                                                         np.mean(train_loss), val_losses))

        if val_losses < lowest_loss:
            lowest_loss = val_losses
            makedir(save_folder_path)
            model_file = os.path.join(save_folder_path, "best_L{}.pt".format(args.seq_len))
            if os.path.exists(model_file):
                os.remove(model_file)
            torch.save(model.state_dict(), model_file)
            print('\t[INFO]**update saved model.')

        # if early_stopper and early_stopper.early_stop_check(val_losses):
        #     print('[INFO]No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        #     print(f'[INFO]The best model at epoch {early_stopper.best_epoch + 1}')
        #     break


def validate(model, valid_dataloader, args):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for x, y, g_label in valid_dataloader:
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            val_loss += loss.item()

    return val_loss


def check_label(dist_lst, thre):
    for num in dist_lst:
        if num > thre:
            return 1
    return 0


def test(model, test_dataloader, args):
    print("[INFO]Start Testing...")
    with torch.no_grad():
        model.eval()
        results = dict()   # collect snapshots predict results
        attack_loss = []
        benign_loss = []
        group_stats = {'attack_loss': dict(), 'benign_loss': dict()}
        for x, y, g_label in tqdm(test_dataloader, desc='Model predicting'):
            x, y = x.to(args.device), y.to(args.device)
            # g_label = g_label[0]  #  g_label is tuple with size batchsize
            pred = model(x)

            loss = F.mse_loss(pred, y, reduction='none')   # tensor (batch_size, emb_dim)
            loss = torch.mean(loss, dim=1)  # (batch_size,)
            for i in range(len(loss)):
                test_loss = loss[i].item()   # float
                label = int(g_label[i])
                if label not in results:
                    results[label] = [test_loss]
                else:
                    results[label].append(test_loss)
                if label > 124:
                    attack_loss.append(test_loss)
                else:
                    benign_loss.append(test_loss)

    group_stats['attack_loss']['max'] = np.max(attack_loss)
    group_stats['attack_loss']['min'] = np.min(attack_loss)
    group_stats['attack_loss']['mean'] = np.mean(attack_loss)
    group_stats['attack_loss']['median'] = np.median(attack_loss)
    group_stats['benign_loss']['max'] = np.max(benign_loss)
    group_stats['benign_loss']['min'] = np.min(benign_loss)
    group_stats['benign_loss']['mean'] = np.mean(benign_loss)
    group_stats['benign_loss']['median'] = np.median(benign_loss)
    # print(group_stats)


    # TODO print each graph's loss
    for g in sorted(results):
        print("{}: max-{}, min-{}, median-{}".
              format(g, np.max(results[g]), np.min(results[g]), np.median(results[g])))
    # exit()

    # TODO search best_threshold
    thre_list = np.arange(0.016, 0.018, 0.0005)
    pair_result = {}
    for threshold in thre_list:
        pred_l, true_l = [], []
        print(f'\n***threshold={threshold}***')
        for g in sorted(results):
            ab_num = 0
            for num in results[g]:
                if num > threshold:
                    ab_num += 1
            ab_rate = ab_num/len(results[g])
            print(f'{g} anomaly num: {ab_num}/{len(results[g])}, ab_rate: {ab_rate}')

            y = 1 if g > 124 else 0
            # y_hat = check_label(results[g], threshold)
            y_hat = 1 if ab_num > 0 else 0
            true_l.append(y)
            pred_l.append(y_hat)
        pair_result[threshold] = [np.array(true_l), np.array(pred_l)]
    return pair_result


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def main(args):
    model = RCNN(embedding_dim=args.embedding_dim,
                 hidden_size=args.hidden_size,
                 hidden_layer_num=args.hidden_layer_num,
                 hidden_size_linear=args.hidden_size_linear,
                 dropout=args.dropout).to(args.device)

    normal = list(range(125))
    train_list = random.sample(normal, 85)  # 85 benign
    val_test_normal = [i for i in normal if i not in train_list]
    val_list = random.sample(val_test_normal, 15)  # 15 benign
    test_list = list(range(125, 150)) + [i for i in val_test_normal if i not in val_list]  # 25 attack + 25 bengin

    train_list.sort()
    val_list.sort()
    test_list.sort()
    print(val_list, test_list)
    # Train
    if args.train:
        train_seq, train_sna, train_label = load_data(args.data_path, train_list, args.seq_len)
        assert args.embedding_dim == train_sna.shape[1]
        val_seq, val_sna, val_label = load_data(args.data_path, val_list, args.seq_len)
        print("[INFO]Load train&val data finished.")

        train_dataset = CustomDataset(train_seq, train_sna, train_label)
        val_dataset = CustomDataset(val_seq, val_sna, val_label)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        train_val(model, optimizer, train_dataloader, val_dataloader, args)
        print('******************** Train Finished ********************\n')

        del train_dataset, train_dataloader, val_dataset, val_dataloader

    # Test
    if args.test:
        # test_seq, test_sna, test_label = load_data(os.path.join(args.data_path, 'test'), args.seq_len)
        test_seq, test_sna, test_label = load_data(args.data_path, test_list,  args.seq_len)
        # test_seq, test_label = load_test_seq(os.path.join(args.data_path, 'test'))
        print("[INFO]Load test data finished.")
        test_dataset = CustomDataset(test_seq, test_sna, test_label)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
        model_dir = os.path.join(args.model_save_path, "rcnn_lr{}_bs{}".format(args.lr, args.batch_size))
        model_path = os.path.join(model_dir, "best_L{}.pt".format(args.seq_len))
        model.load_state_dict(torch.load(model_path))
        print("[INFO]Load model from {}.".format(model_path))

        # true_label, pred_label = test(model, test_dataloader, args)
        test_results = test(model, test_dataloader, args)
        # test_results = test_v1(model, test_seq, test_label, args)
        print('[SUCCESS]Test Finished!')
        print('-------------------- Test Results--------------------')
        # metrics_print(true_label, pred_label)
        for i in test_results:
            print(f'\n***threshold={i}***')
            metrics_print(test_results[i][0], test_results[i][1])

        del test_dataset, test_dataloader
        gc.collect()


def result_fusion():

    test_list = [1, 14, 25, 28, 29, 31, 32, 51, 53, 55, 65, 68, 71, 74, 77, 82, 88, 90, 98, 99, 101,
                 106, 115, 117, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
                 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]  # 25 attack + 25 bengin
    test_list.sort()

    true_l = []
    pred_l = []
    sketch_abnormal = [51, 68, 90, 115, 125, 126, 127, 128, 129, 131, 132, 133, 134, 135, 136, 137,
                 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]
    sketch_abnormal.sort()
    print(f"Histosketch+RCNN predict abnormal(cnt:{len(sketch_abnormal)}): {sketch_abnormal}")
    tgn_abnormal = [14, 32, 53, 74, 98, 124, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
                 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]
    tgn_abnormal.sort()
    print(f"TGN+RCNN predict abnormal(cnt:{len(tgn_abnormal)}): {tgn_abnormal}")

    for i in test_list:
        label = 1 if i > 124 else 0
        pred = 1 if i in sketch_abnormal and i in tgn_abnormal else 0

        true_l.append(label)
        pred_l.append(pred)
    true_l = np.array(true_l)
    pred_l = np.array(pred_l)
    metrics_print(true_l, pred_l)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detector model.")
    # parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--train',default=False)
    parser.add_argument('--test', action='store_true', default=True)

    # data
    parser.add_argument('-i', "--data_path", type=str, default="../../data/shellshock/tgn_embedding")
    # parser.add_argument('-i', "--data_path", type=str, default="../../data/shellshock/graph_sketch")
    parser.add_argument("--model_save_path", type=str, default="./model_saved/")
    # parser.add_argument("--model_save_path", type=str, default="./sketch_model")
    parser.add_argument('--seed', type=int, default=2023)

    # model
    # parser.add_argument("--seq_len", type=int, default=5)
    # parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--seq_len", type=int, default=25)
    # parser.add_argument("--embedding_dim", type=int, default=2000)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_size_linear", type=int, default=128)
    parser.add_argument("--hidden_layer_num", type=int, default=2)
    # parser.add_argument("--hidden_layer_num", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)

    # training
    # parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    # parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--early_stop",action="store_true", default=False)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args)
    print(args)
    main(args)
    # result_fusion()