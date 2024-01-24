import argparse
import pickle
from tqdm import tqdm
import os
import torch
from torch_geometric.data import TemporalData


scene = 'streamspot'
rawlog_path = '../../../../dataset/Streamspot/all.tsv'
data_save_root = '../../data/{}/temporal_data'.format(scene)
if not os.path.exists(data_save_root):
    os.makedirs(data_save_root)


with open('./pkls/nodeType_map.pkl', 'rb') as fr:
    nodeType2id = pickle.load(fr)
with open('./pkls/edegType_map.pkl', 'rb') as fr:
    edgeType2id = pickle.load(fr)
node_type_num = len(nodeType2id)
edge_type_num = len(edgeType2id)



def gen_node_feature(n):
    nodevec = [0] * node_type_num
    nodevec[nodeType2id[n]] = 1
    return torch.tensor(nodevec)


def gen_edge_feature(n):
    edgevec = [0] * edge_type_num
    edgevec[edgeType2id[n]] = 1
    return torch.tensor(edgevec)


def custom_sort(element):
    return int(element[5])


def read_single_graph(target_graph):

    map_id = dict()  # maps original IDs to new IDs, which always start from 0
    new_id = 0
    edge_cnt = 0
    graph = list()  # list of parsed edges

    # file_name = os.path.basename(file_path)
    with open(rawlog_path, 'r') as f:
        # for line in tqdm(f, desc=description):
        for line in f:
            items = line.strip('\n').split('\t')
            if items[-1] != str(target_graph):
                continue
            edge_cnt += 1
            src = items[0]
            dst = items[2]
            # srcType = items[1]
            # dstType = items[3]
            # edgeType = items[4]
            if map_id.setdefault(src, new_id) == new_id:  # new add to node_map_id
                new_id += 1
            items[0] = str(map_id[src])

            if map_id.setdefault(dst, new_id) == new_id:
                new_id += 1
            items[2] = str(map_id[dst])

            items.pop()   # del graph_id
            ts = str(edge_cnt)
            items.append(ts)
            graph.append(items)

    f.close()

    return graph


def gen_vectorized_graph(graph_id):   # filename from current dir
    graph = read_single_graph(graph_id)
    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for i in range(len(graph)):
        cur_e = graph[i]
        src.append(int(cur_e[0]))
        dst.append(int(cur_e[2]))
        src_vec = gen_node_feature(cur_e[1])
        dst_vec = gen_node_feature(cur_e[3])
        e_vec = gen_edge_feature(cur_e[4])
        msg_t = torch.cat([src_vec, e_vec, dst_vec], dim=0)  # 横向拼接
        msg.append(msg_t)
        t.append(int(cur_e[5]))

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)   # dtype: torch.long
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)   # 垂直方向堆叠
    # dataset.src = dataset.src.to(torch.long)
    # dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    # dataset.t = dataset.t.to(torch.long)

    gtype = 'attack' if int(graph_id)>=300 and int(graph_id) < 400 else 'normal'
    save_filename = gtype+'-'+graph_id+'.TemporalData'
    torch.save(dataset, os.path.join(data_save_root, save_filename))


for i in tqdm(range(600), desc='parse ss graph to temporalData'):
    gen_vectorized_graph(str(i))