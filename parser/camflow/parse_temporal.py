import argparse
import pickle
from tqdm import tqdm
import os
import torch
from torch_geometric.data import TemporalData
from file_util import  *


parser = argparse.ArgumentParser(description="Data preprocess.")
parser.add_argument("--data", type=str, default="")
args = parser.parse_args()
assert args.data in ['camflow_apt', 'shellshock']
# scene = 'shellshock'
scene = args.data
data_save_root = '../../data/{}/temporal_data'.format(scene)
if not os.path.exists(data_save_root):
    os.makedirs(data_save_root)
# os.system('cd ../../data/{} && mkdir -p temporal_data'.format(scene))


with open('./pkls/{}/nodeType_map.pkl'.format(scene), 'rb') as fr:
    nodeType2id = pickle.load(fr)
with open('./pkls/{}/edegType_map.pkl'.format(scene), 'rb') as fr:
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


def read_single_graph(file_name):

    map_id = dict()  # maps original IDs to new IDs, which always start from 0
    new_id = 0
    graph = list()  # list of parsed edges

    # file_name = os.path.basename(file_path)

    # description = '[STATUS] Sorting edges in CamFlow data from {}'.format(file_name)
    with open(file_name, 'r') as f:
        # for line in tqdm(f, desc=description):
        for line in f:
            try:
                edge = line.strip().split("\t")
                attributes = edge[2].strip().split(
                    ":")  # [hashed_source_type, hashed_destination_type, hashed_edge_type, edge_logical_timestamp, [timestamp_stats]]
                source_node_type = attributes[0]  # hashed_source_type
                destination_node_type = attributes[1]  # hashed_destination_type
                edge_type = attributes[2]  # hashed_edge_type
                edge_order = attributes[3]  # edge_logical_timestamp
                # now we rearrange the edge vector:
                # edge[0] is source_node_id, as orginally split
                # edge[1] is destination_node_id, as originally split
                edge[2] = source_node_type
                edge.append(destination_node_type)  # edge[3] = hashed_destination_type
                edge.append(edge_type)  # edge[4] = hashed_edge_type
                edge.append(edge_order)  # edge[5] = edge_logical_timestamp

                graph.append(edge)
            except:
                print("{}".format(line))
    f.close()

    # sort the graph edges based on logical timestamps
    graph.sort(key=custom_sort)

    # map nodeid
    # description = '[STATUS] Parsing edges in CamFlow data (final stage) from {}'.format(file_name)
    # for edge in tqdm(graph, desc=description):
    for edge in graph:
        src = edge[0]
        dst = edge[1]

        if map_id.setdefault(src, new_id) == new_id:  # new add to node_map_id
            new_id += 1
        edge[0] = map_id[src]

        if map_id.setdefault(dst, new_id) == new_id:
            new_id += 1
        edge[1] = map_id[dst]

    return graph


def gen_vectorized_graph(filename):   # filename from current dir
    graph = read_single_graph(filename)
    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for i in range(len(graph)):
        cur_e = graph[i]
        src.append(int(cur_e[0]))
        dst.append(int(cur_e[1]))
        src_vec = gen_node_feature(cur_e[2])
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

    gid = filename.split('.')[-1]
    gtype = 'attack' if 'attack' in filename else 'normal'
    save_filename = gtype+'-'+gid+'.TemporalData'
    torch.save(dataset, os.path.join(data_save_root, save_filename))


if 'unicorn' in scene:
    # decompress raw datafile ro ./
    if 'camflow-attack.txt.125' not in os.listdir('./'):
        decompress_data(scene)

    # batch process
    graphlist = [k for k in os.listdir('./') if os.path.isfile(os.path.join('./', k)) and 'camflow-' in k]
    for file in tqdm(graphlist, desc='parse {} grapn to temporalData'.format(scene)):
        gen_vectorized_graph(file)

    os.system('rm camflow-*')


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Data preprocess.")
#     parser.add_argument("--data", type=str, default="shellshock")
#
#     args = parser.parse_args()
#
#     args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
