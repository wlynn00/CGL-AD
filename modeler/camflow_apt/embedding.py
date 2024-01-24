import numpy as np
import torch
import os
import math
from tqdm import tqdm
from torch.nn import Module, Linear
from torch_geometric.nn import TransformerConv

import time


class GraphAttentionEmbedding(Module):
	def __init__(self, in_channels, out_channels, msg_dim, time_enc):
		super(GraphAttentionEmbedding, self).__init__()
		self.time_enc = time_enc
		edge_dim = msg_dim + time_enc.out_channels
		# The graph transformer operator from the paper`"Masked Label Prediction:
		#     Unified Message Passing Model for Semi-Supervised Classification"
		self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
									dropout=0.1, edge_dim=edge_dim)

	def forward(self, x, last_update, edge_index, t, msg):
		last_update.to(device)
		x = x.to(device)
		t = t.to(device)
		rel_t = last_update[edge_index[0]] - t
		rel_t_enc = self.time_enc(rel_t.to(x.dtype))
		edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
		x = self.conv(x, edge_index, edge_attr)
		return x


class LinkPredictor(Module):
	def __init__(self, in_channels):
		super(LinkPredictor, self).__init__()
		self.lin_src = Linear(in_channels, in_channels)
		self.lin_dst = Linear(in_channels, in_channels)
		self.lin_final = Linear(in_channels, 1)

	def forward(self, z_src, z_dst):
		h = self.lin_src(z_src) + self.lin_dst(z_dst)
		h = h.relu()
		return self.lin_final(h)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_root = '../../data/camflow_apt'
temporaldata_dir = '../../data/camflow_apt/temporal_data'
embedding_dir = '../../data/camflow_apt/tgn_embedding'
os.system(f'mkdir -p {embedding_dir}')

BATCHSIZE = 1000
max_node_num = 304593


@torch.no_grad()
def test_new(inference_data, base_size, window_size):
	# 一次处理一张图
	memory.eval()
	gnn.eval()
	link_pred.eval()

	memory.reset_state()  # Start with a fresh memory.
	neighbor_loader.reset_state()  # Start with an empty graph.
	torch.manual_seed(2023)  # Ensure determi|nistic sampling across epochs.

	# min_dst_idx, max_dst_idx = int(inference_data.dst.min()), int(inference_data.dst.max())
	# Helper vector to map global node indices to local ones.
	assoc = torch.empty(max_node_num, dtype=torch.long, device=device)

	total_event_num = inference_data.num_events
	node_emb_map = dict()  # 随着边的流入记录节点的嵌入表示，t时刻图中节点数即为len(node_emb_map)
	sketch_vectors = []

	### process base graph
	inference_data.to(device)
	# base_graph = inference_data[0, base_size]
	base_src = inference_data.src[0:base_size]
	base_dst = inference_data.dst[0:base_size]
	base_msg = inference_data.msg[0:base_size]
	base_t = inference_data.t[0:base_size]
	n_id = torch.cat([base_src, base_dst]).unique()
	n_id, edge_index, e_id = neighbor_loader(n_id)
	assoc[n_id] = torch.arange(n_id.size(0), device=device)
	z, last_update = memory(n_id)
	z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])
	for v in n_id:
		n_vec = z[assoc[v]].cpu().detach().numpy()
		node_emb_map[v.item()] = n_vec
	stacked_array = np.vstack(list(node_emb_map.values()))  # 按垂直方向（行顺序）堆叠
	sketch_vec = np.mean(stacked_array, axis=0)  # 列平均，得到(1,x)
	sketch_vectors.append(sketch_vec)   # base_graph vector

	memory.update_state(base_src, base_dst, base_t, base_msg)
	neighbor_loader.insert(base_src, base_dst)

	### process stream edges
	processed_stream_cnt = 0
	num_stream_instance = total_event_num - base_size
	num_batch = math.ceil(num_stream_instance / BATCHSIZE)

	# for k in tqdm(range(num_batch)):
	for k in range(num_batch):
		# generate mini-batch
		s_idx = k * BATCHSIZE + base_size
		e_idx = min(total_event_num, s_idx + BATCHSIZE)
		if s_idx == e_idx:
			continue

		src = inference_data.src[s_idx:e_idx]
		pos_dst = inference_data.dst[s_idx:e_idx]
		t = inference_data.t[s_idx:e_idx]
		msg = inference_data.msg[s_idx:e_idx]

		n_id = torch.cat([src, pos_dst]).unique()
		n_id, edge_index, e_id = neighbor_loader(n_id)
		assoc[n_id] = torch.arange(n_id.size(0), device=device)

		z, last_update = memory(n_id)
		z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

		for v in n_id:
			n_vec = z[assoc[v]].cpu().detach().numpy()
			node_emb_map[v.item()] = n_vec

		processed_stream_cnt += (e_idx - s_idx)

		## 当处理的边数达到一定数量时聚合节点嵌入得到该时间点的图嵌入
		if processed_stream_cnt % window_size == 0 or \
				(processed_stream_cnt == num_stream_instance and processed_stream_cnt % window_size != 0):
			stacked_array = np.vstack(list(node_emb_map.values()))  # 按垂直方向（行顺序）堆叠
			sketch_vec = np.mean(stacked_array, axis=0)  # 列平均，得到(1,x)
			sketch_vectors.append(sketch_vec)

		memory.update_state(src, pos_dst, t, msg)
		neighbor_loader.insert(src, pos_dst)

	return sketch_vectors


@torch.no_grad()
def test_edge_by_edge(inference_data, base_size, window_size):
	# 一次处理一张图
	memory.eval()
	gnn.eval()
	link_pred.eval()

	memory.reset_state()  # Start with a fresh memory.
	neighbor_loader.reset_state()  # Start with an empty graph.
	torch.manual_seed(2023)  # Ensure determi|nistic sampling across epochs.

	# min_dst_idx, max_dst_idx = int(inference_data.dst.min()), int(inference_data.dst.max())
	# Helper vector to map global node indices to local ones.
	assoc = torch.empty(max_node_num, dtype=torch.long, device=device)

	processed_edge_cnt = 0
	total_event_num = inference_data.num_events
	node_emb_map = dict()  # 随着边的流入记录节点的嵌入表示，t时刻图中节点数即为len(node_emb_map)
	sketch_vectors = []

	# test_loader = TemporalDataLoader(inference_data, batch_size=BATCH)
	for batch in tqdm(inference_data.seq_batches(batch_size=1)):
		batch = batch.to(device)
		src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
		# neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0),),
		# 						dtype=torch.long, device=device)
		# n_id = torch.cat([src, pos_dst, neg_dst]).unique()
		n_id = torch.cat([src, pos_dst]).unique()
		n_id, edge_index, e_id = neighbor_loader(n_id)
		assoc[n_id] = torch.arange(n_id.size(0), device=device)

		z, last_update = memory(n_id)
		z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

		src_vec = z[assoc[src]].cpu().detach().numpy()
		node_emb_map[src[0].item()] = src_vec
		dst_vec = z[assoc[pos_dst]].cpu().detach().numpy()
		node_emb_map[pos_dst[0].item()] = dst_vec

		processed_edge_cnt += BATCHSIZE

		## 当处理的边数达到一定数量时聚合节点嵌入得到该时间点的图嵌入
		if processed_edge_cnt == base_size or (processed_edge_cnt - base_size) % window_size == 0 \
				or ((processed_edge_cnt == total_event_num) and ((processed_edge_cnt - base_size) % window_size != 0)):
			stacked_array = np.vstack(list(node_emb_map.values()))  # 按垂直方向（行顺序）堆叠
			sketch_vec = np.mean(stacked_array, axis=0)  # 列平均，得到(1,x)
			sketch_vectors.append(sketch_vec)

		memory.update_state(src, pos_dst, t, msg)
		neighbor_loader.insert(src, pos_dst)

	return sketch_vectors


def show_time(t):
	return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))


def path2name(path):
	# dir_name = os.path.basename(os.path.dirname(path))
	file_name = os.path.splitext(os.path.basename(path))[0]  # get file name and split filename and extension
	return file_name


print(show_time(time.time()))
m = torch.load("./model_saved/TGN_lr5e-05_epoch10_bceloss_with_neg.pt")
# m = torch.load("./model_saved/TGN_lr0.0001_epoch10_bceloss_with_neg.pt")
# exit()
memory, gnn, link_pred, neighbor_loader = m
print('load model done.', show_time(time.time()))

# j = 0
# for filename in os.listdir(temporaldata_dir):
for filename in tqdm(os.listdir(temporaldata_dir), desc='infer graph embedding'):
	cur_path = os.path.join(temporaldata_dir, filename)
	graph_name = path2name(cur_path)
	# print(j, graph_name)

	cur_g_data = torch.load(cur_path).to(device)
	base_g_size = int(math.ceil(cur_g_data.num_events * 0.1))
	stream_batch_size = 3000
	graph_emb = test_new(cur_g_data, base_g_size, stream_batch_size)

	# print(' ')
	save_emb_path = os.path.join(embedding_dir, graph_name + '.txt')
	fw = open(save_emb_path, 'w')
	for i in range(len(graph_emb)):
		arr_str = ' '.join(map(str, graph_emb[i]))
		fw.write(arr_str + '\n')
	fw.close()

	# j += 1

print('infer embedding done.', show_time(time.time()))
