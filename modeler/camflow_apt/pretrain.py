import os
from tqdm import tqdm
import torch
from torch.nn import Module, Linear, BCEWithLogitsLoss
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
	IdentityMessage,
	LastAggregator,
	LastNeighborLoader
)
import random

random.seed(2023)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
criterion =  BCEWithLogitsLoss()

# max_node_num = 276043  # the max number of nodes in graphs +1
# max_node_num = 255545  # the max number of nodes in graphs +1
# min_dst_idx, max_dst_idx = 0, max_node_num
# # Helper vector to map global node indices to local ones.
# assoc = torch.empty(max_node_num, dtype=torch.long, device=device)


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

LR=0.00005
def init_model(data_msg_size, data_num_nodes):
    global assoc
    memory_dim = time_dim = embedding_dim = 100
    neighbor_size = 10   # Neighborhood Sampling Size

    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data_num_nodes, dtype=torch.long, device=device)
    neighbor_loader = LastNeighborLoader(data_num_nodes, size=neighbor_size, device=device)

    memory = TGNMemory(
        data_num_nodes,
        data_msg_size,  # train_data.msg.size(-1)
        memory_dim,
        time_dim,
        message_module=IdentityMessage(data_msg_size, memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=data_msg_size,
        time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters())
        | set(link_pred.parameters()), lr=LR, eps=1e-08, weight_decay=0.01)

    return memory, gnn, link_pred, optimizer, neighbor_loader, assoc


BATCHSIZE = 1024
def train(train_data):
    global memory, gnn, link_pred, optimizer, neighbor_loader, assoc
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    min_dst_idx, max_dst_idx = int(train_data.dst.min()), int(train_data.dst.max())

    total_loss = 0
    for batch in train_data.seq_batches(batch_size=BATCHSIZE):
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes. 随机整数
        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0),),
                                dtype=torch.long, device=device)
        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)

        assoc[n_id] = torch.arange(n_id.size(0), dtype=torch.long, device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        #         print(z.shape)
        total_loss += float(loss) * batch.num_events
        #     print("trained_stage_data:",train_data)
    return total_loss / train_data.num_events


@torch.no_grad()
def test_new(inference_data):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    total_loss = 0
    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    torch.manual_seed(2023)  # Ensure determi|nistic sampling across epochs.

    pos_o = []
    min_dst_idx, max_dst_idx = int(inference_data.dst.min()), int(inference_data.dst.max())

    # test_loader = TemporalDataLoader(inference_data, batch_size=BATCH)
    for batch in inference_data.seq_batches(batch_size=BATCHSIZE):
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0),),
                                dtype=torch.long, device=device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)

        z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])
        pos_o.append(pos_out)

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))
        total_loss += float(loss) * batch.num_events

        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

    loss = total_loss / inference_data.num_events
    return loss


def load_pretrain_data():
    data_dir = "../../data/camflow_apt/temporal_data"
    graph_5 = torch.load(data_dir + '/normal-5.TemporalData').to(device)
    graph_15 = torch.load(data_dir+'/normal-15.TemporalData').to(device)
    graph_25 = torch.load(data_dir+'/normal-25.TemporalData').to(device)
    graph_100 = torch.load(data_dir+'/normal-100.TemporalData').to(device)   # val

    return [graph_5, graph_15, graph_25], graph_100

EPOCH = 10
pre_data, pre_val_data = load_pretrain_data()
msg_size = pre_data[0].msg.size(-1)
max_node_num = 304593
# memory, gnn, link_pred, optimizer, neighbor_loader, assoc = init_model(msg_size, pre_data[0].num_nodes)
memory, gnn, link_pred, optimizer, neighbor_loader, assoc = init_model(msg_size, max_node_num)

for epoch in tqdm(range(1, EPOCH+1)):
    total_loss = 0.0
    for g in pre_data:
        loss = train(train_data=g)
        total_loss += loss
        print(f"Epoch:{epoch}, g_{g} Loss: {loss:.5f}")
    loss_test = test_new(pre_val_data)
    avg_loss = total_loss / len(pre_data)
    print(f'val Loss: {loss_test:.5f}, average loss: {avg_loss:.5f}')

model = [memory, gnn, link_pred, neighbor_loader]

models_dir = 'model_saved'
os.system(f'mkdir -p {models_dir}')

torch.save(model, os.path.join(models_dir, 'TGN_lr{}_epoch{}_bceloss_with_neg.pt'.format(LR, EPOCH)))

