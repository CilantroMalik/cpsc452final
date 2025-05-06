import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, DMoNPooling
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import pandas as pd

from plotnine import ggplot

N_TRAIN = 300
N_TEST = 60
BATCH_SIZE = 16
NUM_EPOCHS = 8

# load data
metacells = pd.read_csv("./oligo-SCZ-metacellExpr.csv", index_col=0)
metadata = pd.read_csv("./oligo-SCZ-meta.csv", index_col=0)
tom = pd.read_csv("./oligo-SCZ-tom.csv", index_col=0)

le = LabelEncoder()
metadata.loc[:,"disorder_encoded"] = le.fit_transform(metadata["disorder"])

# hyperparameters
TOM_THRESHOLD = 0.03  # value below which we zero out the similarity that will be used as attention priors

# create graph dataset
edges = []
for i in tqdm(range(len(tom.columns))):
    for j in range(i):
        if tom.iloc[i,j] > TOM_THRESHOLD:
            edges.extend([[i,j],[j,i]])


def train_test(metacells, metadata, edges, idx_train, idx_test, bs, device):
    train = []
    test = []
    nGenes = metacells.shape[0]
    print("-- Loading training data --")
    for i in tqdm(idx_train):
        train.append(Data(
            x=torch.tensor(metacells.iloc[:,i].values, dtype=torch.float32).reshape((nGenes,1)).to(device),
            edge_index=torch.tensor(edges).t().contiguous().to(device),
            y=torch.tensor(metadata["disorder_encoded"].iloc[i], dtype=torch.long).to(device)
        ))
    print("-- Loading testing data --")
    for i in tqdm(idx_test):
        test.append(Data(
            x=torch.tensor(metacells.iloc[:,i].values, dtype=torch.float32).reshape((nGenes,1)),
            edge_index=torch.tensor(edges).t().contiguous(),
            y=torch.tensor(metadata["disorder_encoded"].iloc[i], dtype=torch.long)
        )) 
    random.shuffle(train)
    random.shuffle(test)
    return DataLoader(train, batch_size=bs, shuffle=True), DataLoader(test, batch_size=bs, shuffle=True)


# define model
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.num_features = 1
        self.hidden_layers = 1
        self.k = 64
        self.in_heads = 4
        self.mid_heads = 2
        self.out_heads = 1

        self.conv1 = GATv2Conv(self.num_features, self.hidden_layers, heads=self.in_heads, dropout=0.2)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_layers * self.in_heads)
        self.conv2 = GATv2Conv(self.hidden_layers*self.in_heads, self.hidden_layers, heads=self.mid_heads, dropout=0.2)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_layers * self.mid_heads)
        self.conv3 = GATv2Conv(self.hidden_layers*self.mid_heads, self.hidden_layers, heads=self.out_heads, dropout=0.2)
        self.bn3 = torch.nn.BatchNorm1d(self.hidden_layers)

        self.pool = DMoNPooling([self.hidden_layers, self.hidden_layers], k=self.k, dropout=0.2)

        self.proj = torch.nn.Linear(self.hidden_layers, 1)

        self.classifier = torch.nn.Linear(self.k, 2)

    def _dense_adj(self, edge_index, alpha, batch):
        # alpha = alpha.mean(dim=1)
        # adj = to_dense_adj(edge_index, batch=batch, edge_attr=alpha)
        # out = []
        # for i in range(adj.size(0)):
        #     A = adj[i]
        #     deg_inv = A.sum(-1).clamp(min=1e-12).pow(-0.5)
        #     norm_adj = deg_inv.unsqueeze(1) * A * deg_inv.unsqueeze(0)
        #     out.append(norm_adj)
        # return torch.stack(out)
        alpha = alpha.mean(dim=1)
        adj = to_dense_adj(edge_index, batch=batch, edge_attr=alpha)  # shape: [B, N, N]
        deg = adj.sum(-1).clamp(min=1e-12)
        deg_inv_sqrt = deg.pow(-0.5)
        norm_adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        return norm_adj


    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch

        x = F.elu(self.bn1(self.conv1(x, ei)))
        x = F.elu(self.bn2(self.conv2(x, ei)))                     
        x, (ei3, alpha3) = self.conv3(x, ei, return_attention_weights=True) 
        x = F.elu(self.bn3(x))

        # convert to dense batch and corresponding mask
        x_dense, mask = to_dense_batch(x, batch)

        alpha = alpha3.mean(dim=1)
        adj_dense = to_dense_adj(ei3, batch=batch, edge_attr=alpha)
        deg = adj_dense.sum(-1).clamp(min=1e-12)
        deg_inv_sqrt = deg.pow(-0.5)
        adj_dense = deg_inv_sqrt.unsqueeze(-1) * adj_dense * deg_inv_sqrt.unsqueeze(-2)

        # # build dense Laplacian surrogate from layer‑3 attention
        # adj = self._dense_adj(ei3, alpha3, batch)
        # mask = torch.ones(adj.size(0), adj.size(1), dtype=torch.bool, device=adj.device)

        S, x, adj, mod, ort, clu = self.pool(x_dense, adj_dense, mask)

        # read‑out
        x = self.proj(x).squeeze(-1)
        logits = self.classifier(x)

        pool_reg = mod + clu + 0.1 * ort
        return logits, pool_reg


def train_epoch(loader, model, optimizer, device):
    model.train()
    total_loss = 0
    iter_loss = []
    train_iter = tqdm(loader)
    correct_readouts = []
    for data in train_iter:
        data = data.to(device)
        # data.y = data.y.long()
        optimizer.zero_grad()
        out, pool_reg = model(data)
        loss = F.cross_entropy(out, data.y.squeeze()) + pool_reg.mean()
        #print(out)
        pred = out.argmax(dim=1)
        
        correct_readout = ["SCZ" if item == 1 else "CON" for item in data.y]
        for i, item in enumerate(pred == data.y):
            correct_readout[i] += "(√)" if item else "(x)"
        correct_readouts.extend(correct_readout)
        train_iter.set_description(f"(loss {loss.item()}; {(pred == data.y).sum().item()}/{loader.batch_size} [{" ".join(correct_readout)}])")
        
        loss.backward()
        optimizer.step()
        #train_iter.set_description(f"(loss {loss.item()})")
        iter_loss.append(loss.item())
        total_loss += loss.item() * data.num_graphs
    print(iter_loss)
    return total_loss / len(loader.dataset)

def test_epoch(loader, model, device):
    model.eval()
    total_loss = 0
    correct = 0
    correct_readouts = []
    test_iter = tqdm(loader)
    for data in test_iter:
        data = data.to(device)
        # data.y = data.y.long()
        out, pool_reg = model(data)
        #print(out)
        loss = F.cross_entropy(out, data.y.squeeze()) + pool_reg.mean()
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        correct_readout = ["SCZ" if item == 1 else "CON" for item in data.y]
        for i, item in enumerate(pred == data.y):
            correct_readout[i] += "(√)" if item else "(x)"
        correct_readouts.extend(correct_readout)
        test_iter.set_description(f"(loss {loss.item()}; {(pred == data.y).sum().item()}/{loader.batch_size} [{" ".join(correct_readout)}])")
        total_loss += loss.item() * data.num_graphs
    print(correct_readouts)
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


model = GAT()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
device = 'cuda'

con_samples = [metacells.columns.get_loc(c) for c in metadata[metadata["disorder_encoded"] == 1].sample(n=(N_TRAIN+N_TEST)//2).index.values]
scz_samples = [metacells.columns.get_loc(c) for c in metadata[metadata["disorder_encoded"] == 0].sample(n=(N_TRAIN+N_TEST)//2).index.values]

samples_train = con_samples[:(N_TRAIN//2)]
samples_train.extend(scz_samples[:(N_TRAIN//2)])
random.shuffle(samples_train)

samples_test = con_samples[(N_TRAIN//2):]
samples_test.extend(scz_samples[(N_TRAIN//2):])
random.shuffle(samples_test)

train_loader, test_loader = train_test(metacells, metadata, edges, samples_train, samples_test, BATCH_SIZE, device)


for i in range(NUM_EPOCHS):
    print("--> Epoch", i)
    loss = train_epoch(train_loader, model, optimizer, device)
    print(loss)
    print("---------")

total_loss, correct = test_epoch(test_loader, model, device)

print("Accuracy:", correct)

torch.save(model, "./modelOut.pth")


classifier_weights_df = pd.DataFrame(
    model.classifier.weight.detach().numpy(),
    columns=[f"cluster_{i}" for i in range(model.k)]
)
classifier_weights_df.to_csv("classifier_weights.csv", index=False)


def extract_cluster_assignments(model: torch.nn.Module, loader, device):

    model.eval()
    xs = []
    for data in loader.dataset:           
        xs.append(data.x.to(device))
    x = torch.stack(xs, dim=0).mean(dim=0)

    edge_index = loader.dataset[0].edge_index.to(device)

    with torch.no_grad():
        x = F.dropout(x, p=0.6, training=False)
        x = F.elu(model.conv1(x, edge_index))
        x = model.bn1(x)
        x = F.dropout(x, p=0.6, training=False)
        x = F.elu(model.conv2(x, edge_index))
        x = model.bn2(x)
        x, (ei3, alpha3) = model.conv3(x, edge_index, return_attention_weights=True)
        x = model.bn3(F.elu(x))

        # Convert to dense <- not sure if this does anything
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        x_dense, mask = to_dense_batch(x, batch)

        alpha = alpha3.mean(dim=1)
        adj_dense = to_dense_adj(ei3, batch=batch, edge_attr=alpha)
        deg = adj_dense.sum(-1).clamp(min=1e-12)
        inv_sqrt = deg.pow(-0.5)
        adj_dense = inv_sqrt.unsqueeze(-1) * adj_dense * inv_sqrt.unsqueeze(-2)

        S, *_ = model.pool(x_dense, adj_dense, mask)

        clusters = S.argmax(dim=-1).squeeze(0).cpu().tolist()

    return pd.DataFrame({
        "gene_index": list(range(len(clusters))),
        "cluster":      clusters
    })

extract_cluster_assignments(model, test_loader, "cpu").to_csv("./clusters.csv")
