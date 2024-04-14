import os
import sys

# Get the parent directory path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Append the parent directory to sys.path for relative imports
sys.path.append(parent_dir)

import settings as s
from utils import seed_everything

import torch
import pandas as pd
from torch_geometric.nn import LightGCN
from tqdm import tqdm

def train(model, optimizer, train_edges, train_loader, device):
    """Train the model."""
    total_loss = total_examples = 0

    for index in train_loader:
        pos_edge_label_index = train_edges[:, index]
        
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_users, num_users + num_news,
                          (index.numel(), ), device=device)
        ], dim=0)
        
        edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)
        
        optimizer.zero_grad()
        pos_rank, neg_rank = model(train_edges, edge_label_index).chunk(2)
        
        loss = model.recommendation_loss(pos_rank, neg_rank, node_id=edge_label_index.unique())
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()

    return total_loss / total_examples

if __name__ == "__main__":
    # Set seed and device
    seed_everything(s.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    # Load edges data and create DataLoader
    train_edges = torch.load(s.gnn_edges_path).to(device)
    train_loader = torch.utils.data.DataLoader(torch.arange(train_edges.shape[1]), shuffle=True, batch_size=s.gnn_batch_size)

    num_users = train_edges[0].unique().size(0)
    num_news = train_edges[1].unique().size(0)

    # Initialize model and optimizer
    model = LightGCN(num_nodes=num_users+num_news, **s.gnn_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    for epoch in tqdm(range(1, s.gnn_train_epochs+1)):
        loss = train(model, optimizer, train_edges, train_loader, device)

        if epoch % 2 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    # Extract and save user embeddings
    user_embeddings = model.get_embedding(train_edges)[:num_users].detach().cpu()
    pd.DataFrame(user_embeddings.numpy()).add_prefix('user_').to_parquet(s.user_embedding_path)