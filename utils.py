import json
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import settings as s

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f)

def seed_everything(seed: int) -> None:
    """
    Set seed for random number generators in PyTorch, NumPy, and Python.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MLP(nn.Module):
    """Multilayer Perceptron model."""
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

class Recommender(nn.Module):
    """Recommender system."""
    def __init__(self, data_path, user_embedding_path, news_embedding_path, user_map_path, is_inference=False):
        """
        Initialize Recommender object.

        Args:
            data_path (str): Path to DataFrame containing user-item interactions.
            user_embeddings_path (str): Path to DataFrame containing user embeddings.
            news_embeddings_path (str): Path to DataFrame containing news embeddings.
            user_map_path (str): Path to dictionary mapping user IDs to indices.

        """
        super(Recommender, self).__init__()
        
        self.data_path = data_path
        self.user_embedding_path = user_embedding_path
        self.news_embedding_path = news_embedding_path

        self.user_embeddings = pd.read_parquet(self.user_embedding_path)
        self.news_embeddings = pd.read_parquet(self.news_embedding_path)
        self.user_map = load_json(user_map_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = MLP(**s.classifier_params)#.to(self.device)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        if not is_inference: self.prepare_train_data()

    def prepare_train_data(self):
        data = pd.read_parquet(self.data_path)

        data['user_id'] = data['user_id'].map(self.user_map)
        data.dropna(inplace=True)
        data['user_id'] = data['user_id'].astype(int)
        data = data.merge(self.user_embeddings, 'left', left_on='user_id', right_index=True)
        data = data.merge(self.news_embeddings, 'left', 'news_id').drop(columns=['user_id', 'news_id'])
        train, test = train_test_split(data, test_size=0.33, random_state=s.seed)

        self.X_train = torch.from_numpy(train.drop(columns='target').values).float().to(self.device)
        self.y_train = torch.from_numpy(train[['target']].values).float().to(self.device)
        self.X_test = torch.from_numpy(test.drop(columns='target').values).float()
        self.y_test = torch.from_numpy(test[['target']].values).float()
    
    @torch.no_grad()
    def test(self):
        """
        Evaluate model on test data.

        Returns:
            float: AUC score on test data.
        """
        self.model.to('cpu')
        self.model.eval()
        outputs = self.model(self.X_test)
        return roc_auc_score(self.y_test, outputs)

    def train(self):
        """
        Train the model.
        """
        for epoch in range(1, s.classifier_train_epochs+1):
            self.model.to(self.device)
            self.model.train()
            outputs = self.model(self.X_train)
            loss = self.criterion(outputs, self.y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            test_auc = self.test()
            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{s.classifier_train_epochs}], Loss: {loss.item():.4f}',
                      f'Train AUC: {roc_auc_score(self.y_train.cpu(), outputs.detach().cpu()):.4f}, Test AUC: {test_auc:.4f}')

    def recommend(self, user, news, k=5, device='cpu'):
        """
        Recommend top news articles for a user.

        Args:
            user (str): User ID.
            news (list): List of news article IDs.
            k (int): Number of recommendations.
            device (str): Device to run inference on.

        Returns:
            list: Top k recommended news article IDs.
        """
        self.model.to(device)
        self.model.eval()
        df = (self.user_embeddings.loc[[self.user_map[user]]].assign(dummy=1)
              .merge(self.news_embeddings[self.news_embeddings['news_id'].isin(news)].assign(dummy=1), on='dummy')
              .drop(['dummy', 'news_id'], axis=1)
              )
        X = torch.from_numpy(df.values).float().to(device)
        with torch.no_grad():
            outputs = self.model(X)
        return np.array(news)[torch.topk(outputs.reshape(-1), k).indices.cpu()]

def highlight_clicked_rows(row):
    color = 'lightgreen' if row['is_clicked'] else 'white'
    return ['background-color: {}'.format(color)] * len(row)