import os
import sys

# Get the parent directory path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Append the parent directory to sys.path for relative imports
sys.path.append(parent_dir)

import pandas as pd
import torch

from utils import seed_everything, Recommender, load_json
import settings as s

if __name__ == "__main__":
    seed_everything(s.seed)

    df = pd.read_parquet(s.impressions_path)
    news_emb = pd.read_parquet(s.news_embedding_path)
    user_emb = pd.read_parquet(s.user_embedding_path)

    user_map = load_json(s.user_map_path)

    # Create Recommender instance and train the model
    recommender = Recommender(s.impressions_path, s.user_embedding_path, s.news_embedding_path, s.user_map_path)
    recommender.train()

    torch.save(recommender.state_dict(), s.model_path)