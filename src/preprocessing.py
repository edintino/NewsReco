import os
import sys
from pathlib import Path

import pandas as pd
import torch

from ast import literal_eval
from dataclasses import dataclass, field

# Get the parent directory path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Append the parent directory to sys.path for relative imports
sys.path.append(parent_dir)

import settings as s
from utils import seed_everything, write_json

# Create directories if they don't exist
for path in [s.gnn_edges_path, s.impressions_path]:
    Path(os.path.join(*path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)


@dataclass
class DataPreprocessor:
    """
    Class to load and preprocess data for news recommendation.
    """
    behavior_path: str
    entity_path: str
    news_path: str
    behavior_sample_size: float = 0.25
    behavior_names: list = field(default_factory=lambda: ['impression_id', 'user_id', 'time', 'history', 'impressions'])
    news_names: list = field(default_factory=lambda: ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
    news_map: dict = field(default_factory=lambda: {})

    def load_data(self):
        """
        Load data from files.
        """
        self.behavior = pd.read_csv(self.behavior_path, sep='\t', header=None, names=self.behavior_names)
        if self.behavior_sample_size:
            self.behavior = self.behavior.sample(frac=self.behavior_sample_size)
        self.entity = pd.read_csv(self.entity_path, sep='\t', header=None)
        self.news = pd.read_csv(self.news_path, sep='\t', header=None, names=self.news_names)
        self.news[['news_id', 'category', 'subcategory', 'title']].to_parquet(s.news_data_path)

    def preprocess_news_data(self):
        """
        Preprocess news data.
        """
        self.news['title_entities'] = self.news['title_entities'].fillna('[]').apply(literal_eval)
        self.news['entities'] = self.news['title_entities'].apply(lambda row: [entity['WikidataId'] for entity in row])
        self.news = self.news.explode('entities')[['news_id', 'entities']]
        self.news = self.news.merge(self.entity, 'left', left_on='entities', right_on=0).drop(columns=[0, 101, 'entities'])
        self.news.fillna(0, inplace=True)
        self.news = self.news.groupby('news_id').mean()

if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(s.seed)

    # Load and preprocess data
    data = DataPreprocessor(**s.raw_data_paths)
    data.load_data()
    data.preprocess_news_data()

    # Extract relevant data
    news = data.news.add_prefix('news_')
    impressions = data.behavior[['user_id', 'impressions']].copy()
    train_edge = data.behavior[['user_id', 'history']].copy()
    train_edge['history'] = train_edge['history'].str.split()
    train_edge.to_parquet(s.user_history_path)
    train_edge = train_edge.explode('history').rename(columns={'history': 'news_id'}).dropna()

    # Process impressions data
    impressions['impressions'] = impressions['impressions'].str.split()
    impressions = impressions.explode('impressions')
    impressions[['news_id', 'target']] = impressions['impressions'].str.split('-', expand=True)
    impressions['target'] = impressions['target'].astype(int)
    impressions.drop(columns='impressions', inplace=True)

    # Create user and news_id mappings
    user_map = dict(zip(train_edge['user_id'].unique(), range(train_edge['user_id'].nunique())))
    embeddingnews_map = dict(zip(train_edge['news_id'].unique(), range(train_edge['news_id'].nunique())))
    train_edge['user_id'] = train_edge['user_id'].map(user_map)
    train_edge['news_id'] = train_edge['news_id'].map(embeddingnews_map)

    # Save processed data
    torch.save(torch.from_numpy(train_edge.values).T, s.gnn_edges_path)
    impressions.to_parquet(s.impressions_path, index=False)
    news.reset_index().to_parquet(s.news_embedding_path, index=False)

    # Save user map
    write_json(user_map, s.user_map_path)