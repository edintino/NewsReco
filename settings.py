seed = 7843

raw_data_paths = {
    'behavior_path' : './train/behaviors.tsv',
    'entity_path' : './train/entity_embedding.vec',
    'news_path' : './train/news.tsv',
}

news_data_path = './results/news_data.parquet'
impressions_path = './results/impressions.parquet'
news_embedding_path = './results/news_embedding.parquet'
user_embedding_path = './results/user_embedding.parquet'
user_history_path = './processed/user_history.parquet'

model_path = './results/recommender.pt'
user_map_path = './results/user_map.json'
gnn_edges_path = './processed/train_edges.pt'

gnn_batch_size = 8192
gnn_train_epochs = 12

gnn_params = {
    'embedding_dim' : 100,
    'num_layers' : 3,
}

classifier_params = {
    'input_size' : 200,
    'hidden_size' : 32,
    'output_size' : 1,
}

classifier_train_epochs = 5000