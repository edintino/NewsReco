# NewsReco

## Summary

This project showcases the end-to-end development of a News Recommendation System, encompassing data preprocessing, embedding generation, model training, evaluation, and deployment. By leveraging advanced techniques such as GNNs and MLPs, the system can provide personalized news recommendations, enhancing user engagement and satisfaction.

Moreover, this project's architecture is tailored to simulate a real-time inference setup, perfectly suited for online newspapers. Upon publication of a new article, the system can promptly utilize a Named Entity Recognition (NER) model to extract the news' latent representation. For known readers, personalized recommendations can be instantly provided without the necessity of retraining the model. This capability ensures seamless real-time inference, aligning with the dynamic requirements of an online newspaper ecosystem.

## Data Preparation

The data preparation phase is crucial for building an effective news recommendation system. In this phase, raw data collected from user interactions, such as clicks and impressions, along with metadata about news articles, is processed and transformed into a format suitable for model training.

**Behavioral Logs**: The behavioral logs contain information about user interactions with news articles, such as impression IDs, user IDs, timestamps, and lists of previously viewed articles (history). These logs are loaded into a DataFrame and preprocessed to extract relevant features.

**Entity Embeddings**: Entity embeddings represent semantic information about news articles, capturing their content and context. These embeddings are learned from textual data using techniques like Word2Vec or GloVe and provide a dense vector representation of each news article.

**News Articles Metadata**: Metadata about news articles, such as categories, subcategories, titles, and abstracts, is loaded into a DataFrame. This metadata is essential for understanding the content and context of news articles, which is used later in the recommendation process.

## User and News Embeddings

After data preparation, the next step is to generate embeddings for users and news articles. Embeddings are low-dimensional representations of entities in a high-dimensional space, learned through techniques like Graph Neural Networks (GNNs) or collaborative filtering.

**LightGCN Model**: In this project, a LightGCN model is employed to learn user and news embeddings from the interaction graph. LightGCN is a lightweight graph convolutional network designed for collaborative filtering tasks, which efficiently captures user-item interactions and learns embeddings by propagating information through the graph structure.

**Embedding Generation**: The LightGCN model learns embeddings by optimizing an objective function that minimizes the difference between observed user-item interactions and predicted preferences. By iteratively updating the embeddings using gradient descent, the model converges to a set of embeddings that capture the underlying relationships between users and news articles.

## Recommender Model

With user and news embeddings generated, the next step is to build a recommender model that leverages these embeddings to make personalized recommendations to users. In this project, a Multilayer Perceptron (MLP) model is employed for recommendation, which predicts user preferences based on their embedding vectors.

**Model Architecture**: The MLP model consists of multiple layers of neurons, where each layer performs a nonlinear transformation of the input data. The model takes as input the concatenated user and news embeddings and predicts the probability that the user will interact with each news article.

**Training and Evaluation**: The model is trained using a binary classification objective, where the target variable indicates whether the user interacted with a given news article or not. The model is trained using gradient descent optimization and evaluated using metrics such as Area Under the ROC Curve (AUC) to assess its performance in predicting user preferences.

## Streamlit Application

To demonstrate the functionality of the recommendation system, a Streamlit web application is developed. Streamlit provides an easy-to-use interface for building interactive web applications using Python, allowing users to interact with the recommendation system in real-time.

**User Interface**: The Streamlit application features a sidebar where users can select their preferences and user ID. Based on the selected preferences, the application displays recommended news articles tailored to the user's interests.

**Recommendation Display**: The recommended news articles are displayed in a user-friendly format, along with relevant metadata such as titles, categories, and subcategories. Additionally, the application highlights articles that were previously clicked by the user, providing context for the recommendations.

## Results

Given the constraints of utilizing only a fraction of the available data and refraining from additional training data, the achieved results closely rival state-of-the-art performance. Notably, the system's test set AUC of 0.634 compares favorably with the current state-of-the-art methods in news recommendation, as documented in the latest benchmarks available on Papers with Code. Despite these limitations, the system demonstrates promising efficacy and scalability, showcasing its potential for further optimization and enhancement.

## Further improvement ideas

### Enhancing Article Representations with Semantical Text Models
Harness the power of pre-trained transformer-based architectures (BERT, GPT) to extract nuanced semantic information from the textual content of articles, thereby enhancing the depth and richness of the article representations.

### Incorporating Named Entity Recognition (NER) Annotations
Mitigate the impact of missing Named Entity Recognition (NER) annotations for a significant portion of articles in the dataset. Prioritize articles containing named entities, as they often convey more structured and informative content.

### Exploring Alternative Loss Functions
Explore alternative loss functions to improve model optimization. Evaluate the effectiveness of Learning to Rank (LETOR) loss functions, such as Bayesian Personalized Ranking (BPR), specifically designed for recommendation tasks.

## Example usage setup

The dataset can be downloaded from the following link: [MSNews Dataset](https://msnews.github.io/). Once downloaded, please move the data into a folder named "train" within the root directory of the project. It's important to note that only the training data was utilized for this project, and even within that, only a portion was used.

```bash

# Set up a virtual environment using conda
conda create -n myenv python=3.10

# Activate environment
conda activate myenv

# Install dependencies
pip install -r requirements.txt

# Preprocess the data
python3 run.py preprocessing

# Generate user embeddings
python3 run.py user_embedding

# Train the recommendation model
python3 run.py train_model

# Run streamlit application to see the recommender in action
streamlit run app.py

```