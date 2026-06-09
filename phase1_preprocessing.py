import pandas as pd
import numpy as np
import ast
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# Configuration & Setup
# ---------------------------------------------------------
RAW_DATA_DIR = "."
PROCESSED_DATA_DIR = "./processed_data"
CHECKPOINT_DIR = "./checkpoints"
PLOTS_DIR = "./plots"

# Create directories
for directory in [PROCESSED_DATA_DIR, CHECKPOINT_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def clean_string_list(val):
    """Safely parse stringified lists from the features column."""
    if pd.isna(val):
        return ""
    try:
        parsed_list = ast.literal_eval(val)
        if isinstance(parsed_list, list):
            return " ".join(parsed_list)
    except (ValueError, SyntaxError):
        pass
    return str(val)

def preprocess_item_metadata(item_meta_df):
    """Cleans metadata, imputes missing values, and prepares text."""
    print("Cleaning Item Metadata...")
    df = item_meta_df.copy()
    
    # 1. Impute missing categories
    df['main_category'] = df['main_category'].fillna('Unknown')
    
    # 2. Impute missing prices with the median of their respective main_category
    category_medians = df.groupby('main_category')['price'].transform('median')
    # If a category median is entirely NaN, fill with the global median
    global_median = df['price'].median()
    df['price'] = df['price'].fillna(category_medians).fillna(global_median)
    
    # Normalize price
    scaler = MinMaxScaler()
    df['normalized_price'] = scaler.fit_transform(df[['price']])
    
    # 3. Clean Text Columns
    df['features'] = df['features'].apply(clean_string_list)
    df['title'] = df['title'].fillna("")
    df['description'] = df['description'].fillna("")
    
    # 4. Create Unified Text Corpus
    df['unified_text'] = (
        df['title'] + " " + 
        df['features'] + " " + 
        df['description'] + " " + 
        df['main_category']
    ).str.lower()
    
    return df

def generate_text_embeddings(df, n_components=128):
    """Generates from-scratch embeddings using TF-IDF and SVD."""
    print(f"Generating TF-IDF and SVD embeddings (Dim: {n_components})...")
    
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
    sparse_vectors = tfidf.fit_transform(df['unified_text'])
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    dense_vectors = svd.fit_transform(sparse_vectors)
    
    # Save the models for reproducibility
    with open(f"{PROCESSED_DATA_DIR}/tfidf_model.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(f"{PROCESSED_DATA_DIR}/svd_model.pkl", "wb") as f:
        pickle.dump(svd, f)
        
    return dense_vectors

def process_interaction_sequences(train_df, min_interactions=3):
    """Sorts, deduplicates, and groups interactions into sequences."""
    print("Processing Interaction Sequences...")
    df = train_df.copy()
    
    # 1. Sort chronologically
    df = df.sort_values(by=['user_id', 'timestamp'])
    
    # 2. Deduplicate consecutive identical interactions
    # Shift item_id by 1 within each user group and compare
    df['prev_item'] = df.groupby('user_id')['item_id'].shift(1)
    df = df[df['item_id'] != df['prev_item']].drop(columns=['prev_item'])
    
    # 3. Filter pathological users (too few interactions to learn a sequence)
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df['user_id'].isin(valid_users)]
    
    # 4. Group into sequences
    sequences = df.groupby('user_id')['item_id'].apply(list).reset_index()
    sequences.rename(columns={'item_id': 'item_sequence'}, inplace=True)
    
    return sequences

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    # Load Datasets
    print("Loading data...")
    train_df = pd.read_csv(f"{RAW_DATA_DIR}/train.csv")
    item_meta_df = pd.read_csv(f"{RAW_DATA_DIR}/item_meta.csv")
    
    # Process Metadata
    cleaned_meta = preprocess_item_metadata(item_meta_df)
    
    # Generate Dense Text Embeddings
    text_embeddings = generate_text_embeddings(cleaned_meta, n_components=128)
    
    # Combine normalized price and text embeddings into the final feature matrix
    # Shape: [num_items, 129]
    prices = cleaned_meta['normalized_price'].values.reshape(-1, 1)
    final_item_features = np.hstack([text_embeddings, prices])
    
    # Map item_ids to row indices for the feature matrix
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(cleaned_meta['item_id'])}
    
    # Process Interactions
    sequences_df = process_interaction_sequences(train_df)
    
    # Save Processed Artifacts
    print("Saving artifacts...")
    np.save(f"{PROCESSED_DATA_DIR}/item_features.npy", final_item_features)
    
    with open(f"{PROCESSED_DATA_DIR}/item_id_to_idx.pkl", "wb") as f:
        pickle.dump(item_id_to_idx, f)
        
    sequences_df.to_csv(f"{PROCESSED_DATA_DIR}/user_sequences.csv", index=False)
    
    print("Phase 1 Complete! Artifacts saved in ./processed_data/")

if __name__ == "__main__":
    main()