import os
import pandas as pd
import numpy as np
import re
import nltk
import pickle
import json
import hashlib
import time
from datetime import datetime
from collections import defaultdict, Counter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Ensure required directories exist
os.makedirs("rawData", exist_ok=True)
os.makedirs("cleanedData", exist_ok=True)
os.makedirs("pkl_files", exist_ok=True)
os.makedirs("cache", exist_ok=True)
os.makedirs("analytics", exist_ok=True)

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Config
chunk_size = 200  # Increased for faster processing
pooling_method = 'mean'  # Faster than max pooling
cache_size = 1000  # Number of queries to cache

# File mappings for dummy data
file_mappings = {
    "bug_reports.csv": {
        "encoding": "utf-8",
        "mapping": {
            "ID": "ID",
            "Name": "Name",
            "Description": "Description",
            "Cause": "Cause",
            "Solution": "Solution",
            "Verification": "Verification",
            "Deferral Justification": "Deferral Justification",
            "Issue Key": "Issue Key",
            "Area": "Area",
            "Rationale": "Rationale",
            "Application": "Application",
            "Teams": "Teams",
            "Tags": "Tags",
        },
    },
    "feature_requests.csv": {
        "encoding": "utf-8",
        "mapping": {
            "ID": "ID",
            "Name": "Name",
            "Description": "Description",
            "Cause": "Cause",
            "Solution": "Solution",
            "Verification": "Verification",
            "Deferral Justification": "Deferral Justification",
            "Issue Key": "Issue Key",
            "Area": "Area",
            "Rationale": "Rationale",
            "Application": "Application",
            "Teams": "Teams",
            "Tags": "Tags",
        },
    },
    "requirements.csv": {
        "encoding": "utf-8",
        "mapping": {
            "ID": "ID",
            "Name": "Name",
            "Description": "Description",
            "Cause": "Cause",
            "Solution": "Solution",
            "Verification": "Verification",
            "Deferral Justification": "Deferral Justification",
            "Issue Key": "Issue Key",
            "Area": "Area",
            "Rationale": "Rationale",
            "Application": "Application",
            "Teams": "Teams",
            "Tags": "Tags",
        },
    },
    "mixed_documents.csv": {
        "encoding": "utf-8",
        "mapping": {
            "ID": "ID",
            "Name": "Name",
            "Description": "Description",
            "Cause": "Cause",
            "Solution": "Solution",
            "Verification": "Verification",
            "Deferral Justification": "Deferral Justification",
            "Issue Key": "Issue Key",
            "Area": "Area",
            "Rationale": "Rationale",
            "Application": "Application",
            "Teams": "Teams",
            "Tags": "Tags",
        },
    },
}

final_columns = [
    "ID", "Name", "Description", "Cause", "Solution",
    "Verification", "Deferral Justification", "Issue Key", "Area",
    "Rationale", "Application", "Teams", "Tags"
]

class SearchCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key):
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = value
        self.access_times[key] = datetime.now()

class QueryExpander:
    def __init__(self):
        self.synonyms = {
            'bug': ['error', 'issue', 'problem', 'defect', 'fault'],
            'feature': ['functionality', 'capability', 'function', 'tool'],
            'performance': ['speed', 'efficiency', 'optimization', 'fast'],
            'security': ['protection', 'safety', 'vulnerability', 'threat'],
            'ui': ['interface', 'user interface', 'design', 'layout'],
            'database': ['db', 'data', 'storage', 'repository'],
            'api': ['interface', 'endpoint', 'service', 'integration'],
            'mobile': ['phone', 'smartphone', 'app', 'ios', 'android'],
            'web': ['website', 'browser', 'online', 'internet'],
            'login': ['authentication', 'signin', 'access', 'credentials']
        }
        
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
    
    def expand_query(self, query):
        """Expand query with synonyms and related terms"""
        words = query.lower().split()
        expanded_terms = set(words)
        
        for word in words:
            if word in self.synonyms:
                expanded_terms.update(self.synonyms[word])
        
        return ' '.join(expanded_terms)
    
    def clean_query(self, query):
        """Clean and normalize query"""
        # Remove special characters except spaces
        cleaned = re.sub(r'[^\w\s]', ' ', query.lower())
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned

class SearchAnalytics:
    def __init__(self):
        self.queries = []
        self.click_throughs = defaultdict(int)
        self.zero_result_queries = []
        self.popular_queries = Counter()
    
    def log_query(self, query, results_count, response_time):
        """Log search query for analytics"""
        self.queries.append({
            'query': query,
            'timestamp': datetime.now(),
            'results_count': results_count,
            'response_time': response_time
        })
        
        if results_count > 0:
            self.popular_queries[query] += 1
        else:
            self.zero_result_queries.append(query)
    
    def log_click(self, query, document_id):
        """Log document click for relevance feedback"""
        self.click_throughs[f"{query}:{document_id}"] += 1
    
    def get_popular_queries(self, limit=10):
        """Get most popular queries"""
        return [query for query, count in self.popular_queries.most_common(limit)]
    
    def save_analytics(self):
        """Save analytics data"""
        with open("analytics/search_analytics.json", "w") as f:
            json.dump({
                'queries': self.queries[-1000:],  # Keep last 1000 queries
                'popular_queries': dict(self.popular_queries.most_common(100)),
                'zero_result_queries': self.zero_result_queries[-100:],
                'click_throughs': dict(self.click_throughs)
            }, f, default=str)

def extract_columns(source_df, mapping, final_cols):
    new_df = pd.DataFrame()
    for col in final_cols:
        source_col = mapping.get(col)
        if source_col and source_col in source_df.columns:
            new_df[col] = source_df[source_col].apply(
                lambda x: None if (pd.isna(x) or x == "") else str(x)
            )
        else:
            new_df[col] = None
    return new_df

def clean_data(df):
    df['Name'] = df['Name'].fillna('')
    df['Description'] = df['Description'].fillna('')
    df['Tags'] = df['Tags'].fillna('')
    df['combined_text'] = df['Name'] + ". " + df['Description'] + ". " + df['Tags']
    df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'\s+', ' ', x.lower().strip()))
    
    # Add document type classification
    df['doc_type'] = df.apply(classify_document_type, axis=1)
    
    return df

def classify_document_type(row):
    """Classify document type based on content"""
    text = (row['Name'] + ' ' + row['Description']).lower()
    
    if any(word in text for word in ['bug', 'error', 'crash', 'issue', 'problem', 'defect']):
        return 'bug_report'
    elif any(word in text for word in ['feature', 'enhancement', 'request', 'improvement']):
        return 'feature_request'
    elif any(word in text for word in ['requirement', 'specification', 'standard', 'compliance']):
        return 'requirement'
    else:
        return 'general'

def tokenize(text):
    return nltk.word_tokenize(text)

def build_bm25_index(documents):
    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, tokenized_docs

def chunk_text(text, chunk_size=100):
    tokens = nltk.word_tokenize(text)
    return [" ".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]

def pool_embeddings(embeddings, pooling='max'):
    emb_array = np.vstack(embeddings)
    if pooling == 'max':
        return np.max(emb_array, axis=0)
    elif pooling == 'min':
        return np.min(emb_array, axis=0)
    elif pooling == 'mean':
        return np.mean(emb_array, axis=0)
    else:
        raise ValueError("Pooling method not recognized.")

def compute_chunked_embedding(text, model, chunk_size=100, pooling='max'):
    chunks = chunk_text(text, chunk_size)
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True)
    return pool_embeddings(chunk_embeddings, pooling=pooling)

def compute_embeddings_for_df(df, model, chunk_size=100, pooling='max'):
    """Compute embeddings with detailed progress tracking"""
    embeddings = []
    total_docs = len(df)
    start_time = time.time()
    
    print(f"🧠 Computing semantic embeddings for {total_docs:,} documents...")
    print(f"⏱️  Using model: {model}")
    print(f"📊 Estimated time: {total_docs * 0.01:.1f} seconds")
    print("-" * 60)
    
    for i, text in enumerate(df['combined_text']):
        # Progress updates every 100 documents
        if i % 100 == 0 or i == total_docs - 1:
            progress = (i + 1) / total_docs * 100
            elapsed = time.time() - start_time
            
            if i > 0:  # Calculate ETA after first few documents
                avg_time_per_doc = elapsed / (i + 1)
                remaining_docs = total_docs - (i + 1)
                eta_seconds = remaining_docs * avg_time_per_doc
                eta_minutes = eta_seconds / 60
                
                print(f"📈 Progress: {i+1:,}/{total_docs:,} ({progress:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {eta_minutes:.1f}min")
            else:
                print(f"📈 Progress: {i+1:,}/{total_docs:,} ({progress:.1f}%) | Starting...")
        
        embedding = compute_chunked_embedding(text, model, chunk_size, pooling)
        embeddings.append(embedding)
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"✅ Completed embeddings for all {total_docs:,} documents!")
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"⚡ Average speed: {total_docs/total_time:.1f} documents/second")
    
    return np.vstack(embeddings)

def build_tfidf_index(documents):
    """Build TF-IDF index for additional text features"""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

def create_document_clusters(df, n_clusters=50):
    """Create document clusters for better organization"""
    # Use a subset of data for clustering to avoid memory issues
    sample_size = min(10000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    # Simple clustering based on document type and tags
    cluster_features = []
    for _, row in sample_df.iterrows():
        features = [
            1 if row['doc_type'] == 'bug_report' else 0,
            1 if row['doc_type'] == 'feature_request' else 0,
            1 if row['doc_type'] == 'requirement' else 0,
            len(row['Tags'].split(',')) if pd.notna(row['Tags']) else 0,
            1 if 'critical' in str(row['Tags']).lower() else 0,
            1 if 'high' in str(row['Tags']).lower() else 0,
        ]
        cluster_features.append(features)
    
    if len(cluster_features) > n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_features)
        
        # Assign cluster labels to all documents
        df['cluster'] = 0  # Default cluster
        for i, idx in enumerate(sample_df.index):
            df.loc[idx, 'cluster'] = cluster_labels[i]
    
    return df

def save_pickle(obj, filename):
    with open(f"pkl_files/{filename}", "wb") as f:
        pickle.dump(obj, f)

def main():
    print("=== Advanced Document Search System ===")
    print("Building enhanced search index with optimizations...")
    
    # Load and transform data
    frames = []
    for filename, config in file_mappings.items():
        path = os.path.join("rawData", filename)
        if os.path.exists(path):
            print(f"Loading {filename}...")
            df = pd.read_csv(path, encoding=config["encoding"])
            extracted = extract_columns(df, config["mapping"], final_columns)
            frames.append(extracted)
        else:
            print(f"Warning: {filename} not found in rawData/")
    
    if not frames:
        print("No valid files found. Exiting.")
        return

    print("Combining datasets...")
    df_combined = pd.concat(frames, ignore_index=True)
    df_combined.to_csv("cleanedData/combined.csv", index=False)
    print(f"Combined dataset: {len(df_combined)} records")

    print("Cleaning and preprocessing data...")
    df_cleaned = clean_data(df_combined)
    df_cleaned.to_csv("cleanedData/cleaned.csv", index=False)
    
    print("Creating document clusters...")
    df_cleaned = create_document_clusters(df_cleaned)
    
    documents = df_cleaned["combined_text"].tolist()
    print(f"Processing {len(documents)} documents...")

    # Build BM25 index
    print("Building BM25 index...")
    bm25, tokenized_docs = build_bm25_index(documents)
    save_pickle(bm25, "bm25.pkl")
    save_pickle(tokenized_docs, "tokenized_docs.pkl")

    # Build TF-IDF index
    print("Building TF-IDF index...")
    tfidf_vectorizer, tfidf_matrix = build_tfidf_index(documents)
    save_pickle(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    save_pickle(tfidf_matrix, "tfidf_matrix.pkl")

    # Semantic embeddings
    print("Computing semantic embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Much faster model
    doc_embeddings = compute_embeddings_for_df(df_cleaned, model, chunk_size=chunk_size, pooling=pooling_method)
    save_pickle(doc_embeddings, "embeddings.pkl")
    save_pickle(df_cleaned, "cleaned_df.pkl")
    
    # Initialize and save supporting objects
    print("Initializing search components...")
    cache = SearchCache(cache_size)
    query_expander = QueryExpander()
    analytics = SearchAnalytics()
    
    save_pickle(cache, "search_cache.pkl")
    save_pickle(query_expander, "query_expander.pkl")
    save_pickle(analytics, "analytics.pkl")
    
    print("=== Index Building Complete ===")
    print(f"Total documents indexed: {len(df_cleaned)}")
    print(f"Document types: {df_cleaned['doc_type'].value_counts().to_dict()}")
    print("All components saved successfully!")

if __name__ == "__main__":
    main()
