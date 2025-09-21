import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict, Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(
    page_title="Smart Backlog Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- CUSTOM CSS -------------------------
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --text-color: #2d3748;
        --bg-color: #f7fafc;
    }
    
    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header */
    .header-container {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    /* Search input styling */
    .stTextInput > div > div > input {
        border-radius: 15px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 1rem !important;
        font-size: 1.1rem !important;
        background: white !important;
        color: #333 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 15px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .css-1d391kg .stSelectbox > div > div {
        background: white;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    .css-1d391kg .stMultiSelect > div > div {
        background: white;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    .css-1d391kg .stSlider > div > div {
        background: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border-left: 5px solid #28a745;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100px;
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 50%;
        transform: translate(30px, -30px);
        opacity: 0.1;
    }
    
    .search-result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border-left: 5px solid #28a745;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .search-result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        border-left-color: #20c997;
    }
    
    .search-result-card::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 80px;
        height: 80px;
        background: linear-gradient(45deg, #28a745, #20c997);
        border-radius: 50%;
        transform: translate(25px, -25px);
        opacity: 0.1;
    }
    
    .similarity-score {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        text-align: center;
        min-width: 80px;
    }
    
    /* Tag pills */
    .tag-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .tag-critical { background: linear-gradient(45deg, #dc3545, #c82333); color: white; }
    .tag-high { background: linear-gradient(45deg, #fd7e14, #e55a00); color: white; }
    .tag-medium { background: linear-gradient(45deg, #ffc107, #e0a800); color: #212529; }
    .tag-low { background: linear-gradient(45deg, #28a745, #1e7e34); color: white; }
    .tag-database { background: linear-gradient(45deg, #6f42c1, #5a32a3); color: white; }
    .tag-ui { background: linear-gradient(45deg, #20c997, #17a2b8); color: white; }
    .tag-mobile { background: linear-gradient(45deg, #17a2b8, #138496); color: white; }
    .tag-security { background: linear-gradient(45deg, #dc3545, #c82333); color: white; }
    .tag-performance { background: linear-gradient(45deg, #fd7e14, #e55a00); color: white; }
    .tag-default { background: linear-gradient(45deg, #6c757d, #5a6268); color: white; }
    
    /* Purple-Blue Clear All Button */
    div[data-testid="stButton"] button[kind="secondary"] {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Purple-Blue Recent Search Buttons */
    div[data-testid="stButton"] button:not([kind="primary"]):not([kind="secondary"]) {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stButton"] button:not([kind="primary"]):not([kind="secondary"]):hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 3rem 2rem;
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
        border-radius: 20px;
        margin-top: 3rem;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin: 2rem 0;
    }
    
    .empty-state h3 {
        color: #6c757d;
        margin-bottom: 1rem;
    }
    
    .empty-state p {
        color: #adb5bd;
        font-size: 1.1rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .main .block-container {
            padding: 1rem;
        }
    }
    
    /* Input field text visibility */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div > div {
        color: #333 !important;
        background: white !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #666 !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------- HELPER FUNCTIONS -------------------------
def get_search_suggestions(query, popular_queries):
    """Get search suggestions based on query and popular queries"""
    suggestions = []
    
    # Add popular queries that contain the current query
    for popular in popular_queries:
        if query.lower() in popular.lower() and popular.lower() != query.lower():
            suggestions.append(popular)
    
    # Add common search patterns
    common_patterns = [
        f"{query} bug",
        f"{query} issue", 
        f"{query} problem",
        f"{query} error",
        f"{query} feature",
        f"{query} requirement"
    ]
    
    suggestions.extend(common_patterns[:3])  # Limit to 3 suggestions
    
    return suggestions[:5]  # Return max 5 suggestions

def create_analytics_charts(analytics_data):
    """Create analytics charts"""
    try:
        # Popular queries chart
        queries = analytics_data.get('popular_queries', {})
        if queries:
            df_queries = pd.DataFrame(list(queries.items()), columns=['Query', 'Count'])
            df_queries = df_queries.head(10)
            
            fig_queries = px.bar(
                df_queries, 
                x='Count', 
                y='Query',
                orientation='h',
                title="Most Popular Queries",
                color='Count',
                color_continuous_scale='viridis'
            )
            fig_queries.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        else:
            fig_queries = None
        
        # Response time chart
        response_times = analytics_data.get('response_times', [])
        if response_times:
            df_response = pd.DataFrame(response_times, columns=['Time'])
            df_response['Time'] = pd.to_datetime(df_response['Time'])
            df_response['Hour'] = df_response['Time'].dt.hour
            
            hourly_avg = df_response.groupby('Hour').size().reset_index(name='Count')
            
            fig_response = px.line(
                hourly_avg,
                x='Hour',
                y='Count',
                title="Search Activity by Hour",
                markers=True
            )
            fig_response.update_layout(height=400)
        else:
            fig_response = None
            
        return fig_queries, fig_response
        
    except Exception as e:
        st.error(f"Error creating charts: {e}")
        return None, None

# ------------------------- INITIALIZE SESSION STATE -------------------------
if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "expand_states" not in st.session_state:
    st.session_state.expand_states = {}

if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {
        'preferred_search_mode': 'Hybrid',
        'default_results': 10,
        'show_analytics': True
    }

# ------------------------- LOAD FILES (only once) -------------------------
if "bm25" not in st.session_state:
    with st.spinner("Loading advanced search components..."):
        try:
            with open("pkl_files/bm25.pkl", "rb") as f:
                st.session_state.bm25 = pickle.load(f)
            with open("pkl_files/tokenized_docs.pkl", "rb") as f:
                st.session_state.tokenized_docs = pickle.load(f)
            with open("pkl_files/embeddings.pkl", "rb") as f:
                st.session_state.doc_embeddings = pickle.load(f)
            with open("pkl_files/cleaned_df.pkl", "rb") as f:
                st.session_state.df = pickle.load(f)
            with open("pkl_files/tfidf_vectorizer.pkl", "rb") as f:
                st.session_state.tfidf_vectorizer = pickle.load(f)
            with open("pkl_files/tfidf_matrix.pkl", "rb") as f:
                st.session_state.tfidf_matrix = pickle.load(f)
            
            # Load enhanced components
            try:
                with open("pkl_files/query_expander.pkl", "rb") as f:
                    st.session_state.query_expander = pickle.load(f)
            except:
                st.session_state.query_expander = None
            
            try:
                with open("pkl_files/analytics.pkl", "rb") as f:
                    st.session_state.analytics = pickle.load(f)
            except:
                st.session_state.analytics = None
            
            st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")  # Faster model
            st.success("✅ All components loaded successfully!")
            
        except FileNotFoundError as e:
            st.warning("🔄 Model files not found. Building models from scratch...")
            
            # Import required modules for model building
            import os
            import pandas as pd
            import numpy as np
            import re
            import nltk
            from rank_bm25 import BM25Okapi
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Ensure required directories exist
            os.makedirs("rawData", exist_ok=True)
            os.makedirs("cleanedData", exist_ok=True)
            os.makedirs("pkl_files", exist_ok=True)
            os.makedirs("analytics", exist_ok=True)
            
            # Download NLTK data
            nltk.download("punkt_tab", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            
            # Build models directly
            with st.spinner("Building search models... This may take 5-10 minutes on first run."):
                try:
                    # Generate dummy data
                    st.info("📊 Generating dummy data...")
                    from generate_dummy_data import generate_dummy_data, create_csv_files
                    
                    records = generate_dummy_data(25000)  # Smaller dataset for faster deployment
                    create_csv_files(records)
                    
                    # Load and process data
                    st.info("🔄 Processing data...")
                    from model import extract_columns, clean_data, create_document_clusters, build_bm25_index, build_tfidf_index, compute_embeddings_for_df
                    
                    # Load data (replicating main function logic)
                    frames = []
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
                        "incidents.csv": {
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
                        }
                    }
                    final_columns = ["ID", "Name", "Description", "Cause", "Solution", "Verification", "Deferral Justification", "Issue Key", "Area", "Rationale", "Application", "Teams", "Tags"]
                    
                    for filename, config in file_mappings.items():
                        path = os.path.join("rawData", filename)
                        if os.path.exists(path):
                            df = pd.read_csv(path, encoding=config["encoding"])
                            extracted = extract_columns(df, config["mapping"], final_columns)
                            frames.append(extracted)
                    
                    if not frames:
                        st.error("No valid files found in rawData/")
                        st.stop()
                    
                    df_combined = pd.concat(frames, ignore_index=True)
                    df_cleaned = clean_data(df_combined)
                    df_cleaned = create_document_clusters(df_cleaned)
                    
                    # Build BM25 index
                    st.info("🔍 Building BM25 index...")
                    documents = df_cleaned['combined_text'].tolist()
                    bm25, tokenized_docs = build_bm25_index(documents)
                    
                    # Build TF-IDF index
                    st.info("📊 Building TF-IDF index...")
                    tfidf_vectorizer, tfidf_matrix = build_tfidf_index(documents)
                    
                    # Build semantic embeddings
                    st.info("🧠 Building semantic embeddings...")
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    doc_embeddings = compute_embeddings_for_df(df_cleaned, model, chunk_size=200, pooling='mean')
                    
                    # Save all models
                    st.info("💾 Saving models...")
                    with open("pkl_files/bm25.pkl", "wb") as f:
                        pickle.dump(bm25, f)
                    with open("pkl_files/tokenized_docs.pkl", "wb") as f:
                        pickle.dump(tokenized_docs, f)
                    with open("pkl_files/embeddings.pkl", "wb") as f:
                        pickle.dump(doc_embeddings, f)
                    with open("pkl_files/cleaned_df.pkl", "wb") as f:
                        pickle.dump(df_cleaned, f)
                    with open("pkl_files/tfidf_vectorizer.pkl", "wb") as f:
                        pickle.dump(tfidf_vectorizer, f)
                    with open("pkl_files/tfidf_matrix.pkl", "wb") as f:
                        pickle.dump(tfidf_matrix, f)
                    
                    # Store in session state
                    st.session_state.bm25 = bm25
                    st.session_state.tokenized_docs = tokenized_docs
                    st.session_state.doc_embeddings = doc_embeddings
                    st.session_state.df = df_cleaned
                    st.session_state.tfidf_vectorizer = tfidf_vectorizer
                    st.session_state.tfidf_matrix = tfidf_matrix
                    st.session_state.model = model
                    st.session_state.query_expander = None
                    st.session_state.analytics = None
                    
                    st.success("✅ Models built and loaded successfully!")
                    
                except Exception as build_error:
                    st.error(f"❌ Failed to build models: {build_error}")
                    st.stop()

# Assign to working variables
bm25 = st.session_state.bm25
tokenized_docs = st.session_state.tokenized_docs
doc_embeddings = st.session_state.doc_embeddings
df = st.session_state.df
model = st.session_state.model
tfidf_vectorizer = st.session_state.tfidf_vectorizer
tfidf_matrix = st.session_state.tfidf_matrix
query_expander = st.session_state.query_expander
analytics = st.session_state.analytics

# ------------------------- HEADER -------------------------
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🔍 Smart Backlog Search Engine</h1>
    <p class="header-subtitle">Find exactly what you need with AI-powered search</p>
</div>
""", unsafe_allow_html=True)

# ------------------------- SIDEBAR -------------------------
with st.sidebar:
    st.markdown("### 🔍 Search Controls")
    
    # Search mode
    mode = st.selectbox(
        "Search Mode",
        ["Hybrid", "BM25 Only", "Semantic Only", "TF-IDF Only"],
        index=0
    )
    
    # Search mode explanations
    with st.expander("ℹ️ What do these search modes do?", expanded=False):
        st.markdown("""
        **🔀 Hybrid Mode** (Recommended)
        - Combines keyword matching + semantic understanding
        - Best for general searches and finding related content
        - Most accurate and comprehensive results
        
        **🔍 BM25 Only**
        - Pure keyword-based search
        - Fast and precise for exact term matches
        - Good for finding specific technical terms
        
        **🧠 Semantic Only**
        - AI-powered meaning-based search
        - Finds conceptually similar content
        - Great for discovering related topics
        
        **📊 TF-IDF Only**
        - Statistical keyword weighting
        - Balances term frequency and rarity
        - Good for academic or technical documents
        """)
    
    # Number of results
    top_n = st.slider("Number of Results", min_value=5, max_value=50, value=10, step=5)
    
    # Filters
    st.markdown("### 🔧 Filters")
    
    # Document type filter
    doc_types = ['All'] + list(df['doc_type'].unique())
    selected_doc_type = st.selectbox("Document Type", doc_types)
    
    # Area filter
    areas = ['All'] + [area for area in df['Area'].dropna().unique() if area]
    selected_area = st.selectbox("Area", areas)
    
    # Team filter
    teams = ['All'] + [team for team in df['Teams'].dropna().unique() if team]
    selected_team = st.selectbox("Team", teams)
    
    # Application filter
    applications = ['All'] + [app for app in df['Application'].dropna().unique() if app]
    selected_app = st.selectbox("Application", applications)
    
    # Tag filtering
    st.markdown("### 🏷️ Tag Filters")
    
    # Extract all unique tags
    all_tags = set()
    for tags_str in df['Tags'].dropna():
        if pd.notna(tags_str):
            tags = [tag.strip() for tag in str(tags_str).split(',')]
            all_tags.update(tags)
    
    # Priority tags
    priority_tags = [tag for tag in all_tags if tag.lower() in ['critical', 'high', 'medium', 'low', 'urgent']]
    if priority_tags:
        selected_priority = st.multiselect("Priority Tags", priority_tags, default=[])
    
    # Category tags
    category_tags = [tag for tag in all_tags if tag.lower() in ['database', 'ui', 'backend', 'frontend', 'mobile', 'web', 'api', 'security', 'performance']]
    if category_tags:
        selected_categories = st.multiselect("Category Tags", category_tags, default=[])
    
    # Type tags
    type_tags = [tag for tag in all_tags if tag.lower() in ['bug', 'feature', 'enhancement', 'requirement', 'functional', 'non-functional']]
    if type_tags:
        selected_types = st.multiselect("Type Tags", type_tags, default=[])
    
    # Custom tag search
    custom_tag = st.text_input("Custom Tag Search", placeholder="Enter specific tag...")
    
    # Build filters dict
    filters = {}
    if selected_doc_type != 'All':
        filters['doc_type'] = selected_doc_type
    if selected_area != 'All':
        filters['area'] = selected_area
    if selected_team != 'All':
        filters['team'] = selected_team
    if selected_app != 'All':
        filters['application'] = selected_app
    if 'selected_priority' in locals() and selected_priority:
        filters['priority_tags'] = selected_priority
    if 'selected_categories' in locals() and selected_categories:
        filters['category_tags'] = selected_categories
    if 'selected_types' in locals() and selected_types:
        filters['type_tags'] = selected_types
    if custom_tag:
        filters['custom_tag'] = custom_tag
    
    # Analytics toggle
    show_analytics = st.checkbox("Show Analytics", value=True)

# ------------------------- MAIN SEARCH INTERFACE -------------------------
col1, col2 = st.columns([4, 1])

with col1:
    # Search input with suggestions
    query = st.text_input(
        "Search Documents",
        placeholder="Enter your search query... (try 'authentication bug' or 'performance optimization')",
        key="main_search"
    )

with col2:
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)  # Align with input
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

# Recent searches section - moved below search bar
if st.session_state.search_history:
    st.markdown("### 📚 Recent Searches")
    
    # Add clear all button with orange styling
    if st.button("🗑️ Clear All", key="clear_all_history", type="secondary"):
        st.session_state.search_history = []
        st.rerun()
    
    for i, search in enumerate(st.session_state.search_history[-5:]):
        # Create search and remove buttons side by side
        col1, col2 = st.columns([4, 1])
        
        with col1:
            if st.button(f"🔍 {search}", key=f"history_{i}"):
                st.session_state["last_query"] = search
                st.session_state["search_triggered"] = True
                st.rerun()
        
        with col2:
            if st.button("×", key=f"remove_{i}", help="Remove this search"):
                st.session_state.search_history.remove(search)
                st.rerun()
    
    st.markdown("---")  # Add separator after recent searches

# Show search suggestions
if query and len(query) > 2:
    suggestions = get_search_suggestions(query, analytics.get_popular_queries() if analytics else [])
    if suggestions:
        st.markdown("### 💡 Suggestions")
        for suggestion in suggestions:
            if st.button(f"💡 {suggestion}", key=f"suggestion_{suggestion}"):
                st.session_state["last_query"] = suggestion
                st.session_state["search_triggered"] = True
                st.rerun()

# ------------------------- SEARCH LOGIC -------------------------
if search_button or st.session_state.get("search_triggered", False):
    if query or st.session_state.get("last_query"):
        # Use the query from session state if available
        search_query = st.session_state.get("last_query", query)
        
        # Add to search history
        if search_query not in st.session_state.search_history:
            st.session_state.search_history.append(search_query)
        
        # Clear the triggered flag
        st.session_state["search_triggered"] = False
        
        # Start timing
        start_time = time.time()
        
        # Apply filters
        filtered_df = df.copy()
        if filters:
            for key, value in filters.items():
                if key == 'doc_type':
                    filtered_df = filtered_df[filtered_df['doc_type'] == value]
                elif key == 'area':
                    filtered_df = filtered_df[filtered_df['Area'] == value]
                elif key == 'team':
                    filtered_df = filtered_df[filtered_df['Teams'] == value]
                elif key == 'application':
                    filtered_df = filtered_df[filtered_df['Application'] == value]
                elif key == 'priority_tags':
                    filtered_df = filtered_df[filtered_df['Tags'].str.contains('|'.join(value), case=False, na=False)]
                elif key == 'category_tags':
                    filtered_df = filtered_df[filtered_df['Tags'].str.contains('|'.join(value), case=False, na=False)]
                elif key == 'type_tags':
                    filtered_df = filtered_df[filtered_df['Tags'].str.contains('|'.join(value), case=False, na=False)]
                elif key == 'custom_tag':
                    filtered_df = filtered_df[filtered_df['Tags'].str.contains(value, case=False, na=False)]
        
        # Get filtered indices
        filtered_indices = filtered_df.index.tolist()
        
        if len(filtered_indices) == 0:
            st.warning("No documents match the selected filters.")
        else:
            # Search logic based on mode
            if mode == "Hybrid":
                # BM25 search
                bm25_scores = bm25.get_scores(search_query.split())
                bm25_scores = np.array(bm25_scores)
                
                # Semantic search
                query_embedding = model.encode([search_query])
                semantic_scores = cosine_similarity(query_embedding, doc_embeddings)[0]
                
                # TF-IDF search
                query_tfidf = tfidf_vectorizer.transform([search_query])
                tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
                
                # Combine scores
                final_scores = 0.4 * semantic_scores + 0.4 * bm25_scores + 0.2 * tfidf_scores
                
            elif mode == "BM25 Only":
                bm25_scores = bm25.get_scores(search_query.split())
                final_scores = np.array(bm25_scores)
                
            elif mode == "Semantic Only":
                query_embedding = model.encode([search_query])
                final_scores = cosine_similarity(query_embedding, doc_embeddings)[0]
                
            elif mode == "TF-IDF Only":
                query_tfidf = tfidf_vectorizer.transform([search_query])
                final_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
            
            # Get top results from filtered indices only
            filtered_scores = final_scores[filtered_indices]
            top_indices_filtered = np.argsort(filtered_scores)[::-1][:top_n]
            top_indices = [filtered_indices[i] for i in top_indices_filtered]
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Log analytics
            if analytics:
                analytics.log_query(search_query, len(top_indices), response_time)
            
            # Filter out zero-score results
            valid_indices = [i for i in top_indices if final_scores[i] > 0]
            
            if valid_indices:
                results_df = df.iloc[valid_indices].copy()
                results_df["Similarity Score"] = final_scores[valid_indices]
                results_df = results_df.reset_index(drop=True)

                # Display search stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Results Found", len(results_df))
                with col2:
                    st.metric("Response Time", f"{response_time:.3f}s")
                with col3:
                    st.metric("Search Mode", mode)
                with col4:
                    st.metric("Filters Applied", len(filters))
                
                st.markdown("---")
                
                # Display results
                for i, row in results_df.iterrows():
                    with st.container():
                        # Parse tags for display
                        tags_display = ""
                        if pd.notna(row['Tags']) and row['Tags']:
                            tags_list = [tag.strip() for tag in str(row['Tags']).split(',')]
                            tags_html = ""
                            for tag in tags_list:
                                # Enhanced tag styling with CSS classes
                                tag_class = "tag-default"
                                if tag.lower() in ['critical', 'urgent']:
                                    tag_class = "tag-critical"
                                elif tag.lower() in ['high']:
                                    tag_class = "tag-high"
                                elif tag.lower() in ['medium']:
                                    tag_class = "tag-medium"
                                elif tag.lower() in ['low']:
                                    tag_class = "tag-low"
                                elif tag.lower() in ['database', 'backend']:
                                    tag_class = "tag-database"
                                elif tag.lower() in ['ui', 'frontend']:
                                    tag_class = "tag-ui"
                                elif tag.lower() in ['mobile', 'web']:
                                    tag_class = "tag-mobile"
                                elif tag.lower() in ['security']:
                                    tag_class = "tag-security"
                                elif tag.lower() in ['performance']:
                                    tag_class = "tag-performance"
                                
                                tags_html += f'<span class="tag-pill {tag_class}">{tag}</span>'
                            
                            tags_display = f'<div style="margin-top: 1rem;">{tags_html}</div>'
                        
                        st.markdown(f"""
                        <div class="search-result-card">
                            <div style="display: flex; justify-content: space-between; align-items: start;">
                                <div style="flex: 1;">
                                    <h4 style="margin: 0 0 0.5rem 0; color: #333;">{row['Name']}</h4>
                                    <p style="margin: 0; color: #666; font-size: 0.9rem;">
                                        <strong>ID:</strong> {row['ID']} | 
                                        <strong>Type:</strong> {row['doc_type']} | 
                                        <strong>Area:</strong> {row['Area'] or 'N/A'}
                                    </p>
                                    {tags_display}
                                </div>
                                <div class="similarity-score">
                                    {row['Similarity Score']:.3f}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Expandable description
                        toggle_label = "📖 Show Details" if not st.session_state.expand_states.get(i, False) else "📖 Hide Details"
                        if st.button(toggle_label, key=f"toggle_{i}"):
                            st.session_state.expand_states[i] = not st.session_state.expand_states.get(i, False)
                            st.rerun()
                        
                        if st.session_state.expand_states.get(i, False):
                            # Create a clean details section
                            details_content = []
                            
                            if pd.notna(row['Description']) and row['Description']:
                                details_content.append(f"**Description:** {row['Description']}")
                            
                            if pd.notna(row['Cause']) and row['Cause']:
                                details_content.append(f"**Cause:** {row['Cause']}")
                            
                            if pd.notna(row['Solution']) and row['Solution']:
                                details_content.append(f"**Solution:** {row['Solution']}")
                            
                            if pd.notna(row['Verification']) and row['Verification']:
                                details_content.append(f"**Verification:** {row['Verification']}")
                            
                            if pd.notna(row['Teams']) and row['Teams']:
                                details_content.append(f"**Team:** {row['Teams']}")
                            
                            if pd.notna(row['Application']) and row['Application']:
                                details_content.append(f"**Application:** {row['Application']}")
                            
                            if pd.notna(row['Area']) and row['Area']:
                                details_content.append(f"**Area:** {row['Area']}")
                            
                            if details_content:
                                st.markdown("---")
                                for detail in details_content:
                                    st.markdown(detail)
                        
                        st.markdown("---")
            else:
                st.markdown("""
                <div class="empty-state">
                    <h3>🔍 No Results Found</h3>
                    <p>Try adjusting your search terms or filters</p>
                </div>
                """, unsafe_allow_html=True)

# ------------------------- ANALYTICS DASHBOARD -------------------------
if show_analytics and analytics:
    st.markdown("---")
    st.markdown("### 📊 Search Analytics")
    
    # Load analytics data
    try:
        with open("analytics/search_analytics.json", "r") as f:
            analytics_data = json.load(f)
    except:
        analytics_data = None
    
    if analytics_data:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_queries, fig_response = create_analytics_charts(analytics_data)
            if fig_queries:
                st.plotly_chart(fig_queries, use_container_width=True)
        
        with col2:
            if fig_response:
                st.plotly_chart(fig_response, use_container_width=True)
        
        # Tag distribution
        st.markdown("### 🏷️ Tag Distribution")
        
        # Extract all tags and their frequencies
        tag_counts = Counter()
        for tags_str in df['Tags'].dropna():
            if pd.notna(tags_str):
                tags = [tag.strip() for tag in str(tags_str).split(',')]
                tag_counts.update(tags)
        
        # Create tag distribution chart
        if tag_counts:
            tag_df = pd.DataFrame(list(tag_counts.items()), columns=['Tag', 'Count'])
            tag_df = tag_df.head(15)  # Top 15 tags
            
            fig_tags = px.bar(
                tag_df,
                x='Count',
                y='Tag',
                orientation='h',
                title="Most Used Tags",
                color='Count',
                color_continuous_scale='viridis'
            )
            fig_tags.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_tags, use_container_width=True)
        
        # Tag distribution by document type
        st.markdown("### 📊 Tag Distribution by Document Type")
        
        # Create tag-type matrix
        tag_type_data = []
        for doc_type in df['doc_type'].unique():
            type_df = df[df['doc_type'] == doc_type]
            type_tags = Counter()
            for tags_str in type_df['Tags'].dropna():
                if pd.notna(tags_str):
                    tags = [tag.strip() for tag in str(tags_str).split(',')]
                    type_tags.update(tags)
            
            for tag, count in type_tags.most_common(10):
                tag_type_data.append({
                    'Document Type': doc_type,
                    'Tag': tag,
                    'Count': count
                })
        
        if tag_type_data:
            tag_type_df = pd.DataFrame(tag_type_data)
            fig_tag_type = px.bar(
                tag_type_df,
                x='Tag',
                y='Count',
                color='Document Type',
                title="Tag Distribution by Document Type",
                barmode='group'
            )
            fig_tag_type.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_tag_type, use_container_width=True)

# ------------------------- FOOTER -------------------------
# Footer removed - information already shown in header
