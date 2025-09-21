# 🔍 Smart Backlog Search Engine

An AI-powered search system for managing backlog items, bug reports, feature requests, and requirements with semantic understanding and modern UI.

## ✨ Features

- **🔀 Hybrid Search**: Combines keyword matching + AI semantic understanding
- **🏷️ Smart Filtering**: Filter by document type, area, team, priority, tags
- **📚 Search History**: Quick access to recent searches with one-click repeat
- **📊 Analytics**: Track search patterns and performance metrics
- **🎨 Modern UI**: Beautiful, responsive design with smooth animations
- **⚡ Fast Performance**: Intelligent caching and optimized search algorithms

## 🚀 Quick Start

### Local Development
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Streamlit Cloud Deployment
1. Fork this repository to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file: `streamlit_app.py`
5. Deploy!

## 🔧 Search Modes

- **Hybrid** (Recommended): Best overall performance
- **BM25 Only**: Fast keyword matching
- **Semantic Only**: AI-powered meaning search
- **TF-IDF Only**: Statistical text similarity

---

*Built with Streamlit, Sentence Transformers, and modern AI technologies*