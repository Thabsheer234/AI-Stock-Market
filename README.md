# 📈 FinAgent AI — Autonomous Stock Market Financial Desk Terminal

FinAgent AI is a production-ready, multi-desk autonomous financial intelligence platform. It fuses **Retrieval-Augmented Generation (RAG)** over corporate SEC/NSE annual reports with **Quantitative Machine Learning (Ridge Regression)** price forecasting, **Real-Time Yahoo Finance** market streaming, and **Chief Investment Officer (CIO) Executive Synthesis**.

---

## 🏗️ Architecture & Multi-Desk Orchestration

```
                                 [ User Query ]
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   Intent Router & Ticker LLM  │
                       └───────────────┬───────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌──────────────┐               ┌──────────────┐               ┌──────────────┐
│    DESK 1    │               │    DESK 2    │               │    DESK 3    │
│  Annual RAG  │               │ ML Forecast  │               │ Live Prices  │
│ (FAISS &     │               │ (Ridge Model │               │ (Yahoo       │
│  PyMuPDF)    │               │  on yfinance)│               │  Finance)    │
└───────┬──────┘               └───────┬──────┘               └───────┬──────┘
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │  Chief Investment Officer     │
                       │     (Synthesis & Verdict)     │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       [ Streamlit Terminal Frontend ]
```

### 1. 🏢 Desk 1: Deep Corporate Annual Report RAG
- **Vector Search Engine**: FAISS (`faiss_raw_index` with 27,000+ chunks, and `faiss_summary_index` with executive corporate profiles).
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` with automatic GPU (CUDA) or CPU acceleration.
- **Dynamic Ingestion**: Ingests multi-page annual report PDFs via PyMuPDF (`fitz`), automatically extracts metadata (Company, Year), builds executive summaries, and indexes into FAISS.

### 2. 📈 Desk 2: Statistical Machine Learning Predictive Engine
- **Model**: Ridge Regression with standardized feature scaling (`StandardScaler`).
- **Engineered Features**: `Close`, `Volume`, `Return (1d pct)`, `SMA_5` (5-Day Moving Average), `SMA_20` (20-Day Moving Average), and `Volatility_5` (Rolling 5-Day Standard Deviation).
- **Outputs**: Current Price, Projected Price, Expected Return %, and Model $R^2$ Goodness-of-Fit score.

### 3. 💹 Desk 3: Real-Time Market Price Desk
- Interfaces with Yahoo Finance (`yfinance`) for real-time market data, day change %, and 5-day historical closes across Indian and global equities.

### 4. 🏛️ Chief Investment Officer (CIO) Synthesis Desk
- Synthesizes quantitative forecasts, fundamental annual report statements, and market sentiment into a concise, actionable retail investor verdict.

---

## 🚀 Quick Start Guide

### 1. Launch Front-End with One Click (Windows)
Double-click:
```bash
run_frontend.bat
```
Or run directly from terminal:
```powershell
.\venv\Scripts\activate
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

### 2. Launch Interactive CLI Mode
```bash
run_cli.bat
```
Or:
```powershell
.\venv\Scripts\activate
python main.py
```

---

## 🖥️ Front-End Features

| Tab | Feature | Description |
|---|---|---|
| **💬 Interactive Agent Desk** | Real-Time Agentic Chat | Chat with the multi-desk agent. View expandable execution traces for each desk (citations, ML metrics, quotes) and the final CIO verdict. |
| **📈 ML Predictive Studio** | Interactive Forecasting | Select any stock ticker (`CIPLA.NS`, `RELIANCE.NS`, etc.), choose horizon (1–30 days), view interactive Plotly candlestick charts, moving averages, and feature importance. |
| **🏢 Company Profiles** | Corporate Fundamentals | View parsed executive financial profiles for Cipla, GMR Airport, HDFC AMC, Jindal Steel, L&T, and Reliance. |
| **📊 Ensemble Benchmark** | Multi-Model Comparison | Compare $R^2$ scores across Random Forest, XGBoost, LightGBM, Ridge Regression, and Support Vector Regression. |
| **📚 Vector Knowledge Base** | FAISS Search & PDF Ingestion | Test semantic search across 27k+ document chunks and upload new corporate annual report PDFs to expand the vector store. |

---

## 📦 Environment & Dependencies

The virtual environment is set up at `C:\Stock_market_agentic_ai\venv` with Python 3.11:
- `streamlit`
- `langchain`, `langchain-groq`, `langchain-huggingface`, `langchain-community`
- `faiss-cpu`
- `sentence-transformers`, `torch`
- `yfinance`
- `scikit-learn`, `pandas`, `numpy`, `scipy`
- `pymupdf` (fitz)
- `plotly`
