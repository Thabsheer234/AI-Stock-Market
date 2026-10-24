"""
Agentic AI Financial Desk Assistant Backend
Provides multi-agent routing, RAG retrieval over SEC/Corporate Annual Reports,
ML Ridge Regression Forecasting, Real-time Yahoo Finance Data, and CIO Synthesis.
"""

import os
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Generator

import pandas as pd
import numpy as np
import yfinance as yf
import fitz  # PyMuPDF
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Suppress deprecation warnings
warnings.filterwarnings("ignore")

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Default Configuration
DEFAULT_GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DEFAULT_MODEL = "openai/gpt-oss-120b"
AVAILABLE_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "raw_pdfs")
SUMMARY_JSON_FILE = os.path.join(BASE_DIR, "company_summaries.json")
FAISS_RAW_INDEX = os.path.join(BASE_DIR, "faiss_raw_index")
FAISS_SUMMARY_INDEX = os.path.join(BASE_DIR, "faiss_summary_index")
ENSEMBLE_REPORT_FILE = os.path.join(BASE_DIR, "advanced_ensemble_report.csv")


class FinancialAgentBackend:
    """
    Central Orchestrator for the 3-Desk Agentic Stock Market Financial Desk Assistant.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.1
    ):
        self.api_key = api_key or DEFAULT_GROQ_API_KEY
        self.model_name = model_name
        self.temperature = temperature
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key
        )
        
        # Initialize Embeddings
        self.embedding_engine = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": self.device}
        )
        
        # Load Summaries
        self.company_summaries: Dict[str, str] = self._load_summaries()
        self.available_companies = list(set([k.split("_")[0] for k in self.company_summaries.keys()]))
        
        # Load Vector Stores
        self.raw_vector_db: Optional[FAISS] = None
        self.summary_vector_db: Optional[FAISS] = None
        self._load_vector_stores()
        
        # Memory Stores
        self.session_memory_store: Dict[str, ChatMessageHistory] = {}

    def _load_summaries(self) -> Dict[str, str]:
        if os.path.exists(SUMMARY_JSON_FILE):
            try:
                with open(SUMMARY_JSON_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading summaries: {e}")
        return {}

    def _load_vector_stores(self):
        try:
            if os.path.exists(FAISS_RAW_INDEX) and os.path.exists(FAISS_SUMMARY_INDEX):
                self.raw_vector_db = FAISS.load_local(
                    FAISS_RAW_INDEX,
                    self.embedding_engine,
                    allow_dangerous_deserialization=True
                )
                self.summary_vector_db = FAISS.load_local(
                    FAISS_SUMMARY_INDEX,
                    self.embedding_engine,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded FAISS stores: {self.raw_vector_db.index.ntotal} raw chunks, {self.summary_vector_db.index.ntotal} summaries.")
            else:
                print("FAISS vector stores not found on disk. Initializing empty or build needed.")
        except Exception as e:
            print(f"Error loading FAISS stores: {e}")

    def update_model(self, model_name: str, api_key: Optional[str] = None):
        """Update Groq model or API key dynamically."""
        if api_key:
            self.api_key = api_key
        self.model_name = model_name
        self.llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key
        )

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.session_memory_store:
            self.session_memory_store[session_id] = ChatMessageHistory()
        return self.session_memory_store[session_id]

    def clear_session_history(self, session_id: str):
        if session_id in self.session_memory_store:
            del self.session_memory_store[session_id]

    # ==========================================
    # ROUTING & TICKER MAPPING
    # ==========================================

    def classify_intent(self, query: str) -> Dict[str, int]:
        """Classify user query intent into comparison, prediction, historical lookup, and raw RAG."""
        routing_prompt = (
            "Classify user request intent based on vocabulary. "
            "Output ONLY valid JSON matching this exact structure: "
            '{"comparison": 0 or 1, "prediction": 0 or 1, "historical_price_lookup": 0 or 1, "raw": 0 or 1}.\n\n'
            "Criteria:\n"
            "- If user asks for real-time prices, today's price, or historical trends, set historical_price_lookup to 1.\n"
            "- If user is forecasting or guessing future days ahead prices, set prediction to 1.\n"
            "- If user asks to compare 2 or more companies, set comparison to 1.\n"
            "- If user asks about financial statements, annual reports, business risks, management, set raw to 1.\n\n"
            f"User input: {query}\nOutput JSON:"
        )
        try:
            res = self.llm.invoke(routing_prompt).content.strip()
            if res.startswith("```"):
                res = res.replace("```json", "").replace("```", "").strip()
            return json.loads(res)
        except Exception:
            # Fallback heuristic
            q_lower = query.lower()
            return {
                "comparison": 1 if any(w in q_lower for w in ["compare", "vs", "versus", "difference"]) else 0,
                "prediction": 1 if any(w in q_lower for w in ["predict", "forecast", "ahead", "future", "target"]) else 0,
                "historical_price_lookup": 1 if any(w in q_lower for w in ["price", "today", "current", "quote", "cost", "market"]) else 0,
                "raw": 1
            }

    def extract_tickers(self, query_text: str) -> List[str]:
        """Maps query text and company mentions to valid stock tickers."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert market router. Identify companies from the user request and map them to their NSE/BSE tickers.\n"
             "CRITICAL: Use a DOT instead of a colon for exchange suffixes (e.g., '.NS' for National Stock Exchange of India).\n"
             "Standard Mappings:\n"
             "- Cipla -> CIPLA.NS\n"
             "- GMR Airport / GMR Airports Limited -> GMRAIRPORT.NS\n"
             "- HDFC AMC -> HDFCAMC.NS\n"
             "- LTM / L&T / Larsen & Toubro -> LT.NS\n"
             "- Jindal Steel / JSPL -> JINDALSTEL.NS\n"
             "- Reliance / RIL -> RELIANCE.NS\n"
             "- TCS / Tata Consultancy Services -> TCS.NS\n"
             "- Infosys -> INFY.NS\n"
             "- Tata Motors -> TATAMOTORS.NS\n"
             "NEVER use old delisted tickers like GMRINFRA.NS. NEVER use a colon like CIPLA:NS.\n"
             "If the user asks a broad question like 'which is the best company to invest in', return tickers for ALL available companies listed.\n"
             "Output ONLY a valid raw JSON list of strings. Do not include markdown or explanations."
            ),
            ("human", "Available parsed companies: {avail_list}\nUser Request: {query}\nOutput JSON List:")
        ])
        try:
            chain = prompt | self.llm | StrOutputParser()
            raw_output = chain.invoke({
                "query": query_text,
                "avail_list": ", ".join(self.available_companies)
            }).strip()
            if raw_output.startswith("```"):
                raw_output = raw_output.replace("```json", "").replace("```", "").strip()
            tickers = json.loads(raw_output)
            if isinstance(tickers, list):
                return tickers
        except Exception as e:
            print(f"Ticker extraction error: {e}")
        
        # Rule-based fallback
        ticker_map = {
            "cipla": "CIPLA.NS",
            "gmr": "GMRAIRPORT.NS",
            "hdfc": "HDFCAMC.NS",
            "jindal": "JINDALSTEL.NS",
            "reliance": "RELIANCE.NS",
            "ril": "RELIANCE.NS",
            "ltm": "LT.NS",
            "larsen": "LT.NS",
            "tcs": "TCS.NS"
        }
        found = []
        for word, sym in ticker_map.items():
            if word in query_text.lower() and sym not in found:
                found.append(sym)
        return found

    # ==========================================
    # DESK 1: RAG RETRIEVAL ENGINE
    # ==========================================

    def retrieve_raw_documents(self, query: str, k: int = 4) -> List[Document]:
        """Retrieves raw paragraph blocks from annual reports."""
        if not self.raw_vector_db:
            return []
        try:
            return self.raw_vector_db.similarity_search(query, k=k)
        except Exception as e:
            print(f"Raw retrieval error: {e}")
            return []

    def retrieve_summary_documents(self, query: str, k: int = 4) -> List[Document]:
        """Retrieves executive summaries across companies."""
        if not self.summary_vector_db:
            return []
        try:
            return self.summary_vector_db.similarity_search(query, k=k)
        except Exception as e:
            print(f"Summary retrieval error: {e}")
            return []

    def format_docs_for_prompt(self, docs: List[Document]) -> str:
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'Unknown')} | Entity: {doc.metadata.get('company', 'Unknown')} ({doc.metadata.get('year', 'N/A')}) | Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in docs
        )

    # ==========================================
    # DESK 2: STATISTICAL ML FORECASTING
    # ==========================================

    def run_ml_forecast(self, ticker: str, n_days_ahead: int = 5) -> Dict[str, Any]:
        """
        Trains a Ridge Regression model on historical stock features to forecast price n days ahead.
        Returns prediction metrics, expected return, and historical dataframe for interactive charts.
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start="2015-01-01")
            if df.empty or len(df) < 50:
                return {"error": f"Insufficient historical market data for {ticker}."}

            df.index.name = "Date"
            df["Return"] = df["Close"].pct_change()
            df["SMA_5"] = df["Close"].rolling(window=5).mean()
            df["SMA_20"] = df["Close"].rolling(window=20).mean()
            df["Volatility_5"] = df["Return"].rolling(window=5).std()
            df["Target_Price"] = df["Close"].shift(-n_days_ahead)

            latest_today = df.iloc[[-1]].copy()
            df_cleaned = df.dropna()

            feature_cols = ["Close", "Volume", "Return", "SMA_5", "SMA_20", "Volatility_5"]
            X = df_cleaned[feature_cols]
            y = df_cleaned["Target_Price"]
            X_today = latest_today[feature_cols]

            if len(X) < 20:
                return {"error": f"Not enough valid feature samples for {ticker}."}

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_today_scaled = scaler.transform(X_today)

            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X_train_scaled, y_train)
            score = ridge_model.score(X_test_scaled, y_test)

            predicted_price = float(ridge_model.predict(X_today_scaled)[0])
            today_close = float(latest_today["Close"].values[0])
            expected_return = float(((predicted_price - today_close) / today_close) * 100)

            # Feature coefficients
            coef_dict = {col: round(float(c), 4) for col, c in zip(feature_cols, ridge_model.coef_)}

            return {
                "ticker": ticker,
                "Current_Price": round(today_close, 2),
                "Model_Used": "Ridge Regression (Standardized)",
                "Forecast_Horizon_Days": n_days_ahead,
                "Predicted_Price": round(predicted_price, 2),
                "Expected_Return_Pct": round(expected_return, 2),
                "Model_R2_Score": round(score, 4),
                "Feature_Weights": coef_dict,
                "History_DF": df.tail(120),  # last 120 trading sessions for visual charts
                "Last_Updated": latest_today.index[0].strftime("%Y-%m-%d")
            }
        except Exception as e:
            return {"error": f"Failed forecasting for {ticker}: {e}"}

    def run_batch_ml_predictions(self, tickers: List[str], n_days_ahead: int = 5) -> Dict[str, Any]:
        """Runs batch forecasting for multiple tickers."""
        results = {}
        for ticker in tickers:
            fc = self.run_ml_forecast(ticker, n_days_ahead)
            if "error" not in fc:
                results[ticker] = {
                    "Current_Price": fc["Current_Price"],
                    "Model_Used": fc["Model_Used"],
                    "Predicted_Price": fc["Predicted_Price"],
                    "Expected_Return_Pct": fc["Expected_Return_Pct"],
                    "Model_R2_Score": fc["Model_R2_Score"],
                    "Horizon_Days": n_days_ahead
                }
        return results

    # ==========================================
    # DESK 3: REAL-TIME MARKET PRICE DESK
    # ==========================================

    def fetch_market_price_summary(self, ticker: str, time_description: str = "today") -> Dict[str, Any]:
        """Retrieves live or historical market price data and summary metrics."""
        try:
            stock = yf.Ticker(ticker)
            desc = time_description.lower().strip()
            
            # Fast quote info
            fast_info = getattr(stock, "fast_info", None)
            current_price = getattr(fast_info, "last_price", None)
            prev_close = getattr(fast_info, "previous_close", None)
            
            if current_price is None:
                h1d = stock.history(period="1d")
                if not h1d.empty:
                    current_price = float(h1d["Close"].iloc[-1])
                    prev_close = float(h1d["Open"].iloc[0])

            hist_5d = stock.history(period="5d")
            recent_closes = []
            if not hist_5d.empty:
                for idx, row in hist_5d.iterrows():
                    recent_closes.append({
                        "date": idx.strftime("%Y-%m-%d"),
                        "close": round(float(row["Close"]), 2),
                        "volume": int(row["Volume"])
                    })

            price_str = ""
            if "month" in desc:
                target_date = datetime.now() - timedelta(days=30)
                df = stock.history(start=target_date.strftime('%Y-%m-%d'), end=(target_date + timedelta(days=5)).strftime('%Y-%m-%d'))
                if not df.empty:
                    price_str = f"Price for {ticker} roughly 1 month ago ({df.index[0].strftime('%Y-%m-%d')}): ₹{df['Close'].values[0]:.2f}"
            elif "year" in desc:
                target_date = datetime.now() - timedelta(days=365)
                df = stock.history(start=target_date.strftime('%Y-%m-%d'), end=(target_date + timedelta(days=5)).strftime('%Y-%m-%d'))
                if not df.empty:
                    price_str = f"Price for {ticker} roughly 1 year ago ({df.index[0].strftime('%Y-%m-%d')}): ₹{df['Close'].values[0]:.2f}"

            if not price_str and current_price is not None:
                day_change = ((current_price - prev_close) / prev_close * 100) if prev_close else 0.0
                price_str = f"Latest Quote for {ticker}: ₹{current_price:.2f} ({day_change:+.2f}%)"

            return {
                "ticker": ticker,
                "current_price": round(current_price, 2) if current_price else None,
                "prev_close": round(prev_close, 2) if prev_close else None,
                "formatted_summary": price_str,
                "recent_closes": recent_closes
            }
        except Exception as e:
            return {
                "ticker": ticker,
                "current_price": None,
                "formatted_summary": f"Could not retrieve market price data for {ticker}: {e}",
                "recent_closes": []
            }

    # ==========================================
    # DESK 4 / SYNTHESIS: CIO EXECUTIVE VERDICT
    # ==========================================

    def run_agentic_pipeline(
        self,
        user_query: str,
        session_id: str = "default_user_session"
    ) -> Dict[str, Any]:
        """
        Executes the full 3-Desk Multi-Agent Workflow:
        1. Classifies intent & extracts relevant company tickers.
        2. Queries Desk 1 (RAG raw/summary documentation).
        3. Queries Desk 2 (ML statistical prediction).
        4. Queries Desk 3 (Live market pricing).
        5. Chief Investment Officer synthesizes the signals into a final verdict.
        """
        start_time = time.time()
        
        # Step 1: Intent Routing
        intent = self.classify_intent(user_query)
        tickers = self.extract_tickers(user_query)
        
        # Step 2: Desk 1 - RAG Retrieval
        retrieved_raw_docs = []
        retrieved_summary_docs = []
        rag_context_text = ""
        
        if intent.get("comparison") == 1 or "compare" in user_query.lower():
            retrieved_summary_docs = self.retrieve_summary_documents(user_query, k=max(2, len(tickers)))
            rag_context_text = self.format_docs_for_prompt(retrieved_summary_docs)
        else:
            retrieved_raw_docs = self.retrieve_raw_documents(user_query, k=4)
            rag_context_text = self.format_docs_for_prompt(retrieved_raw_docs)

        # Step 3: Desk 2 - ML Forecasts
        ml_predictions = {}
        target_days = 5
        # Extract forecast horizon days if present
        for word in user_query.replace("?", "").split():
            if word.isdigit():
                target_days = int(word)
                break

        if intent.get("prediction") == 1 or any(k in user_query.lower() for k in ["predict", "forecast", "future", "ahead", "target"]):
            if tickers:
                ml_predictions = self.run_batch_ml_predictions(tickers, n_days_ahead=target_days)

        # Step 4: Desk 3 - Live Market Data
        market_quotes = {}
        market_text = ""
        if intent.get("historical_price_lookup") == 1 or tickers or any(k in user_query.lower() for k in ["price", "quote", "cost", "market"]):
            for t in tickers:
                mq = self.fetch_market_price_summary(t, user_query)
                market_quotes[t] = mq
                market_text += mq.get("formatted_summary", "") + "\n"

        # Step 5: CIO Synthesis
        cio_prompt = (
            "You are an elite, concise Chief Investment Officer (CIO) providing advice to executive retail investors.\n"
            "Analyze the signals from the 3 specialized analytical desks below and synthesize a definitive, structured final verdict.\n\n"
            f"User Original Inquiry: {user_query}\n\n"
            "=== DESK 1: RAG Context & Annual Financial Reports ===\n"
            f"{rag_context_text if rag_context_text else 'No specific corporate PDF passages matched.'}\n\n"
            "=== DESK 2: Statistical ML Model Horizon Forecasts ===\n"
            f"{json.dumps(ml_predictions, indent=2) if ml_predictions else 'No ML forecasting requested or active.'}\n\n"
            "=== DESK 3: YFinance Live Market Quotes Desk ===\n"
            f"{market_text if market_text else 'No real-time market price lookup triggered.'}\n\n"
            "INSTRUCTIONS FOR THE VERDICT:\n"
            "1. Give a direct, punchy executive summary answering the question immediately.\n"
            "2. Break down key quantitative insights (Revenue/Profitability from Desk 1, Model Expected Return % from Desk 2, Current Price from Desk 3).\n"
            "3. If cross-comparing companies or years, clearly state winners and structural variance.\n"
            "4. Provide a definitive Actionable Verdict (e.g., Bullish / Bearish / Hold / Accumulate with concise rationale & risks).\n"
            "5. Maintain professional financial rigor with markdown formatting."
        )

        try:
            cio_verdict = self.llm.invoke(cio_prompt).content.strip()
        except Exception as e:
            cio_verdict = f"Error generating CIO verdict: {e}"

        # Update Session History
        history = self.get_session_history(session_id)
        history.add_user_message(user_query)
        history.add_ai_message(cio_verdict)

        total_time = round(time.time() - start_time, 2)

        return {
            "query": user_query,
            "intent": intent,
            "tickers": tickers,
            "forecast_days": target_days,
            "rag_raw_docs": retrieved_raw_docs,
            "rag_summary_docs": retrieved_summary_docs,
            "rag_text": rag_context_text,
            "ml_predictions": ml_predictions,
            "market_quotes": market_quotes,
            "verdict": cio_verdict,
            "execution_time_sec": total_time
        }

    # ==========================================
    # DOCUMENT INGESTION ENGINE
    # ==========================================

    def ingest_new_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a new annual report PDF, generates structural metadata and executive summary,
        chunks text, updates FAISS stores, and persists to disk.
        """
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File {file_path} does not exist."}

        file_name = os.path.basename(file_path)
        try:
            doc = fitz.open(file_path)
            cover_snippet = doc[0].get_text("text")[:1500]

            # LLM Metadata Extraction
            meta_prompt = ChatPromptTemplate.from_template(
                "Analyze this cover text snippet from a financial annual report:\n{cover}\n"
                "Extract the formal Company Name and Financial Year.\n"
                'Output ONLY raw JSON: {{"company_name": "Clean Standard Name", "year": "2024"}}'
            )
            meta_chain = meta_prompt | self.llm | StrOutputParser()
            try:
                raw_meta = meta_chain.invoke({"cover": cover_snippet}).strip()
                if raw_meta.startswith("```"):
                    raw_meta = raw_meta.replace("```json", "").replace("```", "").strip()
                meta_dict = json.loads(raw_meta)
                comp_name = meta_dict.get("company_name", os.path.splitext(file_name)[0]).strip()
                year = str(meta_dict.get("year", "Unknown")).strip()
            except Exception:
                comp_name = os.path.splitext(file_name)[0].split("_")[0].strip()
                year = "Unknown"

            doc_id = f"{comp_name}_{year}"
            
            # Extract full text
            full_pages = [page.get_text("text") for page in doc if page.get_text("text").strip()]
            doc.close()

            # Generate Executive Profile if not present
            if doc_id not in self.company_summaries:
                summary_prompt = ChatPromptTemplate.from_template(
                    "Extract an Executive Financial Profile for {company_name} ({year}).\nExcerpt:\n{text_excerpt}"
                )
                sum_chain = summary_prompt | self.llm | StrOutputParser()
                try:
                    summary_text = sum_chain.invoke({
                        "company_name": comp_name,
                        "year": year,
                        "text_excerpt": "\n".join(full_pages[:3])[:8000]
                    }).strip()
                except Exception:
                    summary_text = f"Profile generated for {comp_name} ({year})."
                self.company_summaries[doc_id] = summary_text
                with open(SUMMARY_JSON_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.company_summaries, f, indent=4, ensure_ascii=False)

            # Chunk document
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            doc_chunks = []
            for idx, text in enumerate(full_pages):
                doc_chunks.append(Document(
                    page_content=text,
                    metadata={"source": file_name, "page": idx + 1, "company": comp_name, "year": year, "doc_id": doc_id}
                ))
            final_chunks = text_splitter.split_documents(doc_chunks)

            # Update FAISS
            if self.raw_vector_db:
                self.raw_vector_db.add_documents(final_chunks)
                self.raw_vector_db.save_local(FAISS_RAW_INDEX)
            else:
                self.raw_vector_db = FAISS.from_documents(final_chunks, self.embedding_engine)
                self.raw_vector_db.save_local(FAISS_RAW_INDEX)

            summary_doc = Document(page_content=self.company_summaries[doc_id], metadata={"doc_id": doc_id, "company": comp_name})
            if self.summary_vector_db:
                self.summary_vector_db.add_documents([summary_doc])
                self.summary_vector_db.save_local(FAISS_SUMMARY_INDEX)
            else:
                self.summary_vector_db = FAISS.from_documents([summary_doc], self.embedding_engine)
                self.summary_vector_db.save_local(FAISS_SUMMARY_INDEX)

            # Refresh available companies
            self.available_companies = list(set([k.split("_")[0] for k in self.company_summaries.keys()]))

            return {
                "success": True,
                "company_name": comp_name,
                "year": year,
                "doc_id": doc_id,
                "chunks_added": len(final_chunks),
                "total_pages": len(full_pages)
            }
        except Exception as e:
            return {"success": False, "error": f"Failed ingesting PDF: {e}"}
