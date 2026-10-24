import torch
import os
import json
import fitz  # PyMuPDF
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Suppress annoying deprecation log messages during presentations
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ==========================================
# CONFIGURATION & FILE SETUP
# ==========================================
INPUT_FOLDER = "./raw_pdfs"
SUMMARY_JSON_FILE = "./company_summaries.json"
FAISS_RAW_INDEX = "./faiss_raw_index"
FAISS_SUMMARY_INDEX = "./faiss_summary_index"

os.makedirs(INPUT_FOLDER, exist_ok=True)

# 1. Initialize LLM Brain (Groq)
model = ChatGroq(
    model="openai/gpt-oss-120b", 
    temperature=0.1, 
    api_key=os.environ.get("GROQ_API_KEY")
)

# 2. Initialize Ultra-lightweight, GPU-Accelerated Local Embedding Model
print("🔧 Initializing Local Vector Embedding Architecture...")
embedding_engine = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": ("cuda" if torch.cuda.is_available() else "cpu")}
)

if os.path.exists(SUMMARY_JSON_FILE):
    with open(SUMMARY_JSON_FILE, "r", encoding="utf-8") as f:
        company_summaries = json.load(f)
else:
    company_summaries = {}

# ==========================================
# DYNAMIC MARKET DATA LOOKUP LAYER
# ==========================================

def fetch_historical_market_price(ticker: str, time_description: str) -> str:
    """Retrieves current or specific past historical prices from yfinance using natural language hints."""
    try:
        stock = yf.Ticker(ticker)
        desc = time_description.lower().strip()
        
        if "today" in desc or "current" in desc or "now" in desc:
            df = stock.history(period="1d")
            if not df.empty:
                return f"Current/Latest Price for {ticker}: ${df['Close'].values[0]:.2f} (As of {df.index[0].strftime('%Y-%m-%d')})"
        
        elif "month" in desc:
            target_date = datetime.now() - timedelta(days=30)
            df = stock.history(start=target_date.strftime('%Y-%m-%d'), end=(target_date + timedelta(days=5)).strftime('%Y-%m-%d'))
            if not df.empty:
                return f"Price for {ticker} roughly 1 month ago ({df.index[0].strftime('%Y-%m-%d')}): ${df['Close'].values[0]:.2f}"
                
        elif "year" in desc:
            target_date = datetime.now() - timedelta(days=365)
            df = stock.history(start=target_date.strftime('%Y-%m-%d'), end=(target_date + timedelta(days=5)).strftime('%Y-%m-%d'))
            if not df.empty:
                return f"Price for {ticker} roughly 1 year ago ({df.index[0].strftime('%Y-%m-%d')}): ${df['Close'].values[0]:.2f}"

        df = stock.history(period="5d")
        if not df.empty:
            return f"Recent historical closes for {ticker}:\n" + "\n".join([f"  {idx.strftime('%Y-%m-%d')}: ${row['Close']:.2f}" for idx, row in df.iterrows()])
            
        return f"Could not pinpoint unique historical price data for {ticker}."
    except Exception as e:
        return f"Error retrieving raw market price data for {ticker}: {e}"

# ==========================================
# ADVANCED DYNAMIC BATCH ML FORECASTING
# ==========================================

def get_multiple_tickers(query_text: str, available_companies: list) -> list:
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert market router. Your job is to identify companies from the user request and map them to their tickers.\n"
         "CRITICAL: You must use a DOT instead of a colon for exchange suffixes (e.g., '.NS' for National Stock Exchange of India).\n"
         "Examples of correct mapping:\n"
         "- Cipla -> CIPLA.NS\n"
         "- GMR Airport / GMR Airports Limited -> GMRAIRPORT.NS\n" # <-- ADD THIS LINE
         "- HDFC AMC -> HDFCAMC.NS\n"
         "- LTM / LTM annual report / Larsen & Toubro -> LT.NS\n"
         "NEVER use a colon like CIPLA:NS, make up suffixes like LTMSHARE.NS, or use old delisted tickers like GMRINFRA.NS.\n" # <-- UPDATED GUARDRAIL
         "If the user asks a broad question like 'which is the best company to invest in', return the tickers for ALL available companies listed.\n"
         "Output ONLY a valid raw JSON list of strings. Do not include any markdown blocks, explanations, or extra characters."
        ),
        ("human", "Available parsed companies: {avail_list}\nUser Request: {query}\nOutput JSON List:")
    ])
    
    chain = prompt | model | StrOutputParser()
    raw_output = chain.invoke({"query": query_text, "avail_list": ", ".join(available_companies)}).strip()
    
    if raw_output.startswith("```"):
        raw_output = raw_output.replace("```json", "").replace("```", "").strip()
        
    try:
        return json.loads(raw_output)
    except:
        return []

def run_batch_ml_predictions(tickers: list, n_days_ahead: int = 5) -> dict:
    results_map = {}
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(start="2015-01-01")
            if df.empty: continue

            df.index.name = "Date"
            df["Return"] = df["Close"].pct_change()
            df["SMA_5"] = df["Close"].rolling(window=5).mean()
            df["SMA_20"] = df["Close"].rolling(window=20).mean()
            df["Volatility_5"] = df["Return"].rolling(window=5).std()
            df["Target_Price"] = df["Close"].shift(-n_days_ahead)

            latest_today_features = df.iloc[[-1]].copy()
            df_cleaned = df.dropna()

            feature_cols = ["Close", "Volume", "Return", "SMA_5", "SMA_20", "Volatility_5"]
            X = df_cleaned[feature_cols]
            y = df_cleaned["Target_Price"]
            X_today = latest_today_features[feature_cols]

            if len(X) < 20: continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_today_scaled = scaler.transform(X_today)

            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X_train_scaled, y_train)
            score = ridge_model.score(X_test_scaled, y_test)

            predicted_price = ridge_model.predict(X_today_scaled)[0]
            today_close = X_today['Close'].values[0]

            results_map[ticker] = {
                "Current_Price": round(today_close, 2),
                "Model_Used": "Ridge Regression (Standardized)",
                "Predicted_Price": round(predicted_price, 2),
                "Expected_Return_Pct": round(((predicted_price - today_close) / today_close) * 100, 2),
                "Model_R2_Score": round(score, 4)
            }
        except Exception as e:
            print(f"     ⚠️ Skipping calculation for {ticker}: {e}")
            
    return results_map

# ==========================================
# STEP 1: SMART STRUCTURAL METADATA PARSING
# ==========================================
print("🚀 Step 1: Scanning PDFs and extracting structural metadata...")
pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".pdf")]

processed_docs = []

metadata_extraction_prompt = ChatPromptTemplate.from_template(
    "Analyze this text snippet from the cover of a financial report.\n"
    "Identify the formal Company Name and the financial Year it represents.\n"
    "Text Excerpt:\n{cover_text}\n\n"
    "Output ONLY a valid raw JSON object matching this structure: "
    '{{"company_name": "Clean Standardized Name", "year": "2024"}}'
)
metadata_chain = metadata_extraction_prompt | model | StrOutputParser()

summary_extraction_prompt = ChatPromptTemplate.from_template(
    "Extract an Executive Financial Profile for {company_name} ({year}).\nExcerpt:\n{text_excerpt}"
)
summary_chain = summary_extraction_prompt | model | StrOutputParser()

# Tracks uniquely indexed combinations to prevent duplicate indexing drops
active_universe_keys = []

for file_name in pdf_files:
    file_path = os.path.join(INPUT_FOLDER, file_name)
    print(f"  -> Ingesting: {file_name}...")
    
    doc = fitz.open(file_path)
    cover_page_text = doc[0].get_text("text")[:1500] # Read cover snippet
    
    # Extract structural meta keys
    try:
        meta_raw = metadata_chain.invoke({"cover_text": cover_page_text}).strip()
        if meta_raw.startswith("```"):
            meta_raw = meta_raw.replace("```json", "").replace("```", "").strip()
        meta_dict = json.loads(meta_raw)
        comp_name = meta_dict.get("company_name", os.path.splitext(file_name)[0]).strip()
        report_year = str(meta_dict.get("year", "Unknown")).strip()
    except Exception as e:
        # Fallback heuristic if LLM schema breaks
        comp_name = os.path.splitext(file_name)[0].split("_")[0].strip()
        report_year = "Unknown"

    unique_document_id = f"{comp_name}_{report_year}"
    active_universe_keys.append(unique_document_id)
    print(f"     🔖 Identified Entity: [Company: {comp_name}] | [Year: {report_year}]")

    company_full_text = [page.get_text("text") for page in doc if page.get_text("text").strip()]
    
    for idx, text in enumerate(company_full_text):
        processed_docs.append(Document(
            page_content=text, 
            metadata={"source": file_name, "page": idx + 1, "company": comp_name, "year": report_year, "doc_id": unique_document_id}
        ))
    
    # Manage Company-Year structural profile caching
    if unique_document_id not in company_summaries:
        print(f"     ✨ Executive Profile missing for {unique_document_id}. Generating dynamically...")
        try:
            summary_output = summary_chain.invoke({
                "company_name": comp_name, 
                "year": report_year, 
                "text_excerpt": "\n".join(company_full_text[:3])[:8000]
            })
            company_summaries[unique_document_id] = summary_output.strip()
        except Exception as e:
            company_summaries[unique_document_id] = f"Profile data extracted for {comp_name} ({report_year})."
    doc.close()

with open(SUMMARY_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(company_summaries, f, indent=4, ensure_ascii=False)

# Collect corporate tokens for standard market query routers
available_companies_list = list(set([key.split("_")[0] for key in company_summaries.keys()]))
print(f"✅ Active Corporate Universe: {available_companies_list}")

print("✂️ Splitting documents into paragraph blocks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_chunks = text_splitter.split_documents(processed_docs)

# ==========================================
# STEP 2 & 3: ADAPTIVE INCREMENTAL FAISS LOADING
# ==========================================
print("\n📦 Step 2: Syncing FAISS Vector Stores with Document Directory...")

index_updated = False

if os.path.exists(FAISS_RAW_INDEX) and os.path.exists(FAISS_SUMMARY_INDEX):
    print("     ⚡ Existing vector cache found on disk. Checking for newly added company files or years...")
    raw_data_vector_db = FAISS.load_local(FAISS_RAW_INDEX, embedding_engine, allow_dangerous_deserialization=True)
    summary_vector_db = FAISS.load_local(FAISS_SUMMARY_INDEX, embedding_engine, allow_dangerous_deserialization=True)
    
    # Dynamically extract tracking list of what is currently resting inside the stored vector indexes
    stored_doc_ids = set([doc.metadata.get("doc_id") for doc in raw_data_vector_db.docstore._dict.values() if doc.metadata])
    
    # Isolate chunks from files whose combination does not exist inside our vector storage block
    new_chunks = [chunk for chunk in final_chunks if chunk.metadata.get("doc_id") not in stored_doc_ids]
    
    if new_chunks:
        print(f"     ✨ New entity or year detected! Appending {len(new_chunks)} fresh chunks to index database...")
        raw_data_vector_db.add_documents(new_chunks)
        
        # Pull only summaries matching the newly injected items
        new_summary_docs = [
            Document(page_content=company_summaries[doc_id], metadata={"doc_id": doc_id, "company": doc_id.split("_")[0]})
            for doc_id in active_universe_keys if doc_id not in stored_doc_ids
        ]
        if new_summary_docs:
            summary_vector_db.add_documents(new_summary_docs)
            
        index_updated = True
    else:
        print("     ✅ Vector store fully synced with folder files. No new files detected.")

else:
    print("     ⏳ Stored vectors not found. Calculating complete local matrix indices via GPU...")
    if not final_chunks:
        print("❌ Error: No text components available to initialize database.")
        exit()
        
    raw_data_vector_db = FAISS.from_documents(final_chunks, embedding_engine)
    
    summary_documents = [
        Document(page_content=text, metadata={"doc_id": doc_id, "company": doc_id.split("_")[0]}) 
        for doc_id, text in company_summaries.items()
    ]
    summary_vector_db = FAISS.from_documents(summary_documents, embedding_engine)
    index_updated = True

if index_updated:
    raw_data_vector_db.save_local(FAISS_RAW_INDEX)
    summary_vector_db.save_local(FAISS_SUMMARY_INDEX)
    print("     💾 FAISS index changes permanently saved to disk files.")

raw_retriever = raw_data_vector_db.as_retriever(search_kwargs={"k": 4})
summary_retriever = summary_vector_db.as_retriever(search_kwargs={"k": max(1, len(company_summaries))})

# ==========================================
# PIPELINE CHAINS WITH MEMORY INTEGRATION
# ==========================================
print("\n🤖 Step 3: Syncing LCEL Context Memory Engines...")

def format_docs(docs):
    return "\n\n".join(f"[Source File: {doc.metadata.get('source')} | Entity: {doc.metadata.get('company')} ({doc.metadata.get('year')})]\n{doc.page_content}" for doc in docs)

session_memory_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_memory_store:
        session_memory_store[session_id] = ChatMessageHistory()
    return session_memory_store[session_id]

deep_retrieval_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an analytical financial desk assistant. Extract context to resolve user statements accurately. Pay close attention to corporate timelines and specific report years.\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

comparison_prompt = ChatPromptTemplate.from_messages([
    ("system", "Compare structural corporate trends across different companies or over multiple years based on background profile documents.\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

raw_rag_chain = (
    {"context": (lambda x: x["input"]) | raw_retriever | format_docs, "input": lambda x: x["input"], "history": lambda x: x["history"]} 
    | deep_retrieval_prompt | model | StrOutputParser()
)

comparison_rag_chain = (
    {"context": (lambda x: x["input"]) | summary_retriever | format_docs, "input": lambda x: x["input"], "history": lambda x: x["history"]} 
    | comparison_prompt | model | StrOutputParser()
)

deep_rag_pipeline = RunnableWithMessageHistory(raw_rag_chain, get_session_history, input_messages_key="input", history_messages_key="history")
comparison_rag_pipeline = RunnableWithMessageHistory(comparison_rag_chain, get_session_history, input_messages_key="input", history_messages_key="history")

print("✅ Execution loop active.")
print("-" * 60)

# ==========================================
# STEP 4: INTERACTIVE RUN LOOP WITH SYNTHESIS
# ==========================================
SESSION_ID = "default_user_session"

while True:
    user_query = input("\nAsk a question about the financial data (or type 'exit' to quit): ")
    if user_query.strip().lower() == 'exit': break
    if not user_query.strip(): continue
        
    print("Analyzing query intent and routing requests...")
    routing_prompt = (
        "Classify user request intent based on vocabulary. "
        "Output ONLY valid JSON matching this structure: "
        '{"comparison": 0 or 1, "prediction": 0 or 1, "historical_price_lookup": 0 or 1, "raw": 0 or 1}.\n\n'
        "CRITICAL Criteria:\n"
        "- If user asks for real-time prices, today's price, or historical trends (last week, 1 month ago, what about X price, etc.), set historical_price_lookup to 1.\n"
        "- If user is forecasting or guessing future days ahead prices, set prediction to 1.\n"
        "- CRITICAL TICKER MAPPING SAFETY: Note that Larsen & Toubro is LT.NS, and GMR Airports Limited is GMRAIRPORT.NS. Never return GMRINFRA.NS.\n" # <-- ADD THIS WARNING LINE
        f"User input: {user_query}\nOutput:"
    )
    routing_content = model.invoke(routing_prompt).content
    if routing_content.startswith("```"):
        routing_content = routing_content.replace("```json", "").replace("```", "").strip()
        
    try:
        intent_routes = json.loads(routing_content)
    except:
        intent_routes = {"comparison": 0, "prediction": 0, "historical_price_lookup": 0, "raw": 1}

    try:
        raw_ml_data = ""
        raw_rag_data = ""
        raw_lookup_data = ""
        
        target_tickers = get_multiple_tickers(user_query, available_companies_list)
        
        if intent_routes.get("historical_price_lookup") == 1 and target_tickers:
            print(f"     📊 Launching real-time market array tracking for: {target_tickers}")
            for ticker in target_tickers:
                raw_lookup_data += fetch_historical_market_price(ticker, user_query) + "\n"

        if intent_routes.get("prediction") == 1 or ("predict" in user_query.lower() and target_tickers):
            target_days = 5
            for word in user_query.replace("?", "").split():
                if word.isdigit():
                    target_days = int(word)
                    break
            
            print(f"     🎯 Active evaluation forecasting targets: {target_tickers}")
            if target_tickers:
                raw_ml_data = json.dumps(run_batch_ml_predictions(target_tickers, n_days_ahead=target_days))

        if intent_routes.get("comparison") == 1 or "compare" in user_query.lower():
            raw_rag_data = comparison_rag_pipeline.invoke({"input": user_query}, config={"configurable": {"session_id": SESSION_ID}})
        else:
            raw_rag_data = deep_rag_pipeline.invoke({"input": user_query}, config={"configurable": {"session_id": SESSION_ID}})

        print("Synthesizing data into a definitive final verdict...")
        final_synthesis_prompt = (
            "You are an elite, concise Chief Investment Officer. Synthesize the incoming signals into a direct retail answer.\n"
            "Identify variance cross-comparisons across years if multiple records are available in the context.\n\n"
            f"User Original Question: {user_query}\n"
            f"Desk 1 (RAG Context Analysis):\n{raw_rag_data}\n"
            f"Desk 2 (Statistical ML Models Future Forecast):\n{raw_ml_data}\n"
            f"Desk 3 (YFinance Real-Time Actual Price Desk):\n{raw_lookup_data}\n\n"
            "Provide the clean, definitive final verdict directly to the investor:"
        )
        
        final_response = model.invoke(final_synthesis_prompt).content
        
        history_obj = get_session_history(SESSION_ID)
        history_obj.add_user_message(user_query)
        history_obj.add_ai_message(final_response)

        print("\n=== VERDICT SYSTEM RESPONSE ===")
        print(final_response)
        print("===============================")
        
    except Exception as e:
        print(f"❌ Execution failed. Error: {e}")