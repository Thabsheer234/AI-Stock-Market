"""
FinAgent AI — Autonomous Multi-Desk Financial Intelligence Platform
Streamlit Front-End Interface
"""

import os
import sys
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Ensure backend can be imported from current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from agent_backend import (
    FinancialAgentBackend,
    DEFAULT_GROQ_API_KEY,
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
    INPUT_FOLDER,
    SUMMARY_JSON_FILE,
    ENSEMBLE_REPORT_FILE
)

# ---------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="FinAgent AI — Stock Market Desk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Financial Terminal Dark Mode CSS
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Card */
    .header-box {
        background: linear-gradient(135deg, #161b22 0%, #1f2937 100%);
        padding: 24px 28px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #58a6ff;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #8b949e;
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Badge Pills */
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 8px;
    }
    .badge-blue { background-color: #1f3a5f; color: #79c0ff; border: 1px solid #388bfd40; }
    .badge-green { background-color: #1a422b; color: #56d364; border: 1px solid #3fb95040; }
    .badge-purple { background-color: #3b2354; color: #d2a8ff; border: 1px solid #a371f740; }
    .badge-amber { background-color: #4a3410; color: #e3b341; border: 1px solid #d2992240; }
    
    /* Metric Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    .metric-sub {
        font-size: 0.82rem;
        margin-top: 4px;
    }
    .positive { color: #3fb950; font-weight: 600; }
    .negative { color: #f85149; font-weight: 600; }
    
    /* Desk Result Accordion */
    .desk-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    /* CIO Verdict Container */
    .verdict-box {
        background: linear-gradient(145deg, #132338 0%, #161b22 100%);
        border: 1px solid #388bfd;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(56, 139, 253, 0.15);
    }
    .verdict-header {
        color: #58a6ff;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Prompt Suggestion Button */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# BACKEND INITIALIZATION & CACHING
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Financial Agent Architecture...")
def get_agent_instance(api_key: str, model_name: str, temperature: float) -> FinancialAgentBackend:
    return FinancialAgentBackend(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature
    )

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = DEFAULT_GROQ_API_KEY
if "groq_model" not in st.session_state:
    st.session_state.groq_model = DEFAULT_MODEL
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.1
if "horizon_days" not in st.session_state:
    st.session_state.horizon_days = 5
if "preset_query" not in st.session_state:
    st.session_state.preset_query = ""


# ---------------------------------------------------------
# SIDEBAR NAVIGATION & SETTINGS
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
    st.markdown("### **FinAgent Terminal**")
    st.caption("Autonomous Agentic AI Desk")
    st.markdown("---")

    st.markdown("#### ⚙️ **Engine Settings**")
    model_choice = st.selectbox(
        "Groq LLM Model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.groq_model) if st.session_state.groq_model in AVAILABLE_MODELS else 0,
        help="Select inference engine model hosted on Groq"
    )
    st.session_state.groq_model = model_choice

    api_key_input = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        help="Groq API Key for high-speed LLM inference"
    )
    st.session_state.groq_api_key = api_key_input

    temp_slider = st.slider(
        "Creativity / Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.05
    )
    st.session_state.temperature = temp_slider

    default_horizon = st.slider(
        "Default Forecast Horizon (Days)",
        min_value=1,
        max_value=30,
        value=st.session_state.horizon_days,
        step=1
    )
    st.session_state.horizon_days = default_horizon

    # Initialize Agent
    agent = get_agent_instance(
        st.session_state.groq_api_key,
        st.session_state.groq_model,
        st.session_state.temperature
    )

    st.markdown("---")
    st.markdown("#### 🏢 **Parsed Corporate Universe**")
    universe_tickers = {
        "Cipla": "CIPLA.NS",
        "GMR Airport": "GMRAIRPORT.NS",
        "HDFC AMC": "HDFCAMC.NS",
        "Jindal Steel": "JINDALSTEL.NS",
        "LTM / L&T": "LT.NS",
        "Reliance Industries": "RELIANCE.NS",
        "TCS": "TCS.NS"
    }
    for comp, sym in universe_tickers.items():
        st.markdown(f"• **{comp}** (`{sym}`)")

    st.markdown("---")
    st.markdown("#### 🚀 **Quick Query Prompts**")
    prompts = [
        "Compare Cipla & Jindal Steel revenue and operational margins",
        "Forecast CIPLA.NS stock price for the next 5 days",
        "What is HDFC AMC today's price and business risk profile?",
        "Synthesize investment recommendations across all available companies",
        "Analyze Reliance Industries annual report highlights and debt ratio"
    ]
    for p in prompts:
        if st.button(f"💬 {p[:38]}...", key=f"btn_{p[:12]}"):
            st.session_state.preset_query = p

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        agent.clear_session_history("default_user_session")
        st.rerun()


# ---------------------------------------------------------
# MAIN BANNER
# ---------------------------------------------------------
raw_total = agent.raw_vector_db.index.ntotal if agent.raw_vector_db else 0
sum_total = agent.summary_vector_db.index.ntotal if agent.summary_vector_db else 0

st.markdown(f"""
<div class="header-box">
    <div class="header-title">📈 FinAgent AI — Autonomous Financial Desk Terminal</div>
    <div class="header-subtitle">Multi-Desk Agentic Architecture with Annual Report RAG, Ridge ML Forecasting & Real-Time Market Lookup</div>
    <div style="margin-top: 10px;">
        <span class="badge-pill badge-blue">⚡ LLM: {st.session_state.groq_model}</span>
        <span class="badge-pill badge-green">🧠 Device: {agent.device.upper()} Accelerated</span>
        <span class="badge-pill badge-purple">📚 RAG Chunks: {raw_total:,}</span>
        <span class="badge-pill badge-amber">🏢 Corporate Summaries: {sum_total}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# MAIN INTERFACE TABS
# ---------------------------------------------------------
tab_chat, tab_ml, tab_fundamentals, tab_ensemble, tab_knowledge = st.tabs([
    "💬 Interactive Agent Desk",
    "📈 ML Predictive Studio",
    "🏢 Company Profiles & Annual Reports",
    "📊 Multi-Model Ensemble Benchmark",
    "📚 Vector Knowledge Base"
])


# =========================================================
# TAB 1: INTERACTIVE AGENT DESK (CHAT & CIO VERDICT)
# =========================================================
with tab_chat:
    st.markdown("#### 💬 **Autonomous 3-Desk Financial Intelligence Assistant**")
    st.caption("Ask questions about company financials, annual reports, future price predictions, or comparative valuations.")

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(f"**{message['content']}**")
            else:
                # Render Detailed Assistant Response
                res_data = message.get("data")
                if res_data:
                    # Collapsible Desk Execution Details
                    with st.expander(f"🔍 Agent Analytical Traces ({res_data.get('execution_time_sec', 0)}s)", expanded=False):
                        col_tr1, col_tr2, col_tr3 = st.columns(3)
                        with col_tr1:
                            st.markdown(f"**Intent Routes:** `{res_data.get('intent')}`")
                        with col_tr2:
                            st.markdown(f"**Extracted Tickers:** `{res_data.get('tickers')}`")
                        with col_tr3:
                            st.markdown(f"**Forecast Horizon:** `{res_data.get('forecast_days')} Days`")

                        # Desk 1: RAG Passages
                        st.markdown("##### 📚 Desk 1: Retrieved Document Excerpts")
                        docs = res_data.get("rag_raw_docs") or res_data.get("rag_summary_docs") or []
                        if docs:
                            for idx, d in enumerate(docs[:3]):
                                meta = d.metadata
                                st.markdown(f"**Passage {idx+1}** `[Source: {meta.get('source')} | Entity: {meta.get('company')} ({meta.get('year')}) | Page {meta.get('page')}]`")
                                st.caption(d.page_content[:350] + "...")
                        else:
                            st.info("No explicit PDF document passages triggered.")

                        # Desk 2: ML Model Forecasts
                        st.markdown("##### 📊 Desk 2: Statistical Machine Learning Signals")
                        ml_data = res_data.get("ml_predictions", {})
                        if ml_data:
                            ml_rows = []
                            for tick, info in ml_data.items():
                                ml_rows.append({
                                    "Ticker": tick,
                                    "Current Price": f"₹{info.get('Current_Price')}",
                                    "Predicted Price": f"₹{info.get('Predicted_Price')}",
                                    "Expected Return": f"{info.get('Expected_Return_Pct'):+0.2f}%",
                                    "Model R² Score": info.get('Model_R2_Score'),
                                    "Horizon": f"{info.get('Horizon_Days')}d"
                                })
                            st.dataframe(pd.DataFrame(ml_rows), use_container_width=True)
                        else:
                            st.info("No quantitative ML forecasting requested in this query.")

                        # Desk 3: Real-Time Market Quotes
                        st.markdown("##### 💹 Desk 3: Real-Time Yahoo Finance Desk")
                        mq = res_data.get("market_quotes", {})
                        if mq:
                            for tick, q in mq.items():
                                st.markdown(f"• **{tick}**: {q.get('formatted_summary')}")
                        else:
                            st.info("No real-time market quote triggered.")

                    # CIO Final Definitive Verdict
                    st.markdown(f"""
                    <div class="verdict-box">
                        <div class="verdict-header">🏛️ Chief Investment Officer Definitive Verdict</div>
                        <div>{res_data.get('verdict')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(message["content"])

    # Handle preset queries
    current_input = ""
    if st.session_state.preset_query:
        current_input = st.session_state.preset_query
        st.session_state.preset_query = ""

    # Chat Input Box
    user_input = st.chat_input("Enter your financial inquiry...", key="main_chat_input")
    query_to_run = user_input or (current_input if current_input else None)

    if query_to_run:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": query_to_run})
        with st.chat_message("user"):
            st.markdown(f"**{query_to_run}**")

        # Execute Pipeline
        with st.chat_message("assistant"):
            with st.status("🤖 Agent Orchestrator Running Across 3 Analytical Desks...", expanded=True) as status:
                st.write("🎯 Step 1: Parsing linguistic intent and mapping company tokens to tickers...")
                time.sleep(0.2)
                st.write("📚 Step 2: Retrieving deep context from FAISS Vector Stores (Annual Reports)...")
                time.sleep(0.2)
                st.write("📊 Step 3: Running Ridge ML Regression models & feature engineering...")
                time.sleep(0.2)
                st.write("💹 Step 4: Interfacing with Yahoo Finance Real-Time Market Desk...")
                time.sleep(0.2)
                st.write("🏛️ Step 5: Chief Investment Officer synthesizing cross-desk signals into verdict...")
                
                res = agent.run_agentic_pipeline(query_to_run, session_id="default_user_session")
                status.update(label=f"✅ Pipeline Completed in {res['execution_time_sec']}s", state="complete", expanded=False)

            # Display traces
            with st.expander(f"🔍 Agent Analytical Traces ({res['execution_time_sec']}s)", expanded=False):
                col_tr1, col_tr2, col_tr3 = st.columns(3)
                with col_tr1:
                    st.markdown(f"**Intent Routes:** `{res.get('intent')}`")
                with col_tr2:
                    st.markdown(f"**Extracted Tickers:** `{res.get('tickers')}`")
                with col_tr3:
                    st.markdown(f"**Forecast Horizon:** `{res.get('forecast_days')} Days`")

                docs = res.get("rag_raw_docs") or res.get("rag_summary_docs") or []
                if docs:
                    st.markdown("##### 📚 Desk 1: Retrieved Document Excerpts")
                    for idx, d in enumerate(docs[:3]):
                        meta = d.metadata
                        st.markdown(f"**Passage {idx+1}** `[Source: {meta.get('source')} | Entity: {meta.get('company')} ({meta.get('year')}) | Page {meta.get('page')}]`")
                        st.caption(d.page_content[:350] + "...")

                ml_data = res.get("ml_predictions", {})
                if ml_data:
                    st.markdown("##### 📊 Desk 2: Statistical Machine Learning Signals")
                    ml_rows = []
                    for tick, info in ml_data.items():
                        ml_rows.append({
                            "Ticker": tick,
                            "Current Price": f"₹{info.get('Current_Price')}",
                            "Predicted Price": f"₹{info.get('Predicted_Price')}",
                            "Expected Return": f"{info.get('Expected_Return_Pct'):+0.2f}%",
                            "Model R² Score": info.get('Model_R2_Score'),
                            "Horizon": f"{info.get('Horizon_Days')}d"
                        })
                    st.dataframe(pd.DataFrame(ml_rows), use_container_width=True)

                mq = res.get("market_quotes", {})
                if mq:
                    st.markdown("##### 💹 Desk 3: Real-Time Yahoo Finance Desk")
                    for tick, q in mq.items():
                        st.markdown(f"• **{tick}**: {q.get('formatted_summary')}")

            # Display CIO Verdict
            st.markdown(f"""
            <div class="verdict-box">
                <div class="verdict-header">🏛️ Chief Investment Officer Definitive Verdict</div>
                <div>{res.get('verdict')}</div>
            </div>
            """, unsafe_allow_html=True)

            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": res.get("verdict"), "data": res})


# =========================================================
# TAB 2: ML PREDICTIVE STUDIO
# =========================================================
with tab_ml:
    st.markdown("#### 📈 **Statistical ML Stock Price Forecasting Studio**")
    st.caption("Train standardized Ridge Regression models on historical price momentum, moving averages (SMA-5, SMA-20), and 5-day volatility.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
    with col_ctrl1:
        selected_ticker = st.selectbox(
            "Select Ticker Symbol",
            ["CIPLA.NS", "RELIANCE.NS", "HDFCAMC.NS", "GMRAIRPORT.NS", "JINDALSTEL.NS", "LT.NS", "TCS.NS", "Custom Ticker..."]
        )
        if selected_ticker == "Custom Ticker...":
            custom_t = st.text_input("Enter Yahoo Finance Ticker (e.g., INFY.NS, AAPL, MSFT):", value="INFY.NS")
            selected_ticker = custom_t.strip().upper()

    with col_ctrl2:
        forecast_horizon = st.slider("Prediction Horizon (Trading Days Ahead)", min_value=1, max_value=30, value=5)

    with col_ctrl3:
        st.write("")
        st.write("")
        run_ml_btn = st.button("🚀 Run ML Model", use_container_width=True)

    if run_ml_btn or selected_ticker:
        with st.spinner(f"Computing Ridge ML Regression model for {selected_ticker}..."):
            fc_res = agent.run_ml_forecast(selected_ticker, n_days_ahead=forecast_horizon)

        if "error" in fc_res:
            st.error(fc_res["error"])
        else:
            cur_p = fc_res["Current_Price"]
            pred_p = fc_res["Predicted_Price"]
            ret_pct = fc_res["Expected_Return_Pct"]
            r2 = fc_res["Model_R2_Score"]
            hist_df = fc_res["History_DF"]

            # Metric Cards
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Current Close</div>
                    <div class="metric-value">₹{cur_p:,.2f}</div>
                    <div class="metric-sub">As of {fc_res['Last_Updated']}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Price ({forecast_horizon}d)</div>
                    <div class="metric-value">₹{pred_p:,.2f}</div>
                    <div class="metric-sub">Ridge Forecast</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m3:
                ret_class = "positive" if ret_pct >= 0 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Expected Return</div>
                    <div class="metric-value {ret_class}">{ret_pct:+0.2f}%</div>
                    <div class="metric-sub">Projected Gain/Loss</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model R² Score</div>
                    <div class="metric-value">{r2:.4f}</div>
                    <div class="metric-sub">Test Fit Accuracy</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m5:
                vol = hist_df['Volatility_5'].iloc[-1] if 'Volatility_5' in hist_df.columns else 0.0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">5-Day Volatility</div>
                    <div class="metric-value">{vol:.4f}</div>
                    <div class="metric-sub">Rolling Std Dev</div>
                </div>
                """, unsafe_allow_html=True)

            # Interactive Plotly Chart
            st.markdown("##### 📊 Historical Candlestick & Forecast Trajectory")
            fig = go.Figure()

            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=hist_df.index,
                open=hist_df['Open'],
                high=hist_df['High'],
                low=hist_df['Low'],
                close=hist_df['Close'],
                name="Price History",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350"
            ))

            # Moving averages
            if 'SMA_5' in hist_df.columns:
                fig.add_trace(go.Scatter(
                    x=hist_df.index,
                    y=hist_df['SMA_5'],
                    mode='lines',
                    line=dict(color='#ff9800', width=1.5),
                    name='5-Day SMA'
                ))
            if 'SMA_20' in hist_df.columns:
                fig.add_trace(go.Scatter(
                    x=hist_df.index,
                    y=hist_df['SMA_20'],
                    mode='lines',
                    line=dict(color='#2196f3', width=1.5),
                    name='20-Day SMA'
                ))

            # Forecast point projection
            last_date = hist_df.index[-1]
            future_date = last_date + pd.Timedelta(days=forecast_horizon)
            fig.add_trace(go.Scatter(
                x=[last_date, future_date],
                y=[cur_p, pred_p],
                mode='lines+markers',
                line=dict(color='#ab47bc', width=2.5, dash='dash'),
                marker=dict(size=9, color='#ab47bc'),
                name=f'{forecast_horizon}-Day Projected Target'
            ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                xaxis_rangeslider_visible=False,
                height=520,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Technical Feature Coefficients
            st.markdown("##### ⚖️ Ridge Regression Feature Standardized Coefficients")
            weights = fc_res.get("Feature_Weights", {})
            w_df = pd.DataFrame(list(weights.items()), columns=["Feature", "Normalized Weight"])
            fig_bar = px.bar(
                w_df,
                x="Feature",
                y="Normalized Weight",
                color="Normalized Weight",
                color_continuous_scale="Tealgrn",
                title=f"Feature Weights Driving {selected_ticker} Forecast"
            )
            fig_bar.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22", height=320)
            st.plotly_chart(fig_bar, use_container_width=True)


# =========================================================
# TAB 3: COMPANY PROFILES & ANNUAL REPORTS
# =========================================================
with tab_fundamentals:
    st.markdown("#### 🏢 **Executive Profiles & Corporate Annual Report Fundamentals**")
    st.caption("Deep financial intelligence extracted from formal SEC/NSE corporate filings and annual reports.")

    available_keys = list(agent.company_summaries.keys())
    if not available_keys:
        st.warning("No company summaries found in `company_summaries.json`.")
    else:
        selected_profile_key = st.selectbox("Select Indexed Company Filing", available_keys)
        profile_content = agent.company_summaries.get(selected_profile_key, "")

        col_p1, col_p2 = st.columns([3, 2])
        with col_p1:
            st.markdown(f"### **Executive Financial Profile: {selected_profile_key}**")
            st.markdown(profile_content)

        with col_p2:
            st.markdown("### **Live Corporate Market Metrics**")
            # Map name to ticker
            sym_guess = "CIPLA.NS"
            for k, s in universe_tickers.items():
                if k.lower() in selected_profile_key.lower():
                    sym_guess = s
                    break

            st.info(f"Target Market Ticker: `{sym_guess}`")
            quote_data = agent.fetch_market_price_summary(sym_guess)
            if quote_data.get("current_price"):
                st.metric("Live Market Price", f"₹{quote_data['current_price']}", quote_data.get("formatted_summary"))
            
            # Show 5-day closes
            closes = quote_data.get("recent_closes", [])
            if closes:
                st.markdown("**Recent Trading Closes:**")
                st.dataframe(pd.DataFrame(closes), use_container_width=True)

            # Available PDFs in raw_pdfs
            st.markdown("---")
            st.markdown("##### 📁 Ingested Corporate PDFs in Library:")
            if os.path.exists(INPUT_FOLDER):
                pdfs = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".pdf")]
                for p in pdfs:
                    st.markdown(f"• 📄 `{p}`")


# =========================================================
# TAB 4: ENSEMBLE BENCHMARK DASHBOARD
# =========================================================
with tab_ensemble:
    st.markdown("#### 📊 **Multi-Model Predictive Ensemble Evaluation Benchmark**")
    st.caption("Evaluating comparative performance across Random Forest, XGBoost, LightGBM, Ridge Regression, and SVR.")

    if os.path.exists(ENSEMBLE_REPORT_FILE):
        ensemble_df = pd.read_csv(ENSEMBLE_REPORT_FILE)
        st.dataframe(ensemble_df, use_container_width=True)

        # Plot comparative R2 scores
        r2_cols = [c for c in ensemble_df.columns if "_R2" in c]
        if r2_cols:
            st.markdown("##### 🏆 Comparative Algorithm R² Goodness of Fit Scores")
            chart_df = ensemble_df.melt(id_vars=["Ticker"], value_vars=r2_cols, var_name="Algorithm", value_name="R2_Score")
            chart_df["Algorithm"] = chart_df["Algorithm"].str.replace("_R2", "")
            
            fig_ens = px.bar(
                chart_df,
                x="Ticker",
                y="R2_Score",
                color="Algorithm",
                barmode="group",
                title="Model Predictive Accuracy by Ticker (R²)",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_ens.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22", height=450)
            st.plotly_chart(fig_ens, use_container_width=True)

        # Download button
        csv_bytes = ensemble_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Ensemble Report (CSV)",
            data=csv_bytes,
            file_name="advanced_ensemble_report.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"Ensemble report file not found at: {ENSEMBLE_REPORT_FILE}")


# =========================================================
# TAB 5: VECTOR KNOWLEDGE BASE & PDF INGESTION
# =========================================================
with tab_knowledge:
    st.markdown("#### 📚 **FAISS Vector Knowledge Architecture & Ingestion**")
    st.caption("Inspect indexed paragraph blocks from financial statements and dynamically ingest new corporate PDFs.")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Raw Report Chunks</div>
            <div class="metric-value">{raw_total:,}</div>
            <div class="metric-sub">Recursive Split Chunks Indexed</div>
        </div>
        """, unsafe_allow_html=True)

    with col_v2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Summary Documents</div>
            <div class="metric-value">{sum_total:,}</div>
            <div class="metric-sub">Executive Profiles Indexed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🔎 **Live FAISS Semantic Search Tester**")
    search_query = st.text_input("Enter a query to test similarity search across corporate filings:", "What are the major business risks and EBITDA growth?")
    top_k = st.slider("Number of matching chunks to retrieve", min_value=1, max_value=10, value=3)

    if st.button("🔍 Execute Vector Search"):
        with st.spinner("Searching FAISS index..."):
            matched_docs = agent.retrieve_raw_documents(search_query, k=top_k)
        if matched_docs:
            for idx, doc in enumerate(matched_docs):
                meta = doc.metadata
                with st.expander(f"Chunk #{idx+1} — {meta.get('company', 'Unknown')} ({meta.get('year', 'N/A')}) | File: {meta.get('source')} | Page: {meta.get('page')}"):
                    st.markdown(doc.page_content)
        else:
            st.info("No matching chunks found.")

    st.markdown("---")
    st.markdown("##### 📥 **Ingest New Corporate Annual Report (PDF)**")
    uploaded_pdf = st.file_uploader("Upload a financial PDF report to index into FAISS", type=["pdf"])

    if uploaded_pdf is not None:
        if st.button("🚀 Ingest & Index Document"):
            save_path = os.path.join(INPUT_FOLDER, uploaded_pdf.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            st.success(f"Saved uploaded PDF to `{save_path}`.")
            
            with st.spinner("Parsing structural metadata, extracting executive profile, and updating FAISS vector stores..."):
                ingest_res = agent.ingest_new_pdf(save_path)
            
            if ingest_res.get("success"):
                st.success(f"✅ Ingestion Successful! Identified Entity: **{ingest_res.get('company_name')}** ({ingest_res.get('year')}). Added **{ingest_res.get('chunks_added')}** chunks across **{ingest_res.get('total_pages')}** pages!")
                st.cache_resource.clear()
            else:
                st.error(f"Failed ingestion: {ingest_res.get('error')}")
