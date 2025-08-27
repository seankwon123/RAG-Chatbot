# chatbot.py
"""
Streamlit chatbot interface for the Bitovi RAG system (cleaned)
- All routing/logic lives in rag_service.answer_question()
- Sidebar sample questions won't double-render
"""

import streamlit as st
from datetime import datetime
import logging
from typing import List, Dict, Any

from rag_service import RagService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Page ----------
st.set_page_config(
    page_title="Bitovi Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Init RAG ----------
@st.cache_resource
def init_rag_service():
    try:
        return RagService()
    except Exception as e:
        st.error(f"Failed to initialize RAG service: {e}")
        st.stop()

rag_service = init_rag_service()

# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []
if "process_question" not in st.session_state:
    st.session_state.process_question = None

# ---------- Query Runner (backend decides everything) ----------
def run_query(prompt: str) -> Dict[str, Any]:
    try:
        return rag_service.answer_question(prompt)
    except Exception as e:
        return {"answer": f"Sorry, I encountered an error: {str(e)}", "sources": []}

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🤖 BitoviBlogBot")

    st.markdown("## 💡 Sample Questions")
    sample_questions = [
        "What is Bitovi's latest blog post about?",
        "Show me all Bitovi articles about DevOps",
        "How many articles does Bitovi have about AI?",
        "What kind of tools does Bitovi recommend for E2E testing?",
        "What are best practices for React development?",
        "Show articles about cloud architecture",
        "What frontend frameworks does Bitovi use?",
        "Find articles about CI/CD pipelines",
    ]
    for i, q in enumerate(sample_questions):
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.process_question = q  # schedule processing
            st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # System status
    st.markdown("### 📊 System Status")
    try:
        latest = rag_service.latest_article()
        if latest:
            st.success("✅ System Connected")
            st.info(
                f"Latest article: {latest.published_date[:10] if latest.published_date else 'Unknown'}"
            )
        else:
            st.warning("⚠️ No articles found")
    except Exception:
        st.error("❌ System Error")

# ---------- Main ----------
st.title("Bitovi Knowledge Assistant")
st.markdown("Ask me anything about Bitovi's blog articles and technical content!")

# Process a scheduled sidebar question without duplicate rendering
if st.session_state.process_question:
    q = st.session_state.process_question
    st.session_state.process_question = None
    with st.spinner("Searching..."):
        result = run_query(q)
    # Append both messages; re-render via loop
    st.session_state.messages.append({"role": "user", "content": q, "ts": datetime.utcnow().isoformat()})
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "sources": result.get("sources", []),
        "ts": datetime.utcnow().isoformat(),
    })
    st.rerun()

# Render complete history once
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(f"- [{source['title']}]({source['url']})")

# Chat input
prompt = st.chat_input("Ask about Bitovi articles...")
if prompt:
    with st.spinner("Searching..."):
        result = run_query(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "ts": datetime.utcnow().isoformat()})
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "sources": result.get("sources", []),
        "ts": datetime.utcnow().isoformat(),
    })
    st.rerun()

# Help
with st.expander("ℹ️ How to use"):
    st.markdown("""
    Ask questions like:
    - **Latest Article**: “What's the latest Bitovi blog post?”
    - **Find All Articles**: “Show me all articles about DevOps”
    - **Count**: “How many articles about AI?”
    - **Topics**: “What tools for E2E testing?”
    - **General**: Any question about Bitovi's technical content

    The assistant searches across titles, excerpts, content, and tags, and includes sources for verification.
    """)
