# chatbot.py
"""
Streamlit chatbot interface for the Bitovi RAG system
"""

import streamlit as st
from datetime import datetime
import logging
from typing import List, Dict, Any
import re

from rag_service import RagService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Bitovi Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize RAG service
@st.cache_resource
def init_rag_service():
    """Initialize the RAG service (cached to avoid reloading)"""
    try:
        service = RagService()
        return service
    except Exception as e:
        st.error(f"Failed to initialize RAG service: {e}")
        st.stop()

# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []
# if "search_mode" not in st.session_state:
st.session_state.search_mode = "auto"
if "process_question" not in st.session_state:
    st.session_state.process_question = None

rag_service = init_rag_service()

# ---------- helper to run a query according to current mode ----------
def run_query(prompt: str) -> Dict[str, Any]:
    mode = st.session_state.search_mode

    try:
        if mode == "auto":
            return rag_service.answer_question(prompt)

        elif mode == "latest":
            article = rag_service.latest_article()
            if article:
                answer = f"**Latest Article:** {article.title}\n\n"
                if article.author:
                    answer += f"**Author:** {article.author}\n"
                if article.published_date:
                    answer += f"**Published:** {article.published_date[:10]}\n"
                answer += f"\n{article.excerpt}\n\n[Read more]({article.url})"
                return {
                    "answer": answer,
                    "sources": [{"title": article.title, "url": article.url}],
                }
            return {"answer": "No articles found.", "sources": []}

        elif mode == "all_matching":
            topic_match = re.search(r"about (.+?)(\?|$)", prompt.lower())
            topic = topic_match.group(1).strip() if topic_match else prompt

            articles = rag_service.list_by_topic_all(topic)
            if articles:
                answer = f"Found **{len(articles)} articles** matching '{topic}':\n\n"
                for i, article in enumerate(articles[:20], 1):  # Limit display
                    pd = article.published_date[:10] if article.published_date else "No date"
                    answer += (
                        f"{i}. **{article.title}**\n"
                        f"   📅 {pd} | 🔍 {article.match_reason}\n"
                        f"   [Read more]({article.url})\n\n"
                    )
                if len(articles) > 20:
                    answer += f"\n… and {len(articles) - 20} more articles"
                return {
                    "answer": answer,
                    "sources": [{"title": a.title, "url": a.url} for a in articles[:5]],
                }
            return {"answer": f"No articles found about '{topic}'.", "sources": []}

        elif mode == "count":
            topic_match = re.search(r"about (.+?)(\?|$)", prompt.lower())
            topic = topic_match.group(1).strip() if topic_match else prompt
            count = rag_service.count_about(topic)
            return {
                "answer": f"Found approximately **{count} articles** related to '{topic}'.",
                "sources": [],
            }

        else:  # hybrid
            return rag_service.hybrid_answer(prompt)

    except Exception as e:
        return {"answer": f"Sorry, I encountered an error: {str(e)}", "sources": []}

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🤖 BitoviBlogBot")

    # Sample questions
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
            # Do NOT render here. Just schedule the processing.
            st.session_state.process_question = q
            st.rerun()

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Stats section
    st.markdown("### 📊 System Status")
    try:
        latest = rag_service.latest_article()
        if latest:
            st.success("✅ System Connected")
            st.info(f"Latest article: {latest.published_date[:10] if latest.published_date else 'Unknown'}")
        else:
            st.warning("⚠️ No articles found")
    except Exception:
        st.error("❌ System Error")

# ---------- Main ----------
st.title("Bitovi Knowledge Assistant")
st.markdown("Ask me anything about Bitovi's blog articles and technical content!")

# If there is a scheduled question (from sidebar), process it now without rendering duplicates.
if st.session_state.process_question:
    q = st.session_state.process_question
    st.session_state.process_question = None
    with st.spinner("Searching..."):
        result = run_query(q)
    # Append both messages to history; do not render here to avoid duplication
    st.session_state.messages.append({"role": "user", "content": q})
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "sources": result.get("sources", []),
        "ts": datetime.utcnow().isoformat(),
    })
    st.rerun()

# Render history exactly once
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
    # Store to history, then rerun so it renders once in the loop above
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "sources": result.get("sources", []),
        "ts": datetime.utcnow().isoformat(),
    })
    st.rerun()

# Footer with instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    ### Search Capabilities

    This assistant can help you find Bitovi blog articles in several ways:

    1. **Latest Article**: Ask "What's the latest Bitovi blog post?"
    2. **Find All Articles**: Ask "Show me all articles about DevOps"
    3. **Count Articles**: Ask "How many articles about AI?"
    4. **Specific Topics**: Ask "What tools for E2E testing?"
    5. **General Questions**: Ask anything about Bitovi's technical content

    ### Search Modes

    - **Auto Mode** (default): Automatically determines the best search type
    - **Latest**: Always returns the most recent article
    - **All Matching**: Lists all articles matching your topic
    - **Count**: Counts articles about a topic
    - **Hybrid**: Uses both semantic and keyword search

    ### Tips

    - Click sample questions in the sidebar to try them
    - The system searches across titles, excerpts, content, and tags
    - Sources are provided for verification
    """)
