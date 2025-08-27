# chatbot.py
"""
Streamlit chatbot interface for the Bitovi RAG system
"""

import streamlit as st
from datetime import datetime
import logging
from typing import List, Dict, Any

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

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "auto"
if "process_question" not in st.session_state:
    st.session_state.process_question = None

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Search mode selection
    search_mode = st.selectbox(
        "Search Mode",
        ["auto", "latest", "all_matching", "count", "hybrid"],
        index=0,
        help="""
        - **auto**: Automatically determines best search type
        - **latest**: Find most recent article
        - **all_matching**: List all matching articles
        - **count**: Count articles about topic
        - **hybrid**: Combined semantic + keyword search
        """
    )
    st.session_state.search_mode = search_mode
    
    # Sample questions
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What is Bitovi's latest blog post about?",
        "Show me all Bitovi articles about DevOps",
        "How many articles does Bitovi have about AI?",
        "What kind of tools does Bitovi recommend for E2E testing?",
        "What are best practices for React development?",
        "Show articles about cloud architecture",
        "What frontend frameworks does Bitovi use?",
        "Find articles about CI/CD pipelines"
    ]
    
    for q in sample_questions:
        if st.button(q, key=f"sample_{q[:20]}"):
            # Add to messages and trigger processing
            st.session_state.messages.append({"role": "user", "content": q})
            st.session_state.process_question = q
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Stats section
    st.markdown("### 📊 System Status")
    rag_service = init_rag_service()
    
    try:
        # Get some basic stats
        latest = rag_service.latest_article()
        if latest:
            st.success("✅ System Connected")
            st.info(f"Latest article: {latest.published_date[:10] if latest.published_date else 'Unknown'}")
        else:
            st.warning("⚠️ No articles found")
    except:
        st.error("❌ System Error")

# Main chat interface
st.title("🤖 Bitovi Knowledge Assistant")
st.markdown("Ask me anything about Bitovi's blog articles and technical content!")

# Process question from sidebar button if needed
if st.session_state.process_question:
    question_to_process = st.session_state.process_question
    st.session_state.process_question = None  # Clear it
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question_to_process)
    
    # Process and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            rag_service = init_rag_service()
            
            try:
                result = rag_service.answer_question(question_to_process)
                st.markdown(result["answer"])
                
                # Store message with sources
                message_data = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", [])
                }
                st.session_state.messages.append(message_data)
                
                # Display sources
                if result.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in result["sources"]:
                            st.markdown(f"- [{source['title']}]({source['url']})")
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- [{source['title']}]({source['url']})")
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about Bitovi articles..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            rag_service = init_rag_service()
            
            try:
                # Process based on search mode
                if st.session_state.search_mode == "auto":
                    # Let the service decide based on the question
                    result = rag_service.answer_question(prompt)
                
                elif st.session_state.search_mode == "latest":
                    # Force latest article search
                    article = rag_service.latest_article()
                    if article:
                        answer = f"**Latest Article:** {article.title}\n\n"
                        if article.author:
                            answer += f"**Author:** {article.author}\n"
                        if article.published_date:
                            answer += f"**Published:** {article.published_date[:10]}\n"
                        answer += f"\n{article.excerpt}\n\n[Read more]({article.url})"
                        result = {
                            "answer": answer,
                            "sources": [{"title": article.title, "url": article.url}]
                        }
                    else:
                        result = {"answer": "No articles found.", "sources": []}
                
                elif st.session_state.search_mode == "all_matching":
                    # Extract topic from question
                    import re
                    topic_match = re.search(r"about (.+?)(\?|$)", prompt.lower())
                    topic = topic_match.group(1).strip() if topic_match else prompt
                    
                    articles = rag_service.list_by_topic_all(topic)
                    if articles:
                        answer = f"Found **{len(articles)} articles** matching '{topic}':\n\n"
                        for i, article in enumerate(articles[:20], 1):  # Limit display
                            answer += f"{i}. **{article.title}**\n"
                            answer += f"   📅 {article.published_date[:10] if article.published_date else 'No date'}"
                            answer += f" | 🔍 {article.match_reason}\n"
                            answer += f"   [Read more]({article.url})\n\n"
                        
                        if len(articles) > 20:
                            answer += f"\n... and {len(articles) - 20} more articles"
                        
                        result = {
                            "answer": answer,
                            "sources": [{"title": a.title, "url": a.url} for a in articles[:5]]
                        }
                    else:
                        result = {"answer": f"No articles found about '{topic}'.", "sources": []}
                
                elif st.session_state.search_mode == "count":
                    # Extract topic and count
                    import re
                    topic_match = re.search(r"about (.+?)(\?|$)", prompt.lower())
                    topic = topic_match.group(1).strip() if topic_match else prompt
                    
                    count = rag_service.count_about(topic)
                    result = {
                        "answer": f"Found approximately **{count} articles** related to '{topic}'.",
                        "sources": []
                    }
                
                else:  # hybrid
                    result = rag_service.hybrid_answer(prompt)
                
                # Display the answer
                st.markdown(result["answer"])
                
                # Store the complete message with sources
                message_data = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", [])
                }
                st.session_state.messages.append(message_data)
                
                # Display sources in an expander
                if result.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in result["sources"]:
                            st.markdown(f"- [{source['title']}]({source['url']})")
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

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

# Run the chatbot
if __name__ == "__main__":
    # This will be executed by: streamlit run chatbot.py
    pass