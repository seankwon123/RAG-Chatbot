import streamlit as st
import time
from rag_system import RAGSystem
import logging

# Configure page
st.set_page_config(
    page_title="Article Knowledge Assistant",
    page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScbAsZzxakZboWw8WrzNIPiQ9zvtXjkSckXg&s",
    layout="wide"
)

# Configure logging to reduce noise
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system (cached to avoid reloading)"""
    return RAGSystem()

def display_sources(sources):
    """Display source articles in a nice format"""
    if sources:
        st.subheader("Sources:")
        for i, source in enumerate(sources, 1):
            # Include publication date in the title if available
            title = source['title']
            published_date = source.get('published_date')
            if published_date and published_date != 'Unknown' and published_date.strip():
                # Extract just the date part from timestamp
                clean_date = published_date.split(' ')[0] if ' ' in published_date else published_date
                title += f" ({clean_date})"
            
            with st.expander(f"{i}. {title} (Score: {source['score']})"):
                st.write(f"**Author:** {source['author']}")
                if published_date and published_date != 'Unknown':
                    clean_date = published_date.split(' ')[0] if ' ' in published_date else published_date
                    st.write(f"**Published:** {clean_date}")
                else:
                    st.write(f"**Published:** Unknown")
                if source['url']:
                    st.write(f"**URL:** {source['url']}")
                else:
                    st.write("**URL:** Not available")

def main():
    """Main chatbot interface"""
    st.title("Article Knowledge Assistant")
    st.write("Ask questions about the articles in our knowledge base!")
    
    # Initialize RAG system
    try:
        with st.spinner("Initializing knowledge base..."):
            rag_system = initialize_rag_system()
        st.success("Knowledge base loaded! Ready to answer questions.")
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.stop()
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                display_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about the articles..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching articles and generating response..."):
                # Get response from RAG system
                result = rag_system.chat(prompt)
                
                if result['success']:
                    # Display response
                    st.write(result['response'])
                    
                    # Display sources
                    display_sources(result['sources'])
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['response'],
                        "sources": result['sources']
                    })
                else:
                    # Display error
                    st.error(result['response'])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {result['response']}"
                    })
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.write("""
        This chatbot can answer questions based on a knowledge base of 444 articles.
        
        **Features:**
        - Semantic search to find relevant articles
        - AI-generated responses based on article content
        - Source attribution for all answers
        
        **Tips:**
        - Ask specific questions for better results
        - Try questions about software development, AI, business strategy, etc.
        - Check the sources to read the full articles
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # Display some example questions
        st.header("Example Questions")
        example_questions = [
            "What is Bitovi's latest blog post about?",
            "Can you show me all Bitovi articles about DevOps?",
            "How many articles does Bitovi have about AI?",
            "What kind of tools does Bitovi recommend for E2E testing?",

            # "What are best practices for software development?",
            # "How can businesses implement AI strategies?",
            # "What tools are used for data pipelines?",
            # "Tell me about DevOps practices",
            # "What is design system methodology?",
            # "How to improve user experience design?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{hash(question)}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

if __name__ == "__main__":
    main()