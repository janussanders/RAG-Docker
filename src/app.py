import streamlit as st
from pathlib import Path
from src.query_docs import DocumentQuerier

class StreamlitApp:
    def __init__(self):
        st.set_page_config(
            page_title="Documentation Assistant",
            page_icon="📚",
            layout="wide"
        )
        
        # Initialize the querier with device support
        if 'querier' not in st.session_state:
            st.session_state.querier = DocumentQuerier(
                docs_dir=Path("./docs"),
                collection_name="docs"
            )
            
            # Show device info in sidebar
            with st.sidebar:
                st.info(f"Using device: {st.session_state.querier.device}")
    
    def render(self):
        st.title("📚 Documentation Assistant")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the documentation"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching documentation..."):
                    # Use synchronous query instead of async
                    response = st.session_state.querier.query_sync(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    app = StreamlitApp()
    app.render()

if __name__ == "__main__":
    main() 