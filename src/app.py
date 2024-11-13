import streamlit as st
from pathlib import Path
from query_docs import DocumentQuerier
import os
import requests
import logging

class StreamlitApp:
    def __init__(self):
        st.set_page_config(
            page_title="Documentation Assistant",
            page_icon="📚",
            layout="wide"
        )
        
        self.ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')
        
        # Initialize querier only once using Streamlit's session state
        if 'querier' not in st.session_state:
            try:
                st.session_state.querier = DocumentQuerier(
                    docs_dir=Path("docs"),
                    collection_name="documentation"
                )
            except Exception as e:
                st.error(f"❌ Error initializing querier: {str(e)}")
                logging.error(f"Initialization error: {str(e)}")
                return
        
        self.querier = st.session_state.querier
                
        # Debug information
        with st.sidebar:
            st.write("System Status:")
            if self.test_ollama_connection():
                st.success("✅ Ollama Connected")
            else:
                st.error("❌ Ollama Connection Failed")
                
            if hasattr(self, 'querier') and self.querier.query_engine is not None:
                st.success("✅ Query Engine Ready")
            else:
                st.error("❌ Query Engine Not Initialized")
    
    def test_ollama_connection(self):
        """Test and display Ollama connection status"""
        with st.sidebar:
            try:
                # Try the /api/tags endpoint
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
                st.write(f"Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    st.success("✅ Connected to Ollama")
                    models = response.json()
                    st.write("Available models:", models)
                else:
                    st.error(f"❌ Error: Status {response.status_code}")
                    st.write("Response:", response.text)
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection Error: {str(e)}")
                
                # Try alternate URLs for debugging
                debug_urls = [
                    "http://localhost:11434",
                    "http://ollama:11434",
                    "http://host.docker.internal:11434"
                ]
                
                st.write("Trying alternate URLs:")
                for url in debug_urls:
                    try:
                        st.write(f"Testing {url}...")
                        r = requests.get(f"{url}/api/tags", timeout=2)
                        st.write(f"- {url}: {r.status_code}")
                    except requests.exceptions.RequestException as e2:
                        st.write(f"- {url}: Failed ({str(e2)})")
    
    def render(self):
        st.title("📚 Documentation Assistant")
        
        # Only show chat interface if everything is initialized
        if hasattr(self, 'querier') and self.querier.query_engine is not None:
            if prompt := st.chat_input("Ask about the documentation"):
                st.chat_message("user").write(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Searching documentation..."):
                        response = self.querier.query(prompt)
                        st.write(response)
        else:
            st.warning("⚠️ System not fully initialized. Please check the sidebar for status.")

def main():
    app = StreamlitApp()
    app.render()

if __name__ == "__main__":
    main() 