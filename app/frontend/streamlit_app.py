import streamlit as st
import requests
from typing import Optional
import os
from io import BytesIO

# Configure the API endpoint
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")

class RAGFrontend:
    def __init__(self):
        st.set_page_config(
            page_title="RAG Document Q&A",
            page_icon="📚",
            layout="wide"
        )
        
        # Initialize session state
        if 'document_uploaded' not in st.session_state:
            st.session_state.document_uploaded = False

    def upload_pdf(self, file: BytesIO) -> Optional[dict]:
        """Upload a PDF file to the API."""
        try:
            files = {
                "file": (
                    file.name,
                    file,
                    "application/pdf"
                )
            }
            
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files=files
            )
            
            if response.status_code != 200:
                print(f"Error response: {response.text}")
                
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            return None

    def ask_question(self, question: str) -> Optional[dict]:
        """Send a question to the API."""
        try:
            response = requests.post(
                f"{API_BASE_URL}/question",
                json={"question": question}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error asking question: {str(e)}")
            return None

    def render_sidebar(self):
        """Render the sidebar with file upload."""
        with st.sidebar:
            st.header("Upload Document")
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            
            if uploaded_file is not None:
                if st.button("Process Document"):
                    with st.spinner("Processing document..."):
                        result = self.upload_pdf(uploaded_file)
                        if result:
                            st.success(f"Document processed successfully! Created {result['num_chunks']} chunks from {result['num_pages']} pages.")
                            st.session_state.document_uploaded = True

    def render_qa_interface(self):
        """Render the Q&A interface."""
        st.header("Ask Questions")
        
        if not st.session_state.document_uploaded:
            st.info("Please upload a document first using the sidebar.")
            return

        question = st.text_input("Enter your question:")
        
        if st.button("Ask") and question:
            with st.spinner("Getting answer..."):
                result = self.ask_question(question)
                if result:
                    # Display answer
                    st.markdown("### Answer")
                    st.write(result['answer'])
                    
                    # Display sources
                    with st.expander("View Sources"):
                        for i, source in enumerate(result['sources'], 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(source)

    def main(self):
        """Main application."""
        st.title("📚 Document Q&A System")
        
        # Render components
        self.render_sidebar()
        self.render_qa_interface()

if __name__ == "__main__":
    app = RAGFrontend()
    app.main()