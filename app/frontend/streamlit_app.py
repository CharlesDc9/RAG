import streamlit as st
import requests
from loguru import logger
from typing import Optional
import os
from io import BytesIO
import traceback
import sys

# Configure Loguru
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Add stderr handler
logger.add("app.log", rotation="10 MB", level="DEBUG")  # Add file handler with rotation

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
        if 'error_logs' not in st.session_state:
            st.session_state.error_logs = []

    def _add_error_log(self, error_message: str):
        """Add an error message to the session state error logs."""
        if 'error_logs' not in st.session_state:
            st.session_state.error_logs = []
        st.session_state.error_logs.append(error_message)
        logger.error(error_message)

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
                error_content = response.json() if response.content else {"detail": "Unknown error"}
                error_msg = f"Server error ({response.status_code}): {error_content.get('detail', str(error_content))}"
                logger.error(f"Upload failed: {error_msg}")
                self._add_error_log(error_msg)
                return None
                
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error: {str(e)}"
            logger.exception(f"Upload failed: {error_msg}")
            self._add_error_log(error_msg)
            return None
        except Exception as e:
            error_msg = f"Error uploading file: {str(e)}"
            logger.exception(f"Upload failed: {error_msg}")
            self._add_error_log(error_msg)
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
            error_msg = f"Error asking question: {str(e)}"
            logger.exception(f"Question failed: {error_msg}")
            self._add_error_log(error_msg)
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
            
            with st.expander("Debug Logs"):
                if st.button("Clear Logs"):
                    st.session_state.error_logs = []
                
                if st.session_state.error_logs:
                    for log in st.session_state.error_logs:
                        st.error(log)
                else:
                    st.info("No errors logged")


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
                else:
                        st.error("Failed to get an answer. Check the debug logs for details.")


    def main(self):
        """Main application."""
        st.title("📚 Document Q&A System")
        logger.info("Application started")
        
        # Render components
        self.render_sidebar()
        self.render_qa_interface()

if __name__ == "__main__":
    try:
        app = RAGFrontend()
        app.main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error(traceback.format_exc())
        logger.critical(f"Application crashed: {str(e)}")
        logger.exception("Critical application error")