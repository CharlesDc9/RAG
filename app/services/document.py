from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import uuid

class DocumentService:
    """Service for document processing."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    async def process_document(self, content: str) -> List[Document]:
        """Process document content into chunks."""
        document = Document(page_content=content, metadata={"id": str(uuid.uuid4())})
        return self.text_splitter.split_documents([document])