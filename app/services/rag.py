from typing import List, Dict, Literal
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated, TypedDict
import hashlib
import logging
import os
import time

logger = logging.getLogger(__name__)

class Search(TypedDict):
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[Literal["beginning", "middle", "end"], ..., "Section to query."]

class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

class RAGEngine:
    """Simplified RAG engine that handles document processing and Q&A."""
    
    def __init__(self, api_key: str):
        # Initialize core components
        self.llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Ensure the persistence directory exists
        self.persist_directory = "./chroma_db"
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Store current collection
        self.current_collection = None
        self.vector_store = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        
        # Define prompt template
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you  don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.

        {context}

        Question: {question}

        Helpful Answer:"""
        self.prompt = PromptTemplate.from_template(template)
        
        # Initialize the graph
        self._setup_graph()

    def get_or_create_collection(self, name: str):
        """Get or create a collection with the given name."""
        # Clean the name to be valid for ChromaDB (alphanumeric and underscores only)
        clean_name = "".join(c if c.isalnum() else "_" for c in name)
        
        # Get or create the collection
        self.current_collection = self.chroma_client.get_or_create_collection(
            name=clean_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Update the vector store with the new collection
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=clean_name,
            embedding_function=self.embeddings
        )
        
        return self.current_collection

    def list_collections(self):
        """List all available collections."""
        return self.chroma_client.list_collections()

    async def process_document(self, content: str, filename: str, metadata: Dict = None) -> List[Document]:
        """Process and index a new document."""
        if metadata is None:
            metadata = {}
        
        # Create a collection for this document
        collection_name = os.path.splitext(filename)[0]  # Remove file extension
        self.get_or_create_collection(collection_name)
        
        # Add document identifier if not present
        if 'source' not in metadata:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            metadata['source'] = content_hash
        
        # Add filename to metadata
        metadata['filename'] = filename
        
        documents = self.text_splitter.split_documents(
            [Document(page_content=content, metadata=metadata)]
        )
        
        # Add section metadata for filtering
        total_documents = len(documents)
        third = total_documents // 3
        
        for i, document in enumerate(documents):
            document.metadata.update({
                "section": "beginning" if i < third else "middle" if i < 2 * third else "end",
                "chunk_index": i,
                "total_chunks": total_documents,
                "source": metadata['source'],
                "filename": filename
            })
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        return documents

    def get_collection_documents(self, collection_name: str) -> List[Dict]:
        """Get all documents from a specific collection."""
        collection = self.chroma_client.get_collection(collection_name)
        return collection.get() if collection else None

    async def get_answer(self, question: str, collection_name: str = None) -> Dict:
        """Get answer for a question from a specific collection or all collections."""
        try:
            if collection_name:
                self.get_or_create_collection(collection_name)
            elif not self.vector_store:
                # If no collection is specified and none is currently selected
                collections = self.list_collections()
                if not collections:
                    return {
                        "answer": "No document collections found. Please upload a document first.",
                        "context": []
                    }
                # Use the first available collection
                self.get_or_create_collection(collections[0])
            
            # Execute graph and get final response
            result = self.graph.invoke({
                "question": question,
                "collection": collection_name
            })
            
            return {
                "answer": result["answer"],
                "context": [doc.page_content for doc in result["context"]],
                "collection": collection_name or "default"
            }
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "context": [],
                "collection": collection_name
            }

    def analyze_query(self, state: State):
        """Analyze the query to determine search parameters."""
        structured_llm = self.llm.with_structured_output(Search)
        query = structured_llm.invoke(state["question"])
        return {"query": query}

    def retrieve(self, state: State):
        """Retrieve relevant documents based on the query."""
        query = state["query"]
        try:
            if not self.vector_store:
                logger.error("No collection selected for retrieval")
                return {"context": [Document(
                    page_content="Please select a collection before asking questions.",
                    metadata={"source": "error"}
                )]}

            # First try with section filter
            try:
                retrieved_docs = self.vector_store.similarity_search(
                    query["query"],
                    filter={"section": query["section"]},
                    k=3
                )
            except Exception as e:
                logger.warning(f"Error searching with section filter: {str(e)}")
                # Try without filter if filtering fails
                retrieved_docs = []

            # If no results, try without section filter
            if not retrieved_docs:
                logger.info("No results with section filter, trying without filter")
                retrieved_docs = self.vector_store.similarity_search(
                    query["query"],
                    k=3
                )
                
            # Still no results? Add a fallback document
            if not retrieved_docs:
                logger.warning("No results found for query")
                retrieved_docs = [Document(
                    page_content="No specific information found on this topic in the current collection.",
                    metadata={"source": "no_results"}
                )]
                
            return {"context": retrieved_docs}
        except Exception as e:
            logger.error(f"Error in retrieve: {str(e)}")
            return {"context": [Document(
                page_content="Unable to retrieve information due to a technical issue. Please ensure a document collection is selected.",
                metadata={"source": "error", "error": str(e)}
            )]}

    def generate(self, state: State):
        """Generate an answer based on retrieved documents."""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def _setup_graph(self):
        """Setup the simplified RAG processing graph."""
        graph_builder = StateGraph(State)
        graph_builder.add_node("analyze_query", self.analyze_query)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("generate", self.generate)
        
        # Configure flow
        graph_builder.add_edge(START, "analyze_query")
        graph_builder.add_edge("analyze_query", "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        
        # Compile the graph
        self.graph = graph_builder.compile()

    def get_all_documents(self) -> List[Dict]:
        """Get all documents stored in the vector store."""
        return self.vector_store.get()

    def get_document_by_metadata(self, metadata_filter: Dict) -> List[Dict]:
        """Get documents that match specific metadata."""
        return self.vector_store.get(where=metadata_filter)

    def delete_document(self, document_id: str) -> None:
        """Delete a specific document by its ID."""
        self.vector_store.delete(ids=[document_id])
        # Persistence is handled automatically by PersistentClient

    def delete_documents_by_metadata(self, metadata_filter: Dict) -> None:
        """Delete documents that match specific metadata."""
        matching_docs = self.vector_store.get(where=metadata_filter)
        if matching_docs and 'ids' in matching_docs:
            self.vector_store.delete(ids=matching_docs['ids'])
            # Persistence is handled automatically by PersistentClient