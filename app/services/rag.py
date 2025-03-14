from typing import List, Dict
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import TypedDict

class MessagesState(TypedDict):
    messages: List[dict]

class RAGEngine:
    """Core RAG engine that handles document processing and Q&A."""
    
    def __init__(self, api_key: str):
        # Initialize core components
        self.llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.vector_store = Chroma(embedding_function=self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        self.memory = MemorySaver()  # Added memory initialization
        
        # Create the retrieve tool
        self.retrieve_tool = tool(response_format="content_and_artifact")(self._retrieve)
        
        # Initialize the graph
        self._setup_graph()

    def _retrieve(self, query: str):
        """Search the database for information."""
        retrieved_docs = self.vector_store.similarity_search(
            query, 
            k=2  # Hardcoded value instead of using settings
        )
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def _setup_graph(self):
        """Setup the RAG processing graph."""
        graph_builder = StateGraph(MessagesState)
        
        # Create and add nodes using the tool instance
        tools = ToolNode([self.retrieve_tool])  # Using retrieve_tool instead of retrieve
        graph_builder.add_node(self.query_or_respond)
        graph_builder.add_node(tools)
        graph_builder.add_node(self.generate)

        # Configure flow
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        # Compile with memory
        self.graph = graph_builder.compile(checkpointer=self.memory)

    def query_or_respond(self, state: MessagesState):
        """Generate tool call for retrieval or respond."""
        system_message = SystemMessage(content=(
            "You are an assistant that MUST search the provided database before answering. "
            "ALWAYS use the 'retrieve' tool first to find relevant information. "
            "Do not rely on your general knowledge without checking the database first. "
            "If you can't find relevant information in the database, say so explicitly."
        ))
        
        messages = [system_message] + state["messages"]
        llm_with_tools = self.llm.bind_tools([self.retrieve_tool])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def generate(self, state: MessagesState):
        """Generate answer."""
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "You MUST base your answer FIRST on the following retrieved context and then on your knowledge. "
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            f"{docs_content}"
        )
        
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        response = self.llm.invoke(prompt)
        return {"messages": [response]}

    async def process_document(self, content: str) -> List[Document]:
        """Process and index a new document."""
        documents = self.text_splitter.split_documents(
            [Document(page_content=content)]
        )
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        return documents
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and normalizing line breaks."""
        # Replace multiple newlines with a single space
        cleaned = ' '.join(text.split())
        # Remove any leading/trailing whitespace
        cleaned = cleaned.strip()
        return cleaned

    async def get_answer(self, question: str, thread_id: str = "default") -> Dict:
        """Get answer for a question using memory."""
        config = {"configurable": {"thread_id": thread_id}}
        
        # Try to fetch thread history
        thread_history = self.memory.get(config)
        
        if thread_history:
            messages = thread_history["messages"] + [
                {"role": "user", "content": question}
            ]
        else:
            messages = [{"role": "user", "content": question}]
        
        # Execute graph and get final response
        responses = []
        for step in self.graph.stream(
            {"messages": messages},
            stream_mode="values",
            config=config
        ):
            last_message = step["messages"][-1]
            responses.append(last_message)
        
        final_answer = self._clean_text(responses[-1].content) if responses else "No response generated"

        # Return the final response
        return {
            "answer": final_answer,
            "thread_id": thread_id
        }