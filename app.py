import streamlit as st
import glob
import os
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.converters import PyPDFToDocument
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
import tempfile
import uuid

# Initialize Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_store" not in st.session_state:
    st.session_state.document_store = InMemoryDocumentStore()
    st.session_state.pipeline_initialized = False
    st.session_state.processed_files = []

# Function to process a PDF file
def process_pdf(file_path, file_name):
    try:
        # Initialize components
        document_embedder = SentenceTransformersDocumentEmbedder(
            model="BAAI/bge-small-en-v1.5")
        
        # Create indexing pipeline
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("converter", PyPDFToDocument())
        indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=2))
        indexing_pipeline.add_component("embedder", document_embedder)
        indexing_pipeline.add_component("writer", DocumentWriter(st.session_state.document_store))
        
        # Connect indexing pipeline components
        indexing_pipeline.connect("converter", "splitter")
        indexing_pipeline.connect("splitter", "embedder")
        indexing_pipeline.connect("embedder", "writer")
        
        # Run indexing pipeline
        indexing_pipeline.run({"converter": {"sources": [file_path], "meta": {"file_name": file_name}}})
        
        return True
    except Exception as e:
        st.error(f"Error processing {file_name}: {str(e)}")
        return False

# Streamlit UI
st.title("Multi-Document Q&A Chat")

# Sidebar for document upload and system setup
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload PDF Documents", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if new_files:
            with st.spinner(f"Processing {len(new_files)} new document(s)..."):
                for uploaded_file in new_files:
                    # Create a temporary file
                    temp_dir = tempfile.mkdtemp()
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    if process_pdf(temp_path, uploaded_file.name):
                        st.session_state.processed_files.append(uploaded_file.name)
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    os.rmdir(temp_dir)
            
            st.success(f"Processed {len(st.session_state.processed_files)} document(s) with {st.session_state.document_store.count_documents()} total chunks")
            
            # Initialize retrieval components if not already done
            if not st.session_state.pipeline_initialized:
                try:
                    # Text embedder and retrievers
                    text_embedder = SentenceTransformersTextEmbedder(model="BAAI/bge-small-en-v1.5")
                    embedding_retriever = InMemoryEmbeddingRetriever(st.session_state.document_store)
                    bm25_retriever = InMemoryBM25Retriever(st.session_state.document_store)
                    document_joiner = DocumentJoiner()
                    ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")
                    
                    # Set up prompt template with citation formatting
                    template = """
                    Act as a senior customer care executive and help users with their queries. Be polite and friendly. Answer the user's questions based on the below context:
                    
                    CONTEXT:
                    {% for document in documents %}
                        {{ document.content }} [Source: {{ document.meta.file_name }}, Page: {{ document.meta.page_number }}]
                    {% endfor %}
                    
                    Make sure to provide all the details. If the answer is not in the provided context just say, 'Answer is not available in the context'. Don't provide wrong information.
                    If the user asks for any external recommendation only provide information related to the documents you received.
                    If user asks you anything other than what's in the context, just say 'Sorry, I can't help you with that'.

                    Question: {{question}}

                    Please explain in detail with a crystal clear format.
                    """
                    
                    prompt_builder = PromptBuilder(template=template)
                    
                    # Set up Ollama Generator
                    generator = OllamaGenerator(model="llama3.2", url="http://localhost:11434")
                    
                    # Create retrieval pipeline
                    st.session_state.retrieval_pipeline = Pipeline()
                    st.session_state.retrieval_pipeline.add_component("text_embedder", text_embedder)
                    st.session_state.retrieval_pipeline.add_component("embedding_retriever", embedding_retriever)
                    st.session_state.retrieval_pipeline.add_component("bm25_retriever", bm25_retriever)
                    st.session_state.retrieval_pipeline.add_component("document_joiner", document_joiner)
                    st.session_state.retrieval_pipeline.add_component("ranker", ranker)
                    st.session_state.retrieval_pipeline.add_component("prompt_builder", prompt_builder)
                    st.session_state.retrieval_pipeline.add_component("llm", generator)
                    
                    # Connect pipeline components
                    st.session_state.retrieval_pipeline.connect("text_embedder", "embedding_retriever")
                    st.session_state.retrieval_pipeline.connect("bm25_retriever", "document_joiner")
                    st.session_state.retrieval_pipeline.connect("embedding_retriever", "document_joiner")
                    st.session_state.retrieval_pipeline.connect("document_joiner", "ranker")
                    st.session_state.retrieval_pipeline.connect("ranker", "prompt_builder.documents")
                    st.session_state.retrieval_pipeline.connect("prompt_builder", "llm")

                    # Additional pipeline for citation extraction
                    text_embedder2 = SentenceTransformersTextEmbedder(model="BAAI/bge-small-en-v1.5")
                    embedding_retriever2 = InMemoryEmbeddingRetriever(st.session_state.document_store)
                    bm25_retriever2 = InMemoryBM25Retriever(st.session_state.document_store)
                    document_joiner2 = DocumentJoiner()
                    ranker2 = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")

                    st.session_state.citation_pipeline = Pipeline()
                    st.session_state.citation_pipeline.add_component("text_embedder", text_embedder2)
                    st.session_state.citation_pipeline.add_component("embedding_retriever", embedding_retriever2)
                    st.session_state.citation_pipeline.add_component("bm25_retriever", bm25_retriever2)
                    st.session_state.citation_pipeline.add_component("document_joiner", document_joiner2)
                    st.session_state.citation_pipeline.add_component("ranker", ranker2)

                    st.session_state.citation_pipeline.connect("text_embedder", "embedding_retriever")
                    st.session_state.citation_pipeline.connect("bm25_retriever", "document_joiner")
                    st.session_state.citation_pipeline.connect("embedding_retriever", "document_joiner")
                    st.session_state.citation_pipeline.connect("document_joiner", "ranker")
                    
                    st.session_state.pipeline_initialized = True
                    
                except Exception as e:
                    st.error(f"Error initializing pipelines: {str(e)}")
    
    # Display processed files
    if st.session_state.processed_files:
        st.header("Processed Documents")
        for file_name in st.session_state.processed_files:
            st.write(f"- {file_name}")
    
    # Reset button
    if st.button("Reset All"):
        st.session_state.document_store = InMemoryDocumentStore()
        st.session_state.pipeline_initialized = False
        st.session_state.processed_files = []
        st.session_state.messages = []
        st.success("All data has been reset!")

# Citation format options
citation_format = st.sidebar.selectbox(
    "Citation Format",
    ["Brief (Filename, Page)", "Detailed (Filename, Page, Score)"],
    index=0
)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response if pipeline is initialized and documents are loaded
    if st.session_state.pipeline_initialized and st.session_state.processed_files:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Run the retrieval pipeline
                    result = st.session_state.retrieval_pipeline.run(
                        {
                            "text_embedder": {"text": prompt},
                            "bm25_retriever": {"query": prompt},
                            "ranker": {"query": prompt},
                            "prompt_builder": {"question": prompt}
                        }
                    )
                    
                    # Get citation information
                    citation_result = st.session_state.citation_pipeline.run(
                        {
                            "text_embedder": {"text": prompt},
                            "bm25_retriever": {"query": prompt},
                            "ranker": {"query": prompt}
                        }
                    )
                    
                    # Extract the response
                    response = result['llm']['replies'][0]
                    
                    # Format citations
                    citations = []
                    seen_citations = set()  # To avoid duplicates
                    
                    for doc in citation_result['ranker']['documents'][:5]:  # Limit to top 5 sources
                        file_name = doc.meta.get('file_name', 'Unknown')
                        page_number = doc.meta.get('page_number', 'Unknown')
                        score = round(doc.score * 100, 1) if hasattr(doc, 'score') else None
                        
                        citation_key = f"{file_name}:{page_number}"
                        if citation_key not in seen_citations:
                            if citation_format == "Brief (Filename, Page)":
                                citations.append(f"[{file_name}, Page {page_number}]")
                            else:
                                citations.append(f"[{file_name}, Page {page_number}, Relevance: {score}%]")
                            seen_citations.add(citation_key)
                    
                    # Add citations to response
                    if citations:
                        citation_text = "\n\n**Sources:**\n" + "\n".join(citations)
                        response += citation_text
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    elif not st.session_state.processed_files:
        with st.chat_message("assistant"):
            message = "Please upload at least one document to start the conversation."
            st.warning(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
    else:
        with st.chat_message("assistant"):
            message = "The system is still initializing. Please wait a moment."
            st.info(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
